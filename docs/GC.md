# NewCormanLisp Garbage Collector — Design

*Status: implemented through start-bit bitmap. Last updated 2026-05-12.*

## What this is

This document is the spec for the NewCormanLisp GC. It is committed
ahead of code because retrofitting a GC onto a system designed without
one is one of the bigger rewrites you can take on, and we are not
doing that.

The design draws on Roger Corman's 32-bit Win32 GC
(`E:\CL\cormanlisp\CormanLispServer\src\Gc.cpp`) — generational
copying, forwarding pointers, per-page dirty tracking. We adopt the
strategy and leave behind the 1996-Win32-x86 implementation specifics:
no hand-written assembly, no `VirtualProtect`-based hardware write
barriers as the primary mechanism, no 32-bit pointer math.

The GC lives entirely in `ncl-runtime`, in pure Rust, with one OS
shim for page allocation. The simplicity rule and the
"FFI-is-a-feature-not-a-foundation" rule both apply.

## Generations and spaces

Two GC-managed generations plus one pinned static area.

```
                allocate
                   │
                   ▼
            ┌──────────────┐
   minor →  │    YOUNG     │     bump pointer; one semispace
            │   (semi)     │     ~16 MB initial, growable
            └──────┬───────┘
                   │ survivors copied
                   ▼ with forwarding pointer
            ┌──────────────┐
   major →  │     OLD      │     two semispaces, swap on full GC
            │   (semi A)   │     ~64 MB initial, growable
            └──────────────┘
            ┌──────────────┐
            │     OLD      │
            │   (semi B)   │
            └──────────────┘
            ┌──────────────┐
            │    STATIC    │     pinned, no GC, used for compiled
            │              │     code, the loaded image, the
            └──────────────┘     stdlib's interned constants
```

Why two generations and not three: modern hardware removes the
constraints (small caches, expensive tenured GC under
page-protection barriers) that drove three-generation
design. A 16 MB young heap fits in L3 on every machine we care
about, and a larger young heap kills more objects before promotion.
If profiling later shows that REPL session bindings are pinning the
old heap and triggering too many full GCs, we add an intermediate
generation — the promotion path is well-defined; this is not a
one-way door.

The static area is for things that never die: JIT-emitted machine
code, the stdlib's interned symbol table, the package registry.
It is allocated once and never moved. The GC walks pointers FROM the
static area into managed heaps (so card-marked) but never moves
anything within it.

## Pointer tagging

64-bit `Word` newtype. Low 3 bits classify; upper 61 bits are payload
or aligned pointer.

| Tag (binary) | Meaning | Payload |
| --- | --- | --- |
| `000` | fixnum | signed 61-bit integer (sign-extended `>> 3`) |
| `001` | cons | pointer to two-`Word` cell, no header |
| `010` | symbol | pointer to `Symbol` |
| `011` | vector / array | pointer to `Vector` |
| `100` | function / closure | pointer to `Function` |
| `101` | string | pointer to `String` |
| `110` | immediate | character / `T` / special markers, in upper bits |
| `111` | forward | GC-internal: this slot has been moved; upper bits are the new address |

`nil` is the bit pattern `0` exactly (fixnum-tagged zero), so
`(eq x nil)` is one `cmp x, 0`. Most CLs do this.

`forward` having its own tag means the GC scanner can detect a stale
slot in a single mask-and-compare. Roger had this; we keep it.

Heap pointers are 8-byte aligned. Untagging a pointer is `word & ~7`
or `word - tag` (equivalent for any single tag), which the compiler
lifts to a constant offset in the load instruction.

## Heap object layout

### Cons cells — headerless

```
+-----+-----+
| car | cdr |    16 bytes total, no header
+-----+-----+
```

Cons cells are by far the most common heap object. They carry no
header — the type is known from the tag of the pointer that reached
them. This costs us one fact: a cons cell can't be distinguished
from an arbitrary 2-word array. The GC scanner handles this by
discriminating at the *pointer* level, not the *object* level: if
you arrive via a `001` tag, you scan two words; if you arrive via a
`011`-tagged vector pointer, you read the header first.

Roger does this. We do this. The space saving is real.

### Everything else — 8-byte header

```
+--------+--------+--------+--------+--------+--------+--------+--------+
|  type  |        length (cells)       | gc bits |       padding         |
+--------+--------+--------+--------+--------+--------+--------+--------+
   5 bits      24 bits                   8 bits         27 bits
```

- **type** — discriminator (Symbol, Vector, Function, String, ...).
  5 bits gives us 32 type codes, plenty for the foreseeable.
- **length** — number of 8-byte cells *after* the header. 24 bits
  caps a single object at 128 MB which is fine; bigger and we use a
  pair of objects.
- **gc bits** — mark bit, age (which GC pass an object survived),
  pinned flag, has-finalizer flag, has-weak-ref flag. We define them
  as we need them.
- **padding** — reserved. Probably consumed by an extended length
  field or a class-pointer slot when CLOS lands.

### Symbol — fixed shape

```
Symbol {
    header:      HeapHeader,        // type=Symbol, length=8
    name:        Word,              // → string
    package:     Word,              // → package
    value:       AtomicWord,        // value cell
    function:    AtomicWord,        // function cell — defun atomically swaps
    plist:       Word,              // → property list
    flags:       Word,              // constant-p, special-p, exported-p, ...
    jump_cache:  AtomicWord,        // optimised dispatch slot (Phase 4+)
}
```

The function cell is `AtomicU64` underneath — a `defun` is a single
`store-release`, the dispatch path on a call is a `load-acquire`.
This is the load-bearing piece of the redefinition story
(MANIFESTO.md, "Note: function redefinition and dispatch").

### Function — fixed shape

```
Function {
    header:        HeapHeader,
    code:          *const u8,       // pointer into the static-code area
    arity:         Word,            // packed: required, optional, rest, keyword
    closure_env:   Word,            // → vector of captured cells, or nil
    name:          Word,            // → symbol or nil (for anonymous)
}
```

The code pointer points into the static area where the JIT emits
machine code. Static-area code is never relocated, so the pointer is
stable across GCs.

## Multi-threading

NewCormanLisp supports **multiple Lisp threads**, matching Corman.
Multi-threading is a design constraint, not an optimisation: a
20-core machine should be able to run 20 Lisp threads doing useful
work concurrently, and the modern hardware case for multi-threaded
Lisp is too strong to ignore. `(make-thread …)`, `mp:process-run-
function`, `with-mutex`, `with-synchronization` — all work.

Verified in upstream `cormanlisp/CormanLispServer/src/Gc.cpp`:
Corman's GC wraps every allocation in `EnterGCCriticalSection`,
keeps per-thread state via `TlsGetValue(Thread_Index)`, and runs a
true stop-the-world by suspending and scanning every thread. We
adopt the same shape with three modernisations:

1. **TLABs (thread-local allocation buffers)** instead of a global
   critical section on every allocation. Each Lisp thread has its
   own slab of young-heap memory and bump-allocates within it
   without locks. Slab refill is the only synchronised operation
   and happens once per ~512 KB of allocation.
2. **Cooperative stop-the-world** instead of preemptive thread
   suspension. The GC sets a global stop flag; each mutator polls
   the flag at safe points and parks itself voluntarily. The GC
   runs after all mutators have parked, signals when done, mutators
   unpark. No `SuspendThread`, no `pthread_kill`, no signal handlers
   walking interrupted stacks — much simpler and Mac-portable.
3. **`AtomicU64` symbol cells** for atomic `defun`. The function
   cell is a single store-release; dispatch is a load-acquire. This
   was already committed in our redefinition design and pays off
   immediately under multi-threading.

The thread-boundary rule still holds: OS-specific code lives only
in the iGui shim. The UI thread remains distinct from all Lisp
threads. What changes is the count of Lisp threads (was: 1; is now:
N). The iGui mailbox is still the only path to/from the UI thread.

## Allocation

Each Lisp thread holds a `MutatorState` containing a TLAB. The TLAB
is a contiguous slab of young-heap cells with its own bump pointer.
Allocation is inline:

```
fn alloc(mutator: &mut MutatorState, size: usize) -> *mut HeapHeader {
    let new_top = mutator.tlab.top + size;
    if new_top > mutator.tlab.limit {
        return alloc_slow(mutator, size);   // refill TLAB or GC
    }
    let p = mutator.tlab.base.add(mutator.tlab.top);
    mutator.tlab.top = new_top;
    p as *mut HeapHeader
}
```

No atomics on the fast path. Each thread bumps its own TLAB pointer.
The slow path acquires the global heap lock to either refill the
TLAB from young or trigger a GC.

TLAB refill is atomic CAS on the global young bump pointer: each
thread grabs (say) 512 KB at a time. When young can't satisfy a
refill, the requesting thread initiates stop-the-world and the GC
runs after all mutators park.

Cons-cell allocation is the hot of the hot path. The compiler
recognises the pattern and emits a specialised inline allocation
that skips the header write (because cons cells have no header).
TLAB-bump is identical for cons; the size is just 2 cells.

## Young start-bit bitmap

Alongside young's cell array, we keep a packed per-cell bitmap —
2 bits per cell, 32 cells per `AtomicU64`, ~3.125% memory overhead.
For cell index `c`, the pair lives at bit positions `(c % 32) * 2`
and `(c % 32) * 2 + 1` within word `c / 32`. Encoding:

```
00 = not a start (the canonical "free / unused" state; abandoned
     TLAB tails and post-GC dead zones are just runs of 00 pairs,
     invisible to bitmap-driven walkers)
01 = header-bearing object start (length lives in the cell at idx)
11 = cons start (2 cells: car at idx, cdr at idx+1)
10 = reserved (candidate uses: forwarded-source marker, pinned-
     header fast-skip, opaque-payload "skip scan" hint — left
     unused until a real use case bites; reach for this code
     before adding another HeapType)
```

Every young allocation path sets the appropriate pair (single
`fetch_or` with `Relaxed` ordering; header = `0b01 << offset`,
cons = `0b11 << offset`). Walkers iterate set start-bits via the
mask `0x5555...` and read the adjacent odd bit to classify each
visit as cons-or-header.

### Why we need it on young

Young is a mixed-format heap: 2-cell headerless cons cells live
right next to header-bearing objects. The structural GC walkers
(`count_pinned`, `rewind_past_pinned`, `clear_pinned_bits`) need
to know where each object begins so they can check the Pinned bit
and advance by `1 + length_cells`. Without a bitmap, the only
way to "guess" object starts is to decode each cell's low 5 bits
as a HeapType — but a cons-car's value will coincidentally decode
to a valid HeapType ~31% of the time, and bits 5..29 of that car
become a fake length that sends the walker stampeding past `top`.
This is exactly the bug we hit at frame 29 of the bouncing demo
on 2026-05-11; the panic-on-suspect-walk diagnostic is preserved
in `Semispace::dump_and_panic` for future walker bugs.

### Why old does NOT need it

A symmetric question with a satisfying answer: **old has no
walker that needs to find object starts.**

Three phases touch old. None of them care about object boundaries:

1. **Minor GC, card-table scan over old.live.** Iterates *every
   cell* in each dirty card, treats it as a candidate `Word`, and
   calls `copy_into`. `copy_into` filters by tag-bits and by
   "points into young", so false positives are impossible — a
   non-pointer Word can't decode as a valid young pointer. It
   doesn't matter whether cell `i` is a header, a car, a cdr, or
   interior payload: it's all just "slots to inspect."

2. **Full GC, queue-driven Cheney copy.** When an object is
   copied into `dest`, `copy_into` pushes a `CopiedObject {
   to_offset, size, is_cons }` onto a queue. `scan_to_completion`
   drains the queue, iterating each object's payload using the
   queue entry's `size` and `is_cons` — never reading the heap
   structurally to recover those.

3. **Pin scan.** Doesn't run on old at all. Pinning exists to
   tolerate conservative stack scans — JIT'd frames hold tagged
   words that might or might not be live pointers, and we pin
   candidates rather than risk moving an object referenced by a
   stale slot. *Old objects never move during minor GC*; they
   only move during full GC, which has precise roots and no
   conservative pass. So no pinning, no Pinned-bit walkers, no
   need to know object starts.

The architecture is symmetric — `Semispace` already has the
bitmap field — we just don't populate or consult it on the old
semispaces. If we ever add a phase that walks old structurally
(a real **sweeper** for mark-sweep, an **in-place compactor**,
**incremental marking** with per-header trace dispatch), the
fix is to start populating the bitmap on old's alloc paths and
extending the walker discipline. Until then, old gets to be the
simpler space.

## Minor GC: copying with forwarding pointers

When the young heap fills:

1. **Stop the world.** The thread that detected the OOM sets the
   coordinator's stop flag. Every other Lisp thread polls the flag
   at its next safe point and parks (saves its own context, signals
   the coordinator, waits on a condvar). The GC runs only once all
   mutators have parked.
2. **Find roots.** Stack roots come from LLVM `gc.statepoint`-emitted
   stack maps **for each parked mutator** — N stacks, N stack-map
   walks. Old-heap-into-young-heap roots come from the dirty card
   table (see write barrier below). Static-into-young roots come
   from a separate set kept by the static-area writer. Global roots
   come from a registered global table (the package registry,
   `*features*`, etc.).
3. **Copy survivors.** Each live young object is copied to the old
   heap. The young location is overwritten with a `forward`-tagged
   pointer to the new location. References to the original location
   are updated as they're encountered. Each mutator's TLAB is
   refilled (as a bump-pointer slab on the empty young).
4. **Reset.** The young heap's bump pointer is reset; the dirty
   card table is cleared.
5. **Resume.** The coordinator clears the stop flag and signals the
   condvar; mutators unpark and continue.

A young object that survives one minor GC is promoted directly to
old. With a 16 MB young heap on modern hardware, the
surviving-but-still-shortlived rate is low enough that immediate
promotion (no intermediate generation) is fine. If profiling
shows otherwise, we add an intermediate generation later.

## Major GC: full collection

Triggered when the old heap fills, or on explicit request
(`(room t)`, scheduled idle GC, etc.):

1. Stop the world (cooperative, as in minor GC).
2. Treat young as roots into old (along with the static-area roots,
   global roots, and per-mutator stack roots from every parked
   thread).
3. Copy live old-heap objects from semi-A to semi-B.
4. Swap A and B; the old A becomes the next "to" space.
5. Reset card table; refill mutator TLABs from the empty young; resume.

The static area participates only as a root source. Code in the
static area that's no longer reachable is *not* collected by the
GC — it's tracked by the loader's retirement-with-quiescent-epoch
mechanism instead (see MANIFESTO.md, "The loader"). The two
mechanisms are deliberately separated.

## Write barrier — software card marking

Card-marking, cards of 512 bytes, one byte per card.

When compiled code stores a `Word` into a heap object whose age is
older-than-young (i.e. into an old-heap object or the static area),
it writes a single byte into the card table at
`(card_table_base + (object_addr - heap_base) / 512)`. No branch
on whether the new value is a young pointer — the bit can be wrong
(false positives are fine; the next minor GC will discover the card
holds no young pointers and clear the bit). False negatives are
not allowed.

The young heap has no barrier — anything in young can point
anywhere; minor GC scans it all.

The static area also has a card table. Pointers from the static
area into managed heaps are normally rare (the static area is for
code and immutable constants), but we don't want to forbid them
outright — interned static symbols may legitimately point at
constants in the heap. Cards work the same way.

Hardware-assisted barriers (page protection on the old heap) are
out of scope for v1. Roger had them off by default for stability
reasons; we honour that lesson.

## Root finding — precise via LLVM `gc.statepoint`

The compiler emits `gc.statepoint` intrinsics at every safe point
in JIT'd code. LLVM produces a stack map describing, for each
safe point, which stack slots and registers hold tagged Lisp
values.

At GC time, the runtime walks frames bottom-up using the stack map,
visits each live `Word`, follows `Word`-typed slots transitively,
and updates them in place if forwarded.

A conservative fallback exists for code paths where stack maps
aren't yet available (early Phase 3 bring-up, certain non-JIT'd
helper code). The fallback treats every i64-aligned stack word that
looks like a tagged pointer as one. This is correct (we never miss
a root) but pins memory (we can't move things that conservative
roots point at). The plan is to eliminate the fallback by the time
`defun` lands.

The Lisp thread carries a single per-thread "GC roots" pointer for
the in-flight allocation receiver and similar runtime-internal
roots. Globals (package registry, *features*, dispatch tables) are
registered in a `RootSet` walked at every GC.

## Cooperation: safe points

Every Lisp thread reaches a "safe point" wherever the GC may run.
These are:

- Allocation slow paths (TLAB refill — checks the stop flag)
- Function call boundaries (compiler emits a flag check)
- Loop back-edges (compiler emits a flag check)
- Explicit `(gc)` calls
- Explicit `safepoint()` runtime calls (used by hand-written
  runtime helpers and tests)

The check is cheap: one atomic load of the stop flag and a branch.
Hot loops pay one extra load per iteration; this is the cost of
admission for multi-threading.

When a thread sees the stop flag set, it **parks**: it saves its
register/stack context (or hands itself off via `gc.statepoint` for
precise scanning), increments the parked counter, and waits on a
condvar. The thread that triggered the GC waits until the parked
counter equals the live-mutator count, then runs the GC. After the
GC completes, the trigger thread clears the stop flag and signals
the condvar; all parked threads wake up, refill their TLABs from
the (now empty) young heap, and resume.

This is **cooperative** stop-the-world: every thread voluntarily
parks itself. We never use OS-level `SuspendThread` / signals /
preemptive interrupts. Cooperative is simpler, more portable
(Windows and macOS use the same code path), and has well-understood
latency bounds (the longest pause is the worst-case time between
two safe points in any thread, which the compiler's safe-point
insertion policy controls).

The thread-boundary rule still holds: the UI thread is NOT a Lisp
thread and never participates in stop-the-world. Lisp threads talk
to the UI thread through the iGui mailbox. Only Lisp threads
allocate, only Lisp threads run JIT'd code, only Lisp threads carry
GC roots.

## What's deliberately not in v1

- **Concurrent GC.** Stop-the-world only. Concurrent collectors
  (mutators run during marking) are an order of magnitude more
  complex and we don't need the latency wins they buy. With small
  young heaps and TLABs, stop-the-world pause times are measured
  in milliseconds.
- **Hardware-assisted write barriers.** Software card-marking only.
  Adding HW barriers later is an opt-in optimisation.
- **Generational beyond two generations.** We add an intermediate
  generation if profiling demands. Default is young + old.
- **Finalisers, weak refs.** They land when something forces them
  (CLOS metaclass with `finalize-instance`, weak hash tables).
  The GC bits include reserved space for the flags.
- **Compaction beyond what copying gives us.** Copying compacts as
  a side effect; that's enough.
- **Pluggable collectors.** One GC. Don't write a framework.

## The discipline — what NOT to do

- **No FFI inside the GC.** OS interaction is page allocation only,
  abstracted behind a `PageAllocator` trait with one impl per OS.
- **No `windows::*` / `core_foundation::*` outside the page
  allocator.** GC code is OS-independent Rust.
- **No bump-allocator-without-headers shortcut.** Every heap object
  carries the standard 8-byte header except cons. Don't add a
  second exception.
- **No "we'll add a write barrier later."** The card-table machinery
  lands in the first GC commit.
- **No conservative scanning of the heap itself.** The heap is
  precisely typed. Conservative scanning is only on stack frames
  whose stack maps don't exist yet, and only as a bring-up fallback.

## Build order

1. `Word` newtype + tag constants + immediate-encode/decode tests.
2. `HeapHeader` + a single managed semispace + bump allocator.
3. Forwarding-pointer-aware copy.
4. Add the second old-semispace; full GC swaps A↔B.
5. **`MutatorState` + TLAB + cooperative stop-the-world coordinator.**
   The single-threaded `Heap` becomes the global heap behind a lock;
   each mutator pulls TLABs from it; safe-point polling drives
   park/unpark. Verified with multiple OS threads allocating
   concurrently.
6. Card table + write barrier; minor GC scans dirty cards.
7. Static area (pinned, separate page allocator).
8. Symbol layout with `AtomicU64` cells, allocation helpers.
9. Stack-map root walking via `gc.statepoint` (drives the compiler
   side too — needs Phase 3).
