# NewCormanLisp Garbage Collector — Design

*Status: agreed, pre-implementation. Last updated 2026-05-10.*

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
page-protection barriers) that drove Roger's three-generation
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

## Allocation

The young heap is one semispace with a bump pointer. Allocation is
inline:

```
fn alloc(size: usize) -> *mut HeapHeader {
    let new_top = young.top - size;        // grows downward
    if new_top < young.limit {
        return alloc_slow(size);            // triggers minor GC
    }
    young.top = new_top;
    new_top as *mut HeapHeader
}
```

The compiler emits this fast path inline at every allocation site.
Slow path triggers a minor GC, retries, and on persistent failure
grows the young heap or aborts.

Allocation is on the Lisp thread only. The Lisp thread owns the
heap; other threads talk to it through the iGui mailbox. This
matches the thread-boundary rule and removes the cost of atomic
ops on the bump pointer.

Cons-cell allocation is the hot of the hot path. The compiler
recognises the pattern and emits a specialised inline allocation
that skips the header write (because cons cells have no header).

## Minor GC: copying with forwarding pointers

When the young heap fills:

1. **Suspend Lisp thread** at a safe point. Compiler-emitted safe
   points appear at every back-edge, every call, and every
   allocation slow path.
2. **Find roots.** Stack roots come from LLVM `gc.statepoint`-emitted
   stack maps. Old-heap-into-young-heap roots come from the dirty
   card table (see write barrier below). Static-into-young roots
   come from a separate set kept by the static-area writer. Global
   roots come from a registered global table (the package
   registry, `*features*`, etc.).
3. **Copy survivors.** Each live young object is copied to the old
   heap. The young location is overwritten with a `forward`-tagged
   pointer to the new location. References to the original location
   are updated as they're encountered.
4. **Reset.** The young heap's bump pointer is reset; the dirty
   card table is cleared.
5. **Resume.**

A young object that survives one minor GC is promoted directly to
old. With a 16 MB young heap on modern hardware, the
surviving-but-still-shortlived rate is low enough that immediate
promotion (no intermediate generation) is fine. If profiling
shows otherwise, we add an intermediate generation later.

## Major GC: full collection

Triggered when the old heap fills, or on explicit request
(`(room t)`, scheduled idle GC, etc.):

1. Suspend Lisp thread.
2. Treat young as roots into old (along with the static-area roots
   and global roots).
3. Copy live old-heap objects from semi-A to semi-B.
4. Swap A and B; the old A becomes the next "to" space.
5. Reset card table; resume.

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

The Lisp thread reaches a "safe point" wherever the GC may run.
These are:

- Allocation slow paths (we just triggered the GC; we're at one)
- Function call boundaries
- Loop back-edges
- Explicit `(gc)` calls

Compiled code doesn't poll for "should I pause for GC?" because
this Lisp thread *is* the only mutator. There are no other Lisp
threads to coordinate with. The GC runs synchronously when an
allocation can't be satisfied.

This is why the thread-boundary rule matters: the GC's life would
be much harder if other threads could be mutating Lisp values.
They can't.

## What's deliberately not in v1

- **Concurrent GC.** Lisp thread is single. Stop-the-world is fine
  because there's no world to stop except itself.
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
5. Card table + write barrier; minor GC scans dirty cards.
6. Static area (pinned, separate page allocator).
7. Symbol layout, allocation helpers.
8. Stack-map root walking via `gc.statepoint` (drives the compiler
   side too — needs Phase 3).
