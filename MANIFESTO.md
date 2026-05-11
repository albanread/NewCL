# NCL — Manifesto and Declaration of Intent

*Drafted 2026-05-10*

## What this is

NCL is a from-scratch reimplementation of the Corman Lisp
**language and user-facing experience**, not its codebase. It is a faithful
tribute that runs Corman Lisp programs and demos, while sharing
none of the original implementation.

We keep the language. We keep the demos. We keep the spirit — a Lisp that
is at home on Windows, that talks fluently to native code, that boots fast
and feels direct. We replace everything underneath.

## The original, and why we are leaving it behind

Roger Corman created the Lisp I could afford, and it was in many ways that
count a better Lisp than the ones that didnt even publish a price list.

Roger Corman's Corman Lisp (1996–2015, now maintained as
[sharplispers/cormanlisp](https://github.com/sharplispers/cormanlisp))
is a Win32 x86 Common Lisp implementation. Its compiler emits 32-bit x86
machine code directly, with no intermediate representation. Substantial
parts of the kernel — the GC barriers, the call sequences, the FFI
shims — are hand-written x86 assembly. 

We replace the compiler.

## Core decisions

NCL is **Rust-first**, **LLVM-based**, **64-bit-first**, and
**Windows-first** — but only the last of those is parochial.

1. **Rust for all native components.** The runtime, GC, image loader,
   reader, evaluator, compiler driver, and FFI bridge are written in safe
   Rust where possible and clearly-scoped `unsafe` where necessary. We
   inherit `unsafe_op_in_unsafe_fn = "deny"` from our sibling project.

2. **No hand-written assembly.** Every piece of the original that *had* to
   be assembly — call frames, GC write barriers, tagged pointer
   manipulation, dynamic dispatch shims — is rewritten in Rust, or lowered
   through LLVM. If a particular sequence demands specific instructions,
   we emit them through `core::arch` intrinsics or LLVM IR, not `.asm`
   files.

3. **LLVM is the code generator.** Corman Lisp had no IR; the compiler
   went straight from Lisp forms to x86. NCL goes Lisp →
   our IR → LLVM IR → machine code, JIT-first, with reviewable textual
   dumps at every phase. This buys us optimization, portability,
   debuggability, and a credible path to AOT later.

4. **64-bit from day one.** Tagged pointers, fixnum range, image format,
   FFI marshalling, and GC layout all assume a 64-bit address space.
   We will not ship a 32-bit build.

5. **Windows-first, not Windows-only.** The first supported target is
   `x86_64-pc-windows-msvc`. The OS-specific surface — windowing,
   graphics, filesystem, threads — lives behind a thin Rust shim. The
   rest of the system has no platform awareness. Porting to macOS or
   Linux is a matter of writing one more shim, not refactoring the core.

6. **JIT-first, image-resident, source-of-truth-on-disk.** Modules are
   compiled into the live process on demand. The **image lives in
   memory; the source lives on disk**, and that is the only direction of
   persistence. There is no image file format. Launching NCL
   means JIT-ing the image into existence from source — and on a 2026
   machine that is fast enough to be the default.

   This is a deliberate departure from the classical Lisp/Smalltalk
   image-based model. Persistent images create real source-control
   problems: you can't meaningfully diff them, review them, or merge
   them, and they drift from the source over years. We keep the
   *interactive workspace* — live editing, hot reload, REPL state — but
   we keep the *artifact* on disk as text. Source files are what git
   sees. The image is what the running process is.

   Workspace ephemera (variable bindings created at the REPL, test
   fixtures, half-built data) lives only in the image and is lost on
   restart. If the user wants it to survive, they write it to a file.
   This matches the SLIME/SLY model and is the model the user already
   knows.

7. **A non-canonical cache, designed in from day one.** Re-deriving the
   whole image from source on every launch is fast enough to be correct
   but not always fast enough to be pleasant. We ship a cache from v1.

   The cache stores compiled artifacts (LLVM bitcode and/or object code)
   keyed by `(source hash, compiler version, codegen flags)`. On launch,
   the loader stats source files, compares against cached stamps, and
   reuses cached artifacts where the key matches. Anything stale is
   recompiled.

   The cache is **never canonical**. It can be deleted at any time
   without loss of correctness. It never round-trips through git. It is
   not an "image format" in disguise. If the cache and source disagree,
   source wins and the cache entry is discarded. Treat the cache the
   way `cargo` treats `target/`.

## The garbage collector

The GC design is pinned in [docs/GC.md](docs/GC.md), committed
ahead of code. Headlines:

- **Multi-threaded mutator, stop-the-world collector.** Multiple
  Lisp threads run concurrently; each has its own TLAB
  (thread-local allocation buffer) so the alloc fast path takes
  no locks. GCs are cooperative stop-the-world: each mutator polls
  a flag at safe points and parks voluntarily; the GC runs once
  all mutators are parked. Modern hardware has 20 cores; this is
  not optional. Matches Corman's design (verified in upstream
  Gc.cpp).
- **Generational copying**, two generations (young + old) plus a
  pinned static area for compiled code and the loaded image.
- **3-bit pointer tagging**, 64-bit word, fixnum tag `000`, cons tag
  `001`, `forward` tag `111` (so a stale slot is one mask-and-compare
  to detect during scanning).
- **Headerless cons cells.** Everything else carries an 8-byte header.
- **Forwarding-pointer-based copying**, inherited from Roger's
  design.
- **Software card-marking write barrier.** Hardware-assist via page
  protection is out of scope for v1 — Roger's `:hardware-gc` was
  off by default for stability and we honour that lesson.
- **Precise root finding** via LLVM `gc.statepoint`-emitted stack
  maps, with a conservative fallback during bring-up that we
  eliminate by the time `defun` lands.
- **Symbol's function cell is `AtomicU64`.** The single-store
  redefinition story we already pinned needs a real atomic to land
  on.

The GC lives entirely in `ncl-runtime` in pure Rust. The OS only
appears via a `PageAllocator` shim. `ncl-runtime` and `ncl-cl`
contain zero FFI imports — including in the GC.

## The loader

The loader is modelled on the [`newcp-loader` crate](file:///E:/NewCP/NewCP/src/newcp-loader/src/lib.rs)
in our sibling project. The shape we are inheriting:

- A **source module graph** walked from a root, with dependency edges
  computed up front so we know the initialization order and can
  invalidate transitively.
- **`SourceFileStamp`** (size + mtime) for cheap per-launch
  dirty-checking. A full content hash is used only when we need to
  match against a cache entry.
- **`DirtyModuleState`** = Clean / Modified / Missing. Drives what gets
  recompiled.
- **Generation numbers per module.** Every recompile bumps the
  generation. This is how hot-reload identifies "which version of `foo`
  is this closure pointing at?"
- **Retirement, not deletion.** When a module is replaced, the old
  compiled image is moved to a retired list with a
  `collect_after_quiescent_epoch`. It is freed only when no live frame
  or closure can still reach it. A long-running computation does not
  get yanked out from under itself by a hot reload.
- **Execution scopes pin generations.** Entering a call site records
  the generation it was entered with; that generation cannot retire
  while the scope is live. This is the load-bearing piece that makes
  hot-reload safe in the face of long-running and recursive code.
- **Staged updates with explicit failure and recovery states.** A
  failed recompile leaves the previous generation running and produces
  a structured diagnostic, not a crash and not a half-installed image.

The cache plugs in at the bottom of this pipeline: when a module is
dirty *or* missing from the live image, the loader first checks the
cache by `(source hash, compiler version, codegen flags)` before
running the full compiler. A hit skips parse, sema, IR, and LLVM
optimization; a miss does the full compile and writes the result back.

The loader is OS-independent Rust. It runs on the Lisp thread, talks
to the GUI through the same event mailbox the rest of the runtime
uses, and never touches Win32 or Cocoa directly.

### Note: function redefinition and dispatch

This is the design we have committed to for global function
redefinition. It is recorded here because it is easy to drift toward
the more general (and wrong-for-Lisp) NewCP-style "retire every
generation" model.

Global functions live in **symbol function cells**, in the Common Lisp
tradition. `(defun foo …)` does not replace `foo`'s code — it
atomically stores a new function object into `foo`'s cell. Compiled
calls to `foo` load the cell and indirect-call through it. Redefining
`foo` is one pointer store; running closures that named `foo` see
the new definition on their next call automatically.

Consequences:

- Hot-reload of a global function is an atomic pointer swap. No
  retirement, no quiescent epoch, no execution-scope pin needed.
- The retirement / quiescent-epoch / scope-pin machinery from the
  loader is reserved for **orphaned compiled code** — anonymous
  lambdas whose closures have been GC'd, old method bodies replaced
  via `defmethod`, JIT buffers from a module that has been unloaded.
  Things where no live cell still points at the code and we need to
  decide when it's safe to free.
- `flet` and `labels` bindings are lexical and skip the cell. The
  compiler may bind them directly to the function object. They are
  not redefinable from outside their lexical scope, which is correct.
- CLOS method redefinition (`add-method` / `remove-method`) follows
  the same cell-swap shape on the generic function's dispatch table.
- Inline caches and self-modifying call sites are optional
  optimizations on top. They do not change the correctness model.

This is the SBCL/CCL/Allegro shape, not the NewCP module-generation
shape. Don't drift.

## Design values

These are tiebreakers, not slogans. When two designs both work, these
decide.

1. **Simplicity wins.** A smaller, more direct design is preferred over a
   larger, more general one — even at the cost of features that may
   never be used. PowerLisp and early Corman Lisp were powerful *because*
   they were simple. When tempted to add a layer (a third IR, a generic
   pass framework, a configuration system), the default answer is no
   until the simpler version has demonstrably failed.

2. **The thread boundary is the OS boundary.** The Lisp runtime runs on
   its own thread, started by the UI thread. That boundary is the only
   place OS-specific code is allowed to live. The UI thread owns the
   Win32 message pump (or, later, the Cocoa run loop, or the Wayland
   event loop); it is the only thread that calls OS APIs that demand a
   particular thread. The Lisp thread sees nothing but the typed event
   mailbox and the drawing surface.

   In practice this means:
   - `ncl-runtime` GC, allocator, scheduler, and reader are
     OS-independent Rust.
   - Anything that touches `windows::*`, `core_foundation::*`,
     `x11::*`, etc. lives in `ncl-runtime/src/igui/` (or its
     equivalents) and is selected by `cfg(target_os = ...)`.
   - The Lisp side never imports a platform-specific symbol. If it needs
     a platform fact (DPI, theme, path separator), it asks the GUI side
     through the event mailbox.

   This is the discipline that makes the Mac port a shim swap rather
   than a redesign.

3. **Mac is the second target, not a hypothetical.** v1 ships on
   `x86_64-pc-windows-msvc`. v2 ships on `aarch64-apple-darwin` and
   `x86_64-apple-darwin`. Every architectural decision is made with
   that v2 in mind — if a choice would make the Mac port harder, we
   pick the other choice now. Linux is welcome but not promised.

4. **The compiler is ours.** Faithful-tribute compatibility is at the
   *language* and *demo* level. The compiler architecture is not. We owe
   nothing to older compiler internals and will redesign them freely
   wherever a simpler or more honest design exists. Reader → small typed
   IR → LLVM IR — and we resist adding stages between those until a
   real demo forces us to.

5. **FFI is a feature, not a foundation.** Common Lisp programs that
   want to call out to C, Win32, or COM can. The Corman demos do, and
   we honour that — `#!…!#` blocks parse, `defun-dll` works, the FFI
   is a fully supported user-facing capability.

   But our own implementation never goes through FFI. `cl:open` uses
   Rust's `std::fs`, not `CreateFileW`. `cl:make-thread` uses
   `std::thread`, not `CreateThread`. `cl:get-internal-real-time` uses
   `std::time::Instant`, not `QueryPerformanceCounter`. The stdlib is
   Rust-backed; the FFI sits beside it for user code to reach.

   Why: relying on FFI for our own internals would pollute the
   language/OS separation — every primitive would carry a "what's the
   Cocoa equivalent?" question into the Mac port, and the iGui
   thread-boundary rule would leak into our stream and time and
   thread code. Keeping FFI out of our foundation keeps the porting
   cost where it belongs (the iGui shim) and nowhere else.

   Concrete consequences:
   - Crate dependencies flow `ncl-ffi → ncl-runtime`, never the
     reverse.
   - `ncl-runtime` and `ncl-cl` contain no FFI imports and no
     `#![allow]` for unsafe FFI patterns.
   - Corman's `Sys/` source tree is not ported. Its FFI-heavy
     implementation of the standard library is not the spec; the
     ANSI surface it presents is. We re-implement the standard
     library natively in Rust, and the demos see the same API
     they always did.

6. **When in doubt, be Lisp.** We borrow architecture from NewCP because
   it has solved many of the same problems we will face — JIT,
   incremental loading, hot reload, OS shim, GUI substrate. But NewCP is
   a Component Pascal system, and Component Pascal is module-shaped and
   statically typed where Common Lisp is symbol-shaped and dynamically
   typed. Where the two traditions disagree on a design choice, we side
   with Lisp. The user wrote in Lisp for a reason; we don't deliver
   them a slightly-Lispy Component Pascal.

   In practice this means: cells, not modules, as the unit of
   redefinition; symbols and packages as first-class; reader macros and
   compile-time evaluation as load-bearing; CLOS metaobject protocol
   semantics respected; `eval` is not a sin and is a real path through
   the system; condition/restart semantics are not collapsed into
   exceptions. When a NewCP design choice would make any of those
   harder, we deviate.

## Compatibility — what "faithful tribute" means

We commit to running:

- Common Lisp as Corman Lisp implements it, including its de-facto
  ANSI-CL surface and the conformance Corman documented.
- The Corman Lisp **demo programs** in `cormanlisp\examples` and the
  packaged tutorials, unmodified except for paths.
- The Corman-specific extensions that the demos rely on — notably the
  Win32 FFI conventions, the `ccl:` package surface, and the IDE
  integration points the demos call into.

We do **not** commit to running:

- Compiled FASL files from the original. Recompile from source.
- Code that depends on the original's 32-bit pointer layout, MFC-specific
  IDE internals, or hand-written assembly entry points.
- Saved images from the original. Ours are a different format.

Compatibility is verified by a regression suite that runs every Corman
demo end-to-end on every CI build.

## The GUI

NCL borrows its GUI substrate from our sibling project
[NewCP](file:///E:/NewCP/NewCP) — specifically the `iGui` event-mailbox
model in `newcp-runtime/src/igui/`. On Windows this sits on Direct2D,
Direct3D 11, DirectWrite, and DXGI; the Lisp side sees a typed event
stream (Key, Char, Mouse, Focus, Resize, Paint, Close, Menu, DpiChange,
ThemeChange) and a drawing surface.

The MFC IDE of the original is not ported. Its replacement is built in
NCL itself, on top of `iGui`. The original IDE's *feel* — the
inline REPL, the workspace, the inspector, the "hot edit and reload"
loop — is the spec.

The GUI shim is small enough that retargeting it to Cocoa/Metal or to
X11/Wayland is a future shim swap, not a redesign.

## What is explicitly out of scope (for v1)

- 32-bit support.
- A direct-from-Lisp x86 backend (LLVM is the only backend).
- MFC, ATL, COM beyond what the demos require.
- Loading the original Corman `.img` or `.fasl` files.
- A persistent NCL image format. There is none. Images live
  only in memory; source is the only persistence. The launch-time
  artifact cache is not an image format — it can be deleted at any
  time without loss.
- `cl::save-application` and friends as image-dump operations. If a
  demo relies on them, we may re-map them to "write the relevant
  source out" — but a literal image dump is not in scope.
- A pure interpreter mode — everything compiles, even at the REPL.

## Repository layout (planned)

```
E:\CL\
  cormanlisp\          upstream — read-only reference, port FROM
  NCL\       this project — port TO
    MANIFESTO.md       this file
    src\               Rust workspace
      ncl-driver       compiler + REPL entry point
      ncl-reader       s-expression reader
      ncl-ir           our high-level IR
      ncl-compiler     IR → LLVM IR
      ncl-llvm         LLVM bindings + JIT
      ncl-loader       source graph, dirty tracking, generations, cache
      ncl-runtime      GC, image construction, FFI, iGui shim
      ncl-cl           Common Lisp standard library (Lisp-side)
    Lisp\              Lisp-side sources (CL stdlib, IDE, demos)
    tests\
      ncl-tests        unit + integration
      ncl-corman-demos faithful-tribute regression suite
```

## The promise

A user who wrote Corman Lisp code in 2005 should be able to open it in
NCL in 2026, hit compile, and watch it run — faster, on a
64-bit process, on a modern toolchain, with a debugger that understands
the source — without changing a line.

Everything underneath is new. Everything on top is theirs.
