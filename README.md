# NCL 

### !Beware this compiler and GC are all new ###


A from-scratch reimplementation of the Common Lisp / Corman Lisp **language and
user-facing experience** — Rust core, LLVM-based JIT, 64-bit,
Windows-first with a Mac port planned.

NCL ports some Corman Lisp source code and demos. It does not run
the original implementation's compiled artifacts (`.img`, `.fasl`).
Recompile from source.

See [MANIFESTO.md](MANIFESTO.md) for the design and what we have
committed to. This file will fill in as the system grows; the
manifesto is the spec.

## Status

**Working JIT-compiled Common Lisp** (v0.0.0 — pre-1.0; internals and
interfaces still move). NCL self-hosts its standard library and runs real
programs. It is **not** a complete ANSI implementation yet — see
*Conformance* and *Known gaps* below.

### What works

- **Compiler** — JIT-first, no interpreter. Lisp → IR → LLVM MCJIT (`-O2`),
  compiled per function. Optimization passes: self-tail-call elimination,
  unboxed `double-float` representation inference, no-capture closure
  elision, and IR-level inlining of `declaim inline` functions.
- **Runtime / GC** — a custom generational page-heap collector (G0 / G1 /
  Tenured), multi-mutator, with conservative stack pinning plus a precise
  inline root stack and card marking.
- **Language** — the numeric tower (fixnum, bignum, ratio, double-float,
  complex); the full macro system (`defmacro`, `macrolet`,
  `symbol-macrolet`, `&environment`); the condition system
  (`handler-case`, `unwind-protect`, restarts); CLOS (`defclass`,
  `defmethod`, generic dispatch; closette-derived); `format`, sequences,
  lists, strings, hash tables, structures.
- **GUI (Windows)** — *iGui*, a Direct2D MDI shell with a pixel canvas,
  menus, and a tick/event loop. Demos include a live neural-net + genetic
  algorithm tank simulation and assorted animation demos.
- **Standard library** — ~800 forms, JIT-compiled at startup from
  `Lisp/core.lisp` (embedded) + `clos.lisp` + `Library/` (loaded from disk,
  user-extensible).

### Performance

NCL emits real native code, not a tree-walk. Honest numbers (2026-06,
measured against SBCL 2.6.4 on the same machine):

- **Float kernels** — ~3–4× faster via the unboxing pass; matches a hand
  `(declare (double-float …))` without needing the declaration.
- **Symbolic / heavy-backtracking** (Norvig Prolog solving the Zebra
  puzzle, `demos/prolog.lisp`) — **≈10× SBCL** (1.08 s vs 0.10 s) after
  stdlib hot-path tuning; was 18× before.
- Allocation is competitive (`cons` ≈ 2× SBCL) and GC is usually *not* the
  bottleneck. The remaining gap is per-call overhead from a late-bound,
  boxed calling convention — the next compiler lever is an unboxed /
  known-call ABI.

### Conformance

The Corman/ANSI test chapters (`demos/ansi-runner.lisp`) currently pass
**≈493 / fail ≈56**. Known weak spots: parts of the type system
(`subtypep`, `typep` on compound types), some `coerce` targets, a few CLOS
corners (`with-slots` / `with-accessors` edge cases), and the full extended
`loop`. The performance "gauntlet" (`bench/gauntlet.lisp`) is ALL-PASS.

### Known gaps

- Not a complete ANSI CL (see *Conformance*).
- Windows-only today; the Mac port is planned, not started.
- No image / fasl save-and-load — recompile from source.
- Some GC roots are found conservatively (precise stack maps are future
  work).

## Building & running

Requires a Rust toolchain and LLVM (see [MANIFESTO.md](MANIFESTO.md) /
the build notes).

```
cargo build --release                              # console REPL  -> target/release/ncl.exe
cargo build --release --features gui-app -p ncl-driver   # GUI build -> target/release/ncl.exe
```

In a packaged release the console binary is shipped as `nclterm.exe` and
the GUI binary as `ncl.exe`. Run a program with `ncl.exe -l file.lisp`.
