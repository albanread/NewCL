# NCL Test-and-Improve Journal

A running log of testing, debugging, and improvement work on
NewCormanLisp. Newest entries at the top.

---

## 2026-06-03 — Session 1

### Orientation

Started from a working tree with two uncommitted, GC-touching changes:

- `src/ncl-runtime/src/mutator.rs` — the per-thread precise-root stack
  changed from `Mutex<Vec<Word>>` to `RefCell<Vec<Word>>`. Reasoning in
  the diff: the root stack is only touched by its owning thread (each
  `MutatorState` is `!Send`); the collector reads peers' roots through
  raw slices they publish while parked inside `poll_safepoint`, not
  through their `RefCell`s. So a non-atomic borrow (~1 ns) replaces a
  mutex lock (~25 ns) on the per-call hot path. Claimed win: fib(30)
  184 ms → 20 ms with rooting disabled, i.e. the lock was ~90 % of
  per-call cost.
- `src/ncl-llvm/src/lib.rs` — "Lever 2a": inline the symbol-call
  dispatch in IR (load function cell, tag-check, indirect-call the
  fast path; fall back to `ncl_call` only for unbound / not-a-function)
  plus an `NCL_NO_ROOTS` diagnostic and a gated-off `optimize_module`
  harness.

**Verdict on the uncommitted changes:** I audited the RefCell change
against the threading model and it is sound. I audited the inline
dispatch against `ncl_call` (abi.rs) and the offsets/ABI match. All 208
`ncl-runtime` unit tests pass with both changes in place.

### BLOCKER found: stdlib bootstrap stack-overflows

`./target/release/ncl.exe -e '(+ 1 2)'` → **STACK_OVERFLOW**. So does
`--repl`, `-c`, and every non-`--lean` entry. `--lean -e '(+ 1 2)'`
returns `3`. So the crash is in **stdlib bootstrap**, and it reproduces
on the committed HEAD too (stashed the perf changes and rebuilt to
confirm) — *not* caused by the uncommitted work.

Bisected the `init.lisp` require sequence: crash hits when loading
`hash-tables` (after `numbers`). The crash backtrace is a tight cycle:

```
%HT-BUCKET-INDEX → GETHASH → %NEW-TYPEP → TYPEP → FLOATP → MOD → %HT-BUCKET-INDEX → …
```

Root cause: `Library/numbers.lisp` (commit a25a5ab) redefines `MOD`/`REM`
to be **polymorphic** — they call `(floatp a)` to dispatch. `FLOATP`
is `(typep x 'float)`; `%NEW-TYPEP` (types.lisp) looks the type symbol
up with `(gethash head *type-expanders*)`; and `GETHASH`'s bucket index
is `(mod hash nbuckets)` — which now resolves to the polymorphic `MOD`.
That closes an infinite cycle. types.lisp even carries a "CRITICAL
ARCHITECTURE NOTE" about the `symbolp→typep` version of this loop, but
the `mod→floatp` path slipped through because `MOD` only becomes
dangerous once numbers.lisp is layered on top.

### Fix

`%ht-bucket-index` (core.lisp) now uses the **native integer `REM`**,
captured into `%ht-native-rem` via `(symbol-function 'rem)` *before*
numbers.lisp can redefine it. The hash is a non-negative fixnum
(`%word-hash` is documented non-negative) and nbuckets is positive, so
`rem ≡ mod` here. This breaks the cycle at its only closure point and
also takes every `gethash` off the polymorphic-dispatch path (a perf
win in the hash hot loop).

Pure-Lisp change to `core.lisp` (embedded via `include_str!`, so it
needs a rebuild); does not touch GC or compiler code.

(Verification + commit to follow in this entry once the rebuild lands.)
