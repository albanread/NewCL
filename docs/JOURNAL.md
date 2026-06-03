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

Committed as `6c56628`.

### Post-fix verification

**Broad Lisp smoke (full CLI binary):** CLOS multiple dispatch +
`call-next-method`, `handler-case`, extended `LOOP`, vector sequences
(`reverse`/`some`/`every` — all *segfaulted* in the May review), bignums,
ratios, `string-upcase`, `equal`-keyed hash tables, and the Prolog
Zebra-puzzle demo (correct answer: Norwegian drinks water, Japanese owns
the zebra) — all run.

**GC reliability suite (all green):**
- `cargo test -p ncl-runtime` — 208/208 (26 mutator + 21 heap GC tests).
- `conservative-pin/alloc-test.lisp` — 8 threads × 2000, exact value
  `32000000` (no lost increments under concurrent minor GC).
- `conservative-pin/stress.lisp` — 16 threads × 1500 heavy mixed alloc,
  all joined cleanly.
- `conservative-pin/gc-watch.lisp` — 8 workers, 5 s, 62 M iterations,
  3 minor GCs, 803 MB promoted, 0 pinned/residual (no pin leaks).
- `seh-unwind/{exit,deep-panic,preempt,preempt-full}-test.lisp` — all
  print DONE, exit 0; deep-panic unwinds 1000 JIT frames cleanly.
  `panic-test.lisp` now exits 2 (clean driver-level catch), not the
  README's documented 101 — the unwind itself is healthy (no
  `__fastfail`); the exit code changed with the "unhandled conditions
  exit cleanly" commit (a40ccee). README is stale, not the code.

### Integration suite: 1 pre-existing failure (not from this fix)

`ncl-tests::characters::predicates_reach_beyond_ascii` fails:
`(alphanumericp (code-char #x0969))` (Devanagari digit ३) returns `nil`,
test expects `T`. Root cause: `core.lisp:2021` shadows the Unicode-aware
native `alphanumericp_shim` (`is_alphanumeric`) with the spec-literal
`(or (alpha-char-p c) (digit-char-p c))`; `digit-char-p` is ASCII
radix-based, so a non-ASCII decimal digit yields `nil`. Identical at
HEAD~1, and there is no call path from the predicate to `%ht-bucket-index`,
so the hash fix did not cause it. Candidate compat fix (see below).

### Perf: precise-root stack `Mutex` → `RefCell`

Re-evaluated the stashed perf WIP now that the stdlib loads. Adopted the
first half: `MutatorState.roots` changes from `Mutex<Vec<Word>>` to
`RefCell<Vec<Word>>`.

Soundness: `MutatorState` is `!Send` — each Lisp thread owns its own, and
the only borrower of a given root stack is its owning thread. When a peer
drives a stop-the-world minor GC, every other mutator is parked *inside*
`poll_safepoint`, holding its own `borrow_mut` guard across the call; the
collector visits peers' roots through the raw slices they published to
newgc-core while parked, never through their `RefCell`s. No cross-thread
`RefCell` access, no reentrant borrow (the collecting thread runs no Lisp
code mid-collection). So the mutex was pure overhead on the hot path.

Per the GC_LESSONS warning ("Rust tests pass, GC still wrong"), I gated
this on the *semantic* tests, not just the 208 unit tests:
- alloc-test 8×2000 exact value `32000000`; stress 16×1500 clean;
  gc-watch 8-worker 5 s, 0 pinned / 0 residual. GC reliability identical
  to baseline.

Speedup (this machine, release):

| bench         | baseline | RefCell | speedup |
|---------------|----------|---------|---------|
| fib 30        | 0.189 s  | 0.049 s | 3.9×    |
| fib 32        | 0.502 s  | 0.128 s | 3.9×    |
| tak 24 16 8   | 0.300 s  | 0.060 s | 5.0×    |
| ack 3 7       | 0.042 s  | 0.010 s | 4.2×    |

The push/pop happens once per live variable around *every* call, so the
~25 ns mutex lock dominated; the ~1 ns non-atomic borrow erases it.

(The second half of the WIP — inline call dispatch in the LLVM backend —
is evaluated separately next; kept stashed for now.)

