# ANSI chapter-killers — map of remaining conformance blockers

*Last measured 2026-06-17. Suite: `demos/ansi-runner.lisp` over the Corman
ANSI hyperspec-examples chapters (`E:/CL/cormanlisp/test/ansi-chapter-*.lisp`).*

```
passed: 751   failed: 77   errors: 91   total: 919   (every chapter loads fully)
```

## The "chapter-killer" pattern (mostly retired)

The suite loads each chapter file form-by-form at **compile time**. A single
unsupported construct — an unknown LOOP clause, a reader dispatch we don't
handle, a compiler `NotImplemented`, or a worker-thread **panic** — used to
raise during load and **abort the rest of that chapter**: every test after it
never ran. So a handful of missing features hid a large fraction of the count.

As of this session **all seven chapters now load to completion** — no aborts,
no panics. The remaining `failed`/`errors` are forms that *actually executed*
and produced a wrong/no answer, which is honest: the gaps are visible
per-form rather than masked by an early chapter abort.

Work so far took the count from **622 → 751 (+129)** and, more importantly,
took the suite from 784 forms run to **919** (chapters 5, 6, 8 previously
aborted partway). What remains is no longer quick clause-adds — each is a
substantial single feature.

## Fixed (later pass)

- **Top-level `(progn …)` sequencing** (`ncl-compiler`: `eval_value` +
  `macroexpand_toplevel`) — CLHS §3.2.3.1. The loader now detects a top-level
  progn via a *head-only* macro expansion BEFORE the full recursive
  `macroexpand_all`, and processes each subform sequentially. This makes one
  subform's compile-time side effects (defmacro/defstruct registration) visible
  to later siblings within the same top-level form — which fixes `defstruct
  :include` of a sibling inside the same `dotests` block (astronaut/truck/
  pickup), and the `typep`-on-struct-subtype test. (+6, and removes a whole
  class of "same-form ordering" surprises.)
- **`typep` on struct subtypes** (`Library/types.lisp`) — `%new-typep` now
  recognises registered DEFSTRUCT types and walks the `:include` chain via
  `%ds-isa`, guarded so it is inert before `structures.lisp` loads.
- **Full `defstruct`** (`Library/structures.lisp`, +30) — option-list name
  form with `(:conc-name)`, `(:constructor name [boa-arglist])` incl.
  BOA + `&optional`/`&rest`/`&aux`, `(:copier)`, `(:predicate)`, `(:include
  parent overrides…)`, `(:type list)`, `:named`, `(:initial-offset n)`, and
  per-slot `:read-only`/`:type`. The plain symbol-name path stays compatible
  with the bootstrap structs. Most of chapter 8 now passes (~38/56). Two
  residual limits: (a) `=> #S(...)` comparisons still fail because NCL prints
  structs as `SIMPLE-VECTOR` (the value is a tagged vector, the expected is a
  quoted cons), and (b) `:include` of a sibling defined in the *same* top-level
  form (see the macroexpand-sequencing note below).
- **`multiple-value-call`** macro, **`function-lambda-expression`**, the
  `call-arguments-limit`/`lambda-parameters-limit`/`multiple-values-limit`
  constants, **`%setf-symbol-value`**, and **`loop-finish`** (terminate a
  LOOP normally from body or `initially`) — all in `Library/`.

## Fixed (first pass)

- **LOOP conditional sublanguage** (`Library/loop.lisp`, +36) — `else`, the
  anaphoric `it`, the `end` preposition, nested `when/when/else`, and the
  **parallel-stepping** fix for `and`-joined `for` clauses (each sibling now
  reads the *old* values via a temp-capture step block). Chapter 6 fully
  unlocked.
- **`#S(...)` read-time struct literals** (`ncl-reader/src/parser.rs`) —
  reader now expands `#S(NAME :k v …)` into the constructor call
  `(make-NAME :k v …)`, mirroring `#C` → `(complex …)`. Unblocks the chapter-8
  *load* (see the chapter-8 caveat below for why it yields few *passes*).
- **explicit-keyword `&key`** (`ncl-compiler/src/lib.rs`,
  `parse_param_list_inner`) — `((keyword-name var) [default [supplied-p]])`.
  The keyword string is built from the real symbol (`:NAME` for a keyword
  symbol, bare `NAME` otherwise — NCL's runtime symbol table is flat by name).
- **`aref`/`(setf aref)` no longer panic** (`ncl-runtime/src/abi.rs` +
  `ncl-llvm`) — a bad index / out-of-range / non-array now signals a
  **catchable** condition (CLHS type-error) instead of crashing the worker
  thread. `ncl_aref_generic` gained a `mutator` param for the error path; the
  fast path is unchanged (no new abort-check). This is what lets the suite
  *complete* instead of dying in chapter 8 on a stale-cons accessor.
- **N-index `aref`/`(setf aref)` compile** (`ncl-compiler/src/lower.rs`) —
  the 2-arg form stays the fast primitive; ≥3-arg falls through to an
  ordinary late-bound call / the `%SETF-AREF` rewrite, so a 2-D `(aref a i j)`
  *compiles* (and runtime-signals, since multidim arrays are unimplemented)
  rather than aborting chapter 5 at load with `BadArity`.
- **setf-expander protocol** (`Library/places.lisp`):
  - `rplaca` / `rplacd` were genuinely **undefined** — now defined over
    `(setf (car …))` / `(setf (cdr …))`. This alone unblocked `middleguy`
    (its writer calls `rplaca`).
  - `define-setf-expander` + a `*setf-expanders*` registry that
    `get-setf-expansion` now consults before its generic syntactic fallback.
  - `push` / `pop` redefined for **once-only** subform evaluation (the
    core.lisp versions evaluated the place twice).
  - `rotatef` / `shiftf` rewritten via `get-setf-expansion` for once-only
    semantics (side-effecting subforms like `(nth (incf n) x)` now evaluate
    exactly once). Shared helper `%collect-setf-expansions`.

## Live blockers (now per-form, not chapter-aborting)

### Multidimensional arrays  *(blocks the chapter-5 2-D `xy` setf demo)*
- **Symptom:** `(make-array '(10 10))` →
  `"multidimensional arrays (2 dimensions) are not yet supported"`
  (`abi.rs:make_array_dimension`); N-index `aref` now compiles but
  runtime-signals.
- **Why hard:** arrays are flat `Tag::Vector`s with a single length and **no
  dimension metadata**. Real support needs a representation carrying
  rank+dims (header change or a wrapper struct) plus row-major index math in
  `aref`/`aset` and `array-rank`/`array-dimension`. Invasive to the array/GC
  layout — a dedicated effort, not a clause-add.

### Struct⇄print parity  *(~11 false-fails in chapter 8)*
- The `=> #S(TYPE …)` comparison tests still fail. The actual value is a
  tagged `SIMPLE-VECTOR`; the expected `#S(...)` is read (by the cheap-partial
  reader) into a quoted `(make-TYPE …)` cons, so `equalp(vector, cons)` is NIL.
  Closing these needs real `#S` reading (a struct value at read time) +
  printing structs as `#S(...)` + structural `equalp` on structs — a coherent
  "real structs print/read/compare" feature. Until then they are known
  false-fails (the accessor/predicate/constructor tests all pass).

### `getf` / `ldb` setf places  *(last setf-expander corners)*
- `(setf (getf place k) v)` needs an expander that rewrites the *underlying*
  place (getf's first subform is itself a place); it currently lowers to a
  nonexistent `%SETF-GETF`. `(setf (ldb (byte …) int) v)` needs an `ldb`
  setf-expander registered in `bits.lisp`. Direct `(setf (custom …) v)` for a
  `define-setf-expander` place also isn't rerouted through the registry —
  NCL's `SETF` is a hardwired special form (`lower.rs`) using the
  `%SETF-NAME` writer-function convention, so a *complete* fix needs a small
  `macroexpand.rs` `SETF` arm that routes non-native compound places through
  a `%setf-expand` Lisp macro (the registry already exists to back it).
- Note the `dotests` harness compiles a whole block before running it, so a
  `define-setf-expander` registered at *runtime* in one test entry isn't
  visible to the *compile* of a sibling `(setf (lastguy …) …)` entry — the
  expander **definition** test passes; the direct-setf use after it does not.

## Notable *non*-bugs (harness/printer mismatches — don't chase)

- `#C(4.56 0.0)` vs expected `(COMPLEX 4.56 0.0)` — NCL prints standard
  reader syntax; arguably *more* correct than the hyperspec example text.
- `(type-of (expt 2 40))` → `FIXNUM` not `BIGNUM` — NCL fixnums are 61-bit;
  impl-defined and correct for our representation.
- Full-precision float printing (`3.2911733607066247` vs `3.291173`).
- `(< (ZAP 5 3) 3)` — a documented *flaky* hyperspec example (`zap` returns
  `(random 4)`); fails ~1 run in 4. See the header of `demos/ansi-runner.lisp`.
- `with-output-to-string` returns `""` (string-stream capture is a separate,
  unrelated bug surfaced while testing LOOP — not a LOOP problem).
- `(defmacro …)` / `define-modify-macro` / `define-setf-expander` *inside* a
  `dotests` lambda don't register (NCL registers macros at compile time, but
  these sit in a runtime lambda body) — a harness artifact, not a feature gap.
