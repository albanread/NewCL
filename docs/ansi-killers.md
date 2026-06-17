# ANSI chapter-killers — map of remaining conformance blockers

*Last measured 2026-06-17. Suite: `demos/ansi-runner.lisp` over the Corman
ANSI hyperspec-examples chapters (`E:/CL/cormanlisp/test/ansi-chapter-*.lisp`).*

```
passed: 622   failed: 64   errors: 98   total: 784
```

## The "chapter-killer" pattern

The suite loads each chapter file form-by-form at **compile time**. A single
unsupported construct — an unknown LOOP clause, a reader dispatch we don't
handle, a compiler `NotImplemented` — raises during load and **aborts the
rest of that chapter**: every test after it never runs. So a handful of
missing features hide a large fraction of the `errors` count. The suite is a
deliberate checklist (each `dotests` block exercises one feature), which is
why it surfaces gaps one at a time, and why each fix tends to unlock a *burst*
of passes rather than one.

This session took the count from **493 → 622 (+129)** by killing killers in
chapters 5 and 6. What remains is no longer quick clause-adds — each is a
substantial single feature.

## Live killers (each aborts its chapter at the cited point)

### Chapter 8 — `#S(...)` read-time struct literals  *(≈56 locked, 100% of chapter)*
- **Symptom:** `read error: UnsupportedSharpDispatch { ch: 'S', … "struct
  literals (#S) not supported" }` — `src/ncl-reader/src/parser.rs:286`.
- **Why hard:** CLHS `#S` is *read-time construction* — `#S(door :x 1)` must
  produce an actual `door` instance while parsing. But `ncl-reader` is a
  standalone parser crate with no access to the runtime struct registry /
  constructors. Needs a reader→runtime hook (or a deferred-construction node
  the compiler materialises). Even then, the many `=> #S(...)` comparison
  tests need struct value↔`test-equalp` parity (NCL prints structs as
  `SIMPLE-VECTOR` today, so `type-of` and printed form differ from CL).
- **Cheap partial:** have the reader expand `#S(NAME :k v …)` into the form
  `(make-NAME :k v …)`. Unblocks the chapter LOAD so all the
  accessor/predicate/copier/`defstruct`-option tests run; only the direct
  `=> #S(...)` data-position comparisons stay failing.

### Chapter 6 — LOOP conditional sublanguage  *(≈41 locked from CLAUSE-GROUPING on)*
- **Symptom:** `MacroError("while expanding macro (LOOP …): #<SIMPLE-ERROR>")`
  at the `CLAUSE-GROUPING` block (`ansi-chapter-6.lisp:414`).
- **Missing:** `else`, the anaphoric `it`, the `end` preposition, and nested
  `when … when … else`. Also parallel `and`-joined **for**-clauses
  (`for x … and y …` step in parallel — currently mis-stepped sequentially,
  a *correctness* bug, not an abort).
- **Self-contained:** all in `Lisp/Library/loop.lisp` (disk-loaded — no
  rebuild to iterate). Build on `%parse-when-unless` (already handles
  `when … and …` since this session) and the `result-override` plan slot.
- Behind it sit DO/DO*, DOTIMES, DOLIST, a second LOOP block, and LOOP-FINISH
  — mostly non-LOOP and likely to pass once the abort clears.

### Chapter 5 — explicit-keyword `&key`, then the setf-expander protocol  *(≈42 locked)*
- **Symptom:** `NotImplemented("init-form name must be a symbol, got (#:X
  #:X)")` at `(defun xy (&key ((x x) 0) …) …)` (`ansi-chapter-5.lisp:1076`).
- **Missing #1:** the `((keyword-name var) [default])` form of `&key`
  parameters (compiler lambda-list parser, `parse_param_list`).
- **Missing #2 (next in line):** `define-setf-expander` and
  `get-setf-expansion` — the full subform-evaluated-once setf-expander
  protocol. Long-form `defsetf` already lands (this session) via the
  `%SETF-NAME` writer-function shim; the expander protocol is the harder,
  separate piece.

## Notable *non*-bugs among the 64 FAILEDs (harness/printer mismatches)

Do **not** chase these as code bugs:
- `#C(4.56 0.0)` vs expected `(COMPLEX 4.56 0.0)` — NCL prints standard
  reader syntax; arguably *more* correct than the hyperspec example text.
- `(type-of (expt 2 40))` → `FIXNUM` not `BIGNUM` — NCL fixnums are 61-bit;
  impl-defined and correct for our representation.
- Full-precision float printing (`3.2911733607066247` vs `3.291173`).
- `(< (ZAP 5 3) 3)` — a documented *flaky* hyperspec example (`zap` returns
  `(random 4)`); fails ~1 run in 4. See the header of `demos/ansi-runner.lisp`.

## Fixed this session (for reference)
- `(cond (test))` test-only clause (`+71`) — `lower.rs`.
- Variadic `append` (was binary; dropped args past the 2nd) + `PROG`/`PROG*`
  (`+26`) — `core.lisp`.
- Nested `(defun (setf X) …)` + long-form `defsetf` (`+2`) — `lower.rs`,
  `places.lisp`.
- LOOP `nconc`/`of-type`/`always`/`never`/`thereis`/bare-type-designators
  (`+21`) and `and`-conjoined sub-clauses (`+6`) — `loop.lisp`.
