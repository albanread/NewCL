;;;; Lisp/Library/loop.lisp
;;;;
;;;; Extended LOOP macro. CL's iconic iteration form.
;;;;
;;;; This file REDEFINES the `loop` macro to dispatch:
;;;;
;;;;   (loop body...)                — simple loop (existing behaviour)
;;;;   (loop for x in '(1 2 3) ...)  — extended LOOP
;;;;
;;;; The dispatch is based on whether the first form is a recognised
;;;; loop keyword. The simple form remains the primitive (a thin
;;;; wrapper around %native-loop) and is what core.lisp's other
;;;; macros (`dotimes`, manual loops in CLOS, etc.) expand to.
;;;;
;;;; Subset implemented:
;;;;
;;;;   Variable / iteration:
;;;;     (for VAR in LIST)
;;;;     (for VAR on LIST)
;;;;     (for VAR from N [to|below|downto|above M] [by STEP])
;;;;     (for VAR = EXPR [then EXPR])
;;;;     (for VAR across VECTOR)   ; via length+svref
;;;;     (repeat N)
;;;;     (as ...)                  ; synonym for `for`
;;;;
;;;;   Outer bindings:
;;;;     (with VAR [= INIT])
;;;;
;;;;   Termination:
;;;;     (while EXPR)
;;;;     (until EXPR)
;;;;     for-clause range bounds (to, below, downto, above)
;;;;     (repeat N)
;;;;
;;;;   Accumulation (no `into` form — assume single accumulator):
;;;;     (collect EXPR)        ; nconc-style; result is the list
;;;;     (sum EXPR)            ; running total
;;;;     (count EXPR)          ; count of true values
;;;;     (minimize EXPR)
;;;;     (maximize EXPR)
;;;;     (append EXPR)         ; like collect but append items
;;;;
;;;;   Control:
;;;;     (initially FORM ...)  ; run once before loop
;;;;     (finally FORM ...)    ; run once after natural completion
;;;;     (do FORM ...)         ; side effects, no accumulation
;;;;     (when EXPR CLAUSE)    ; conditional single sub-clause
;;;;     (unless EXPR CLAUSE)
;;;;     (named NAME)          ; name the implicit block for return-from
;;;;     (return EXPR)         ; exit with EXPR
;;;;
;;;; What's NOT in this slice:
;;;;
;;;;   * `into VAR` for accumulators (single-accumulator only).
;;;;   * `for VAR being the hash-keys/values of HT`.
;;;;   * Destructuring `for (A B) in PAIRS`.
;;;;   * `else` branches on `when/unless` (single clause only).
;;;;   * Multiple sequential `for` clauses are treated as parallel
;;;;     stepping; loop ends when ANY iterator exhausts.
;;;;
;;;; These all land in a future pass — the most common idioms are
;;;; covered.

;; ── Recogniser ────────────────────────────────────────────────────────────

(defun %loop-keyword-p (sym)
  "T iff SYM is a LOOP keyword (the head of an extended-loop clause)."
  (and (symbolp sym)
       (member sym
               '(for as with while until repeat named
                 initially finally do doing return
                 collect collecting append appending
                 sum summing count counting
                 minimize minimizing maximize maximizing
                 when unless if))))

;; ── Tokeniser-as-cursor ───────────────────────────────────────────────────
;;
;; The cursor is a one-cell holder: (car cur) is the REMAINING list of
;; tokens. Eat updates (car cur) to its cdr and returns the old head.
;; This shape lets callers share the cursor by reference (passing the
;; one-cell) and observe each others' advances.

(defun %make-cursor (lst) (cons lst nil))   ; (REMAINING . _)
(defun %cur-peek (cur)  (car (car cur)))
(defun %cur-empty? (cur) (null (car cur)))

(defun %cur-eat! (cur)
  "Pop the head off the cursor and return it."
  (let ((lst (car cur)))
    (cond
      ((null lst) nil)
      (t (let ((head (car lst)))
           (setf (car cur) (cdr lst))
           head)))))

(defun %cur-eat-keyword? (cur kw)
  "If the cursor's head is KW (a symbol), advance and return T."
  (cond
    ((eq (%cur-peek cur) kw) (%cur-eat! cur) t)
    (t nil)))

;; ── Plan structure ────────────────────────────────────────────────────────
;;
;; The plan accumulates as we parse. Each field is a list we add to.
;; Final code emission walks the plan.

(defstruct loop-plan
  (name nil)            ; named block tag, or NIL
  (with-bindings nil)   ; ((var . init-form) ...)
  (iter-bindings nil)   ; ((var . init-form) ...) for for-clause vars
  (iter-tests nil)      ; pre-body tests — for/while/until BEFORE first do
  (post-body-tests nil) ; post-body tests — while/until AFTER do clauses.
                        ; CL `(loop do EXPR until COND)` runs EXPR first,
                        ; then tests COND — Corman's quicksort relies on
                        ; this. The split tracks which side of the body a
                        ; test landed in.
  (iter-steps nil)      ; forms — run after body to advance iterators
  (initially nil)       ; forms — run before main loop
  (body nil)            ; forms — main per-iteration body
  (body-seen nil)       ; flipped to T once a do/doing clause was parsed;
                        ; subsequent while/until tests are routed to
                        ; post-body-tests.
  (finally nil)         ; forms — run after natural completion
  (accumulators nil)    ; ((kind var . extras) ...) for the result computation
  )

(defun %plan-add-with (plan var init)
  (setf (loop-plan-with-bindings plan)
        (append (loop-plan-with-bindings plan)
                (list (cons var init)))))

(defun %plan-add-iter-binding (plan var init)
  (setf (loop-plan-iter-bindings plan)
        (append (loop-plan-iter-bindings plan)
                (list (cons var init)))))

(defun %plan-add-test (plan form)
  (cond
    ((loop-plan-body-seen plan)
     (setf (loop-plan-post-body-tests plan)
           (append (loop-plan-post-body-tests plan) (list form))))
    (t
     (setf (loop-plan-iter-tests plan)
           (append (loop-plan-iter-tests plan) (list form))))))

(defun %plan-add-step (plan form)
  (setf (loop-plan-iter-steps plan)
        (append (loop-plan-iter-steps plan) (list form))))

(defun %plan-add-body (plan form)
  (setf (loop-plan-body plan)
        (append (loop-plan-body plan) (list form)))
  (setf (loop-plan-body-seen plan) t))

(defun %plan-add-initially (plan form)
  (setf (loop-plan-initially plan)
        (append (loop-plan-initially plan) (list form))))

(defun %plan-add-finally (plan form)
  (setf (loop-plan-finally plan)
        (append (loop-plan-finally plan) (list form))))

(defun %plan-add-accumulator (plan kind var extras)
  (setf (loop-plan-accumulators plan)
        (append (loop-plan-accumulators plan)
                (list (cons kind (cons var extras))))))

;; ── Clause parsers ────────────────────────────────────────────────────────

(defun %parse-for (plan cur)
  "(for VAR <spec>). spec is one of:
     in LIST                          → walk list
     on LIST                          → walk cons cells
     = EXPR [then EXPR]               → bind, optionally step
     from N [to|below|downto|above M] [by STEP]
     across VECTOR                    → walk vector by index

   `of-type TYPE` may appear between VAR and the spec keyword:
     (loop for a of-type integer in '(1 2 3) collect a)
   The type declaration is accepted but ignored — NCL's compiler
   doesn't yet act on CL type declarations, and accepting+ignoring
   matches what the test corpus assumes."
  (let ((var (%cur-eat! cur)))
    ;; Skip optional `of-type TYPE`. Token-eater eats both; the
    ;; declared type is discarded.
    (when (%cur-eat-keyword? cur 'of-type)
      (%cur-eat! cur))
    (cond
      ;; for VAR in LIST
      ((%cur-eat-keyword? cur 'in)
       (let* ((list-form (%cur-eat! cur))
              (tail-var (gensym "TAIL-")))
         (%plan-add-iter-binding plan tail-var list-form)
         (%plan-add-iter-binding plan var `(car ,tail-var))
         (%plan-add-test plan `(null ,tail-var))
         (%plan-add-step plan `(setq ,tail-var (cdr ,tail-var)))
         (%plan-add-step plan `(setq ,var (car ,tail-var)))))
      ;; for VAR on LIST
      ((%cur-eat-keyword? cur 'on)
       (let ((list-form (%cur-eat! cur)))
         (%plan-add-iter-binding plan var list-form)
         (%plan-add-test plan `(null ,var))
         (%plan-add-step plan `(setq ,var (cdr ,var)))))
      ;; for VAR across VECTOR
      ((%cur-eat-keyword? cur 'across)
       (let* ((vec-form (%cur-eat! cur))
              (vec-var (gensym "VEC-"))
              (i-var (gensym "I-"))
              (n-var (gensym "N-")))
         (%plan-add-iter-binding plan vec-var vec-form)
         (%plan-add-iter-binding plan i-var 0)
         (%plan-add-iter-binding plan n-var `(length ,vec-var))
         (%plan-add-iter-binding plan var `(svref ,vec-var 0))
         (%plan-add-test plan `(>= ,i-var ,n-var))
         (%plan-add-step plan `(setq ,i-var (+ ,i-var 1)))
         (%plan-add-step plan `(when (< ,i-var ,n-var)
                                 (setq ,var (svref ,vec-var ,i-var))))))
      ;; for VAR = EXPR [then EXPR]
      ((%cur-eat-keyword? cur '=)
       (let ((init (%cur-eat! cur)))
         (cond
           ((%cur-eat-keyword? cur 'then)
            (let ((step (%cur-eat! cur)))
              (%plan-add-iter-binding plan var init)
              (%plan-add-step plan `(setq ,var ,step))))
           (t
            ;; No `then` → re-evaluate INIT every iteration.
            (%plan-add-iter-binding plan var init)
            (%plan-add-step plan `(setq ,var ,init))))))
      ;; for VAR from N [to|below|downto|above M] [by STEP]
      ((%cur-eat-keyword? cur 'from)
       (%parse-from-clause plan cur var))
      (t (error "loop: unknown for-spec after ~A" var)))))

(defun %parse-from-clause (plan cur var)
  "Parse (for VAR from N [bound-keyword M] [by STEP])."
  (let* ((start (%cur-eat! cur))
         (cmp nil)
         (limit nil)
         (step 1)
         (direction 1))
    ;; Optional bound keyword.
    (cond
      ((%cur-eat-keyword? cur 'to)     (setq cmp '<=) (setq limit (%cur-eat! cur)))
      ((%cur-eat-keyword? cur 'below)  (setq cmp '<)  (setq limit (%cur-eat! cur)))
      ((%cur-eat-keyword? cur 'downto)
       (setq cmp '>=) (setq direction -1) (setq limit (%cur-eat! cur)))
      ((%cur-eat-keyword? cur 'above)
       (setq cmp '>)  (setq direction -1) (setq limit (%cur-eat! cur))))
    ;; Optional `by STEP`.
    (when (%cur-eat-keyword? cur 'by)
      (setq step (%cur-eat! cur)))
    (%plan-add-iter-binding plan var start)
    (cond
      (cmp
       (%plan-add-test plan `(not (,cmp ,var ,limit))))
      ;; No bound → infinite range; user must use while/until/return.
      )
    (cond
      ((eql direction 1)  (%plan-add-step plan `(setq ,var (+ ,var ,step))))
      (t                  (%plan-add-step plan `(setq ,var (- ,var ,step)))))))

(defun %parse-with (plan cur)
  "(with VAR [= INIT])."
  (let ((var (%cur-eat! cur)))
    (cond
      ((%cur-eat-keyword? cur '=)
       (let ((init (%cur-eat! cur)))
         (%plan-add-with plan var init)))
      (t (%plan-add-with plan var nil)))))

(defun %parse-while (plan cur)
  (let ((expr (%cur-eat! cur)))
    (%plan-add-test plan `(not ,expr))))

(defun %parse-until (plan cur)
  (let ((expr (%cur-eat! cur)))
    (%plan-add-test plan expr)))

(defun %parse-repeat (plan cur)
  (let ((expr (%cur-eat! cur))
        (counter (gensym "REPEAT-")))
    (%plan-add-iter-binding plan counter expr)
    (%plan-add-test plan `(<= ,counter 0))
    (%plan-add-step plan `(setq ,counter (- ,counter 1)))))

(defun %parse-collect (plan cur)
  (let ((expr (%cur-eat! cur))
        (acc (gensym "COLLECT-")))
    (%plan-add-with plan acc nil)
    (%plan-add-body plan `(setq ,acc (cons ,expr ,acc)))
    (%plan-add-accumulator plan 'collect acc nil)))

(defun %parse-append (plan cur)
  (let ((expr (%cur-eat! cur))
        (acc (gensym "APPEND-")))
    (%plan-add-with plan acc nil)
    (%plan-add-body plan `(setq ,acc (append* ,acc ,expr)))
    (%plan-add-accumulator plan 'append acc nil)))

(defun %parse-sum (plan cur)
  (let ((expr (%cur-eat! cur))
        (acc (gensym "SUM-")))
    (%plan-add-with plan acc 0)
    (%plan-add-body plan `(setq ,acc (+ ,acc ,expr)))
    (%plan-add-accumulator plan 'sum acc nil)))

(defun %parse-count (plan cur)
  (let ((expr (%cur-eat! cur))
        (acc (gensym "COUNT-")))
    (%plan-add-with plan acc 0)
    (%plan-add-body plan `(when ,expr (setq ,acc (+ ,acc 1))))
    (%plan-add-accumulator plan 'count acc nil)))

(defun %parse-minimize (plan cur)
  (let ((expr (%cur-eat! cur))
        (acc (gensym "MIN-"))
        (val (gensym "V-")))
    (%plan-add-with plan acc nil)
    (%plan-add-body plan
                    `(let ((,val ,expr))
                       (when (or (null ,acc) (< ,val ,acc))
                         (setq ,acc ,val))))
    (%plan-add-accumulator plan 'min acc nil)))

(defun %parse-maximize (plan cur)
  (let ((expr (%cur-eat! cur))
        (acc (gensym "MAX-"))
        (val (gensym "V-")))
    (%plan-add-with plan acc nil)
    (%plan-add-body plan
                    `(let ((,val ,expr))
                       (when (or (null ,acc) (> ,val ,acc))
                         (setq ,acc ,val))))
    (%plan-add-accumulator plan 'max acc nil)))

;; (do FORM ...) — run forms each iteration, no accumulation.
;; Reads forms until the next clause keyword or end of body.
(defun %parse-do (plan cur)
  (loop
    (cond
      ((%cur-empty? cur) (return nil))
      ((%loop-keyword-p (%cur-peek cur)) (return nil))
      (t (%plan-add-body plan (%cur-eat! cur))))))

(defun %parse-initially (plan cur)
  (loop
    (cond
      ((%cur-empty? cur) (return nil))
      ((%loop-keyword-p (%cur-peek cur)) (return nil))
      (t (%plan-add-initially plan (%cur-eat! cur))))))

(defun %parse-finally (plan cur)
  (loop
    (cond
      ((%cur-empty? cur) (return nil))
      ((%loop-keyword-p (%cur-peek cur)) (return nil))
      (t (%plan-add-finally plan (%cur-eat! cur))))))

;; (return EXPR) — exit with EXPR. Returns from the implicit block.
(defun %parse-return (plan cur)
  (let ((expr (%cur-eat! cur))
        (name (or (loop-plan-name plan) 'nil)))
    (%plan-add-body plan `(return-from ,name ,expr))))

;; (when EXPR CLAUSE) / (unless EXPR CLAUSE) — single sub-clause.
;; We bury the sub-clause's effect inside an (if ...) in the body.
;; Implementation: temporarily save the body length, parse the
;; sub-clause's body additions, then wrap them in a conditional.
(defun %parse-when-unless (plan cur negate)
  (let* ((expr (%cur-eat! cur))
         (saved-len (length (loop-plan-body plan))))
    (%parse-one-clause plan cur)
    (let ((added (subseq (loop-plan-body plan) saved-len)))
      ;; Trim plan body back to its pre-sub-clause length.
      (setf (loop-plan-body plan) (subseq (loop-plan-body plan) 0 saved-len))
      ;; Wrap the additions in an if.
      (let ((wrapped
             (cond
               ((null added) nil)
               (negate `(unless ,expr ,@added))
               (t      `(when ,expr ,@added)))))
        (when wrapped
          (%plan-add-body plan wrapped))))))

;; Helper used by when/unless to parse exactly one sub-clause.
(defun %parse-one-clause (plan cur)
  (let ((head (%cur-eat! cur)))
    (cond
      ((eq head 'do)         (%plan-add-body plan (%cur-eat! cur)))
      ((eq head 'collect)    (%parse-collect plan cur))
      ((eq head 'collecting) (%parse-collect plan cur))
      ((eq head 'append)     (%parse-append plan cur))
      ((eq head 'appending)  (%parse-append plan cur))
      ((eq head 'sum)        (%parse-sum plan cur))
      ((eq head 'summing)    (%parse-sum plan cur))
      ((eq head 'count)      (%parse-count plan cur))
      ((eq head 'counting)   (%parse-count plan cur))
      ((eq head 'minimize)   (%parse-minimize plan cur))
      ((eq head 'minimizing) (%parse-minimize plan cur))
      ((eq head 'maximize)   (%parse-maximize plan cur))
      ((eq head 'maximizing) (%parse-maximize plan cur))
      ((eq head 'return)     (%parse-return plan cur))
      (t (error "loop: unsupported sub-clause head ~A" head)))))

;; ── Top-level parser ──────────────────────────────────────────────────────

(defun %parse-loop (body)
  (let ((plan (make-loop-plan))
        (cur (%make-cursor body)))
    ;; Optional leading (named NAME) clause.
    (when (eq (%cur-peek cur) 'named)
      (%cur-eat! cur)
      (setf (loop-plan-name plan) (%cur-eat! cur)))
    (loop
      (cond
        ((%cur-empty? cur) (return nil))
        (t (let ((head (%cur-eat! cur)))
             (cond
               ((or (eq head 'for) (eq head 'as)) (%parse-for plan cur))
               ((eq head 'with)       (%parse-with plan cur))
               ((eq head 'while)      (%parse-while plan cur))
               ((eq head 'until)      (%parse-until plan cur))
               ((eq head 'repeat)     (%parse-repeat plan cur))
               ((eq head 'do)         (%parse-do plan cur))
               ((eq head 'doing)      (%parse-do plan cur))
               ((eq head 'initially)  (%parse-initially plan cur))
               ((eq head 'finally)    (%parse-finally plan cur))
               ((eq head 'collect)    (%parse-collect plan cur))
               ((eq head 'collecting) (%parse-collect plan cur))
               ((eq head 'append)     (%parse-append plan cur))
               ((eq head 'appending)  (%parse-append plan cur))
               ((eq head 'sum)        (%parse-sum plan cur))
               ((eq head 'summing)    (%parse-sum plan cur))
               ((eq head 'count)      (%parse-count plan cur))
               ((eq head 'counting)   (%parse-count plan cur))
               ((eq head 'minimize)   (%parse-minimize plan cur))
               ((eq head 'minimizing) (%parse-minimize plan cur))
               ((eq head 'maximize)   (%parse-maximize plan cur))
               ((eq head 'maximizing) (%parse-maximize plan cur))
               ((eq head 'when)       (%parse-when-unless plan cur nil))
               ((eq head 'unless)     (%parse-when-unless plan cur t))
               ((eq head 'if)         (%parse-when-unless plan cur nil))
               ((eq head 'return)     (%parse-return plan cur))
               (t (error "loop: unknown clause head ~A" head)))))))
    plan))

;; ── Code generation ───────────────────────────────────────────────────────

(defun %loop-rewrite-return (form name)
  "Replace `(return X)` and `(return)` inside FORM with
   `(return-from NAME X)` / `(return-from NAME nil)`. Stops
   descending into nested (loop ...) and (block ...) forms so
   their own returns target their own blocks."
  (cond
    ((atom form) form)
    ((eq (car form) 'loop) form)
    ((eq (car form) 'block) form)
    ((eq (car form) 'return)
     (cond
       ((null (cdr form)) `(return-from ,name nil))
       (t `(return-from ,name ,(cadr form)))))
    (t (mapcar (lambda (sub) (%loop-rewrite-return sub name)) form))))

(defun %loop-result-form (plan)
  "Pick the final-value form. CL says the value of the last
   accumulator clause; collect's accumulator was built with cons-
   prepend so we reverse it back."
  (let ((accs (loop-plan-accumulators plan)))
    (cond
      ((null accs) nil)
      (t
       (let ((last (car (last accs))))
         (let ((kind (car last))
               (var (cadr last)))
           (cond
             ((eq kind 'collect) `(reverse ,var))
             (t var))))))))

(defun %loop-emit (plan)
  "Walk PLAN, emit the full expansion. Shape:
     (block NAME
       (let (with-bindings + iter-bindings + accumulator vars)
         (initially-forms ...)
         (loop
           (when (or test1 test2 ...) (return nil))
           body-forms ...
           step-forms ...)
         finally-forms ...
         result-form))"
  (let* ((name (or (loop-plan-name plan) 'nil))
         (all-bindings
          (append
           ;; with-bindings come first (user-visible scope).
           (mapcar (lambda (pair) `(,(car pair) ,(cdr pair)))
                   (loop-plan-with-bindings plan))
           ;; iter-bindings come second (loop-internal).
           (mapcar (lambda (pair) `(,(car pair) ,(cdr pair)))
                   (loop-plan-iter-bindings plan))))
         (tests (loop-plan-iter-tests plan))
         (post-tests (loop-plan-post-body-tests plan))
         (test-form
          (cond
            ((null tests) 'nil)            ; no automatic termination
            ((null (cdr tests)) (car tests))
            (t `(or ,@tests))))
         (post-test-form
          (cond
            ((null post-tests) 'nil)       ; no post-body termination
            ((null (cdr post-tests)) (car post-tests))
            (t `(or ,@post-tests))))
         ;; Rewrite (return X) → (return-from NAME X) so user
         ;; code's `return` inside body/initially/finally reaches
         ;; the implicit block. The body's accumulator-setq forms
         ;; we added don't contain `return`, but it's harmless to
         ;; walk them.
         (body (mapcar (lambda (f) (%loop-rewrite-return f name))
                       (loop-plan-body plan)))
         (steps (loop-plan-iter-steps plan))
         (initially (mapcar (lambda (f) (%loop-rewrite-return f name))
                            (loop-plan-initially plan)))
         (finally (mapcar (lambda (f) (%loop-rewrite-return f name))
                          (loop-plan-finally plan)))
         (result (%loop-result-form plan)))
    ;; The simple `loop`'s `return` doesn't unwind — subsequent
    ;; body forms run after it (see core.lisp's caveat). So we
    ;; structure each iteration as cond: if the test fires, return;
    ;; else run body + steps. The cond branch with `return` is the
    ;; ONLY form in that branch — nothing follows to execute after
    ;; the return-flag is set.
    ;; Iteration shape:
    ;;   pre-body tests first (for / while / until that appeared BEFORE
    ;;   any do clause), then body, then post-body tests (while / until
    ;;   that appeared AFTER do clauses), then steps. Each test bails
    ;;   out of the implicit block. With only pre-tests this collapses
    ;;   to the original shape.
    ;;
    ;; A non-trivial subtlety: when STEPS is empty, the inner
    ;; `(t ,@steps)` clause becomes `(t)`, which NCL's lowerer rejects
    ;; ("cond clause with only a test"). Splice a NIL body in that
    ;; case so the resulting cond is always well-formed.
    (let* ((step-clause (if steps `(t ,@steps) '(t nil)))
           (post-test-cond
            (cond
              ;; No post-test → skip the inner cond entirely; just
              ;; run steps after body.
              ((null post-tests) `(progn ,@steps))
              (t
               `(cond
                  (,post-test-form (return nil))
                  ,step-clause)))))
      `(block ,name
         (let* ,all-bindings
           ,@initially
           (loop
             (cond
               (,test-form (return nil))
               (t ,@body
                  ,post-test-cond)))
           ,@finally
           ,result)))))

;; ── Macro redefinition ────────────────────────────────────────────────────
;;
;; The dispatch: simple-loop usage (first form is a list or
;; non-keyword atom) goes to the old %native-loop path; extended-
;; loop usage goes through the parser + emitter.

(defmacro loop (&rest body)
  (cond
    ((null body)              `(%native-loop (lambda ())))
    ((consp (car body))       `(%native-loop (lambda () ,@body)))
    ((null (car body))        `(%native-loop (lambda () ,@body)))
    ((%loop-keyword-p (car body))
     (%loop-emit (%parse-loop body)))
    (t                        `(%native-loop (lambda () ,@body)))))

(provide 'loop)
nil
