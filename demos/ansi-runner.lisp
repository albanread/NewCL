;;;; ansi-runner.lisp — driver for Corman's ANSI hyperspec-examples
;;;; test suite (E:\CL\cormanlisp\test\ansi-chapter-*.lisp).
;;;;
;;;; The corman suite ships as `ansi-examples.lisp` (defines `dotests`
;;;; + verify, loads chapters 2-8). This wrapper recreates that
;;;; harness in a form NCL can run today: stubs `in-package` (we have
;;;; no real package system yet), defines `dotests` / `verify` /
;;;; `passed-test` / `failed-test`, then loads each chapter file by
;;;; absolute path. Counts pass/fail at the end.

;; ── No-op shims ─────────────────────────────────────────────────────

(defmacro in-package (&rest args)
  (declare (ignore args))
  nil)

;; ── Counters (so we get a tally at the end) ─────────────────────────

(defparameter *ansi-pass-count* 0)
(defparameter *ansi-fail-count* 0)
(defparameter *ansi-error-count* 0)

;; ── Test reporters ──────────────────────────────────────────────────

(defun passed-test (expr result)
  (setq *ansi-pass-count* (+ *ansi-pass-count* 1))
  ;; Quiet by default; uncomment for verbose.
  ;; (format t "PASSED: ~S => ~S~%" expr result)
  nil)

(defun failed-test (expr result expected-result)
  (setq *ansi-fail-count* (+ *ansi-fail-count* 1))
  (format t "~%FAILED: ~S~%  got:      ~S~%  expected: ~S~%"
          expr result expected-result))

;; equalp on lists with true/false designators (CL's hyperspec-example
;; result vocabulary). Same as the corman version, minimised.
(defun test-equalp (a b)
  (or (equalp a b)
      (and (listp a) (listp b)
           (every (lambda (x y)
                    (or (and x (eq y 'true))
                        (and (not x) (eq y 'false))
                        (equalp x y)))
                  a b))))

;; ── check + dotests ─────────────────────────────────────────────────
;;
;; The corman version of `dotests` builds a quoted list of test forms
;; and walks it with `eval`. NCL has no Lisp-side eval — every form
;; goes through the JIT-compile pipeline at read time. We rewrite
;; `dotests` as a code-generating macro: each (expr => expected) triple
;; expands into a `(check-one '<expr-quoted> <expr> '<expected>)` call,
;; so the test expression is compiled and executed directly. Errors
;; during the expression are caught with handler-case so one bad test
;; doesn't kill the suite.

(defun check-one-result (expr result-list expected-result)
  (case expected-result
    (true
     (if (car result-list)
         (passed-test expr (car result-list))
         (failed-test expr (car result-list) expected-result)))
    (false
     (if (car result-list)
         (failed-test expr (car result-list) expected-result)
         (passed-test expr (car result-list))))
    (implementation-dependent
     (passed-test expr (car result-list)))
    (t
     (cond
       ((and (consp expected-result)
             (eq (car expected-result) 'values))
        (if (test-equalp result-list (cdr expected-result))
            (passed-test expr (cons 'values result-list))
            (failed-test expr (cons 'values result-list) expected-result)))
       ((equalp (car result-list) expected-result)
        (passed-test expr (car result-list)))
       (t
        (failed-test expr (car result-list) expected-result))))))

(defmacro check-one (expr-quoted expr-thunk expected)
  ;; expr-thunk is a (lambda () expr); we call it inside handler-case.
  `(handler-case
       (check-one-result
         ,expr-quoted
         (multiple-value-list (funcall ,expr-thunk))
         ,expected)
     (error (c)
       (setq *ansi-error-count* (+ *ansi-error-count* 1))
       (format t "~%ERROR in ~S: ~A~%" ,expr-quoted c))))

(defmacro dotests (symbol &rest examples)
  ;; Walk EXAMPLES three at a time: (expr => expected) (expr => expected) ...
  ;; The macro is hand-rolled because `loop … on … by 'cdddr` and
  ;; `destructuring-bind` aren't on solid ground here yet.
  ;;
  ;; Corman's chapter files use bare commas as English punctuation
  ;; after the expected result, e.g.
  ;;
  ;;     (function-keywords *)  =>  (:C :DEE :E EFF), false
  ;;
  ;; Standard CL readers parse `,false` as `(UNQUOTE FALSE)` and
  ;; reject it outside backquote (we do). Corman's runner happens to
  ;; collect these as "secondary value" hints which it then ignores.
  ;; To stay aligned with the (expr => expected) triple cadence we
  ;; filter UNQUOTE / UNQUOTE-SPLICING forms out of the example list
  ;; before grouping — same net effect.
  (labels ((unquote-form-p (x)
             (and (consp x)
                  (consp (cdr x))
                  (symbolp (car x))
                  (or (eq (car x) 'unquote)
                      (eq (car x) 'unquote-splicing))))
           (strip-unquotes (xs)
             (cond
               ((null xs) nil)
               ((unquote-form-p (car xs)) (strip-unquotes (cdr xs)))
               (t (cons (car xs) (strip-unquotes (cdr xs))))))
           (triples (xs acc)
             (cond
               ((null xs) (nreverse acc))
               ((or (null (cdr xs)) (null (cddr xs)))
                (nreverse acc)) ; trailing partial group: drop
               (t (let ((expr (first xs))
                        (arrow (second xs))
                        (expected (third xs)))
                    (declare (ignore arrow))
                    (triples (cdddr xs)
                             (cons (list expr expected) acc)))))))
    (let ((groups (triples (strip-unquotes examples) nil)))
      `(progn
         (format t "~&Testing ~A " ',symbol)
         (force-output)
         ,@(mapcar (lambda (g)
                     (let ((expr (first g))
                           (expected (second g)))
                       `(check-one ',expr (lambda () ,expr) ',expected)))
                   groups)
         (format t "~%")))))

;; ── Load the chapters ──────────────────────────────────────────────

(defparameter *ansi-base*
  "E:/CL/cormanlisp/test/")

(defun run-ansi-chapter (n)
  (let ((path (format nil "~Aansi-chapter-~A.lisp" *ansi-base* n)))
    (format t "~%==== chapter ~A (~A) ====~%" n path)
    (force-output)
    (handler-case
        (load path)
      (error (c)
        (format t "~%load error in chapter ~A: ~A~%" n c)))))

(format t "~%~%==== Corman ANSI hyperspec-examples suite ====~%")
(force-output)
(setq *ansi-pass-count* 0)
(setq *ansi-fail-count* 0)
(setq *ansi-error-count* 0)

(run-ansi-chapter 2)
(run-ansi-chapter 3)
(run-ansi-chapter 4)
(run-ansi-chapter 5)
(run-ansi-chapter 6)
(run-ansi-chapter 8)
;; Chapter 7 last: it exercises method-combination paths that
;; currently trigger an infinite recursion in CLOS's FIND/
;; SUBCLASSP (visible as a Windows stack overflow that kills the
;; process). Running it last preserves the suite's pass/fail tally
;; for chapters 2-6 and 8. Fix lands when the FIND recursion is
;; properly bounded.
(run-ansi-chapter 7)

(format t "~%~%==== ANSI suite summary ====~%")
(format t "  passed: ~A~%" *ansi-pass-count*)
(format t "  failed: ~A~%" *ansi-fail-count*)
(format t "  errors: ~A~%" *ansi-error-count*)
(format t "  total:  ~A~%"
        (+ *ansi-pass-count* *ansi-fail-count* *ansi-error-count*))
(force-output)
