;;;; bench/gauntlet.lisp — fast correctness gauntlet for performance work.
;;;;
;;;; Run:  ./target/release/ncl.exe --load bench/gauntlet.lisp
;;;;       (or the release nclterm.exe — any console build).
;;;;
;;;; Prints PASS/FAIL per check and a final "GAUNTLET ALL-PASS" / "N FAILS".
;;;;
;;;; A *targeted* regression gate that complements the broad ANSI suite
;;;; (demos/ansi-runner.lisp): it exercises exactly the surfaces the perf
;;;; changes touch, so a behavioural regression shows up in <1s instead of
;;;; in a 600-case conformance run. Keep it fast, and extend it whenever a
;;;; perf change lands. Sections below map 1:1 to landed work:
;;;;   LOOP plan        -> LOOP append->cons+nreverse rewrite
;;;;   strings          -> O(n) string-upcase/downcase/capitalize
;;;;   eql / cmp tower  -> "decode tags once" in eql_values / ncl_cmp_full
;;;;   apply            -> ncl_apply <=16-arg stack buffer (17/20 cross it)
;;;;   alloc + GC churn -> cheap safepoint poll + inline root stack
;;;; Next to add: unboxed-float arithmetic — see
;;;; docs/performance-unbox-float.md (a float-correctness section is the
;;;; gate for that work's Sprint 1).

(defparameter *fails* 0)
(defun chk (name got want)
  (cond
    ((equal got want) (format t "PASS ~A~%" name))
    (t (setq *fails* (+ *fails* 1))
       (format t "FAIL ~A: got ~S want ~S~%" name got want))))

;; ── LOOP plan ordering ──────────────────────────────────────────────
(chk "loop-collect-order" (loop for i from 1 to 5 collect i) '(1 2 3 4 5))
(chk "loop-multi-with"
     (loop with a = 1 with b = 2 with c = 3 for i from 0 below 1 collect (list a b c))
     '((1 2 3)))
(chk "loop-body-order"
     (let ((acc nil))
       (loop for i from 1 to 3 do (setq acc (cons i acc)) do (setq acc (cons (* i 10) acc)))
       (reverse acc))
     '(1 10 2 20 3 30))
(chk "loop-when-collect" (loop for i from 1 to 10 when (evenp i) collect i) '(2 4 6 8 10))
(chk "loop-unless-collect" (loop for i from 1 to 6 unless (evenp i) collect i) '(1 3 5))
(chk "loop-last-accum-wins" (loop for i from 1 to 4 collect i sum i) 10)
(chk "loop-sum-then-collect" (loop for i from 1 to 4 sum i collect i) '(1 2 3 4))
(chk "loop-into-finally"
     (loop for i from 1 to 4 collect i into xs finally (return (reverse xs)))
     '(4 3 2 1))
(chk "loop-post-until"
     (let ((n 0)) (loop do (setq n (+ n 1)) until (>= n 5)) n)
     5)
(chk "loop-initially"
     (let ((acc nil))
       (loop initially (setq acc (cons :init acc))
             for i from 1 to 2 do (setq acc (cons i acc)))
       (reverse acc))
     '(:init 1 2))
(chk "loop-max-min" (list (loop for i in '(3 1 4 1 5) maximize i)
                          (loop for i in '(3 1 4 1 5) minimize i))
     '(5 1))
(chk "loop-while-steps"
     (let ((x 0)) (loop for i from 0 below 100 while (< i 3) do (setq x i)) x)
     2)
(chk "loop-append" (loop for x in '((1 2) (3) (4 5)) append x) '(1 2 3 4 5))
(chk "loop-count" (loop for i from 1 to 10 count (evenp i)) 5)
(chk "loop-named"
     (loop named outer for i from 0 below 10
           do (when (= i 3) (return-from outer (* i 7))))
     21)

;; ── strings ─────────────────────────────────────────────────────────
(chk "upcase" (string-upcase "hello world") "HELLO WORLD")
(chk "downcase" (string-downcase "HeLLo") "hello")
(chk "upcase-range" (string-upcase "abcdef" :start 2 :end 4) "abCDef")
(chk "downcase-range" (string-downcase "ABCDEF" :start 1 :end 3) "AbcDEF")
(chk "capitalize" (string-capitalize "hello world 3rd time") "Hello World 3rd Time")
(chk "capitalize-range" (string-capitalize "hello world" :start 6) "hello World")
(chk "upcase-empty" (string-upcase "") "")
(chk "upcase-symbol" (string-upcase 'foo) "FOO")

;; ── eql / cmp numeric tower ─────────────────────────────────────────
(chk "eql-float" (eql 1.5 1.5) t)
(chk "eql-float-ne" (eql 1.5 2.5) nil)
(chk "eql-cross-type" (eql 3 3.0) nil)
(chk "eql-bignum" (eql (expt 2 100) (expt 2 100)) t)
(chk "eql-ratio" (eql 1/3 1/3) t)
(chk "eql-ratio-ne" (eql 1/3 2/3) nil)
(chk "eql-complex" (eql #c(3 -4) #c(3 -4)) t)
(chk "eql-complex-ne" (eql #c(3 -4) #c(3 4)) nil)
(chk "eql-fixnum" (eql 42 42) t)
(chk "eql-char" (eql #\a #\a) t)
(chk "eql-string-not" (eql "ab" "ab") nil)
(chk "=-mixed" (= 1 1.0) t)
(chk "<-mixed" (< 1 2.5 3) t)
(chk "=-big-ratio" (= (/ (expt 2 100) (expt 2 99)) 2) t)
(chk "cmp-complex-eq" (= #c(2 3) #c(2 3)) t)
(chk "equal-float" (equal 3.0 3.0) t)

;; ── apply: stack path + heap path ───────────────────────────────────
(chk "apply-small" (apply #'+ 1 2 '(3 4)) 10)
(chk "apply-empty-tail" (apply #'+ 1 2 '()) 3)
(chk "apply-no-prefix" (apply #'list '(1 2 3)) '(1 2 3))
(chk "apply-17-args"
     (apply #'+ 1 2 3 4 5 6 7 8 '(9 10 11 12 13 14 15 16 17))
     153)
(chk "apply-20-args" (apply #'list 1 2 3 '(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20))
     '(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20))

;; ── alloc + GC under churn (safepoint early-out + inline root stack) ─
(chk "cons-churn-gc"
     (let ((keep nil))
       (dotimes (i 200)
         (let ((l nil))
           (dotimes (j 5000) (setq l (cons j l)))
           (when (= (mod i 50) 0) (setq keep (cons (length l) keep)))))
       keep)
     '(5000 5000 5000 5000))
(chk "sort-after-churn"
     (sort (list 5 3 9 1 7 2 8 4 6) #'<)
     '(1 2 3 4 5 6 7 8 9))

(format t "~%GAUNTLET ~A~%" (if (= *fails* 0) "ALL-PASS" (format nil "~A FAILS" *fails*)))
