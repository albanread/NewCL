;;;; core.lisp — the user-Lisp portion of NewCormanLisp's standard
;;;; library. This file is loaded by Session::load_core_stdlib at
;;;; session start.
;;;;
;;;; Everything in this file is plain Lisp using the primitives
;;;; defined by the compiler (cons/car/cdr, equal, +/-, if/cond,
;;;; let, defun, lambda, funcall, setq, setf). When a function
;;;; appears here it lives as a defun whose code is JIT-compiled at
;;;; load time and installed in the symbol's function cell — the
;;;; same path user code goes through.
;;;;
;;;; Conventions:
;;;;   - Helpers prefixed with % are internal; don't depend on them
;;;;     in user code.
;;;;   - Predicates that return T or NIL match Common Lisp.
;;;;   - Test-equality default is EQUAL (deep), not EQL. CL's exact
;;;;     EQL/EQUAL/EQUALP/:test distinction lands when keyword
;;;;     arguments do.

;; -- Trivial accessors --------------------------------------------------------

(defun first (lst) (car lst))
(defun rest (lst) (cdr lst))

(defun second (lst) (car (cdr lst)))
(defun third (lst) (car (cdr (cdr lst))))
(defun fourth (lst) (car (cdr (cdr (cdr lst)))))

(defun caar (lst) (car (car lst)))
(defun cadr (lst) (car (cdr lst)))
(defun cdar (lst) (cdr (car lst)))
(defun cddr (lst) (cdr (cdr lst)))

(defun identity (x) x)

;; -- reverse, append ---------------------------------------------------------

(defun %revappend (lst acc)
  ;; (revappend lst acc) ≡ (append (reverse lst) acc), tail recursive.
  (if (null lst)
      acc
      (%revappend (cdr lst) (cons (car lst) acc))))

(defun reverse (lst)
  (%revappend lst nil))

(defun append (a b)
  ;; Binary append. Variadic CL append lands when &rest does.
  (if (null a)
      b
      (cons (car a) (append (cdr a) b))))

;; -- mapcar, mapc, every, some -----------------------------------------------

(defun mapcar (fn lst)
  (if (null lst)
      nil
      (cons (funcall fn (car lst))
            (mapcar fn (cdr lst)))))

(defun mapc (fn lst)
  ;; Like mapcar but returns the original list and is called for
  ;; effect.
  (if (null lst)
      lst
      (progn (funcall fn (car lst))
             (mapc fn (cdr lst))
             lst)))

(defun every (pred lst)
  ;; True iff pred is non-nil for every element.
  (cond
    ((null lst) t)
    ((funcall pred (car lst)) (every pred (cdr lst)))
    (t nil)))

(defun some (pred lst)
  ;; Returns the first non-nil value of pred over the list, or nil.
  (cond
    ((null lst) nil)
    (t (let ((v (funcall pred (car lst))))
         (if v v (some pred (cdr lst)))))))

;; -- member, position, find, assoc -------------------------------------------

(defun member (item lst)
  ;; CL's `member` returns the tail of lst starting at the first
  ;; match (or nil). Comparison uses equal — CL's default is eql,
  ;; but until we have keyword args, equal is the more useful
  ;; default.
  (cond
    ((null lst) nil)
    ((equal item (car lst)) lst)
    (t (member item (cdr lst)))))

(defun %position-from (item lst i)
  (cond
    ((null lst) nil)
    ((equal item (car lst)) i)
    (t (%position-from item (cdr lst) (+ i 1)))))

(defun position (item lst)
  (%position-from item lst 0))

(defun find (item lst)
  (cond
    ((null lst) nil)
    ((equal item (car lst)) (car lst))
    (t (find item (cdr lst)))))

(defun assoc (key alist)
  ;; Walk an alist; return the first entry whose car is `equal` to
  ;; key, or nil.
  (cond
    ((null alist) nil)
    ((equal key (car (car alist))) (car alist))
    (t (assoc key (cdr alist)))))

;; -- nth, nthcdr, last -------------------------------------------------------

(defun nthcdr (n lst)
  (if (= n 0)
      lst
      (nthcdr (- n 1) (cdr lst))))

(defun nth (n lst)
  (car (nthcdr n lst)))

(defun last (lst)
  ;; CL's `last` returns the LAST CONS CELL of lst, not the last
  ;; element. (last '(1 2 3)) is (3), not 3.
  (cond
    ((null lst) nil)
    ((null (cdr lst)) lst)
    (t (last (cdr lst)))))

(defun butlast (lst)
  ;; Returns lst with its last cons removed.
  (cond
    ((null lst) nil)
    ((null (cdr lst)) nil)
    (t (cons (car lst) (butlast (cdr lst))))))

;; -- list construction helpers -----------------------------------------------

(defun copy-list (lst)
  (if (null lst)
      nil
      (cons (car lst) (copy-list (cdr lst)))))

(defun list-length (lst)
  ;; Same as the LENGTH primitive on lists; provided for symmetry.
  (length lst))

;; -- Numeric helpers ---------------------------------------------------------

(defun zerop (n) (= n 0))
(defun plusp (n) (> n 0))
(defun minusp (n) (< n 0))

;; CL `mod` matches the sign of the divisor; `rem` matches the
;; sign of the dividend. They differ only when divisor and
;; dividend have opposite signs and the remainder is non-zero.
(defun mod (a b)
  (let ((r (rem a b)))
    (if (zerop r)
        0
        (if (eq (minusp r) (minusp b))
            r
            (+ r b)))))

(defun evenp (n) (zerop (rem n 2)))
(defun oddp (n) (not (evenp n)))

;; (floor a b): largest integer k such that k*b <= a (when b > 0;
;; flips for b < 0). Differs from truncate only when sign(a) !=
;; sign(b) and there's a non-zero remainder, in which case floor
;; rounds further from zero.
(defun floor (a b)
  (let ((q (truncate a b))
        (r (rem a b)))
    (if (and (not (zerop r))
             (not (eq (minusp r) (minusp b))))
        (- q 1)
        q)))

(defun 1+ (n) (+ n 1))
(defun 1- (n) (- n 1))

;; CL `min` and `max` are variadic; until &rest lands these are the
;; binary forms.
(defun min2 (a b) (if (< a b) a b))
(defun max2 (a b) (if (> a b) a b))

(defun abs (n) (if (< n 0) (- n) n))
