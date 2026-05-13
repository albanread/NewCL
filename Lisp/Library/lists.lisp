;;;; Lisp/Library/lists.lisp — CL list-traversal extras.
;;;;
;;;; Ported from Roger Corman's Sys/lists.lisp. Provides:
;;;;
;;;;   * The mapping family — MAPL, MAPLIST, MAPCAN, MAPCON. (MAPCAR
;;;;     and MAPC are already in core.lisp; this module adds the
;;;;     four cdr-chain / concatenating variants.)
;;;;   * Alist helpers — PAIRLIS, ACONS.
;;;;   * Tail-sharing helpers — TAILP, LDIFF.
;;;;
;;;; Scope note: CL's mapping family is variadic over the input
;;;; lists — (MAPCAR fn a b c) walks three lists in parallel. The
;;;; Corman implementation expresses this with DO loops and
;;;; RETURN-FROM blocks, neither of which NCL has yet. This module
;;;; ports the unary (one input list) shapes only, which cover the
;;;; overwhelming majority of real uses. Variadic versions land
;;;; when DO / named-block support arrives.

;; ── Map family ─────────────────────────────────────────────────────────
;;
;; The four operators differ along two axes:
;;
;;          element-arg          cdr-chain-arg
;;   collect    MAPCAR              MAPLIST
;;   nconc      MAPCAN              MAPCON
;;   discard    MAPC                MAPL
;;
;; MAPCAR / MAPC live in core.lisp. We add the other four.

(defun mapl (fn lst)
  "Apply FN successively to LST, (CDR LST), (CDDR LST), … until the
   list is exhausted. Called for effect; returns LST unchanged."
  (cond
    ((null lst) lst)
    (t (funcall fn lst)
       (mapl fn (cdr lst))
       lst)))

(defun maplist (fn lst)
  "Apply FN to LST, (CDR LST), (CDDR LST), … collecting each
   result into a fresh list. Like MAPCAR but the function receives
   the *tail* at each step, not the head element."
  (cond
    ((null lst) nil)
    (t (cons (funcall fn lst)
             (maplist fn (cdr lst))))))

(defun mapcan (fn lst)
  "Apply FN to each element of LST and NCONC the results together.
   FN must return a list at every step; an empty return contributes
   no elements."
  (cond
    ((null lst) nil)
    (t (let ((head (funcall fn (car lst))))
         (nconc head (mapcan fn (cdr lst)))))))

(defun mapcon (fn lst)
  "Apply FN to LST, (CDR LST), … and NCONC the results. Each FN
   call receives a tail; each return is concatenated into the
   final list."
  (cond
    ((null lst) nil)
    (t (let ((head (funcall fn lst)))
         (nconc head (mapcon fn (cdr lst)))))))

;; ── Alist helpers ──────────────────────────────────────────────────────

(defun acons (key datum alist)
  "Prepend (KEY . DATUM) to ALIST and return the new list. The
   one-line definition is the CL spec verbatim."
  (cons (cons key datum) alist))

(defun %pairlis-recur (keys vals tail)
  (cond
    ((or (null keys) (null vals)) tail)
    (t (cons (cons (car keys) (car vals))
             (%pairlis-recur (cdr keys) (cdr vals) tail)))))

(defun pairlis (keys vals &optional tail)
  "Zip KEYS and VALS into an alist of (key . value) cells. TAIL, if
   supplied, is the cdr of the last cell — i.e. the new pairs are
   prepended to it. The spec leaves the order of the resulting
   pairs unspecified (Corman builds them via MAPCAR + NCONC so the
   first key/value ends up at the head); we preserve that order."
  (%pairlis-recur keys vals tail))

;; ── Tail-sharing tests ─────────────────────────────────────────────────

(defun tailp (object lst)
  "T iff OBJECT is one of the cons cells reachable by repeated
   CDR from LST (including LST itself). Comparison is EQL on the
   cons cells. Useful for asking whether one list is a structural
   suffix of another."
  (cond
    ((eq object lst) t)
    ((atom lst) nil)
    (t (tailp object (cdr lst)))))

(defun %ldiff-recur (lst object acc)
  ;; Collect cars onto ACC in reverse order until we EQ-hit OBJECT
  ;; or run off the end (atom-cdr). Reverse ACC for the result;
  ;; if we ran off the end with a non-nil dotted tail, append it.
  (cond
    ((eq object lst)
     (nreverse acc))
    ((atom lst)
     ;; The list ended at an atom that isn't OBJECT. Per CL spec
     ;; LDIFF returns a *copy* — the dotted tail is appended only
     ;; if it isn't nil.
     (if (null lst)
         (nreverse acc)
         (let ((front (nreverse acc)))
           ;; Splice the dotted tail onto the end via setf on the
           ;; last cons. If FRONT is nil we just return the tail.
           (if (null front)
               lst
               (progn (setf (cdr (last front)) lst) front)))))
    (t (%ldiff-recur (cdr lst) object (cons (car lst) acc)))))

(defun ldiff (lst object)
  "Return a fresh list of the elements of LST that precede OBJECT
   (compared as cons-cell identity). If OBJECT is not a tail of
   LST, return a copy of LST. Useful for cutting a prefix off a
   list while preserving the suffix's identity."
  (%ldiff-recur lst object nil))

(provide 'lists)
nil
