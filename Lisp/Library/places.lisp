;;;; Lisp/Library/places.lisp — extensible SETF places.
;;;;
;;;; Two things in one module:
;;;;
;;;;   1. The standard CL `(setf accessor)` declarations for the
;;;;      cons-cell accessor family — FIRST, SECOND, …, REST,
;;;;      CAAR, CADR, CDAR, CDDR, CAAAR through CDDDDR.  Each is a
;;;;      one-line `(defun (setf NAME) (val place) (setf <expansion>
;;;;      val))` that maps the named accessor onto the underlying
;;;;      `(setf (car …) …)` / `(setf (cdr …) …)` primitives.
;;;;
;;;;      Without these, code like `(setf (first xs) 99)` failed
;;;;      with "undefined function: %SETF-FIRST" — the generic-
;;;;      setf-fallback mangled FIRST correctly but no setter was
;;;;      registered.  These add the registration; the fallback
;;;;      now resolves.
;;;;
;;;;   2. A DEFSETF macro (the short-form variant) — sugar for the
;;;;      common case where the setter is just another function
;;;;      call:
;;;;
;;;;        (defsetf access-fn setter-fn)
;;;;
;;;;      installs an inverse of ACCESS-FN such that
;;;;
;;;;        (setf (access-fn arg1 … argN) val)
;;;;
;;;;      expands to
;;;;
;;;;        (setter-fn arg1 … argN val)
;;;;
;;;;      The long form `(defsetf access-fn lambda-list (store-var)
;;;;      body)` and the fully general
;;;;      `define-setf-expander` are out of scope for this slice —
;;;;      both can layer on top of `defsetf` later without changing
;;;;      the underlying machinery.

;; (No (in-package …) — NCL uses a flat symbol namespace at this
;; tier; everything interns into COMMON-LISP-USER by default.)

;; ── Named cons-cell accessors ───────────────────────────────────────────

(defun (setf first)  (val x) (setf (car x) val))
(defun (setf second) (val x) (setf (car (cdr x)) val))
(defun (setf third)  (val x) (setf (car (cdr (cdr x))) val))
(defun (setf fourth) (val x) (setf (car (cdr (cdr (cdr x)))) val))

(defun (setf rest)   (val x) (setf (cdr x) val))

(defun (setf caar) (val list) (setf (car (car list)) val))
(defun (setf cadr) (val list) (setf (car (cdr list)) val))
(defun (setf cdar) (val list) (setf (cdr (car list)) val))
(defun (setf cddr) (val list) (setf (cdr (cdr list)) val))

(defun (setf caaar) (val list) (setf (car (car (car list))) val))
(defun (setf caadr) (val list) (setf (car (car (cdr list))) val))
(defun (setf cadar) (val list) (setf (car (cdr (car list))) val))
(defun (setf caddr) (val list) (setf (car (cdr (cdr list))) val))
(defun (setf cdaar) (val list) (setf (cdr (car (car list))) val))
(defun (setf cdadr) (val list) (setf (cdr (car (cdr list))) val))
(defun (setf cddar) (val list) (setf (cdr (cdr (car list))) val))
(defun (setf cdddr) (val list) (setf (cdr (cdr (cdr list))) val))

(defun (setf caaaar) (val list) (setf (car (car (car (car list)))) val))
(defun (setf caaadr) (val list) (setf (car (car (car (cdr list)))) val))
(defun (setf caadar) (val list) (setf (car (car (cdr (car list)))) val))
(defun (setf caaddr) (val list) (setf (car (car (cdr (cdr list)))) val))
(defun (setf cadaar) (val list) (setf (car (cdr (car (car list)))) val))
(defun (setf cadadr) (val list) (setf (car (cdr (car (cdr list)))) val))
(defun (setf caddar) (val list) (setf (car (cdr (cdr (car list)))) val))
(defun (setf cadddr) (val list) (setf (car (cdr (cdr (cdr list)))) val))

(defun (setf cdaaar) (val list) (setf (cdr (car (car (car list)))) val))
(defun (setf cdaadr) (val list) (setf (cdr (car (car (cdr list)))) val))
(defun (setf cdadar) (val list) (setf (cdr (car (cdr (car list)))) val))
(defun (setf cdaddr) (val list) (setf (cdr (car (cdr (cdr list)))) val))
(defun (setf cddaar) (val list) (setf (cdr (cdr (car (car list)))) val))
(defun (setf cddadr) (val list) (setf (cdr (cdr (car (cdr list)))) val))
(defun (setf cdddar) (val list) (setf (cdr (cdr (cdr (car list)))) val))
(defun (setf cddddr) (val list) (setf (cdr (cdr (cdr (cdr list)))) val))

;; NTH / NTHCDR
;;
;; `(setf (nth n list) val)` and `(setf (nthcdr n list) val)` are
;; the obvious setf-able forms over the indexed-access functions.
;; CL spec: nth indexes from 0; same for nthcdr.

(defun (setf nth) (val n list)
  (setf (car (nthcdr n list)) val))

(defun (setf nthcdr) (val n list)
  ;; Replace the entire tail starting at position N.
  ;; CL doesn't strictly mandate this since nthcdr returns a tail
  ;; and setf-ing a tail is dodgy; we provide it via (setf (cdr
  ;; ...))-of-prefix for symmetry with nth.
  (cond
    ((zerop n)
     ;; Can't setf a binding through the car of the head — the
     ;; caller's list variable wouldn't update. Signal.
     (error "(setf nthcdr) with N=0 cannot mutate the variable; assign directly instead"))
    (t (setf (cdr (nthcdr (- n 1) list)) val))))

;; LAST
;;
;; `(setf (last list) val)` replaces the final cons cell. CL's
;; `last` returns the LAST cons cell of LIST; setting it via setf
;; means rplaca'ing the head of the last cell.

(defun (setf last) (val list)
  (setf (car (last list)) val))

;; ── DEFSETF (short form) ────────────────────────────────────────────────
;;
;; The convention: (defsetf foo set-foo) expands to
;;
;;   (defun (setf foo) (val arg1 … argN)
;;     (set-foo arg1 … argN val))
;;
;; So (setf (foo a b) v) → (%setf-foo v a b) [via lower.rs fallback]
;; → (set-foo a b v) [via the body we install].
;;
;; We don't know SET-FOO's exact arglist at macro-expansion time,
;; but we know it takes one more arg than FOO (the new value, last).
;; So we accept the place-fn's lambda-list implicitly via a single
;; &REST capture and apply through it.

(defmacro defsetf (access-fn setter-fn)
  "Short-form DEFSETF. (defsetf place-fn setter-fn) means: a
   (setf (place-fn arg…) val) form evaluates to (setter-fn arg… val).
   SETTER-FN must take one more argument than PLACE-FN, with the
   new value as its last positional argument.

   Long-form DEFSETF and DEFINE-SETF-EXPANDER are not yet
   implemented; use (defun (setf NAME) (val args…) …) directly
   for cases the short form doesn't cover."
  ;; Wrap in progn so the top-level recogniser walks the inner
  ;; defun — a bare (defun …) produced by a macro hits the
  ;; \"defun only at top level\" guard otherwise.
  (let ((val (gensym "VAL-"))
        (args (gensym "ARGS-")))
    `(progn
       (defun (setf ,access-fn) (,val &rest ,args)
         (apply (function ,setter-fn) (append ,args (list ,val)))))))

(provide 'places)
nil
