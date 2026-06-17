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

;; ── RPLACA / RPLACD ──────────────────────────────────────────────────────
;;
;; The destructive cons mutators. NCL lowers (setf (car x) v) / (setf
;; (cdr x) v) natively, so these are thin wrappers over that. They
;; return the CONS (not the new value) per CLHS. Defined here because
;; the setf accessors used below (and long-form DEFSETF templates such
;; as those in the ANSI suite) call them.

(defun rplaca (cons new-car)
  "Replace the car of CONS with NEW-CAR; return CONS."
  (setf (car cons) new-car)
  cons)

(defun rplacd (cons new-cdr)
  "Replace the cdr of CONS with NEW-CDR; return CONS."
  (setf (cdr cons) new-cdr)
  cons)

(defun %strip-backquote (form)
  "Turn a (BACKQUOTE template) read form with plain (UNQUOTE x) markers
   into the equivalent literal code: `(progn (f ,a ,b) ,a) => (progn (f
   a b) a). Used by long-form DEFSETF, whose template body — evaluated
   at macroexpand time in the real protocol — is run directly at the
   setter's call time here instead (equivalent for side-effect-free
   place subforms, the same assumption the short form makes). Nested
   backquote and ,@ splicing are not handled."
  (cond
    ((not (consp form)) form)
    ((and (symbolp (car form)) (string= (symbol-name (car form)) "UNQUOTE"))
     (car (cdr form)))
    ((and (symbolp (car form)) (string= (symbol-name (car form)) "BACKQUOTE"))
     (%strip-backquote (car (cdr form))))
    (t (cons (%strip-backquote (car form))
             (%strip-backquote (cdr form))))))

(defmacro defsetf (access-fn arg2 &rest rest)
  "DEFSETF — register how (setf (ACCESS-FN …) val) is computed.

   Short form: (defsetf access-fn setter-fn) — (setf (access-fn arg…)
   val) evaluates to (setter-fn arg… val); SETTER-FN takes one more
   argument than ACCESS-FN, the new value last.

   Long form: (defsetf access-fn (lambda-list) (store-var…) body…) —
   body is a template producing the update form. We install it as the
   `(setf access-fn)` writer with params (store-var… . lambda-list);
   NCL's setf fallback routes (setf (access-fn a…) v) to
   (%setf-access-fn v a…), matching that order. Place subforms are
   assumed side-effect-free (as in the short form).

   DEFINE-SETF-EXPANDER (full subform-once protocol) is still separate."
  (if (and (null rest) (symbolp arg2))
      ;; ── short form ──
      ;; Wrap in progn so the top-level recogniser walks the inner defun.
      (let ((val (gensym "VAL-"))
            (args (gensym "ARGS-")))
        `(progn
           (defun (setf ,access-fn) (,val &rest ,args)
             (apply (function ,arg2) (append ,args (list ,val))))
           ',access-fn))
      ;; ── long form ──
      (let ((lambda-list arg2)
            (store-vars (car rest))
            (body (cdr rest)))
        `(progn
           (defun (setf ,access-fn) ,(append store-vars lambda-list)
             ,@(mapcar #'%strip-backquote body))
           ',access-fn))))

;; ── INCF / DECF ─────────────────────────────────────────────────────────
;;
;; CL's INCF and DECF are sugar for (setf place (+ place delta)) and
;; (setf place (- place delta)) — delta defaults to 1. Same caveat
;; about side-effecting subforms as ROTATEF below; CL's "full" form
;; uses get-setf-expansion to evaluate subforms once.

(defmacro incf (place &optional (delta 1))
  "Increment PLACE by DELTA (default 1). Returns the new value."
  `(setf ,place (+ ,place ,delta)))

(defmacro decf (place &optional (delta 1))
  "Decrement PLACE by DELTA (default 1). Returns the new value."
  `(setf ,place (- ,place ,delta)))

;; ── ROTATEF / SHIFTF ────────────────────────────────────────────────────
;;
;; ROTATEF rotates the values stored at N places one step to the LEFT:
;;
;;   (rotatef A B)         ; swap   — A gets B, B gets A
;;   (rotatef A B C)       ; rotate — A←B, B←C, C←A
;;
;; SHIFTF shifts values RIGHT through N places, returning the leftmost
;; value (the one that "falls off"):
;;
;;   (shiftf A B 99)       ; A's old value is returned; A←B, B←99
;;
;; CL's full versions use get-setf-expansion so subforms of each place
;; are evaluated only once. We don't have that yet, so the macros below
;; assume the places are either bare symbols or simple accessor forms
;; whose subforms are side-effect-free. Corman's `quicksort.lisp` uses
;; `(rotatef (elt vec i) (elt vec j))` which qualifies — `elt`'s
;; subforms are just variable reads.
;;
;; If a caller hands us a place with side-effecting subforms (rare
;; outside CL bootstrap code), the result will be wrong; that's the
;; same trade-off our PUSH and INCF make today.

(defun %collect-setf-expansions (places)
  "Expand each place in PLACES via get-setf-expansion, returning four
   parallel lists: a flat (var val) binding list (subforms once, in
   left-to-right order), and per-place lists of store gensyms, writers,
   and readers."
  (let ((binds nil) (stores nil) (writers nil) (readers nil))
    (dolist (p places)
      (multiple-value-bind (vars vals strs writer reader) (get-setf-expansion p)
        (dolist (vv (mapcar #'list vars vals)) (setq binds (cons vv binds)))
        (setq stores  (cons (car strs) stores))
        (setq writers (cons writer writers))
        (setq readers (cons reader readers))))
    (values (nreverse binds) (nreverse stores)
            (nreverse writers) (nreverse readers))))

(defmacro rotatef (&rest places)
  "Rotate values stored at PLACES one step to the left.
   (rotatef a b c) ≡ A←B, B←C, C←A. Returns NIL. Each place's subforms
   are evaluated exactly once (CLHS 5.1.1.1) via get-setf-expansion."
  (cond
    ((null places) nil)
    ((null (cdr places)) nil)  ; single place — no-op
    (t
     (multiple-value-bind (binds stores writers readers)
         (%collect-setf-expansions places)
       ;; store_i := OLD value of the next place (last wraps to first);
       ;; all stores bound (reading old values) before any writer runs.
       (let* ((rotated (append (cdr readers) (list (car readers))))
              (store-binds (mapcar #'list stores rotated)))
         (cons 'let*
               (cons (append binds store-binds)
                     (append writers (list nil)))))))))

(defmacro shiftf (&rest args)
  "Shift values stored at PLACES one step to the right; returns the
   leftmost old value (the one that falls off).
   (shiftf a b c new) ≡ saves a, then A←B, B←C, C←new; returns saved.
   At least 2 args required: N-1 places + 1 incoming value."
  (cond
    ((or (null args) (null (cdr args)))
     (error "SHIFTF needs at least 2 args (one place + one value), got ~A"
            (length args)))
    (t
     (let* ((places (let ((all args) (out nil))
                      ;; All but last: places to shift through.
                      (do ()
                          ((null (cdr all)) (nreverse out))
                        (setq out (cons (car all) out))
                        (setq all (cdr all)))))
            (new-val (car (last args))))
       (multiple-value-bind (binds stores writers readers)
           (%collect-setf-expansions places)
         ;; result := OLD value of place1 (returned). store_i := OLD value
         ;; of place_{i+1}; the last place gets NEW-VAL. Subforms once.
         (let* ((result (gensym "SHIFT-"))
                (shifted (append (cdr readers) (list new-val)))
                (store-binds (mapcar #'list stores shifted)))
           (cons 'let*
                 (cons (append binds
                               (list (list result (car readers)))
                               store-binds)
                       (append writers (list result))))))))))

;; ── PSETQ / PSETF ────────────────────────────────────────────────────────────
;;
;; Parallel assignment: all right-hand sides are evaluated first (in
;; left-to-right order), then all assignments are performed.  Returns NIL.
;;
;; PSETQ assigns bare symbols; PSETF is the general version that handles
;; any setf-able place (same caveat as ROTATEF: compound places with
;; side-effecting subforms are evaluated in the binding phase, which is
;; good — each subform is evaluated once in order).

(defmacro psetq (&rest pairs)
  "Parallel SETQ: evaluate all right-hand sides, then set all variables.
   Returns NIL.  Unlike SETQ, assignment order does not matter — all
   old values are captured first."
  (when (oddp (length pairs))
    (error "PSETQ: odd number of arguments (~A)" (length pairs)))
  (let ((vs nil) (es nil) (ts nil))
    (do ((p pairs (cddr p)))
        ((null p))
      (push (car p)  vs)
      (push (cadr p) es)
      (push (gensym) ts))
    (let ((vs (nreverse vs))
          (es (nreverse es))
          (ts (nreverse ts)))
      `(let ,(mapcar #'list ts es)
         ,@(mapcar (lambda (v tmp) `(setq ,v ,tmp)) vs ts)
         nil))))

(defmacro psetf (&rest pairs)
  "Parallel SETF: evaluate all right-hand sides, then set all places.
   Returns NIL.  All old values are captured before any assignment."
  (when (oddp (length pairs))
    (error "PSETF: odd number of arguments (~A)" (length pairs)))
  (let ((places nil) (es nil) (ts nil))
    (do ((p pairs (cddr p)))
        ((null p))
      (push (car p)  places)
      (push (cadr p) es)
      (push (gensym) ts))
    (let ((places (nreverse places))
          (es     (nreverse es))
          (ts     (nreverse ts)))
      `(let ,(mapcar #'list ts es)
         ,@(mapcar (lambda (place tmp) `(setf ,place ,tmp)) places ts)
         nil))))

;; ── GET-SETF-EXPANSION ───────────────────────────────────────────────────────
;;
;; Returns five values describing how to read and write a generalised place:
;;
;;   vars       — list of gensyms, one per subform of the accessor call
;;   vals       — list of the actual subforms (parallel to vars)
;;   stores     — list of one gensym: the "new value" variable
;;   writer     — form: using the vars and stores, writes the new value
;;   reader     — form: using the vars, reads the current value
;;
;; Callers bind `(let* ,(mapcar #'list vars vals) ...)` to evaluate
;; each accessor subform exactly once, then bind `(let ((store ...)) ...)`
;; for the new value, evaluate `writer` to commit it, and use `reader`
;; to observe the old value.
;;
;; NCL's setf model: `(defun (setf f) (new-val arg1 ... argN) ...)` registers
;; as `%SETF-F`; `(setf (f arg1 ...argN) val)` calls `(%SETF-F val arg1...argN)`.
;; For standard built-in places (car, cdr, aref, gethash, …) the compiler
;; handles them natively without going through %SETF-*.
;;
;; The ENVIRONMENT argument is accepted for CL compatibility but ignored;
;; NCL does not yet have compile-time environment objects.

;; ── Custom setf-expander registry ─────────────────────────────────────────
;;
;; DEFINE-SETF-EXPANDER installs an expander function here, keyed by the
;; place-operator symbol. get-setf-expansion consults it before the
;; generic syntactic fallback. An expander takes the place's ARGS (the
;; cdr of the place form) and returns the 5 setf-expansion values.

(defvar *setf-expanders* (make-hash-table :test 'eq)
  "Place-operator symbol -> expander fn of the place's args.")

(defun %register-setf-expander (name fn)
  (setf (gethash name *setf-expanders*) fn)
  name)

(defun %setf-expander-for (name)
  (gethash name *setf-expanders*))

(defun get-setf-expansion (place &optional environment)
  "Return (values vars vals stores writer reader) for the generalised place PLACE.
   ENVIRONMENT is accepted for compatibility but ignored in NCL. A place
   whose operator has a registered expander (DEFINE-SETF-EXPANDER) uses it."
  (declare (ignore environment))
  (cond
    ;; Bare symbol — read/write directly.
    ((symbolp place)
     (let ((store (gensym "STORE-")))
       (values nil nil (list store)
               `(setq ,place ,store)
               place)))
    ((consp place)
     (let ((fn (car place)) (args (cdr place)))
       (let ((expander (%setf-expander-for fn)))
         (if expander
             ;; Custom expander is responsible for its own gensyms /
             ;; once-only semantics.
             (apply expander args)
             ;; (accessor arg1 ... argN) — generic syntactic fallback.
             (let ((vars  (mapcar (lambda (a) (declare (ignore a)) (gensym "G-")) args))
                   (store (gensym "STORE-")))
               (values vars
                       args
                       (list store)
                       `(setf (,fn ,@vars) ,store)
                       `(,fn ,@vars)))))))
    (t
     (error "get-setf-expansion: cannot expand place ~S" place))))

;; ── DEFINE-SETF-EXPANDER ───────────────────────────────────────────────────
;;
;; (define-setf-expander ACCESS-FN LAMBDA-LIST body…) — body is the
;; expander: given the place's subforms (bound by LAMBDA-LIST), it
;; returns the 5 setf-expansion values. We compile body into a function
;; and register it. NCL has no compile-time environment objects, so a
;; trailing/embedded &environment VAR is stripped and VAR is bound to
;; NIL inside the body (same convention as define-modify-macro).
;;
;; NOTE: this registers an EXPANDER consulted by get-setf-expansion (and
;; thus by incf/push/pop/rotatef/shiftf/psetf). Direct (setf (ACCESS …) v)
;; goes through NCL's hardwired SETF special form, which uses the
;; %SETF-ACCESS writer-function convention rather than this registry —
;; so direct setf of a define-setf-expander place is not rerouted here.

(defmacro define-setf-expander (access-fn lambda-list &rest body)
  (let ((env-var nil)
        (clean nil)
        (ll lambda-list))
    (do ()
        ((null ll))
      (cond
        ((eq (car ll) '&environment)
         (setq env-var (cadr ll))
         (setq ll (cddr ll)))
        (t (setq clean (cons (car ll) clean))
           (setq ll (cdr ll)))))
    (setq clean (nreverse clean))
    `(progn
       (%register-setf-expander ',access-fn
         (function (lambda ,clean
                     ,(if env-var
                          ;; bind &environment VAR to NIL around the body
                          (cons 'let (cons (list (list env-var nil)) body))
                          (cons 'progn body)))))
       ',access-fn)))

;; ── DEFINE-MODIFY-MACRO ──────────────────────────────────────────────────────
;;
;; Creates a read-modify-write macro analogous to INCF or PUSH.
;;
;;   (define-modify-macro name lambda-list function &optional doc)
;;
;; Generates a macro NAME that reads a generalised place, applies
;; FUNCTION to the old value and any extra arguments, then writes the
;; result back.  Example:
;;
;;   (define-modify-macro appendf (&rest more) append)
;;   (appendf my-list '(1 2 3))   ; ≡ (setf my-list (append my-list '(1 2 3)))
;;
;; The lambda-list may contain &optional and &rest (but not &key or &aux).
;; NOTE: the generated macro does NOT accept &environment in NCL because
;; NCL does not thread compile-time environments through defmacro.

(defmacro define-modify-macro (name lambda-list function &optional doc-string)
  "Define a read-modify-write macro named NAME.
   When NAME is called as (NAME place arg…), it is equivalent to
   (setf place (FUNCTION place arg…)).  The generalised place is
   evaluated only once."
  (let ((other-args nil)
        (rest-arg   nil)
        (reference  (gensym "PLACE-")))
    ;; Walk lambda-list to collect extra argument names.
    (do ((ll lambda-list (cdr ll)))
        ((null ll))
      (let ((arg (car ll)))
        (cond
          ((eq arg '&optional) nil)      ; skip keyword itself
          ((eq arg '&rest)
           (cond
             ((null (cdr ll))
              (error "DEFINE-MODIFY-MACRO ~S: &rest must be followed by a name" name))
             ((symbolp (cadr ll))
              (setq rest-arg (cadr ll))
              (return))
             (t
              (error "DEFINE-MODIFY-MACRO ~S: &rest arg must be a symbol" name))))
          ((member arg '(&key &allow-other-keys &aux))
           (error "DEFINE-MODIFY-MACRO ~S: ~S not allowed in lambda list" name arg))
          ((symbolp arg)
           (push arg other-args))
          ((and (consp arg) (symbolp (car arg)))
           ;; optional-with-default: (var default) — just want the var name
           (push (car arg) other-args))
          (t
           (error "DEFINE-MODIFY-MACRO ~S: bad lambda-list element ~S" name arg)))))
    (setq other-args (nreverse other-args))
    `(defmacro ,name (,reference ,@lambda-list)
       ,@(when doc-string (list doc-string))
       (multiple-value-bind (vars vals stores writer reader)
           (get-setf-expansion ,reference)
         (let ((let-list (mapcar #'list vars vals)))
           (push (list (car stores)
                       ,(if rest-arg
                            `(list* ',function reader ,@other-args ,rest-arg)
                            `(list ',function reader ,@other-args)))
                 let-list)
           ;; Build (let* (binding...) writer) without nested backquote —
           ;; NCL does not support nested backquotes yet.
           (cons 'let* (cons (nreverse let-list) (list writer))))))))

;; ── PUSH / POP (once-only) ───────────────────────────────────────────────
;;
;; core.lisp defines PUSH/POP textually (evaluating PLACE twice). Now
;; that get-setf-expansion exists, redefine them here (places.lisp loads
;; after core.lisp) to evaluate each place subform exactly once — CLHS
;; 5.1.1.1. Built with (cons 'let* …) like define-modify-macro because
;; NCL has no nested backquote.

(defmacro push (value place)
  "Prepend VALUE to the list at PLACE. VALUE is evaluated first, then
   the place's subforms (each exactly once), then the store happens."
  (multiple-value-bind (vars vals stores writer reader) (get-setf-expansion place)
    (let ((vtmp (gensym "V-")))
      (cons 'let*
            (cons (append (list (list vtmp value))
                          (mapcar #'list vars vals)
                          (list (list (car stores) (list 'cons vtmp reader))))
                  (list writer))))))

(defmacro pop (place)
  "Remove and return the head of the list at PLACE; the place's subforms
   are evaluated exactly once."
  (multiple-value-bind (vars vals stores writer reader) (get-setf-expansion place)
    (let ((old (gensym "OLD-")))
      (cons 'let*
            (cons (append (mapcar #'list vars vals)
                          (list (list old reader)
                                (list (car stores) (list 'cdr old))))
                  (list (list 'prog1 (list 'car old) writer)))))))

(provide 'places)
nil
