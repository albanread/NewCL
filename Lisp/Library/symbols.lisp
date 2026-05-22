;;;; Lisp/Library/symbols.lisp — symbol property lists and standard macros.
;;;;
;;;; Provides:
;;;;
;;;;   Property lists (CL §10.1):
;;;;     symbol-plist   (setf symbol-plist)
;;;;     get            (setf get)
;;;;     remprop        putprop
;;;;
;;;;   Standard control macros:
;;;;     prog1   prog2
;;;;
;;;;   Destructuring:
;;;;     destructuring-bind   — required, &optional, &rest, dotted rest,
;;;;                            nested sub-patterns, &key (basic form).
;;;;
;;;; Implementation note — property lists:
;;;; The NCL runtime's symbol struct has a plist cell (gc_symbol.rs §[5]),
;;;; but that cell is documented as write-once-at-intern-time, so it is not
;;;; safe to mutate from Lisp.  We simulate property lists with a global
;;;; EQ hash table instead.  Functionally equivalent for all common uses.

;; ── Property lists ───────────────────────────────────────────────────────────

(defvar *symbol-plists* (make-hash-table :test 'eq)
  "Global table: symbol → property-list (a flat key/value list).")

(defun symbol-plist (sym)
  "Return the property list of symbol SYM."
  (gethash sym *symbol-plists*))

(defun (setf symbol-plist) (new-plist sym)
  "Replace the property list of SYM with NEW-PLIST (a flat key/value list)."
  (if (null new-plist)
      (remhash sym *symbol-plists*)
      (setf (gethash sym *symbol-plists*) new-plist))
  new-plist)

;; Internal: walk PLIST looking for KEY; return VAL or DEFAULT.
(defun %plist-get (plist key default)
  (cond
    ((null plist) default)
    ((eq (car plist) key) (car (cdr plist)))
    (t (%plist-get (cdr (cdr plist)) key default))))

;; Internal: return a plist like PLIST but with KEY → VAL (non-destructive).
(defun %plist-set (plist key val)
  (cond
    ((null plist)
     (list key val))
    ((eq (car plist) key)
     (cons key (cons val (cdr (cdr plist)))))
    (t
     (cons (car plist)
           (cons (car (cdr plist))
                 (%plist-set (cdr (cdr plist)) key val))))))

;; Internal: return (values new-plist foundp) with KEY removed.
(defun %plist-remove (plist key)
  (cond
    ((null plist) (values nil nil))
    ((eq (car plist) key) (values (cdr (cdr plist)) t))
    (t (multiple-value-bind (rest found)
           (%plist-remove (cdr (cdr plist)) key)
         (values (cons (car plist) (cons (car (cdr plist)) rest)) found)))))

(defun get (sym indicator &optional default)
  "Return the value of INDICATOR on SYM's property list, or DEFAULT."
  (%plist-get (symbol-plist sym) indicator default))

(defun (setf get) (new-value sym indicator)
  "Set INDICATOR on SYM's property list to NEW-VALUE."
  (setf (symbol-plist sym)
        (%plist-set (symbol-plist sym) indicator new-value))
  new-value)

(defun remprop (sym indicator)
  "Remove INDICATOR from SYM's property list.
   Returns T if the indicator was present, NIL otherwise."
  (multiple-value-bind (new found)
      (%plist-remove (symbol-plist sym) indicator)
    (when found
      (setf (symbol-plist sym) new))
    found))

(defun putprop (sym val indicator)
  "Corman Lisp compat: (putprop SYM VAL INDICATOR) sets SYM's INDICATOR property.
   Equivalent to (setf (get SYM INDICATOR) VAL)."
  (setf (get sym indicator) val))

;; ── prog1 / prog2 ────────────────────────────────────────────────────────────

(defmacro prog1 (first-form &rest more-forms)
  "Evaluate FIRST-FORM and MORE-FORMS in order; return the value of FIRST-FORM."
  (let ((g (gensym "P1")))
    `(let ((,g ,first-form))
       ,@more-forms
       ,g)))

(defmacro prog2 (first-form second-form &rest more-forms)
  "Evaluate FIRST-FORM, SECOND-FORM, and MORE-FORMS; return the value of SECOND-FORM."
  (let ((g (gensym "P2")))
    `(progn
       ,first-form
       (let ((,g ,second-form))
         ,@more-forms
         ,g))))

;; ── destructuring-bind ───────────────────────────────────────────────────────
;;
;; Handles:
;;   required args             (a b c)
;;   optional args             (a &optional (b 0) c)
;;   rest / dotted rest        (a &rest r)  or  (a . r)
;;   nested sub-patterns       ((x y) b)
;;   basic &key                (a &key x (y 0))
;;
;; &allow-other-keys and &aux are accepted and silently skipped.

(defmacro destructuring-bind (pattern form &body body)
  "Bind variables from PATTERN to the parts of FORM, then execute BODY.
   PATTERN is a lambda-list-like tree of variable names; &optional, &rest,
   &key, and nested sub-lists are all supported."
  (let ((g (gensym "DBB")))
    `(let ((,g ,form))
       ,(%dbb-expand pattern g body))))

;; ── compile-time helpers (called only during macro-expansion) ────────────────

(defun %dbb-lambda-keyword-p (x)
  "True if X is one of the &KEYWORD symbols used in lambda lists."
  (member x '(&optional &rest &key &allow-other-keys &aux &body &whole &environment)))

(defun %dbb-expand (pattern form-sym body)
  "Return Lisp code that binds PATTERN from FORM-SYM, then evaluates BODY forms."
  (cond
    ;; () — just run body (match against empty list)
    ((null pattern)
     `(progn ,@body))
    ;; bare symbol — bind the whole form to it
    ((symbolp pattern)
     `(let ((,pattern ,form-sym)) ,@body))
    ;; list/dotted-list pattern
    ((consp pattern)
     (%dbb-list-expand pattern form-sym body 0))
    (t
     (error "destructuring-bind: bad pattern element ~S" pattern))))

(defun %dbb-list-expand (pattern form-sym body index)
  "Walk LIST-PATTERN left to right, generating let forms around BODY."
  (cond
    ;; End of required section
    ((null pattern)
     `(progn ,@body))
    ;; Dotted rest: (a b . rest-var)
    ((atom pattern)
     `(let ((,pattern (nthcdr ,index ,form-sym)))
        ,@body))
    ;; &rest
    ((eq (car pattern) '&rest)
     (let ((rest-sym (cadr pattern)))
       `(let ((,rest-sym (nthcdr ,index ,form-sym)))
          ,@body)))
    ;; &body (treated as &rest)
    ((eq (car pattern) '&body)
     (let ((body-sym (cadr pattern)))
       `(let ((,body-sym (nthcdr ,index ,form-sym)))
          ,@body)))
    ;; &optional section
    ((eq (car pattern) '&optional)
     (%dbb-opt-expand (cdr pattern) form-sym body index))
    ;; &key section
    ((eq (car pattern) '&key)
     (%dbb-key-expand (cdr pattern) form-sym body index))
    ;; skip &allow-other-keys, &aux, &environment, &whole
    ((%dbb-lambda-keyword-p (car pattern))
     `(progn ,@body))
    ;; Nested sub-pattern
    ((consp (car pattern))
     (let ((sub (gensym "DBB-N"))
           ;; generate code for the remaining list elements
           (rest-code (%dbb-list-expand (cdr pattern) form-sym body (1+ index))))
       ;; Bind (nth index form-sym) to a temp, then destructure it
       `(let ((,sub (nth ,index ,form-sym)))
          ,(%dbb-expand (car pattern) sub (list rest-code)))))
    ;; Simple required variable
    (t
     `(let ((,(car pattern) (nth ,index ,form-sym)))
        ,(%dbb-list-expand (cdr pattern) form-sym body (1+ index))))))

(defun %dbb-opt-expand (opts form-sym body index)
  "Expand &optional variables, then hand off to the rest of the pattern."
  (cond
    ((null opts)
     `(progn ,@body))
    ;; Hit another &keyword — hand back to list expander
    ((%dbb-lambda-keyword-p (car opts))
     (%dbb-list-expand opts form-sym body index))
    (t
     (let* ((opt (car opts))
            (sym (if (consp opt) (car opt) opt))
            (def (if (consp opt) (cadr opt) nil)))
       `(let ((,sym (if (nthcdr ,index ,form-sym)
                        (nth ,index ,form-sym)
                        ,def)))
          ,(%dbb-opt-expand (cdr opts) form-sym body (1+ index)))))))

(defun %dbb-key-expand (keys form-sym body index)
  "Expand &key variables, looking them up in the plist tail of FORM-SYM.
   INDEX is the position where keyword arguments begin."
  (cond
    ((null keys)
     `(progn ,@body))
    ((%dbb-lambda-keyword-p (car keys))
     ;; &allow-other-keys etc — done with keys
     `(progn ,@body))
    (t
     (let* ((key-spec (car keys))
            (sym      (if (consp key-spec) (car key-spec) key-spec))
            (def      (if (consp key-spec) (cadr key-spec) nil))
            ;; keyword indicator: the :NAME keyword matching symbol NAME.
            ;; We build it at macro-expansion time.
            (key-kw   (intern (string-concat ":" (symbol-name sym)))))
       `(let ((,sym (let ((tail (member ',key-kw (nthcdr ,index ,form-sym))))
                      (if tail (cadr tail) ,def))))
          ,(%dbb-key-expand (cdr keys) form-sym body index))))))

(provide 'symbols)
nil
