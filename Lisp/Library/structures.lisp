;;;; Lisp/Library/structures.lisp — full DEFSTRUCT.
;;;;
;;;; Redefines the bootstrap `defstruct` (core.lisp, simple symbol-name
;;;; form) with the option-list surface: (:conc-name) (:constructor name
;;;; [boa-arglist]) (:copier) (:predicate) (:include parent overrides…)
;;;; (:type list|vector) :named (:initial-offset n), per-slot :read-only
;;;; / :type, and BOA constructors (incl. &optional/&rest/&aux). The
;;;; plain `(defstruct NAME slot…)` path stays compatible with the
;;;; bootstrap version (vector with the type symbol at slot 0).

(defvar *defstruct-info* (make-hash-table :test 'eq)
  "struct-name -> plist (:slots EFF :conc-name S :type T :named B :offset N
   :include PARENT). EFF is a list of parsed slots (name default read-only).")

(defun %ds-pslot-name (s) (car s))
(defun %ds-pslot-default (s) (cadr s))
(defun %ds-pslot-ro (s) (caddr s))

(defun %ds-sym (prefix name)
  (intern (concatenate 'string prefix (string name))))

(defun %ds-parse-slot (spec)
  "spec: SYMBOL | (name [default] {:type T | :read-only B}*) -> (name default ro)."
  (if (symbolp spec)
      (list spec nil nil)
      (let ((nm (car spec))
            (default (if (cdr spec) (cadr spec) nil))
            (ro nil))
        (do ((o (cddr spec) (cddr o)))
            ((null o))
          (when (eq (car o) :read-only) (setq ro (cadr o))))
        (list nm default ro))))

(defun %ds-arglist-vars (arglist)
  "Variable names bound by a (BOA) lambda-list (skip &markers, take car of
   (var …) forms)."
  (let ((vars nil))
    (dolist (x arglist)
      (cond
        ((member x '(&optional &rest &aux &key &allow-other-keys)) nil)
        ((symbolp x) (setq vars (cons x vars)))
        ((consp x) (setq vars (cons (car x) vars)))))
    (nreverse vars)))

(defun %ds-split-aux (arglist)
  "Split a BOA lambda-list at &aux. Returns (values before-aux aux-specs).
   NCL's defun rejects &aux, so the constructor binds aux vars in a let*."
  (let ((before nil) (aux nil) (in-aux nil))
    (dolist (x arglist)
      (cond
        ((eq x '&aux) (setq in-aux t))
        (in-aux (setq aux (cons x aux)))
        (t (setq before (cons x before)))))
    (values (nreverse before) (nreverse aux))))

;; Layout math. Vector struct: slot 0 is the type tag, data at 1.. .
;; Typed (:type list/vector): `offset` leading NILs, then (if :named) the
;; type tag, then data.
(defun %ds-data-base (type named offset)
  (cond ((null type) 1) (named (+ offset 1)) (t offset)))
(defun %ds-name-pos (type named offset)
  (cond ((null type) 0) (named offset) (t nil)))
(defun %ds-total-len (type named offset n)
  (cond ((null type) (+ n 1)) (named (+ offset 1 n)) (t (+ offset n))))

(defun %ds-isa (x type)
  "T iff struct-type X is TYPE or (transitively) :includes it."
  (cond
    ((null x) nil)
    ((eq x type) t)
    (t (let ((info (gethash x *defstruct-info*)))
         (and info (%ds-isa (getf info :include) type))))))

(defun %ds-build-body (name slot-vals type named offset)
  "Build the constructor body that materialises the instance from SLOT-VALS
   (a value form per effective slot, in order)."
  (let* ((n (length slot-vals))
         (len (%ds-total-len type named offset n))
         (base (%ds-data-base type named offset))
         (namepos (%ds-name-pos type named offset)))
    (if (null type)
        ;; tagged vector
        (let ((setfs nil) (i base))
          (dolist (sv slot-vals)
            (setq setfs (cons (list 'setf (list 'svref '__v i) sv) setfs))
            (setq i (+ i 1)))
          (cons 'let
                (cons (list (list '__v (list 'make-array len :initial-element nil)))
                      (append (list (list 'setf (list 'svref '__v namepos)
                                          (list 'quote name)))
                              (nreverse setfs)
                              (list '__v)))))
        ;; :type list — the struct IS a list
        (let ((elems nil))
          (dotimes (k offset) (setq elems (cons nil elems)))
          (when named (setq elems (cons (list 'quote name) elems)))
          (dolist (sv slot-vals) (setq elems (cons sv elems)))
          (cons 'list (nreverse elems))))))

(defun %ds-build-ctor (ctor-name arglist-spec name eff-slots type named offset)
  (if (eq arglist-spec :keyword)
      (let* ((keyargs (mapcar (lambda (s)
                                (list (%ds-pslot-name s) (%ds-pslot-default s)))
                              eff-slots))
             (body (%ds-build-body name (mapcar #'%ds-pslot-name eff-slots)
                                   type named offset)))
        (cons 'defun (cons ctor-name (cons (cons '&key keyargs) (list body)))))
      ;; BOA: arglist-spec is a lambda-list
      (multiple-value-bind (before aux) (%ds-split-aux arglist-spec)
        (let* ((argvars (append (%ds-arglist-vars before) (%ds-arglist-vars aux)))
               (slot-vals (mapcar (lambda (s)
                                    (if (member (%ds-pslot-name s) argvars)
                                        (%ds-pslot-name s)
                                        (%ds-pslot-default s)))
                                  eff-slots))
               (body (%ds-build-body name slot-vals type named offset))
               (wrapped (if aux
                            (list 'let*
                                  (mapcar (lambda (a) (if (consp a) a (list a nil))) aux)
                                  body)
                            body)))
          (cons 'defun (cons ctor-name (cons before (list wrapped))))))))

(defun %ds-build-accessor (acc idx type)
  (list 'defun acc (list 'obj)
        (if (null type) (list 'svref 'obj idx) (list 'nth idx 'obj))))

(defun %ds-build-setter (acc idx type)
  (let ((setter (%ds-sym "%SETF-" acc)))
    (list 'defun setter (list 'val 'obj)
          (if (null type)
              (list 'setf (list 'svref 'obj idx) 'val)
              (list 'setf (list 'nth idx 'obj) 'val))
          'val)))

(defun %defstruct-expand (name-and-options slot-specs)
  (let* ((name (if (symbolp name-and-options) name-and-options (car name-and-options)))
         (options (if (symbolp name-and-options) nil (cdr name-and-options)))
         (conc-name (concatenate 'string (string name) "-"))
         (ctors nil) (ctor-specified nil)
         (copier (%ds-sym "COPY-" name)) (copier-spec nil)
         (predicate (%ds-sym "" (concatenate 'string (string name) "-P")))
         (pred-spec nil)
         (include nil) (overrides nil)
         (type nil) (named nil) (offset 0))
    ;; ── parse options ──
    (dolist (opt options)
      (cond
        ((eq opt :named) (setq named t))
        ((symbolp opt) nil)
        (t (let ((k (car opt)) (v (cdr opt)))
             (cond
               ((eq k :conc-name)
                (setq conc-name (if (or (null v) (null (car v))) "" (string (car v)))))
               ((eq k :constructor)
                (setq ctor-specified t)
                (cond
                  ((null v) nil)             ; (:constructor) — keep default-style? treat as none extra
                  ((null (car v)) nil)        ; (:constructor nil) — suppress
                  (t (setq ctors (cons (cons (car v) (if (cdr v) (cadr v) :keyword)) ctors)))))
               ((eq k :copier) (setq copier-spec t) (setq copier (car v)))
               ((eq k :predicate) (setq pred-spec t) (setq predicate (car v)))
               ((eq k :include) (setq include (car v)) (setq overrides (cdr v)))
               ((eq k :type) (setq type (car v)))
               ((eq k :initial-offset) (setq offset (car v)))
               (t nil))))))
    ;; ── effective slots (inherited + own, with overrides) ──
    (let* ((parent-info (and include (gethash include *defstruct-info*)))
           (parent-slots (and parent-info (getf parent-info :slots)))
           (inherited (mapcar (lambda (ps)
                                (let ((ov (assoc (%ds-pslot-name ps) overrides)))
                                  (if ov (%ds-parse-slot ov) ps)))
                              parent-slots))
           (own (mapcar #'%ds-parse-slot slot-specs))
           (eff-slots (append inherited own))
           (base (%ds-data-base type named offset)))
      ;; ── register info (side effect during macroexpansion — :include of a
      ;;    sibling defined earlier in the same compile unit can see it) ──
      (setf (gethash name *defstruct-info*)
            (list :slots eff-slots :conc-name conc-name :type type
                  :named named :offset offset :include include))
      ;; ── constructors ──
      (let ((ctor-forms nil))
        (if ctor-specified
            (dolist (c (reverse ctors))
              (setq ctor-forms
                    (cons (%ds-build-ctor (car c) (cdr c) name eff-slots type named offset)
                          ctor-forms)))
            (setq ctor-forms
                  (list (%ds-build-ctor (%ds-sym "MAKE-" name) :keyword name eff-slots
                                        type named offset))))
        (setq ctor-forms (nreverse ctor-forms))
        ;; ── accessors / setters ──
        (let ((acc-forms nil) (idx base))
          (dolist (s eff-slots)
            (let ((acc (%ds-sym conc-name (%ds-pslot-name s))))
              (setq acc-forms (cons (%ds-build-accessor acc idx type) acc-forms))
              (unless (%ds-pslot-ro s)
                (setq acc-forms (cons (%ds-build-setter acc idx type) acc-forms))))
            (setq idx (+ idx 1)))
          (setq acc-forms (nreverse acc-forms))
          ;; ── predicate (named structs only) ──
          (let ((pred-form
                 (when (and predicate (or (null type) named))
                   (let ((npos (%ds-name-pos type named offset)))
                     (if (null type)
                         (list 'defun predicate (list 'obj)
                               (list 'and (list 'vectorp 'obj)
                                     (list '> (list 'length 'obj) npos)
                                     (list '%ds-isa (list 'svref 'obj npos) (list 'quote name))))
                         (list 'defun predicate (list 'obj)
                               (list 'and (list 'consp 'obj)
                                     (list 'eq (list 'nth npos 'obj) (list 'quote name)))))))))
            ;; ── copier (vector only; lists copied with copy-list) ──
            (let ((copier-form
                   (when copier
                     (list 'defun copier (list 'obj)
                           (if (null type) (list 'copy-seq 'obj) (list 'copy-list 'obj))))))
              (append
               (list 'progn)
               ctor-forms
               acc-forms
               (when pred-form (list pred-form))
               (when copier-form (list copier-form))
               (list (list 'quote name))))))))))

(defmacro defstruct (name-and-options &rest slot-specs)
  (%defstruct-expand name-and-options slot-specs))

(provide (quote structures))
nil
