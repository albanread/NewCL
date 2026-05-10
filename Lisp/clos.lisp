;;;; clos.lisp — CLOS port from Corman Lisp's Closette.
;;;;
;;;; Original: Closette 1.0 (Xerox, 1991), as elaborated in
;;;; Roger Corman's Corman Lisp Sys/clos.lisp. Heavy editing by
;;;; RGC/LC over decades — we're porting the latest shape.
;;;;
;;;; Staging:
;;;;   Stage A — utilities + std-instance representation (this file
;;;;             at first; later stages append).
;;;;   Stage B — class metaobjects + bootstrap.
;;;;   Stage C — defclass + finalize-inheritance.
;;;;   Stage D — make-instance + slot-value.
;;;;   Stage E — defgeneric + dispatch.
;;;;   Stage F — defmethod.
;;;;   Stage G — call-next-method, before/after/around.
;;;;   Stage H — EQL specializers.
;;;;
;;;; Closette uses Corman-specific primitives (uref, alloc-clos-
;;;; instance, with-synchronization). We adapt:
;;;;   - CLOS instances are 4-element Vectors with a marker.
;;;;   - Synchronisation is dropped; we're single-threaded.
;;;;   - The (setf NAME) function-naming convention becomes
;;;;     %SETF-NAME (matches our chunk-6 generic-setf fallback).

;; ─── Stage A: utilities + std-instance representation ───────────────────────

;; -- CLOS instance layout ---------------------------------------------------
;;
;; A CLOS instance is a 4-cell Vector:
;;   slot 0: '%CLOS-INSTANCE marker — distinguishes CLOS objects
;;           from ordinary defstruct vectors and bare vectors.
;;   slot 1: the class (itself a CLOS instance — meta-circular).
;;   slot 2: the slot-storage vector (one cell per effective slot).
;;   slot 3: the class-signature snapshot, for class-redefinition
;;           detection. NIL until set.
;;
;; Closette uses `uref` + offset constants to read these. We use
;; `svref` with named offsets.

(defparameter %clos-instance-marker '%clos-instance)
(defparameter %clos-instance-class-offset 1)
(defparameter %clos-instance-slots-offset 2)
(defparameter %clos-instance-signature-offset 3)
(defparameter %clos-instance-cell-count 4)

(defun clos-instance-p (x)
  "True iff X is a CLOS instance (a 4-vector tagged with our
   marker in slot 0). False for ordinary vectors, defstructs, or
   anything else — defstructs use their own type symbol in slot
   0, never the %CLOS-INSTANCE marker."
  (and (vectorp x)
       (= (length x) %clos-instance-cell-count)
       (eq (svref x 0) %clos-instance-marker)))

(defun alloc-clos-instance ()
  "Allocate an unfilled CLOS instance. Slot 0 is set to the
   marker; the other cells stay NIL until allocate-std-instance
   fills them in."
  (let ((v (make-array %clos-instance-cell-count :initial-element nil)))
    (setf (svref v 0) %clos-instance-marker)
    v))

;; Direct accessors — used internally where we know the arg is a
;; CLOS instance and don't want the type check overhead. Closette
;; uses `clos-instance-class` etc. for the same purpose.

(defun clos-instance-class (x) (svref x %clos-instance-class-offset))
(defun clos-instance-slots (x) (svref x %clos-instance-slots-offset))

;; std-instance-class / -slots / -signature are the public
;; accessors with type checks. Closette wraps each with a
;; clos-instance-p check; we do the same.

(defun std-instance-class (x)
  (cond
    ((clos-instance-p x) (clos-instance-class x))
    (t (error "Not a CLOS instance: ~A" x))))

(defun %setf-std-instance-class (val x)
  (cond
    ((clos-instance-p x)
     (setf (svref x %clos-instance-class-offset) val))
    (t (error "Not a CLOS instance: ~A" x))))

(defun std-instance-slots (x)
  (cond
    ((clos-instance-p x) (clos-instance-slots x))
    (t (error "Not a CLOS instance: ~A" x))))

(defun %setf-std-instance-slots (val x)
  (cond
    ((clos-instance-p x)
     (setf (svref x %clos-instance-slots-offset) val))
    (t (error "Not a CLOS instance: ~A" x))))

(defun std-instance-signature (x)
  (cond
    ((clos-instance-p x)
     (svref x %clos-instance-signature-offset))
    (t (error "Not a CLOS instance: ~A" x))))

(defun %setf-std-instance-signature (val x)
  (cond
    ((clos-instance-p x)
     (setf (svref x %clos-instance-signature-offset) val))
    (t (error "Not a CLOS instance: ~A" x))))

;; -- Allocation -------------------------------------------------------------

(defun allocate-std-instance (class slots)
  "Build a CLOS instance bound to CLASS, with SLOTS as its
   per-instance slot vector. Used during make-instance and when
   bootstrap manually constructs the meta-class tower."
  (let ((x (alloc-clos-instance)))
    (setf (svref x %clos-instance-class-offset) class)
    (setf (svref x %clos-instance-slots-offset) slots)
    x))

;; A unique sentinel for "this slot has never been written".
;; Anything `eql`-equal to this is treated as unbound.
(defparameter secret-unbound-value (list "slot unbound"))

(defun allocate-slot-storage (size initial-value)
  "Make-array wrapper. CLOS instances all use simple-vectors for
   their slot storage — one cell per effective slot."
  (make-array size :initial-element initial-value))

;; -- General-purpose utilities ----------------------------------------------

(defmacro push-on-end (value location)
  "Append VALUE to the end of the list at LOCATION. Closette
   uses this heavily for accumulators where order must be
   preserved (e.g. method ordering)."
  `(setf ,location (nconc ,location (list ,value))))

(defun %setf-getf* (new-value plist key)
  "(setf (getf* plist key) new-value) — like (setf getf) but
   destructively modifies PLIST in place when KEY is already
   there, else appends. PLIST must be non-nil. Closette uses
   this for canonicalising initargs.

   Internal note: we copy PLIST into a let-local before the
   walk + appends because the compiler doesn't yet box mutated
   parameters. The destructive nconc reaches the caller via
   PLIST's existing cons-cell tail, so the semantics still
   match CL even though we never reassign the param itself."
  (let ((p plist))
    (block body
      (let ((x p))
        (loop
          (cond
            ((null x) (return nil))
            ((eq (car x) key)
             (setf (car (cdr x)) new-value)
             (return-from body new-value))
            (t (setq x (cdr (cdr x)))))))
      (setq p (nconc p (list key)))
      (setq p (nconc p (list new-value)))
      new-value)))

(defun mapappend (fn &rest args)
  "Like mapcar, but the per-call results are appended together.
   Standard Closette workhorse for flattening nested results."
  (cond
    ((some #'null args) nil)
    (t (append (apply fn (mapcar #'car args))
               (apply #'mapappend fn (mapcar #'cdr args))))))

(defun mapplist (fn x)
  "mapcar over a property list, calling FN with each (key value)
   pair in turn."
  (cond
    ((null x) nil)
    (t (cons (funcall fn (car x) (cadr x))
             (mapplist fn (cddr x))))))

;; -- Method table -----------------------------------------------------------
;;
;; Each generic function carries a method-table. The table holds
;; the registered methods plus a single-entry cache of the most
;; recently looked-up (types → method) pair — Closette's hot-path
;; optimisation. Stage E's dispatch logic reads this; for now we
;; just install the data structure and its CRUD helpers.
;;
;; Closette wraps every access in `with-synchronization` over a
;; per-table critical section. We're single-threaded and skip
;; that — restore when concurrent dispatch becomes a concern.

(defstruct method-table
  (method-list nil)
  (cached-method nil)
  (cached-method-types nil)
  (eql-specializers nil))

(defun clear-method-table (table)
  (setf (method-table-method-list table) nil)
  (setf (method-table-cached-method table) nil)
  (setf (method-table-cached-method-types table) nil)
  table)

(defun add-method-table-method (table types method)
  "Push (TYPES, METHOD) onto the table and prime the cache
   with this newest entry."
  (setf (method-table-method-list table)
        (cons types (cons method (method-table-method-list table))))
  (setf (method-table-cached-method table) method)
  (setf (method-table-cached-method-types table) types)
  table)

(defun lists-match (list1 list2)
  "Element-wise EQ comparison of two equal-length lists.
   Returns T iff every pair matches; NIL on first mismatch or
   length difference."
  (let ((x list1) (y list2))
    (loop
      (cond
        ((null x) (return (null y)))
        ((null y) (return nil))
        ((not (eq (car x) (car y))) (return nil))
        (t (setq x (cdr x)) (setq y (cdr y)))))))

(defun find-method-table-method (table eqls-classes)
  "Walk TABLE's method-list (laid out as types1 method1 types2
   method2 …) and return the first method whose types match
   EQLS-CLASSES, or NIL."
  (let ((p (method-table-method-list table)))
    (loop
      (cond
        ((null p) (return nil))
        ((lists-match (car p) eqls-classes) (return (cadr p)))
        (t (setq p (cdr (cdr p))))))))

;; -- EQL specializer cache --------------------------------------------------
;;
;; Used by Stage H. Maps an eql-target object → a synthetic class
;; object representing the (eql obj) specialiser. One global table.

(defparameter *clos-singleton-specializers*
  (make-hash-table :test 'eql))

;; -- Predicates -------------------------------------------------------------
;;
;; Stage A defines the signatures; the predicates start as
;; constant-NIL because no CLOS classes exist yet. Stage B
;; promotes them once the meta-class tower is wired and
;; STANDARD-CLASS / STANDARD-GENERIC-FUNCTION become real
;; class objects.

(defun standard-class-p (x)
  (declare (ignore x))
  nil)

(defun standard-generic-function-p (x)
  (declare (ignore x))
  nil)

;; ─── Stage B: class metaobjects + bootstrap ─────────────────────────────────
;;
;; The chicken-and-egg: defclass needs the metaclass to exist, but
;; the metaclass is itself a class. Closette resolves this with a
;; manual construction sequence at the end of the file (its "10
;; easy steps"). We follow the same shape — the data (the-defclass-
;; standard-class) is declared up front, then the bootstrap section
;; at the bottom of this stage hand-builds standard-class and T.

;; -- find-class registry ----------------------------------------------------

(defparameter *clos-class-table* (make-hash-table :test 'eq))

(defun find-class (name &optional (errorp t))
  "Return the class object registered under NAME. Signals an
   error if no such class is registered and ERRORP is true;
   returns NIL otherwise."
  (let ((c (gethash name *clos-class-table*)))
    (cond
      (c c)
      (errorp (error "find-class: no class named ~A" name))
      (t nil))))

(defun %setf-find-class (new-value name &optional errorp)
  (declare (ignore errorp))
  (setf (gethash name *clos-class-table*) new-value)
  new-value)

(defun forget-all-classes ()
  (clrhash *clos-class-table*)
  nil)

(defparameter *clos-generic-function-table* (make-hash-table :test 'eq))

(defun forget-all-generic-functions ()
  (clrhash *clos-generic-function-table*)
  nil)

;; -- Slot definitions (plist representation) --------------------------------
;;
;; Closette represents slot definitions as plists with :name,
;; :initargs, :initform, :initfunction, :readers, :writers,
;; :allocation, optionally :documentation and :shared-slot. The
;; accessors are getf/setf-getf*. See clos.lisp:747.

(defun make-direct-slot-definition (&key name (initargs nil) (initform nil)
                                          (initfunction nil) (readers nil)
                                          (writers nil) (allocation :instance))
  (let ((slot (list ':name name
                    ':initargs initargs
                    ':initform initform
                    ':initfunction initfunction
                    ':readers readers
                    ':writers writers
                    ':allocation allocation)))
    (when (eq allocation :class)
      (setf slot (nconc slot (list ':shared-slot (list secret-unbound-value)))))
    slot))

(defun make-effective-slot-definition (&key name (initargs nil) (initform nil)
                                            (initfunction nil) (allocation :instance))
  (list ':name name
        ':initargs initargs
        ':initform initform
        ':initfunction initfunction
        ':allocation allocation))

(defun slot-definition-name (slot) (getf slot ':name))
(defun slot-definition-initargs (slot) (getf slot ':initargs))
(defun slot-definition-initform (slot) (getf slot ':initform))
(defun slot-definition-initfunction (slot) (getf slot ':initfunction))
(defun slot-definition-readers (slot) (getf slot ':readers))
(defun slot-definition-writers (slot) (getf slot ':writers))
(defun slot-definition-allocation (slot) (getf slot ':allocation))
(defun slot-definition-documentation (slot) (getf slot ':documentation))
(defun slot-definition-shared-slot (slot) (getf slot ':shared-slot))

(defun instance-slot-p (slot)
  (eq (slot-definition-allocation slot) ':instance))

;; -- Standard-class slot positions (the meta-circular shortcut) -------------
;;
;; standard-class's effective-slots layout — the index of each
;; slot in the slot-storage vector. Closette hard-codes these in
;; slot-location; we mirror the choice. See the-defclass-standard-
;; class below for the source of truth, but ORDER MATTERS for the
;; bootstrap so we list them once explicitly.

(defparameter *standard-class-slot-names*
  '(name documentation direct-subclasses direct-superclasses
    class-precedence-list direct-methods direct-slots
    effective-slots shared-slot-definitions shared-slots
    direct-default-initargs effective-default-initargs))

;; the-class-standard-class is filled in by the bootstrap below.
;; the-slots-of-standard-class is the list of effective-slot-
;; definitions for standard-class.
(defparameter the-slots-of-standard-class nil)
(defparameter the-class-standard-class nil)

(defparameter the-defclass-standard-class
  '(defclass standard-class (class)
     ((name :initarg :name)
      (documentation :initform () :initarg :documentation)
      (direct-subclasses :initform ())
      (direct-superclasses :initarg :direct-superclasses)
      (class-precedence-list)
      (direct-methods :initform ())
      (direct-slots)
      (effective-slots)
      (shared-slot-definitions :initform ())
      (shared-slots :initform ())
      (direct-default-initargs :initform () :initarg :direct-default-initargs)
      (effective-default-initargs :initform ()))))

;; -- Slot access ------------------------------------------------------------
;;
;; slot-location walks the class's effective-slots list looking
;; for the named slot, returning its position-among-instance-slots
;; (shared slots are counted separately). Closette short-circuits
;; the lookup of 'effective-slots in standard-class because that
;; query would recurse forever otherwise.

(defun slot-location (class slot-name)
  "Return the index of SLOT-NAME within CLASS's instance-slot
   storage, or NIL if it's a shared slot or absent. Special case:
   the lookup of 'effective-slots in standard-class returns 7 by
   construction — without this short-circuit, finding any slot in
   standard-class would recurse infinitely."
  (cond
    ((and (eq slot-name 'effective-slots)
          (eq class the-class-standard-class))
     7)
    (t
     (let ((slots (class-effective-slots class))
           (pos 0)
           (result nil))
       (loop
         (cond
           ((null slots) (return nil))
           (t (let ((s (car slots)))
                (cond
                  ((eq (slot-definition-name s) slot-name)
                   (setq result pos)
                   (setq slots nil))
                  (t (when (instance-slot-p s)
                       (setq pos (+ pos 1)))
                     (setq slots (cdr slots))))))))
       result))))

(defun shared-slot-location (class slot-name)
  (let ((slots (class-shared-slot-definitions class))
        (pos 0)
        (result nil))
    (loop
      (cond
        ((null slots) (return nil))
        (t (let ((s (car slots)))
             (cond
               ((eq (slot-definition-name s) slot-name)
                (setq result pos)
                (setq slots nil))
               (t (setq pos (+ pos 1))
                  (setq slots (cdr slots))))))))
    result))

;; slot-contents is just svref. Closette uses a uref-with-offset
;; trick for inline access; we don't have that and svref is fine.
(defun slot-contents (slots location) (svref slots location))
(defun %setf-slot-contents (new-value slots location)
  (setf (svref slots location) new-value)
  new-value)

(defun std-slot-value (instance slot-name)
  "Read SLOT-NAME from INSTANCE via slot-location. Errors if
   the slot doesn't exist or is unbound."
  (let ((class (class-of instance)))
    (let ((location (slot-location class slot-name))
          (val nil))
      (cond
        (location
         (setq val (slot-contents (std-instance-slots instance) location)))
        (t
         (let ((sloc (shared-slot-location class slot-name)))
           (cond
             (sloc
              (setq val (car (slot-contents (class-shared-slots class) sloc))))
             (t (error "The slot ~A is missing from the class ~A."
                       slot-name class))))))
      (when (eq secret-unbound-value val)
        (error "The slot ~A is unbound in the object ~A." slot-name instance))
      val)))

(defun %setf-std-slot-value (new-value instance slot-name)
  (let ((class (class-of instance)))
    (let ((location (slot-location class slot-name)))
      (cond
        (location
         (setf (slot-contents (std-instance-slots instance) location)
               new-value))
        (t
         (let ((sloc (shared-slot-location class slot-name)))
           (cond
             (sloc
              (setf (car (slot-contents (class-shared-slots class) sloc))
                    new-value))
             (t (error "The slot ~A is missing from the class ~A."
                       slot-name class))))))))
  new-value)

;; slot-value / (setf slot-value) — Stage D adds the full
;; standard-class-p dispatch path and slot-value-using-class.
;; For Stage B we just route to std-slot-value.

(defun slot-value (object slot-name)
  (std-slot-value object slot-name))

(defun %setf-slot-value (new-value object slot-name)
  (%setf-std-slot-value new-value object slot-name))

;; -- Class metaobject accessors --------------------------------------------
;;
;; Closette implements these as plain defuns calling slot-value.
;; They become generic functions in stage E (which dispatches via
;; standard-class-p to the std-slot-value fast path); for now they
;; just call slot-value directly.

(defun class-name (class) (slot-value class 'name))
(defun %setf-class-name (new-value class)
  (setf (slot-value class 'name) new-value))

(defun class-documentation (class) (slot-value class 'documentation))
(defun %setf-class-documentation (new-value class)
  (setf (slot-value class 'documentation) new-value))

(defun class-direct-superclasses (class)
  (slot-value class 'direct-superclasses))
(defun %setf-class-direct-superclasses (new-value class)
  (setf (slot-value class 'direct-superclasses) new-value))

(defun class-direct-slots (class) (slot-value class 'direct-slots))
(defun %setf-class-direct-slots (new-value class)
  (setf (slot-value class 'direct-slots) new-value))

(defun class-precedence-list (class)
  (slot-value class 'class-precedence-list))
(defun %setf-class-precedence-list (new-value class)
  (setf (slot-value class 'class-precedence-list) new-value))

(defun class-effective-slots (class) (slot-value class 'effective-slots))
(defun %setf-class-effective-slots (new-value class)
  (setf (slot-value class 'effective-slots) new-value))

(defun class-direct-subclasses (class)
  (slot-value class 'direct-subclasses))
(defun %setf-class-direct-subclasses (new-value class)
  (setf (slot-value class 'direct-subclasses) new-value))

(defun class-direct-methods (class) (slot-value class 'direct-methods))
(defun %setf-class-direct-methods (new-value class)
  (setf (slot-value class 'direct-methods) new-value))

(defun class-shared-slots (class) (slot-value class 'shared-slots))
(defun %setf-class-shared-slots (new-value class)
  (setf (slot-value class 'shared-slots) new-value))

(defun class-shared-slot-definitions (class)
  (slot-value class 'shared-slot-definitions))
(defun %setf-class-shared-slot-definitions (new-value class)
  (setf (slot-value class 'shared-slot-definitions) new-value))

;; -- subclassp / sub-specializer-p -----------------------------------------

(defun subclassp (c1 c2)
  (not (null (find c2 (class-precedence-list c1)))))

(defun sub-specializer-p (c1 c2 c-arg)
  (let ((cpl (class-precedence-list c-arg)))
    (not (null (find c2 (cdr (member c1 cpl)))))))

;; -- class-of + built-in-class-of ------------------------------------------
;;
;; Initial built-in-class-of returns NIL for built-in types until
;; Stage C runs the full bootstrap that defclasses INTEGER /
;; SYMBOL / etc. After that, all built-ins resolve.

(defun class-of (x)
  (cond
    ((clos-instance-p x) (std-instance-class x))
    (t (built-in-class-of x))))

(defun built-in-class-of (x)
  "Slow but straightforward typecase. Stage C extends the
   class table to cover all the built-in types this checks for."
  (typecase x
    (null      (find-class 'null nil))
    (symbol    (find-class 'symbol nil))
    (integer   (find-class 'integer nil))
    (cons      (find-class 'cons nil))
    (character (find-class 'character nil))
    (string    (find-class 'string nil))
    (vector    (find-class 'vector nil))
    (function  (find-class 'function nil))
    (t         (find-class 't nil))))

;; -- Standard-instance allocation (used during bootstrap and later) ---------

(defun std-allocate-instance (class)
  (allocate-std-instance
    class
    (allocate-slot-storage
      (length (class-effective-slots class))
      secret-unbound-value)))

;; ─── Bootstrap steps 1-5: skeleton standard-class + T ───────────────────────
;;
;; Each defclass slot in the-defclass-standard-class becomes an
;; effective-slot-definition (just :name + :allocation :instance
;; for now; initforms get plumbed in stage C). Then we manually
;; allocate standard-class and patch up the circular class-of
;; link — once that's done class-effective-slots etc. all work
;; on standard-class itself.

(forget-all-classes)
(forget-all-generic-functions)

;; Step 1: build effective-slot-definitions for standard-class's slots.
(setq the-slots-of-standard-class
      (mapcar (lambda (slotd)
                (make-effective-slot-definition
                  :name (car slotd)
                  :initargs (let ((a (getf (cdr slotd) ':initarg)))
                              (cond (a (list a))
                                    (t nil)))
                  :initform (getf (cdr slotd) ':initform)
                  :allocation ':instance))
              (nth 3 the-defclass-standard-class)))

;; Step 2: hand-allocate standard-class with placeholder class link.
(setq the-class-standard-class
      (allocate-std-instance
        'tba
        (make-array (length the-slots-of-standard-class)
                    :initial-element secret-unbound-value)))
;; Step 3: install the circular class-of link.
(setf (std-instance-class the-class-standard-class)
      the-class-standard-class)
;; (now slot-value on standard-class works — slot-location's
;; effective-slots short-circuit kicks in)

;; Step 4: fill in standard-class's class-effective-slots so that
;; lookups for OTHER slot names also work.
(setf (class-effective-slots the-class-standard-class)
      the-slots-of-standard-class)
;; Step 5: hand-build the class T. T has no superclasses, no
;; methods, no slots — it's the root of the type hierarchy.
(setf (gethash 't *clos-class-table*)
      (let ((class (std-allocate-instance the-class-standard-class)))
        (setf (class-name class) 't)
        (setf (class-documentation class) nil)
        (setf (class-direct-subclasses class) nil)
        (setf (class-direct-superclasses class) nil)
        (setf (class-direct-methods class) nil)
        (setf (class-direct-slots class) nil)
        (setf (class-precedence-list class) (list class))
        (setf (class-effective-slots class) nil)
        (setf (class-shared-slot-definitions class) nil)
        (setf (class-shared-slots class) nil)
        class))

;; Return a printable sentinel so Session::eval's last-value
;; format_word doesn't try to render the circular T-class
;; instance (the printer doesn't yet handle cycles). Stage I
;; will introduce a print-object hook that breaks the cycle
;; properly; until then, a leading-NIL works.
nil
