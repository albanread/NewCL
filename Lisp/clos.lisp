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
