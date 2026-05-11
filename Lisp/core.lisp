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
(defun caddr   (lst) (car (cdr (cdr lst))))
(defun cdddr   (lst) (cdr (cdr (cdr lst))))
(defun cadddr  (lst) (car (cdr (cdr (cdr lst)))))
(defun cddddr  (lst) (cdr (cdr (cdr (cdr lst)))))
(defun cadar   (lst) (car (cdr (car lst))))
(defun caaar   (lst) (car (car (car lst))))

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

;; -- Sequence/list searches with :test and :key -----------------------------
;;
;; CL's family of search functions (member, find, position, assoc,
;; ...) accept :test (the predicate, default eql) and :key (an
;; accessor applied to each candidate before comparison, default
;; identity). Closette uses both heavily — assoc with :key #'cadr
;; for example. The implementations below desugar each candidate
;; check to (funcall test item (funcall key elem)).

(defun member (item lst &key (test #'eql) (key #'identity))
  "Return the tail of LST starting at the first ELEM where
   (funcall TEST ITEM (funcall KEY ELEM)) is true, or NIL."
  (cond
    ((null lst) nil)
    ((funcall test item (funcall key (car lst))) lst)
    (t (member item (cdr lst) :test test :key key))))

(defun find (item lst &key (test #'eql) (key #'identity))
  "Return the first element of LST that matches ITEM under TEST
   (after KEY is applied to the element), or NIL."
  (cond
    ((null lst) nil)
    ((funcall test item (funcall key (car lst))) (car lst))
    (t (find item (cdr lst) :test test :key key))))

(defun position (item lst &key (test #'eql) (key #'identity))
  "Return the index in LST of the first matching element, or NIL."
  (%position-from item lst 0 test key))

(defun %position-from (item lst i test key)
  (cond
    ((null lst) nil)
    ((funcall test item (funcall key (car lst))) i)
    (t (%position-from item (cdr lst) (+ i 1) test key))))

(defun assoc (item alist &key (test #'eql) (key #'identity))
  "Walk ALIST; return the first entry whose CAR matches ITEM
   under TEST (with KEY applied to the entry's car). Default
   TEST is eql to match CL — earlier this was equal because we
   didn't have keyword args; callers that relied on equal should
   pass `:test #'equal`."
  (cond
    ((null alist) nil)
    ((funcall test item (funcall key (car (car alist)))) (car alist))
    (t (assoc item (cdr alist) :test test :key key))))

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

;; (list* a b c lst) ≡ (cons a (cons b (cons c lst))).
;; CL's variadic list* — the last arg is used as the tail; earlier
;; args are consed onto the front. (list* x) ≡ x.
(defun %list*-build (head r)
  (if (null r)
      head
      (cons head (%list*-build (car r) (cdr r)))))
(defun list* (head &rest r)
  (%list*-build head r))

;; Variadic append: (append a b c d) ≡ (append a (append b (append c d))).
;; Reuses the binary `append` defined above.
(defun %append-many (lst rest-of-lists)
  (if (null rest-of-lists)
      lst
      (append lst (%append-many (car rest-of-lists) (cdr rest-of-lists)))))
(defun append* (&rest lists)
  ;; Named `append*` to coexist with the binary `append`. When &rest
  ;; argument unpacking matures we'll merge them.
  (cond
    ((null lists) nil)
    ((null (cdr lists)) (car lists))
    (t (%append-many (car lists) (cdr lists)))))

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

(defun min2 (a b) (if (< a b) a b))
(defun max2 (a b) (if (> a b) a b))

;; Variadic min / max via &rest. (min) is an error in CL — we
;; return nil for the empty case instead, until conditions exist.
(defun %min-of (a r)
  (if (null r) a (%min-of (min2 a (car r)) (cdr r))))
(defun min (a &rest r) (%min-of a r))

(defun %max-of (a r)
  (if (null r) a (%max-of (max2 a (car r)) (cdr r))))
(defun max (a &rest r) (%max-of a r))

(defun abs (n) (if (< n 0) (- n) n))

;; -- Loops -------------------------------------------------------------------
;;
;; (loop body...) repeats body forever; (return v) exits the
;; immediately enclosing loop with value v. Both wrap the
;; %native-loop / %loop-return primitives.
;;
;; CAVEAT: like (error ...), (return) doesn't unwind — code
;; *after* the (return) call but still inside the same iteration
;; body still runs. Idiomatic CL puts return at the end of a
;; cond/case clause, which sidesteps this:
;;
;;   (loop (cond ((stop?) (return result))
;;               (t (do-work))))
;;
;; works correctly. The dual-form
;;
;;   (loop (when (stop?) (return result))
;;         (do-work))                      ; <-- still runs after return
;;
;; doesn't, because (do-work) is a sibling of (when ...) and
;; runs once the when's expansion has stashed the return value.

(defmacro loop (&rest body)
  `(%native-loop (lambda () ,@body)))

(defmacro return (&rest args)
  ;; (return)   → exit with nil
  ;; (return v) → exit with v
  (cond
    ((null args) `(%loop-return nil))
    (t `(%loop-return ,(car args)))))

(defmacro let* (bindings &rest body)
  "Sequential let — each binding sees the earlier bindings.
   Expands to nested `let` forms."
  (cond
    ((null bindings) `(progn ,@body))
    (t `(let (,(car bindings))
          (let* ,(cdr bindings) ,@body)))))

;; -- Tier 1 macros (chunk 7) -------------------------------------------------
;;
;; CL forms Closette uses heavily. None of these need new compiler
;; or runtime machinery — every expansion lands on something that
;; already exists.

(defmacro declare (&rest decls)
  "No-op. CL `(declare …)` carries metadata for the compiler
   (ignore, type, dynamic-extent, …). We don't act on any of it
   yet, so the form just expands to NIL — sufficient because
   declarations always appear in implicit-progn position and a
   stray NIL there is harmless."
  nil)

(defmacro prog1 (first &rest rest)
  "Evaluate FIRST, save its value, then evaluate REST in order
   and return FIRST's value."
  (let ((r (gensym "PROG1")))
    `(let ((,r ,first))
       ,@rest
       ,r)))

(defmacro prog2 (first second &rest rest)
  "Evaluate the three sections, return SECOND's value."
  `(progn ,first (prog1 ,second ,@rest)))

(defmacro defvar (name &rest rest)
  "Declare a special variable. CL distinguishes defvar (assign
   only if currently unbound) from defparameter (always assign);
   without a boundp primitive yet we treat them identically.
   Closette's defvars are at file load time and aren't reset
   later, so this is safe."
  (cond
    ((null rest) `(defparameter ,name nil))
    (t `(defparameter ,name ,(car rest)))))

(defmacro push (value place)
  "Prepend VALUE to the list stored at PLACE. PLACE is evaluated
   twice — fine for symbol or simple-accessor places (which is
   what Closette uses) but not safe for places with side effects.
   A proper get-setf-expansion-based version lands when we have
   one."
  `(setf ,place (cons ,value ,place)))

(defmacro pop (place)
  "Remove and return the head of the list at PLACE. Same
   double-evaluation caveat as `push`."
  `(prog1 (car ,place) (setf ,place (cdr ,place))))

(defmacro case (keyform &rest clauses)
  "(case KEY (KEYS body…) (KEYS body…) …) — dispatch on KEY.
   KEYS is either a symbol/atom matched with EQL, a list of
   atoms (matches if KEY EQL any of them), or T / OTHERWISE for
   the catch-all clause."
  (let ((k (gensym "CASE-KEY")))
    `(let ((,k ,keyform))
       (cond
         ,@(mapcar (lambda (clause)
                     (case-clause-expand clause k))
                   clauses)))))

(defun case-clause-expand (clause keyvar)
  "Helper for the CASE macro. Builds one cond-clause from a
   case-clause."
  (let ((keys (car clause))
        (body (cdr clause)))
    (cond
      ((or (eq keys 'otherwise) (eq keys 't))
       `(t ,@body))
      ((listp keys)
       `((or ,@(mapcar (lambda (key) `(eql ,keyvar ',key)) keys))
         ,@body))
      (t `((eql ,keyvar ',keys) ,@body)))))

(defmacro ecase (keyform &rest clauses)
  "Like CASE but with no fall-through — signals an error if no
   clause matches. No OTHERWISE clause expected."
  (let ((k (gensym "ECASE-KEY")))
    `(let ((,k ,keyform))
       (cond
         ,@(mapcar (lambda (clause)
                     (case-clause-expand clause k))
                   clauses)
         (t (error "ecase: no matching clause for ~A" ,k))))))

(defmacro typecase (keyform &rest clauses)
  "(typecase KEY (TYPE body…) …) — clause matches if KEY is of
   TYPE per `typep`. T matches anything (catch-all)."
  (let ((k (gensym "TYPECASE-KEY")))
    `(let ((,k ,keyform))
       (cond
         ,@(mapcar (lambda (clause)
                     (typecase-clause-expand clause k))
                   clauses)))))

(defun typecase-clause-expand (clause keyvar)
  (let ((type (car clause))
        (body (cdr clause)))
    (cond
      ((eq type 't) `(t ,@body))
      (t `((typep ,keyvar ',type) ,@body)))))

(defmacro dolist (binding &rest body)
  "(dolist (var list [result-form]) body…) — bind VAR to each
   element of LIST in turn, evaluate BODY. Returns RESULT-FORM
   (or NIL if absent)."
  (let* ((var (car binding))
         (list-form (car (cdr binding)))
         (result-form (cond
                        ((null (cdr (cdr binding))) nil)
                        (t (car (cdr (cdr binding))))))
         (rem (gensym "DOLIST-REM")))
    `(let ((,rem ,list-form)
           (,var nil))
       (loop
         (cond
           ((null ,rem) (return ,result-form))
           (t (setq ,var (car ,rem))
              ,@body
              (setq ,rem (cdr ,rem))))))))

(defmacro block (name &rest body)
  "(block NAME body…) — establish a named exit point. Inside
   BODY, `(return-from NAME val)` immediately exits the block
   with VAL. Without a matching return-from the block returns
   the last body form's value.

   Implementation: wraps BODY in a thunk passed to
   %native-block, which sets a setjmp at entry and longjmps
   from a matching %return-from. The longjmp aborts the rest
   of the body — unlike loop's flag-based (return), this is a
   real non-local exit."
  `(%native-block ',name (lambda () ,@body)))

(defmacro return-from (name value)
  "(return-from NAME val) — non-locally exit the innermost
   enclosing (block NAME …) with VAL."
  `(%return-from ',name ,value))

(defmacro dotimes (binding &rest body)
  "(dotimes (var count [result-form]) body…) — bind VAR to
   0, 1, …, COUNT-1; evaluate BODY each time. Returns RESULT-FORM."
  (let* ((var (car binding))
         (count-form (car (cdr binding)))
         (result-form (cond
                        ((null (cdr (cdr binding))) nil)
                        (t (car (cdr (cdr binding))))))
         (limit (gensym "DOTIMES-LIMIT")))
    `(let ((,var 0)
           (,limit ,count-form))
       (loop
         (cond
           ((>= ,var ,limit) (return ,result-form))
           (t ,@body
              (setq ,var (+ ,var 1))))))))

;; -- Property lists ----------------------------------------------------------

(defun getf (plist key)
  "Walk PLIST, returning the value paired with KEY, or nil if not
   found. The plist is a flat list of alternating keys and values:
   (:a 1 :b 2 :c 3)."
  (cond
    ((null plist) nil)
    ((eq (car plist) key) (car (cdr plist)))
    (t (getf (cdr (cdr plist)) key))))

;; -- Conditions --------------------------------------------------------------
;;
;; (error condition-or-message) signals; (handler-case body
;; (error (var) recovery)) catches. The condition is whatever was
;; passed to error — typically a string. Conditions as typed
;; objects with class hierarchies wait on CLOS.

;; -- Tier-2 utilities (chunk 12) ---------------------------------------------
;;
;; A pile of small CL functions Closette pulls in. None need new
;; compiler or runtime support; each is a thin wrapper over
;; primitives we already have.

(defun third (x) (car (cdr (cdr x))))

(defun complement (pred)
  "Return a predicate that negates PRED. PRED is currently
   assumed unary; CL allows variadic but Closette only uses
   the unary case."
  (lambda (x) (not (funcall pred x))))

(defun fdefinition (name)
  "Return the function bound to NAME. Same as `symbol-function`
   for plain symbols; CL also accepts (setf NAME) function-name
   lists which we don't yet support."
  (symbol-function name))

(defun make-symbol (name)
  "Return a fresh symbol whose name is NAME. CL's make-symbol
   produces an UNINTERNED symbol; we don't have uninterned
   symbols, so this just calls intern on a uniqued name and
   returns. Same compromise as gensym."
  (intern (format nil "~A~A" name (gensym ""))))

(defun nreverse (lst)
  "Same as REVERSE for now. CL nreverse is allowed to mutate
   LST's cons cells; we just allocate fresh — slower but
   semantically equivalent for non-shared lists. Closette
   doesn't depend on the destructive behaviour."
  (reverse lst))

(defun nconc (&rest lists)
  "Concatenate LISTS. CL nconc destructively splices each
   non-last list's tail; we use append* (allocating). Closette
   doesn't depend on the destructive behaviour."
  (apply #'append* lists))

(defun find-if (pred lst &key (key #'identity))
  "Return the first element of LST for which (PRED (KEY elem))
   is true, or NIL."
  (cond
    ((null lst) nil)
    ((funcall pred (funcall key (car lst))) (car lst))
    (t (find-if pred (cdr lst) :key key))))

(defun remove-if (pred lst &key (key #'identity))
  "Return a fresh list of LST's elements for which (PRED (KEY
   elem)) is FALSE. Order preserved."
  (cond
    ((null lst) nil)
    ((funcall pred (funcall key (car lst)))
     (remove-if pred (cdr lst) :key key))
    (t (cons (car lst) (remove-if pred (cdr lst) :key key)))))

(defun remove-if-not (pred lst &key (key #'identity))
  "Return a fresh list of LST's elements for which (PRED (KEY
   elem)) is TRUE — the keep-matching counterpart of remove-if."
  (cond
    ((null lst) nil)
    ((funcall pred (funcall key (car lst)))
     (cons (car lst) (remove-if-not pred (cdr lst) :key key)))
    (t (remove-if-not pred (cdr lst) :key key))))

(defun remove (item lst &key (test #'eql) (key #'identity))
  "Return a fresh list with all elements matching ITEM removed."
  (cond
    ((null lst) nil)
    ((funcall test item (funcall key (car lst)))
     (remove item (cdr lst) :test test :key key))
    (t (cons (car lst) (remove item (cdr lst) :test test :key key)))))

;; -- sort -------------------------------------------------------------------
;;
;; Mergesort variant. CL's `sort` is destructive but allowed to
;; share — we just return a fresh list. Comparator returns T iff
;; the first arg should come before the second.

(defun %split-list (lst)
  "Split LST into two halves; returns (cons left right)."
  (let ((slow lst) (fast lst) (n 0))
    (loop
      (cond
        ((or (null fast) (null (cdr fast))) (return nil))
        (t (setq slow (cdr slow))
           (setq fast (cdr (cdr fast)))
           (setq n (+ n 1)))))
    (let ((left nil) (rest lst))
      (loop
        (cond
          ((zerop n) (return nil))
          (t (setq left (cons (car rest) left))
             (setq rest (cdr rest))
             (setq n (- n 1)))))
      (cons (reverse left) rest))))

(defun %merge (a b cmp)
  (cond
    ((null a) b)
    ((null b) a)
    ((funcall cmp (car a) (car b))
     (cons (car a) (%merge (cdr a) b cmp)))
    (t (cons (car b) (%merge a (cdr b) cmp)))))

(defun sort (lst cmp)
  "Mergesort by CMP. CMP returns true iff its first arg should
   come before its second. Returns a fresh list (we don't share
   tails) — CL's destructive variant is harmless here because we
   never read the input again."
  (cond
    ((null lst) nil)
    ((null (cdr lst)) lst)
    (t (let* ((split (%split-list lst))
              (left  (sort (car split) cmp))
              (right (sort (cdr split) cmp)))
         (%merge left right cmp)))))

(defun copy-list (lst)
  (cond ((null lst) nil)
        (t (cons (car lst) (copy-list (cdr lst))))))

(defun remove-duplicates (lst &key (test #'eql) (key #'identity))
  "Return a fresh list with duplicates removed. CL default
   drops the EARLIER occurrence of any pair of matches, so the
   LAST occurrence wins. (Pass :from-end to invert — not yet
   supported here.)"
  (cond
    ((null lst) nil)
    ((member (funcall key (car lst)) (cdr lst) :test test :key key)
     (remove-duplicates (cdr lst) :test test :key key))
    (t (cons (car lst)
             (remove-duplicates (cdr lst) :test test :key key)))))

(defun set-difference (xs ys &key (test #'eql) (key #'identity))
  "Elements of XS not in YS (under TEST + KEY). Order preserved."
  (cond
    ((null xs) nil)
    ((member (funcall key (car xs)) ys :test test :key key)
     (set-difference (cdr xs) ys :test test :key key))
    (t (cons (car xs)
             (set-difference (cdr xs) ys :test test :key key)))))

(defun intersection (xs ys &key (test #'eql) (key #'identity))
  "Elements of XS that are also in YS."
  (cond
    ((null xs) nil)
    ((member (funcall key (car xs)) ys :test test :key key)
     (cons (car xs)
           (intersection (cdr xs) ys :test test :key key)))
    (t (intersection (cdr xs) ys :test test :key key))))

(defun union (xs ys &key (test #'eql) (key #'identity))
  "Union as a fresh list — XS first (with duplicates removed
   relative to YS), then YS."
  (cond
    ((null xs) ys)
    ((member (funcall key (car xs)) ys :test test :key key)
     (union (cdr xs) ys :test test :key key))
    (t (cons (car xs)
             (union (cdr xs) ys :test test :key key)))))

(defun subseq (seq start &optional end)
  "Substring/sublist from START (inclusive) to END (exclusive,
   defaults to length of seq). Works on strings (delegates to
   substring) and lists; vectors not yet."
  (cond
    ((stringp seq)
     (substring seq start (cond ((null end) (length seq)) (t end))))
    ((listp seq)
     (let ((lst (nthcdr start seq))
           (n (cond ((null end) nil) (t (- end start)))))
       (cond
         ((null n) (copy-list lst))
         (t (subseq-take lst n)))))
    (t (error "subseq: unsupported sequence type: ~A" seq))))

(defun subseq-take (lst n)
  (cond
    ((or (null lst) (<= n 0)) nil)
    (t (cons (car lst) (subseq-take (cdr lst) (- n 1))))))

(defun coerce (object result-type)
  "Limited coerce: list <-> simple list/vector identity-ish;
   STRING from a list of characters; LIST from a string. CL's
   full coerce is overloaded — we cover what Closette uses.
   RESULT-TYPE is a symbol."
  (cond
    ((eq result-type 'list)
     (cond
       ((listp object) object)
       ((stringp object) (coerce-string-to-list object 0 (length object)))
       (t (error "coerce: cannot coerce ~A to LIST" object))))
    ((eq result-type 'string)
     (cond
       ((stringp object) object)
       ((listp object) (coerce-list-to-string object))
       (t (error "coerce: cannot coerce ~A to STRING" object))))
    ((or (eq result-type 'vector) (eq result-type 'simple-vector))
     (cond
       ((vectorp object) object)
       ((listp object) (apply #'vector object))
       (t (error "coerce: cannot coerce ~A to VECTOR" object))))
    (t (error "coerce: unsupported result-type ~A" result-type))))

(defun coerce-string-to-list (s i n)
  (cond
    ((>= i n) nil)
    (t (cons (char s i) (coerce-string-to-list s (+ i 1) n)))))

(defun coerce-list-to-string (chars)
  (cond
    ((null chars) "")
    (t (string-concat (string-append-char "" (car chars))
                       (coerce-list-to-string (cdr chars))))))

;; -- Type predicates ---------------------------------------------------------
;;
;; Each is a one-line wrapper around (typep x 'KIND). We could
;; install them as native shims for speed, but most callers are
;; already inside Lisp-level code paths and the indirection is
;; cheap. Direct calls to (typep x 'foo) work too — these are
;; just the conventional CL spellings.

(defun symbolp (x) (typep x 'symbol))
(defun stringp (x) (typep x 'string))
(defun vectorp (x) (typep x 'vector))
(defun listp (x) (typep x 'list))
(defun consp (x) (typep x 'cons))
(defun integerp (x) (typep x 'integer))
(defun numberp (x) (typep x 'number))
(defun bignump (x) (typep x 'bignum))
(defun fixnump (x) (typep x 'fixnum))
(defun characterp (x) (typep x 'character))
(defun functionp (x) (typep x 'function))

;; -- defstruct ---------------------------------------------------------------
;;
;; A struct instance is laid out as a Vector with the type tag in
;; slot 0 and the user slots in slots 1..N. Each (defstruct NAME
;; (slot1 default1) (slot2 default2) ...) generates:
;;
;;   - constructor:  (make-NAME &key slot1 slot2 ...) returning an
;;                   instance with the given inits (or defaults).
;;   - predicate:    (NAME-p obj) → T iff obj is a vector tagged
;;                   with 'NAME at slot 0.
;;   - accessors:    (NAME-slot1 obj), (NAME-slot2 obj), ...
;;   - setf-accessors: (%setf-NAME-slot1 val obj), ...
;;     reached via the generic (setf (NAME-slot1 obj) val) lowering
;;     in the compiler.
;;
;; This is enough for Closette's `method-table` defstruct. Things
;; not yet supported: :type / :include / :print-function /
;; :predicate / :constructor options, BOA-style positional
;; constructors, and the (slot default :type T :read-only T)
;; per-slot keyword args.

(defun defstruct-slot-name (spec)
  (cond
    ((symbolp spec) spec)
    (t (car spec))))

(defun defstruct-slot-default (spec)
  (cond
    ((symbolp spec) nil)
    ((null (cdr spec)) nil)
    (t (car (cdr spec)))))

(defun defstruct-symbol (prefix name)
  "Intern a fresh symbol whose name is PREFIX concatenated with
   NAME's printer text. Used by the defstruct macro to build the
   constructor / accessor / setter symbols at expansion time."
  (intern (format nil "~A~A" prefix name)))

(defun defstruct-build-constructor (name slots)
  (let ((ctor (defstruct-symbol "MAKE-" name))
        (n-slots (length slots)))
    `(defun ,ctor (&key
                    ,@(mapcar (lambda (s)
                                (list (defstruct-slot-name s)
                                      (defstruct-slot-default s)))
                              slots))
       ;; Allocate length n-slots+1 (1 extra cell for the type tag).
       (let ((__v (make-array ,(+ n-slots 1) :initial-element nil)))
         (setf (svref __v 0) ',name)
         ,@(defstruct-build-init-stmts slots 1)
         __v))))

(defun defstruct-build-init-stmts (slots i)
  "Emit a list of (setf (svref __v i) slot-name) forms, one per
   slot, with i counting up from 1. Recursive walk so we don't
   need a multi-list mapcar."
  (cond
    ((null slots) nil)
    (t (cons `(setf (svref __v ,i) ,(defstruct-slot-name (car slots)))
             (defstruct-build-init-stmts (cdr slots) (+ i 1))))))

(defun defstruct-build-predicate (name)
  (let ((pred (defstruct-symbol "" (format nil "~A-P" name))))
    `(defun ,pred (obj)
       (and (vectorp obj)
            (eq (svref obj 0) ',name)))))

(defun defstruct-build-accessors (name slots)
  (defstruct-build-accessors-iter name slots 1))

(defun defstruct-build-accessors-iter (name slots i)
  (cond
    ((null slots) nil)
    (t
     (let* ((slot (defstruct-slot-name (car slots)))
            (acc (defstruct-symbol (format nil "~A-" name) slot)))
       (cons `(defun ,acc (obj) (svref obj ,i))
             (defstruct-build-accessors-iter name (cdr slots) (+ i 1)))))))

(defun defstruct-build-setters (name slots)
  (defstruct-build-setters-iter name slots 1))

(defun defstruct-build-setters-iter (name slots i)
  (cond
    ((null slots) nil)
    (t
     (let* ((slot (defstruct-slot-name (car slots)))
            (acc-name (format nil "~A-~A" name slot))
            (setter (intern (format nil "%SETF-~A" acc-name))))
       (cons `(defun ,setter (val obj)
                (setf (svref obj ,i) val)
                val)
             (defstruct-build-setters-iter name (cdr slots) (+ i 1)))))))

(defmacro defstruct (name &rest slots)
  "Define a struct type NAME with the given SLOTS (each a symbol
   or a (name default) list). Generates a make-NAME constructor
   that takes the slot names as &key args, a NAME-P type
   predicate, and per-slot accessors NAME-SLOT plus matching
   setf-accessors %SETF-NAME-SLOT (the latter reached via the
   generic (setf (NAME-SLOT obj) val) lowering)."
  `(progn
     ,(defstruct-build-constructor name slots)
     ,(defstruct-build-predicate name)
     ,@(defstruct-build-accessors name slots)
     ,@(defstruct-build-setters name slots)
     ',name))

;; -- Hash tables -------------------------------------------------------------
;;
;; A hash table is a Vector laid out as:
;;   slot 0 — test symbol (one of EQ / EQL / EQUAL)
;;   slot 1 — current count of entries (fixnum, mutable via setf-svref)
;;   slot 2..N+1 — N buckets, each a list of (key . value) cons cells
;;
;; Closette and the GUI demos only need EQ / EQL tables, so we
;; don't yet content-hash for EQUAL (the bit-mix in %word-hash
;; gives different strings of equal contents different bucket
;; indices). EQUAL is tracked as the test for completeness so
;; callers can opt in once content-hash lands.
;;
;; The whole structure lives on the GC heap (vector + cons cells),
;; so old-to-young pointer marking and the ordinary trace pass take
;; care of survival across GC.

(defun make-hash-table (&key (test 'eql) (size 16))
  "Allocate a new hash table. TEST is one of EQ, EQL, or EQUAL
   (defaults to EQL). SIZE is the initial bucket count (defaults
   to 16). Returns the table."
  (let* ((nbuckets (max size 4))
         (v (make-array (+ nbuckets 2) :initial-element nil)))
    (setf (svref v 0) test)
    (setf (svref v 1) 0)
    v))

(defun %ht-test (ht) (svref ht 0))
(defun %ht-count (ht) (svref ht 1))
(defun %ht-bump-count (ht delta)
  (setf (svref ht 1) (+ (svref ht 1) delta)))
(defun %ht-nbuckets (ht) (- (length ht) 2))
(defun %ht-bucket (ht i) (svref ht (+ i 2)))
(defun %ht-set-bucket (ht i v) (setf (svref ht (+ i 2)) v))

(defun %ht-bucket-index (ht key)
  (mod (%word-hash key) (%ht-nbuckets ht)))

(defun %ht-keys-match (test k1 k2)
  "Compare K1 and K2 under TEST. EQUAL falls back to EQUAL on
   conses/strings; EQL handles fixnums/chars/symbols/T/NIL same
   as EQ in our current value set; EQ is identity."
  (cond
    ((eq test 'eq) (eq k1 k2))
    ((eq test 'eql) (eql k1 k2))
    ((eq test 'equal) (equal k1 k2))
    (t (eql k1 k2))))

(defun hash-table-count (ht)
  "Return the number of key/value pairs currently in HT."
  (%ht-count ht))

(defun hash-table-test (ht)
  "Return the test symbol HT was created with."
  (%ht-test ht))

(defun gethash (key ht &optional default)
  "Look up KEY in HT. Returns the associated value, or DEFAULT
   if none. Returns NIL as the secondary value when the key was
   absent, T when it was found."
  (let ((bucket (%ht-bucket ht (%ht-bucket-index ht key)))
        (test (%ht-test ht))
        (result default)
        (found nil))
    (loop
      (cond
        ((null bucket) (return nil))
        (t (let ((pair (car bucket)))
             (cond
               ((%ht-keys-match test (car pair) key)
                (setq result (cdr pair))
                (setq found t)
                (setq bucket nil))
               (t (setq bucket (cdr bucket))))))))
    (if found
        (values result t)
        (values default nil))))

(defun %hash-set (ht key val)
  "Insert or update KEY → VAL. Returns VAL. Used by
   `(setf (gethash ...) ...)` lowering."
  (let* ((bi (%ht-bucket-index ht key))
         (bucket (%ht-bucket ht bi))
         (test (%ht-test ht))
         (cur bucket)
         (done nil))
    (loop
      (cond
        ((or done (null cur))
         ;; Not found in walk — prepend a fresh pair to the
         ;; bucket. Inserting at the head is O(1) and keeps the
         ;; small-bucket hot path tight.
         (cond
           ((not done)
            (%ht-set-bucket ht bi (cons (cons key val) bucket))
            (%ht-bump-count ht 1)))
         (return val))
        (t (let ((pair (car cur)))
             (cond
               ((%ht-keys-match test (car pair) key)
                (setf (cdr pair) val)
                (setq done t))
               (t (setq cur (cdr cur))))))))))

(defun remhash (key ht)
  "Remove KEY from HT. Returns T if it was present, NIL otherwise."
  (let* ((bi (%ht-bucket-index ht key))
         (bucket (%ht-bucket ht bi))
         (test (%ht-test ht)))
    ;; Two cases: head-of-bucket vs middle. Handle head first.
    (cond
      ((null bucket) nil)
      ((%ht-keys-match test (car (car bucket)) key)
       (%ht-set-bucket ht bi (cdr bucket))
       (%ht-bump-count ht -1)
       t)
      (t
       ;; Walk with prev/cur so we can splice cur out by setting
       ;; (cdr prev) = (cdr cur).
       (let ((prev bucket)
             (cur (cdr bucket))
             (found nil))
         (loop
           (cond
             ((null cur) (return nil))
             ((%ht-keys-match test (car (car cur)) key)
              (setf (cdr prev) (cdr cur))
              (%ht-bump-count ht -1)
              (setq found t)
              (setq cur nil))
             (t (setq prev cur)
                (setq cur (cdr cur)))))
         found)))))

(defun clrhash (ht)
  "Empty HT. Returns HT."
  (let ((i 0)
        (n (%ht-nbuckets ht)))
    (loop
      (cond
        ((>= i n) (return nil))
        (t (%ht-set-bucket ht i nil)
           (setq i (+ i 1)))))
    (setf (svref ht 1) 0)
    ht))

(defun maphash (fn ht)
  "Call FN with each key and value of HT. Returns NIL."
  (let ((i 0)
        (n (%ht-nbuckets ht)))
    (loop
      (cond
        ((>= i n) (return nil))
        (t
         (let ((bucket (%ht-bucket ht i)))
           (loop
             (cond
               ((null bucket) (return nil))
               (t (funcall fn (car (car bucket)) (cdr (car bucket)))
                  (setq bucket (cdr bucket))))))
         (setq i (+ i 1)))))))

(defmacro multiple-value-list (form)
  "Evaluate FORM and return a fresh list of all the values it
   produced. If FORM returned a single value, the list has one
   element; if FORM was `(values v1 v2 ... vN)` in tail position,
   the list has N elements.

   Implementation: clear the multi-value slot before FORM runs, so
   that constants / variable lookups / native shim calls (which
   don't write the slot) are observable as such. Then JIT'd
   function calls in FORM either set the slot via `Expr::Values`
   (tail-position `(values ...)`) or via `EnsureSingleMv` (every
   other function exit). Either way `%multiple-value-list-of`
   reads the slot afterward and falls back to `(primary)` when it
   was never written."
  (let ((p (gensym "MV-PRIMARY")))
    `(progn
       (%mv-clear)
       (let ((,p ,form))
         (%multiple-value-list-of ,p)))))

(defmacro multiple-value-bind (vars form &rest body)
  "Evaluate FORM, then bind the symbols in VARS to its primary,
   secondary, … values. Excess vars get NIL; extra values are
   discarded. BODY is run in the new bindings.

   Expansion mirrors `multiple-value-list`: clear, evaluate form,
   snapshot, then destructure."
  (let ((p (gensym "MV-PRIMARY"))
        (l (gensym "MV-LIST")))
    `(progn
       (%mv-clear)
       (let ((,p ,form))
         (let ((,l (%multiple-value-list-of ,p)))
           ,(multiple-value-bind-build-bindings vars l body))))))

(defun multiple-value-bind-build-bindings (vars list-sym body)
  "Helper for the multiple-value-bind macro. Builds a chain of
   let-bindings that pull successive elements out of LIST-SYM and
   bind them to the names in VARS. Each step guards with `(if l
   ...)` so a list shorter than VARS binds the trailing names to
   NIL instead of crashing on (car nil). The generated form ends
   in BODY."
  (cond
    ((null vars) `(progn ,@body))
    (t
     (let ((var (car vars))
           (rest (cdr vars))
           (l (gensym "MV-TAIL")))
       `(let ((,var (if ,list-sym (car ,list-sym) nil))
              (,l (if ,list-sym (cdr ,list-sym) nil)))
          ,(multiple-value-bind-build-bindings rest l body))))))

(defmacro handler-case (body-form &rest clauses)
  "(handler-case body
      (error (var) recovery))
   For now only the ERROR clause is supported. The single-clause
   form is enough to demonstrate the unwind-and-bind mechanism;
   typed condition dispatch lands when CLOS does."
  (cond
    ((null clauses)
     ;; No clauses — the body's value is just returned.
     body-form)
    (t
     (let ((clause (car clauses)))
       (let ((var-list (car (cdr clause)))
             (handler-body (cdr (cdr clause))))
         (let ((var (car var-list)))
           `(%handler-case
              (lambda () ,body-form)
              (lambda (,var) ,@handler-body))))))))

;; -- iGui drawing ------------------------------------------------------------
;;
;; Colors are packed fixnums: 0xRRGGBBAA. (rgb r g b) sets alpha to
;; 255; (rgba r g b a) lets the caller specify it.

(defun rgb (r g b)
  "Pack a fully-opaque color into a fixnum."
  (+ (* r 16777216)        ; r << 24
     (* g 65536)            ; g << 16
     (* b 256)              ; b << 8
     255))

(defun rgba (r g b a)
  (+ (* r 16777216)
     (* g 65536)
     (* b 256)
     a))

;; A handful of named colors. Match common-CL/Win32 conventions
;; loosely; users who want their own should just call (rgb ...).
(defparameter +black+   (rgb 0 0 0))
(defparameter +white+   (rgb 255 255 255))
(defparameter +red+     (rgb 220 50 50))
(defparameter +green+   (rgb 50 180 80))
(defparameter +blue+    (rgb 50 100 200))
(defparameter +yellow+  (rgb 220 200 60))
(defparameter +slate+   (rgb 46 51 57))
(defparameter +panel+   (rgb 30 33 38))

(defmacro with-batch (child-id &rest body)
  "Open a drawing batch for CHILD-ID, evaluate BODY (which calls
   clear/fill-rect/draw-line/etc.), and submit on exit.

   Each new submit replaces the child's previous on-screen batch
   (latest-wins) — so the body should re-emit the entire pane,
   not just changes."
  `(progn
     (%begin-batch ,child-id)
     ,@body
     (%submit-batch)))

(defun clear (color)
  "Fill the active pane with COLOR."
  (%emit-clear color))

(defun fill-rect (x y w h color)
  (%emit-fill-rect x y w h color))

(defun stroke-rect (x y w h thickness color)
  (%emit-stroke-rect x y w h thickness color))

(defun draw-line (x1 y1 x2 y2 thickness color)
  (%emit-draw-line x1 y1 x2 y2 thickness color))

(defun draw-text (x y text size color)
  "Render TEXT at (X, Y) in Segoe UI at SIZE px. Y is the
   baseline-ish top of the text run. SIZE and coords are
   fixnums for now (sub-pixel waits on float support)."
  (%emit-draw-text x y text size color))

(defun draw-text-styled (x y text size color &rest opts)
  "Like draw-text but with styling. OPTS is a flat property list
   of any of:
     :family   STRING    — font family, e.g. \"Consolas\"
     :weight   FIXNUM    — 100..900 (regular = 400, bold = 700)
     :style    KEYWORD   — :normal | :italic | :oblique
     :stretch  FIXNUM    — 1 (ultra-condensed) .. 9 (ultra-expanded)
     :align    KEYWORD   — :leading | :trailing | :center | :justified
   Unrecognised keys are ignored. Missing keys take the same
   defaults as `draw-text`.

   Example:
     (draw-text-styled 10 20 \"Code\" 14 +white+
                       :family \"Consolas\" :weight 700 :style :italic)"
  (%emit-draw-text-styled x y text size color opts))

(defun fill-oval (x y w h color)
  "Filled ellipse, axis-aligned, with the given bounding box."
  (%emit-fill-oval x y w h color))

(defun stroke-oval (x y w h thickness color)
  (%emit-stroke-oval x y w h thickness color))

(defun fill-circle (cx cy radius color)
  (%emit-fill-circle cx cy radius color))

(defun stroke-circle (cx cy radius thickness color)
  (%emit-stroke-circle cx cy radius thickness color))

(defun draw-arc (cx cy radius rotation-deg aperture-deg thickness color)
  "Outlined circular arc centered at (CX, CY). ROTATION-DEG is the
   midpoint angle (0 points right, 90 points down) in degrees;
   APERTURE-DEG is the full angular span. Both are fixnums for now;
   floats land when the compiler grows them."
  (%emit-draw-arc cx cy radius rotation-deg aperture-deg thickness color))

(defun measure-text (child-id text size &rest opts)
  "Measure TEXT as it would render in CHILD-ID's pane. Returns a
   plist `(:width W :height H :ascent A :line-count N)` (all
   fixnums, rounded to nearest pixel) or NIL on failure.

   OPTS takes the same keys as `draw-text-styled` so layout sees
   the same metrics drawing will produce."
  (%measure-text child-id text size opts))

;; -- Log view ----------------------------------------------------------------

(defun log (control &rest args)
  "Format CONTROL with ARGS (same directives as `format`) and push
   the result as a single line into the iGui log overlay. Open
   the overlay via Tools → Log or Ctrl+Shift+L."
  (log-write (apply #'format nil control args)))

;; -- Text-view (terminal-style monospaced child) -----------------------------
;;
;; The native text-window primitives, rolled up into one place:
;;   open-text-window TITLE       → child-id (fixnum) or NIL
;;   text-write ID STRING         → write at cursor (handles \n \r \t \b)
;;   text-write-char ID CHAR      → single-char convenience
;;   text-clear ID                → wipe whole grid, cursor → (0,0)
;;   text-clear-eol ID            → clear cursor → end of line
;;   text-clear-eos ID            → clear cursor → bottom-right
;;   text-newline ID              → CR + LF, scroll if at bottom
;;   text-scroll-up ID N          → scroll grid up N rows
;;   text-set-cursor ID ROW COL   → move cursor (clamped)
;;   text-set-pen ID FG BG        → packed-RGBA colours
;;   text-reset-pen ID            → restore defaults
;;   text-show-caret ID FLAG      → caret visibility
;;
;; Colours are packed fixnums via (rgb r g b) / (rgba r g b a),
;; same encoding the geometry primitives use.

(defun text-format (id control &rest args)
  "Format CONTROL with ARGS (using `format` directives) and write
   the result into text window ID at the cursor. Returns T."
  (text-write id (apply #'format nil control args)))

(defun text-print (id obj)
  "Write OBJ's printed form into text window ID at the cursor."
  (text-write id (format nil "~A" obj)))

(defun text-println (id obj)
  "Like `text-print` but also issues a newline."
  (text-write id (format nil "~A" obj))
  (text-newline id))

;; -- String helpers ----------------------------------------------------------

(defun string-concat (a b)
  "Return a fresh string with B appended to A."
  (format nil "~A~A" a b))

(defun string-append-char (s c)
  "Return a fresh string with C appended to S."
  (format nil "~A~A" s c))

(defun string-without-last (s)
  "Return S with its last codepoint removed; empty string stays empty."
  (let ((n (length s)))
    (if (zerop n) s (substring s 0 (- n 1)))))

;; -- File I/O ----------------------------------------------------------------
;;
;; The native primitives are:
;;   open-input-file path        → handle (or 0 if open fails)
;;   open-output-file path       → handle (truncates existing)
;;   open-append-file path       → handle (creates or appends)
;;   close-stream handle         → t
;;   read-line handle            → string or nil at EOF
;;   read-char-from handle       → char or nil at EOF
;;   write-string-to handle s    → s
;;   file-position handle        → fixnum or -1
;;   file-length handle          → fixnum or -1
;;   file-exists path            → t / nil
;;   delete-file path            → t / nil
;;
;; The Lisp wrappers below add ergonomics — line-at-a-time text
;; iteration, RAII via with-open-file, file-as-string slurping.

(defun write-line (stream s)
  "Write S to STREAM followed by a newline. Returns S."
  (write-string-to stream s)
  (write-string-to stream (format nil "~%"))
  s)

(defmacro with-open-file (binding-and-mode &rest body)
  "(with-open-file (var path direction) body...)
   Direction is one of the keywords :input, :output, :append.
   Opens path, binds the handle to var, evaluates body, and closes
   the handle on the way out. (Without conditions we can't yet
   guarantee close on non-local exit; the body just isn't allowed
   to escape via a condition until those land.)"
  (let ((var (car binding-and-mode))
        (path (car (cdr binding-and-mode)))
        (direction (car (cdr (cdr binding-and-mode)))))
    ;; Dispatch at macro-expansion time: compare the keyword the
    ;; user passed against the literal direction keywords.
    (let ((open-fn (cond
                     ((eq direction ':input)  'open-input-file)
                     ((eq direction ':output) 'open-output-file)
                     ((eq direction ':append) 'open-append-file)
                     (t 'open-input-file))))
      `(let ((,var (,open-fn ,path)))
         (let ((result (progn ,@body)))
           (close-stream ,var)
           result)))))

(defun %read-lines-from (stream acc)
  ;; Tail-recursive line reader. Acc is built reversed; caller flips.
  (let ((line (read-line stream)))
    (if (null line)
        (reverse acc)
        (%read-lines-from stream (cons line acc)))))

(defun read-file-lines (path)
  "Read every line of PATH into a list of strings (newlines stripped)."
  (let ((stream (open-input-file path)))
    (let ((result (%read-lines-from stream nil)))
      (close-stream stream)
      result)))

(defun read-file-string (path)
  "Read the entire contents of PATH as a single string. Lines are
   joined with newlines."
  (let ((lines (read-file-lines path)))
    (cond
      ((null lines) "")
      ((null (cdr lines)) (car lines))
      (t (%join-lines lines)))))

(defun %join-lines (lines)
  ;; Concatenate lines with \n separators using format.
  (cond
    ((null lines) "")
    ((null (cdr lines)) (car lines))
    (t (format nil "~A~%~A" (car lines) (%join-lines (cdr lines))))))

(defun write-file-string (path s)
  "Write the string S to PATH, replacing any existing file."
  (let ((stream (open-output-file path)))
    (write-string-to stream s)
    (close-stream stream)
    s))

(defun write-file-lines (path lines)
  "Write each string in LINES to PATH, one per line, replacing
   any existing file."
  (let ((stream (open-output-file path)))
    (%write-lines-to stream lines)
    (close-stream stream)
    lines))

(defun %write-lines-to (stream lines)
  (cond
    ((null lines) nil)
    (t (write-line stream (car lines))
       (%write-lines-to stream (cdr lines)))))

;; -- Loader: load / require / provide / *load-path* / *modules* --------------
;;
;; Closely models CL's load / require / provide. The Rust primitive
;; %load-file (registered in ncl-compiler) reads a UTF-8 source file
;; and runs every top-level form through the active session — this
;; layer adds path resolution and the load-once memo.
;;
;; *load-path* is a list of directory strings searched left-to-right
;; by REQUIRE when a bare module name is passed. The driver mutates
;; it at startup to point into the exe-relative `Library/` directory;
;; user code can push extra paths.
;;
;; *modules* is a list of symbols REQUIRE has loaded. PROVIDE adds
;; to it. A REQUIRE for an already-provided module is a no-op.

(defparameter *load-path* '())
(defparameter *modules* '())
(defparameter *verbose-load* nil)

(defun load (path)
  "Read every top-level form from PATH and evaluate it. Returns T
   on success; the underlying %load-file shim signals on read or
   eval failure."
  (when *verbose-load*
    (format t ";;; loading ~A~%" path))
  (%load-file path))

(defun provide (module-name)
  "Record MODULE-NAME (a symbol) as loaded. REQUIRE for the same
   name afterwards becomes a no-op. Returns MODULE-NAME."
  (cond
    ((member module-name *modules*) module-name)
    (t (setq *modules* (cons module-name *modules*))
       module-name)))

(defun %require-search (name dirs)
  "Walk DIRS; return the first existing 'DIR/NAME.lisp' or NIL."
  (cond
    ((null dirs) nil)
    (t (let ((candidate (format nil "~A/~A.lisp" (car dirs) name)))
         (cond
           ((file-exists candidate) candidate)
           (t (%require-search name (cdr dirs))))))))

(defun require (module-name &optional explicit-path)
  "If MODULE-NAME is in *modules*, do nothing. Otherwise, load it:
   if EXPLICIT-PATH is given, load that file directly; else search
   *load-path* for MODULE-NAME.lisp. Returns MODULE-NAME on
   successful load, NIL if already provided."
  (cond
    ((member module-name *modules*) nil)
    (explicit-path
     (load explicit-path)
     (provide module-name))
    (t
     (let ((found (%require-search module-name *load-path*)))
       (cond
         (found
          (load found)
          (provide module-name))
         (t (error "require: cannot find ~A.lisp in *load-path* (~A)"
                   module-name *load-path*)))))))
