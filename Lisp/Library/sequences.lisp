;;;; Lisp/Library/sequences.lisp — generic CL sequence operations.
;;;;
;;;; The CL spec defines position / find / count / remove / search
;;;; / mismatch / etc. as polymorphic over lists, vectors, and
;;;; strings. core.lisp shipped list-only versions of the common
;;;; ones. This file:
;;;;
;;;;   * adds the generic accessor (elt seq i)
;;;;   * adds copy-seq, fill, replace, concatenate
;;;;   * adds map (multi-sequence), reduce, search, mismatch
;;;;   * adds substitute / delete
;;;;   * adds vector support to subseq (was list + string only)
;;;;   * REDEFINES position / find / count / remove / find-if /
;;;;     position-if / remove-if / remove-if-not so they dispatch
;;;;     by sequence type
;;;;
;;;; Strategy: the redefined functions check (listp seq) first
;;;; and call the original list-only path (preserving performance
;;;; for the common case), else use the generic elt/length walker.
;;;; The generic walker is a few lines slower per call but
;;;; correct for every type.

;; ── Generic accessors ─────────────────────────────────────────────────────

(defun elt (seq i)
  "Return the I-th element of SEQ. Works on lists, vectors, and
   strings. Bounds-error is the underlying primitive's choice
   (currently: segfault for vectors out of range — bounds-check
   lands when arrays grow up)."
  (cond
    ((listp seq)   (nth i seq))
    ((stringp seq) (char seq i))
    ((vectorp seq) (svref seq i))
    (t (error "elt: not a sequence: ~A" seq))))

(defun %set-elt (seq i val)
  "Mutate seq[i]. Mirrors (setf (elt ...)). Returns val."
  (cond
    ((listp seq)
     (setf (car (nthcdr i seq)) val)
     val)
    ((stringp seq)
     (setf (char seq i) val)
     val)
    ((vectorp seq)
     (setf (svref seq i) val)
     val)
    (t (error "(setf elt): not a sequence: ~A" seq))))

;; CL spec: (setf (elt seq i) v) — handled via the %setf-elt
;; generic-setf-fallback name.
(defun %setf-elt (val seq i)
  (%set-elt seq i val))

;; ── do-seq: iteration helper ─────────────────────────────────────────────
;;
;; Calls FN with each element of SEQ in order, regardless of
;; sequence type. Used internally by everything below.

(defun %seq-foreach (fn seq)
  (cond
    ((listp seq)
     (let ((p seq))
       (loop
         (cond
           ((null p) (return nil))
           (t (funcall fn (car p))
              (setq p (cdr p)))))))
    (t
     (let ((i 0) (n (length seq)))
       (loop
         (cond
           ((>= i n) (return nil))
           (t (funcall fn (elt seq i))
              (setq i (+ i 1)))))))))

;; ── copy-seq / subseq / concatenate ──────────────────────────────────────

(defun copy-seq (seq)
  "Shallow copy of SEQ, preserving type."
  (cond
    ((listp seq) (copy-list seq))
    ((stringp seq) (substring seq 0 (length seq)))
    ((vectorp seq)
     (let* ((n (length seq))
            (v (make-array n)))
       (let ((i 0))
         (loop
           (cond
             ((>= i n) (return v))
             (t (setf (svref v i) (svref seq i))
                (setq i (+ i 1))))))))
    (t (error "copy-seq: not a sequence: ~A" seq))))

;; Replace the list+string subseq from core.lisp with a vector-
;; aware version. The defparameter capture-then-redefine pattern
;; isn't needed since the old version had a (t error) fall-through
;; for vectors; we're just adding that branch.
(defun subseq (seq start &optional end)
  "Substring/sublist/subvector from START (inclusive) to END
   (exclusive, defaults to (length seq))."
  (let ((real-end (cond ((null end) (length seq)) (t end))))
    (cond
      ((stringp seq) (substring seq start real-end))
      ((listp seq)
       (let ((lst (nthcdr start seq))
             (n (- real-end start)))
         (cond
           ((<= n 0) nil)
           (t (%list-take lst n)))))
      ((vectorp seq)
       (let* ((n (- real-end start))
              (v (make-array n)))
         (let ((i 0))
           (loop
             (cond
               ((>= i n) (return v))
               (t (setf (svref v i) (svref seq (+ start i)))
                  (setq i (+ i 1))))))))
      (t (error "subseq: not a sequence: ~A" seq)))))

(defun %list-take (lst n)
  (cond
    ((or (null lst) (<= n 0)) nil)
    (t (cons (car lst) (%list-take (cdr lst) (- n 1))))))

(defun concatenate (result-type &rest seqs)
  "Concatenate the given sequences into a fresh sequence of
   RESULT-TYPE (one of 'list, 'string, 'vector)."
  (cond
    ((eq result-type 'list)
     (let ((out nil))
       (dolist (s seqs)
         (%seq-foreach (lambda (x) (setq out (cons x out))) s))
       (reverse out)))
    ((eq result-type 'string)
     ;; Build via format ~A on each char.
     (let ((out ""))
       (dolist (s seqs)
         (%seq-foreach (lambda (c) (setq out (string-append-char out c))) s))
       out))
    ((eq result-type 'vector)
     (let ((total 0))
       (dolist (s seqs) (setq total (+ total (length s))))
       (let ((v (make-array total))
             (i 0))
         (dolist (s seqs)
           (%seq-foreach (lambda (x)
                           (setf (svref v i) x)
                           (setq i (+ i 1)))
                         s))
         v)))
    (t (error "concatenate: unsupported result-type ~A" result-type))))

;; ── Iteration / aggregation ──────────────────────────────────────────────

(defun map (result-type fn &rest seqs)
  "Apply FN to each parallel tuple of elements across SEQS.
   RESULT-TYPE selects what to build:
     'list     → fresh list of results
     'string   → fresh string from char results
     'vector   → fresh vector
     NIL       → discard results, return NIL (side-effect form)

   Stops at the shortest input."
  (let* ((lengths (mapcar (lambda (s) (length s)) seqs))
         (n (cond ((null lengths) 0)
                  (t (%list-min lengths))))
         (out nil))
    (let ((i 0))
      (loop
        (cond
          ((>= i n) (return nil))
          (t
           (let ((args (mapcar (lambda (s) (elt s i)) seqs)))
             (let ((r (apply fn args)))
               (cond
                 ((null result-type) nil)
                 (t (setq out (cons r out))))))
           (setq i (+ i 1))))))
    (cond
      ((null result-type) nil)
      ((eq result-type 'list)   (reverse out))
      ((eq result-type 'vector)
       (let* ((rev (reverse out))
              (v (make-array (length rev)))
              (i 0))
         (dolist (x rev)
           (setf (svref v i) x)
           (setq i (+ i 1)))
         v))
      ((eq result-type 'string)
       (let ((s "")
             (rev (reverse out)))
         (dolist (c rev)
           (setq s (string-append-char s c)))
         s))
      (t (error "map: unsupported result-type ~A" result-type)))))

(defun %list-min (lst)
  (cond
    ((null (cdr lst)) (car lst))
    (t (let ((rest (%list-min (cdr lst))))
         (cond ((< (car lst) rest) (car lst)) (t rest))))))

;; Sentinel for "no :initial-value supplied". Our compiler doesn't
;; yet support the &key (var default supplied-p) shape, so we use
;; a unique cons that the user can't pass.
(defparameter %reduce-no-init (cons 'no 'init))

(defun reduce (fn seq &key (initial-value %reduce-no-init) (from-end nil)
                              (key #'identity) (start 0) end)
  "Combine elements of SEQ via FN. Without :initial-value, the
   first element is the seed; with it, INITIAL-VALUE is the
   seed and every element is folded in."
  (declare (ignore from-end))   ; right-fold deferred
  (let* ((real-end (cond ((null end) (length seq)) (t end)))
         (i start)
         (acc (cond
                ((eq initial-value %reduce-no-init)
                 (cond
                   ((>= start real-end)
                    (error "reduce: empty sequence and no :initial-value"))
                   (t (let ((v (funcall key (elt seq start))))
                        (setq i (+ start 1))
                        v))))
                (t initial-value))))
    (loop
      (cond
        ((>= i real-end) (return acc))
        (t (setq acc (funcall fn acc (funcall key (elt seq i))))
           (setq i (+ i 1)))))))

;; ── Search / position / find / count (generic) ──────────────────────────

;; Stash the original list-only versions captured at LOAD time, so
;; the redefinitions below can fast-path lists without re-walking
;; everything.

(defparameter %list-find     (symbol-function 'find))
(defparameter %list-position (symbol-function 'position))
(defparameter %list-remove   (symbol-function 'remove))
;; COUNT didn't exist before this file — no list-only fast path
;; to capture. The generic walker below handles all sequence types.

(defun find (item seq &key (test #'eql) (key #'identity))
  (cond
    ((listp seq) (funcall %list-find item seq :test test :key key))
    (t
     (let ((i 0) (n (length seq)) (result nil) (found nil))
       (loop
         (cond
           (found (return result))
           ((>= i n) (return nil))
           (t (let ((el (elt seq i)))
                (cond
                  ((funcall test item (funcall key el))
                   (setq result el)
                   (setq found t))
                  (t (setq i (+ i 1))))))))))))

(defun position (item seq &key (test #'eql) (key #'identity))
  (cond
    ((listp seq) (funcall %list-position item seq :test test :key key))
    (t
     (let ((i 0) (n (length seq)) (result nil) (found nil))
       (loop
         (cond
           (found (return result))
           ((>= i n) (return nil))
           (t (cond
                ((funcall test item (funcall key (elt seq i)))
                 (setq result i)
                 (setq found t))
                (t (setq i (+ i 1)))))))))))

(defun count (item seq &key (test #'eql) (key #'identity))
  "Count elements equal to ITEM under TEST + KEY."
  (let ((c 0))
    (%seq-foreach
     (lambda (el)
       (when (funcall test item (funcall key el))
         (setq c (+ c 1))))
     seq)
    c))

(defun remove (item seq &key (test #'eql) (key #'identity))
  "Return a fresh sequence with elements equal to ITEM removed.
   Result preserves SEQ's type."
  (cond
    ((listp seq) (funcall %list-remove item seq :test test :key key))
    ((stringp seq)
     (let ((out ""))
       (%seq-foreach
        (lambda (c)
          (unless (funcall test item (funcall key c))
            (setq out (string-append-char out c))))
        seq)
       out))
    ((vectorp seq)
     ;; Two-pass: count kept, allocate, fill.
     (let ((kept 0))
       (%seq-foreach
        (lambda (el)
          (unless (funcall test item (funcall key el))
            (setq kept (+ kept 1))))
        seq)
       (let ((v (make-array kept))
             (i 0))
         (%seq-foreach
          (lambda (el)
            (unless (funcall test item (funcall key el))
              (setf (svref v i) el)
              (setq i (+ i 1))))
          seq)
         v)))
    (t (error "remove: not a sequence: ~A" seq))))

;; delete is the destructive variant of remove. Our remove is
;; already non-shared for vectors and strings; for lists CL allows
;; modifying cons cells but our list-remove builds fresh. Same
;; observable behaviour either way; delete is just an alias.
(defun delete (item seq &key (test #'eql) (key #'identity))
  (remove item seq :test test :key key))

(defun substitute (new-item old-item seq
                            &key (test #'eql) (key #'identity))
  "Fresh sequence with every element equal to OLD-ITEM replaced
   by NEW-ITEM. Type-preserving."
  (cond
    ((listp seq)
     (let ((out nil))
       (%seq-foreach
        (lambda (el)
          (setq out
                (cons (cond
                        ((funcall test old-item (funcall key el)) new-item)
                        (t el))
                      out)))
        seq)
       (reverse out)))
    ((stringp seq)
     (let ((out ""))
       (%seq-foreach
        (lambda (c)
          (setq out (string-append-char out
                                        (cond
                                          ((funcall test old-item
                                                    (funcall key c))
                                           new-item)
                                          (t c)))))
        seq)
       out))
    ((vectorp seq)
     (let* ((n (length seq))
            (v (make-array n))
            (i 0))
       (%seq-foreach
        (lambda (el)
          (setf (svref v i)
                (cond
                  ((funcall test old-item (funcall key el)) new-item)
                  (t el)))
          (setq i (+ i 1)))
        seq)
       v))
    (t (error "substitute: not a sequence: ~A" seq))))

;; ── Predicate variants (find-if / position-if / remove-if) ──────────────

(defparameter %list-find-if (symbol-function 'find-if))
(defparameter %list-remove-if (symbol-function 'remove-if))
(defparameter %list-remove-if-not (symbol-function 'remove-if-not))

(defun find-if (pred seq &key (key #'identity))
  (find t seq :test (lambda (_ x) (funcall pred x)) :key key))

(defun position-if (pred seq &key (key #'identity))
  (position t seq :test (lambda (_ x) (funcall pred x)) :key key))

(defun count-if (pred seq &key (key #'identity))
  (count t seq :test (lambda (_ x) (funcall pred x)) :key key))

(defun remove-if (pred seq &key (key #'identity))
  (remove t seq :test (lambda (_ x) (funcall pred x)) :key key))

(defun remove-if-not (pred seq &key (key #'identity))
  (remove t seq :test (lambda (_ x) (not (funcall pred x))) :key key))

;; ── mismatch / search ────────────────────────────────────────────────────

(defun mismatch (seq1 seq2 &key (test #'eql) (key #'identity))
  "Return the position of the first element-by-element difference
   between SEQ1 and SEQ2, or NIL if they're equal up to the
   shorter one's length. The mismatch position is in SEQ1's
   indexing scheme."
  (let* ((n1 (length seq1))
         (n2 (length seq2))
         (n  (cond ((< n1 n2) n1) (t n2)))
         (i 0)
         (result nil)
         (done nil))
    (loop
      (cond
        (done (return result))
        ((>= i n)
         ;; Walked through the shorter; if lengths differ, mismatch
         ;; is at index n (one past the shorter end). Else nil.
         (setq result (cond ((= n1 n2) nil) (t n)))
         (setq done t))
        (t (cond
             ((funcall test
                       (funcall key (elt seq1 i))
                       (funcall key (elt seq2 i)))
              (setq i (+ i 1)))
             (t (setq result i)
                (setq done t))))))))

(defun search (sub-seq seq &key (test #'eql) (key #'identity))
  "Return the index in SEQ of the first occurrence of SUB-SEQ,
   or NIL if absent. O(n*m) — naive scan, sufficient for the
   typical short-needle case."
  (let* ((nsub (length sub-seq))
         (nseq (length seq))
         (limit (- nseq nsub))
         (start 0)
         (result nil)
         (done nil))
    (cond
      ((> nsub nseq) nil)
      (t
       (loop
         (cond
           (done (return result))
           ((> start limit)
            (setq done t))                              ; not found
           (t (let ((ok t) (j 0))
                (loop
                  (cond
                    ((>= j nsub) (return nil))
                    ((not (funcall test
                                   (funcall key (elt sub-seq j))
                                   (funcall key (elt seq (+ start j)))))
                     (setq ok nil)
                     (return nil))
                    (t (setq j (+ j 1)))))
                (cond
                  (ok (setq result start) (setq done t))
                  (t (setq start (+ start 1))))))))))))

;; ── fill / replace ──────────────────────────────────────────────────────

(defun fill (seq item &key (start 0) end)
  "Destructively replace each element of SEQ in [START, END) with
   ITEM. Returns SEQ. For lists this mutates car cells; for
   vectors/strings it writes via setf elt."
  (let ((real-end (cond ((null end) (length seq)) (t end)))
        (i start))
    (loop
      (cond
        ((>= i real-end) (return seq))
        (t (%set-elt seq i item)
           (setq i (+ i 1)))))))

(defun replace (target source
                       &key (start1 0) end1 (start2 0) end2)
  "Destructively copy elements of SOURCE into TARGET. Returns
   TARGET. Bounds are clamped at the shorter of the two
   sub-ranges."
  (let* ((re1 (cond ((null end1) (length target)) (t end1)))
         (re2 (cond ((null end2) (length source)) (t end2)))
         (n   (cond ((< (- re1 start1) (- re2 start2))
                     (- re1 start1))
                    (t (- re2 start2))))
         (i 0))
    (loop
      (cond
        ((>= i n) (return target))
        (t (%set-elt target (+ start1 i) (elt source (+ start2 i)))
           (setq i (+ i 1)))))))

(provide 'sequences)
nil
