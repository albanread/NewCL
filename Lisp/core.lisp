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
