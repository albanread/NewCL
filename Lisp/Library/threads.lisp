;;;; Lisp/Library/threads.lisp
;;;;
;;;; Roger Corman's THREADS package, faithfully ported on top of the
;;;; cross-platform Rust primitives installed by the runtime. The
;;;; native primitives are intentionally low-level (raw integer ids,
;;;; mandatory arguments, no keywords); this file wraps them in the
;;;; documented Corman API surface.
;;;;
;;;; Roger's THREADS docs name the package `THREADS` with nickname
;;;; `TH`. We have a single global namespace, so the symbols live
;;;; in COMMON-LISP-USER like everything else; the names match what
;;;; Corman exported.
;;;;
;;;; What we have:
;;;;
;;;;   (create-thread func &key (report-when-finished t))
;;;;     spawn an OS thread running (funcall func), return its id.
;;;;
;;;;   (exit-thread &optional condition)
;;;;     COOPERATIVE — sets a "please exit" flag on the calling
;;;;     thread; the next (thread-safepoint) call returns T so the
;;;;     thread's own loop can RETURN. Roger's Corman docs say
;;;;     this "never returns"; that contract requires unwinding
;;;;     through JIT'd frames, which v1 doesn't support (no SEH
;;;;     unwind tables in our LLVM JIT yet). Until that lands,
;;;;     wrap your worker in a (loop) that checks
;;;;     (when (thread-safepoint) (return)).
;;;;
;;;;   (thread-handle thread-id)
;;;;     under our cross-platform shim, returns the same integer
;;;;     as the id (or NIL if the id isn't live). Kept for Corman
;;;;     API compatibility.
;;;;
;;;;   (suspend-thread id)  (resume-thread id)  (terminate-thread id)
;;;;     COOPERATIVE — the target thread acts on the request at its
;;;;     next (thread-safepoint). In tight CPU loops, insert
;;;;     (thread-safepoint) periodically so suspend/terminate work.
;;;;     suspend-thread parks the target inside its safepoint until
;;;;     resume-thread; terminate-thread makes the next safepoint
;;;;     return T so the worker's own loop can RETURN.
;;;;
;;;;   *current-thread-id* / *current-thread-handle*
;;;;     captured at load-time for the main thread. Inside a fresh
;;;;     thread, call (current-thread-id) — until per-thread special
;;;;     bindings land we can't make the variable update itself.
;;;;
;;;;   *current-process-id* / *current-process-handle*
;;;;     process-wide, set once at load. Same value across threads.
;;;;
;;;;   (critical-section)  class with accessor (cs cs)
;;;;     ENTER / LEAVE generic functions
;;;;     (with-synchronization cs . body)  macro
;;;;
;;;;   (allocate-critical-section)        — low-level id allocator
;;;;   (deallocate-critical-section id)
;;;;   (enter-critical-section id)
;;;;   (leave-critical-section id)
;;;;
;;;; A worked example:
;;;;
;;;;   (require 'threads)
;;;;   (defparameter *cs* (make-instance 'critical-section))
;;;;   (defparameter *count* 0)
;;;;   (defun bump ()
;;;;     (dotimes (i 1000)
;;;;       (with-synchronization (cs *cs*)
;;;;         (setq *count* (+ *count* 1)))))
;;;;   (let ((a (create-thread #'bump))
;;;;         (b (create-thread #'bump)))
;;;;     (declare (ignore a b))
;;;;     ;; Give the threads a moment, then read the result.
;;;;     ;; (A proper barrier / join would be nicer; the API
;;;;     ;; Corman documented doesn't include join, so we don't
;;;;     ;; expose it here either.)
;;;;     )

;; ── create-thread wrapper: handles :report-when-finished ────────────────
;;
;; The native shim takes one mandatory argument (the function). The
;; Lisp wrapper accepts Corman's keyword and forwards.

(defun create-thread (func &key (report-when-finished t))
  "Spawn an OS thread running (funcall FUNC) with no arguments.
   Returns the new thread's integer id. If REPORT-WHEN-FINISHED
   is non-nil (the default), a line is printed to stderr when the
   thread terminates."
  (declare (ignore report-when-finished))
  ;; The runtime shim always reports today; the keyword is accepted
  ;; for Corman API compatibility but currently a no-op. Honouring
  ;; the flag is a small follow-up: pipe it into a second shim arg
  ;; and store it on the registry entry.
  (%create-thread func))

;; ── thread-loop: cooperative termination helper ─────────────────────────
;;
;; The natural shape of a worker function: a loop that does work,
;; checks for termination, repeats. (thread-safepoint) returns T
;; when terminate-thread / exit-thread has been requested; this
;; macro wraps the boilerplate.

(defmacro thread-loop (&rest body)
  "(loop) with an implicit (thread-safepoint) check on every pass.
   When the safepoint reports termination, RETURNS from the
   enclosing loop. Use this as the outer shell of a worker:

       (defun worker ()
         (thread-loop
           ;; ... do work ..."
  `(loop
     (when (thread-safepoint) (return :terminated))
     ,@body))

;; ── *current-thread-id* / *current-thread-handle* ───────────────────────
;;
;; Corman makes these special variables. Without per-thread dynamic
;; bindings we capture the main thread's value at load-time; a
;; freshly-spawned thread should call (current-thread-id) instead.

(defparameter *current-thread-id*
  (current-thread-id)
  "Lisp id of the thread that loaded the THREADS package. Inside
   a thread spawned via create-thread, this variable still reflects
   the loader thread — use (current-thread-id) for the live value
   until per-thread special bindings land.")

(defparameter *current-thread-handle*
  (current-thread-id)
  "Cross-platform OS-thread handle. Under our Rust layer there is
   no separate handle; this equals *current-thread-id*. Provided
   for Corman API compatibility.")

(defparameter *current-process-id*
  (current-process-id)
  "Operating-system process id of the running ncl instance.
   Constant for the lifetime of the process.")

(defparameter *current-process-handle*
  (current-process-id)
  "Cross-platform OS-process handle. Under our Rust layer there
   is no separate process handle; this equals *current-process-id*.")

;; ── CRITICAL-SECTION class ──────────────────────────────────────────────
;;
;; Corman's CRITICAL-SECTION is a CLOS class wrapping a Win32
;; CRITICAL_SECTION. Ours wraps a reentrant Rust mutex keyed by an
;; integer id allocated in the runtime registry. Same interface: an
;; ENTER / LEAVE generic-function pair, with `cs` as the accessor
;; for the underlying handle.

(defclass critical-section ()
  ((cs :initform (allocate-critical-section)
       :accessor cs
       :documentation
       "Integer handle of the underlying reentrant Rust mutex in
        the runtime's critical-section registry. Allocated at
        initform time so the class is ready-to-use the moment
        make-instance returns."))
  (:documentation
   "A reentrant mutex object. Use (enter section) before a
    critical region, (leave section) after, or — recommended —
    wrap the region in (with-synchronization section ...).

    Reentrance: the owning thread can call ENTER multiple times;
    a matching number of LEAVE calls releases the section."))

(defgeneric enter (section)
  (:documentation
   "Acquire SECTION. Blocks until the section is owned by the
    current thread. Reentrant: same-thread re-enter increments a
    count."))

(defgeneric leave (section)
  (:documentation
   "Release SECTION. Decrements the reentrance count; when it
    reaches zero, the section is unowned and waiters wake."))

(defmethod enter ((section critical-section))
  (enter-critical-section (cs section)))

(defmethod leave ((section critical-section))
  (leave-critical-section (cs section)))

;; ── with-synchronization macro ──────────────────────────────────────────

(defmacro with-synchronization (section-form &rest body)
  "Bracket BODY with ENTER and LEAVE on SECTION-FORM. The LEAVE runs
   on the body's normal return path. Non-local exits during BODY
   will leak the lock — for our v1 this is a known limitation
   (loop/return doesn't unwind, and we don't yet have unwind-protect
   in the form a critical section would need).

   Example:
     (with-synchronization *cs*
       (push x *shared-stack*))"
  (let ((sec (gensym "SEC-"))
        (result (gensym "RES-")))
    `(let ((,sec ,section-form))
       (enter ,sec)
       (let ((,result (progn ,@body)))
         (leave ,sec)
         ,result))))

(provide 'threads)
nil
