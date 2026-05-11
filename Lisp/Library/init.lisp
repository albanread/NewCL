;;;; Lisp/Library/init.lisp
;;;;
;;;; Loaded automatically at startup by `ncl` (the driver looks for
;;;; this file next to the executable, or under <repo>/Lisp/Library/
;;;; for dev builds). This file is the place to wire up things that
;;;; should be available in every session — analogous to Corman's
;;;; init.lisp.
;;;;
;;;; The core stdlib (Lisp/core.lisp) and Closette (Lisp/clos.lisp)
;;;; are already baked into the binary and loaded BEFORE this file
;;;; runs. So everything we need — defun, defmacro, defclass, format,
;;;; require, load, *load-path*, *modules* — is already in place.
;;;;
;;;; Out of the box this file does very little. Drop new .lisp files
;;;; next to it and add `(require :module)` lines below to have them
;;;; loaded automatically. Modules are loaded exactly once per
;;;; session (REQUIRE checks *modules*).

;;; Example: load a personal utilities module if present.
;;; Uncomment and rename to taste.
;; (require 'my-utils)

;;; A user can verify the loader picked this file up by checking
;;; *modules* and *load-path* at the REPL.
(provide 'init)

nil
