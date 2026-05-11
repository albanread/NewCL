;;;; Lisp/Library/scratch.lisp — playground for hot-reload smoke testing.
;;;;
;;;; Used by the hot-reload demo:
;;;;
;;;;   ncl> (require 'scratch)
;;;;   ncl> (start-hot-reload)
;;;;   ncl> (scratch-message)
;;;;   "v1"
;;;;   ;;; edit this file, save it ...
;;;;   ncl> (scratch-message)
;;;;   "v2"   ; reloaded automatically between prompts

(defun scratch-message () "v1")
(provide 'scratch)
