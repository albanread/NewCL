;;;; zebra-sbcl.lisp — same Zebra solve under SBCL, for A/B vs NCL.
;;;; Run:  sbcl --script bench/zebra-sbcl.lisp
(load "demos/prolog.lisp")
(format t "~%=== SBCL timed zebra solve (3 runs) ===~%")
(dotimes (i 3) (time (solve '(zebra ?houses ?water ?zebra))))
