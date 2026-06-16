;;;; zebra-time.lisp — timed Zebra-puzzle benchmark.
;;;;
;;;; Loads the Norvig-Prolog engine + zebra database from
;;;; demos/prolog.lisp (which also runs the demo once, untimed), then
;;;; times ONLY the (solve …) of the zebra puzzle with the `time` macro
;;;; from Library/time.lisp. This is the canonical "heavy symbolic /
;;;; backtracking" workload we track against SBCL.
;;;;
;;;; Run:  ncl.exe -l bench/zebra-time.lisp

(require 'time)
(load "demos/prolog.lisp")

(format t "~%=== Timed zebra solve (heavy backtracking) ===~%")
(time (solve '(zebra ?houses ?water ?zebra)))
(format t "done.~%")
