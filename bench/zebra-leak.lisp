;;;; zebra-leak.lisp — find what zebra allocates and never reclaims.
(require 'time)
(load "demos/prolog.lisp")
(defun stat (k) (getf (gc-stats) k))

(defun snap () (list :static (stat :static-used)
                     :old (stat :old-used)
                     :young (stat :young-used)
                     :varctr *var-counter*))
(defun delta (a b) (- b a))

(format t "~%=== zebra: where does the memory go? ===~%")
(let ((s0 (snap)))
  (solve '(zebra ?houses ?water ?zebra))
  (let ((s1 (snap)))
    (format t "  new-variable calls (var-counter delta): ~A~%"
            (delta (getf s0 :varctr) (getf s1 :varctr)))
    (format t "  STATIC-USED delta : ~A bytes~%" (delta (getf s0 :static) (getf s1 :static)))
    (format t "  OLD-USED   delta : ~A bytes~%" (delta (getf s0 :old) (getf s1 :old)))
    (format t "  bytes/new-variable (static): ~A~%"
            (let ((n (delta (getf s0 :varctr) (getf s1 :varctr))))
              (if (zerop n) 0 (truncate (delta (getf s0 :static) (getf s1 :static)) n))))
    ;; Solve a SECOND time: if static keeps growing, interned vars leak monotonically.
    (let ((s2a (snap)))
      (solve '(zebra ?houses ?water ?zebra))
      (let ((s2b (snap)))
        (format t "  --- 2nd solve ---~%")
        (format t "  new-variable calls: ~A~%" (delta (getf s2a :varctr) (getf s2b :varctr)))
        (format t "  STATIC-USED delta : ~A bytes (leaks again => never reclaimed)~%"
                (delta (getf s2a :static) (getf s2b :static)))))))
(format t "done.~%")
