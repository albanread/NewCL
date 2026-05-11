;;;; Lisp/demos/paint-and-log.lisp — two windows, one event loop.
;;;;
;;;; Port of NewBCPL's examples/paint-and-log.bcl. Demonstrates a
;;;; multi-window iGui app:
;;;;
;;;;   * a graphics canvas where left-clicks drop coloured dots
;;;;   * a text-pane log that prints the click number + coords
;;;;
;;;; Both windows register with the persistent event filter so the
;;;; main loop only sees their events (plus globals like
;;;; :frame-close). Events for other children (a separate REPL pane,
;;;; the iGui log overlay, etc.) park in the stash and never
;;;; pollute the dispatch.
;;;;
;;;; Run:
;;;;
;;;;   ncl --load Lisp/demos/paint-and-log.lisp --eval "(run-paint-and-log)"
;;;;
;;;; Close either window to exit.

(defparameter +canvas-bg+ (rgb 20 25 41))      ; very dark blue
(defparameter +log-bg+    (rgb 32 40 64))
(defparameter +log-fg+    (rgb 102 224 224))   ; soft cyan
(defparameter +log-banner-bg+ (rgb 32 40 64))

;; Six dot colours cycled through on each click.
(defparameter +dot-colors+
  (list (rgb 242 90 60)     ; warm orange-red
        (rgb 250 180 60)    ; amber
        (rgb 240 235 80)    ; yellow
        (rgb 120 220 100)   ; green
        (rgb 80 160 240)    ; sky blue
        (rgb 200 120 240))) ; lavender

(defun %dot-color (n)
  (nth (rem n 6) +dot-colors+))

(defun paint-and-log-banner (log)
  (text-set-pen log +log-fg+ +log-banner-bg+)
  (text-write log "+--------------------------+~%")
  (text-write log "|   paint-and-log demo     |~%")
  (text-write log "+--------------------------+~%")
  (text-reset-pen log)
  (text-write log "Click the canvas to drop dots.~%")
  (text-write log "Close either window to exit.~%~%"))

(defun run-paint-and-log ()
  "Open the canvas and log windows, run the unified event loop."
  (igui-start)
  (let ((canvas (open-child "Canvas"))
        (log    (open-text-window "Log")))
    (cond
      ((or (null canvas) (null log))
       (format t "** failed to open windows (canvas=~A log=~A)~%" canvas log)
       :failed)
      (t
       ;; Register both windows with the persistent filter. After
       ;; this, plain (next-event) only sees events for these two
       ;; (plus globals).
       (filter-on-window canvas)
       (filter-on-window log)

       ;; Initial canvas paint.
       (with-batch canvas
         (clear +canvas-bg+))

       ;; Banner in the log.
       (paint-and-log-banner log)

       (let ((count 0))
         ;; We use plain event-loop (not event-loop-for) because
         ;; we're watching TWO child windows; dispatch by :child-id
         ;; inside the :mouse clause picks the canvas.
         (event-loop
           (:frame-close
            (clear-event-filter)
            (return :done))
           (:close
            ;; Either window closed → exit.
            (clear-event-filter)
            (return :done))
           (:mouse
            (when (and (eq (getf ev :op) :LEFT-DOWN)
                       (= (getf ev :child-id) canvas))
              (setq count (+ count 1))
              ;; Draw a new dot on the canvas.
              (with-batch canvas
                (fill-circle (getf ev :x) (getf ev :y) 7
                             (%dot-color count)))
              ;; Log the click into the text pane.
              (text-write log
                          (format nil "  click ~D at (~D, ~D)~%"
                                  count
                                  (getf ev :x)
                                  (getf ev :y))))))
         (format t "[paint-and-log] exited after ~D clicks~%" count))))))
