;;;; hello-igui.lisp — minimal "pixels on screen" demo.
;;;;
;;;; Opens an MDI child, paints a panel with some shapes, runs an
;;;; event loop that re-paints on every :resize and exits when
;;;; the frame is closed.
;;;;
;;;; Run with:  ncl --eval "(load \"Lisp/demos/hello-igui.lisp\")"
;;;; (the loader is still TODO — for now, paste this into a session
;;;; that's already loaded the stdlib, or invoke the body via --eval.)

(defun paint-hello (child-id width height)
  "Paint a simple scene into CHILD-ID's surface, sized to WIDTH × HEIGHT."
  (with-batch child-id
    (clear +slate+)
    ;; Header bar.
    (fill-rect 0 0 width 40 +panel+)
    ;; Centred-ish swatches.
    (fill-rect 60 80 100 60 +red+)
    (fill-rect 200 80 100 60 +green+)
    (fill-rect 340 80 100 60 +blue+)
    ;; A cross of lines through the swatch row.
    (draw-line 0 110 width 110 1 +white+)
    (draw-line 250 60 250 160 1 +white+)
    ;; An outlined rectangle around the whole drawing area.
    (stroke-rect 4 4 (- width 8) (- height 8) 2 +yellow+)))

(defun run-hello-igui ()
  "Open a child, paint it, drive the event loop until the frame closes."
  (igui-start)
  (let ((id (open-child "hello-igui")))
    ;; Initial paint at a guessed size — the first :RESIZE event
    ;; will repaint at the actual size.
    (paint-hello id 480 320)
    (loop
      (let ((ev (next-event -1)))
        (cond
          ((null ev) nil)
          ((eq (getf ev :kind) :frame-close) (return :done))
          ((and (eq (getf ev :kind) :resize)
                (= (getf ev :child-id) id))
           (paint-hello id (getf ev :width) (getf ev :height)))
          ((and (eq (getf ev :kind) :close)
                (= (getf ev :child-id) id))
           (close-child id)
           (return :done))
          (t nil))))))

;; Calling code:
;;   (run-hello-igui)
