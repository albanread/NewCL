;;;; mandelbrot.lisp — direct pixel access via the NCL canvas API.
;;;;
;;;; Run:  ./target/release/ncl.exe --windows --load demos/mandelbrot.lisp
;;;;
;;;; The Mandelbrot set is the canonical direct-pixel-access demo: every
;;;; pixel is an independent escape-time computation, written STRAIGHT
;;;; into a host-owned framebuffer with no per-pixel boundary crossing.
;;;;
;;;;   (canvas-open child-id w h)  -> base address of a BGRA32 buffer
;;;;   (buffer-set-u32 base off v) -> poke one pixel: off = (y*w + x)*4
;;;;   (canvas-present child-id)    -> draw the frame (one GPU blit)

(defun argb (r g b)
  "Opaque BGRA32 pixel 0xAARRGGBB (alpha ignored by the canvas)."
  (logior #xFF000000
          (ash (logand r 255) 16)
          (ash (logand g 255) 8)
          (logand b 255)))

(defun mandel-iter (cx cy max)
  "Escape-time iteration count for the point c = cx + i*cy.
   Returns MAX for points in the set (never escape)."
  (let ((zx 0.0) (zy 0.0) (n 0))
    (loop
      (let ((zx2 (* zx zx))
            (zy2 (* zy zy)))
        (when (or (>= n max) (> (+ zx2 zy2) 4.0))
          (return n))
        (let ((new-zx (+ (- zx2 zy2) cx))
              (new-zy (+ (* 2.0 (* zx zy)) cy)))
          (setq zx new-zx)
          (setq zy new-zy)))
      (setq n (+ n 1)))))

(defun mandel-color (n max)
  "Map an escape count to a colour: black inside the set, a cyclic
   banded palette outside."
  (if (>= n max)
      #xFF000000
      (argb (mod (* n 5) 256)
            (mod (* n 7) 256)
            (mod (+ 64 (* n 11)) 256))))

(defun render-mandelbrot (base w h max)
  "Compute the whole set and poke each pixel directly into the buffer."
  (let* ((x-min -2.5) (x-max 1.0)
         (y-min -1.25) (y-max 1.25)
         (x-step (/ (- x-max x-min) w))
         (y-step (/ (- y-max y-min) h)))
    (dotimes (py h)
      (let ((cy (+ y-min (* py y-step)))
            (row (* py w)))
        (dotimes (px w)
          (let* ((cx (+ x-min (* px x-step)))
                 (n  (mandel-iter cx cy max)))
            (buffer-set-u32 base (* (+ row px) 4) (mandel-color n max))))))))

(defun run-mandelbrot (&optional (w 480) (h 360) (max 100))
  (igui-start)
  (let* ((id   (open-child-sized "Mandelbrot — direct pixel access" w h))
         (base (canvas-open id w h)))
    (cond
      ((null base) (format t "canvas-open failed~%"))
      (t
       (format t "rendering ~Ax~A Mandelbrot, max-iter ~A ...~%" w h max)
       (render-mandelbrot base w h max)
       (canvas-present id)
       (format t "done — close the window to exit.~%"))))
  (igui-wait))

(run-mandelbrot)
