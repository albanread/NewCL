;;;; canvas-demo.lisp — fast pixel-direct rendering via the NCL canvas API.
;;;;
;;;; Run:  ./target/release/ncl.exe --windows --load demos/canvas-demo.lisp
;;;;
;;;; The canvas is a host-owned BGRA32 framebuffer bound to a render-host
;;;; child window. CANVAS-OPEN returns the buffer's BASE ADDRESS; the Lisp
;;;; program then writes pixels DIRECTLY into it with BUFFER-SET-U32 — no
;;;; per-pixel boundary crossing — and CANVAS-PRESENT snapshots the buffer
;;;; and draws one frame (a single GPU blit). An animated plasma shows the
;;;; throughput: every pixel is poked every frame.
;;;;
;;;;   (canvas-open  child-id w h) -> base-address (fixnum) | nil
;;;;   (buffer-set-u32 base byte-offset argb)   ; pixel (x,y) at (y*w + x)*4
;;;;   (canvas-present child-id)   -> base-address (for the next frame)

(defun argb (r g b)
  "An opaque BGRA32 pixel word, 0xAARRGGBB. Alpha is ignored by the
   canvas (pixels are drawn opaque), so we always set it to 0xFF."
  (logior #xFF000000
          (ash (logand r 255) 16)
          (ash (logand g 255) 8)
          (logand b 255)))

(defun draw-plasma (base w h tick)
  "Poke a shifting plasma directly into the canvas buffer for this TICK.
   Pure integer arithmetic — one BUFFER-SET-U32 per pixel."
  (dotimes (y h)
    (let ((row (* y w)))
      (dotimes (x w)
        (let ((r (logand (+ x tick) 255))
              (g (logand (+ y tick) 255))
              (b (logand (+ x y (* 2 tick)) 255)))
          (buffer-set-u32 base (* (+ row x) 4) (argb r g b)))))))

(defun run-canvas-demo (&optional (frames 240))
  "Open a 256x256 canvas child and animate a plasma into it."
  (igui-start)
  (let* ((w 256)
         (h 256)
         (id (open-child-sized "Canvas — fast pixel poke" w h))
         (base (canvas-open id w h)))
    (cond
      ((null base) (format t "canvas-open failed~%"))
      (t
       (dotimes (tick frames)
         (draw-plasma base w h tick)
         (canvas-present id)
         (sleep 0.016))          ; ~60 fps
       (format t "canvas demo done (~A frames)~%" frames)))
    ;; Bonus: tidy any minimized child icons (the MDI window verbs).
    (mdi-arrange-icons))
  ;; Keep the frame up; close the window to exit.
  (igui-wait))

(run-canvas-demo)
