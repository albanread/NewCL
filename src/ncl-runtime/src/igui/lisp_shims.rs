//! Lisp-callable shims for iGui — the integration layer between
//! the renderer/window subsystem and the rest of the runtime's
//! `extern "C-unwind"` shim conventions.
//!
//! Each function here has the standard JIT calling convention
//! `(mutator, env, args, n_args) -> u64` so it can be installed in
//! a Symbol's function cell via `install_native`. It does any
//! Word-to-native translation, calls the corresponding iGui
//! function (which lives on the GUI thread or marshals onto it),
//! and converts the result back.
//!
//! First slice (this commit): start/quit, the window-management
//! trio (open-child / close-child / set-title), and next-event.
//! Drawing primitives (with-batch / fill-rect / ...) come next.

#![cfg(windows)]

use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{PostMessageW, WM_CLOSE};

use crate::gc_string;
use crate::word::{Tag, Word};
use crate::GcCoordinator;

use super::batch::{self as batch_mod, Point, Rect, Rgba, SurfaceCmd};
use super::channels::{kind, IGuiEvent};
use super::cp_exports::FRAME_HWND;
use super::{channels, log_view, window};

/// JoinHandle for the GUI thread, taken on `(igui-wait)`. Tracking
/// it here means the Lisp thread can block until the message pump
/// exits — useful for one-shot demos via `--eval`, where without a
/// wait the main thread would return immediately and tear the
/// process (and its GUI thread) down with it.
static GUI_THREAD: OnceLock<Mutex<Option<JoinHandle<()>>>> = OnceLock::new();

// -- helpers ----------------------------------------------------------------

fn arg(args: *const u64, i: u64) -> Word {
    Word::from_raw(unsafe { *args.add(i as usize) })
}

fn arg_fixnum(args: *const u64, i: u64) -> Option<i64> {
    arg(args, i).as_fixnum()
}

fn arg_string(args: *const u64, i: u64) -> Option<String> {
    let w = arg(args, i);
    if w.tag() != Tag::String {
        return None;
    }
    Some(gc_string::chars_of(w).collect())
}

/// Build a runtime keyword (`:NAME`) from a name string. Uses the
/// same colon-prefixed convention the compiler uses for keyword
/// literals so user code can `eq` them.
fn kw(coord: &GcCoordinator, name: &str) -> Word {
    let mut buf = String::with_capacity(name.len() + 1);
    buf.push(':');
    buf.push_str(name);
    coord.intern(&buf)
}

// -- shim entry points ------------------------------------------------------

/// `(igui-start)` — spawn the GUI thread, wait for the frame to be
/// up, return T. Idempotent: subsequent calls are no-ops.
/// Returns NIL on startup failure.
///
/// The JoinHandle for the spawned thread is stashed; `(igui-wait)`
/// blocks the calling thread on it.
pub extern "C-unwind" fn igui_start_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    let slot = GUI_THREAD.get_or_init(|| Mutex::new(None));
    {
        let guard = slot.lock().unwrap();
        if guard.is_some() {
            return Word::T.raw();
        }
    }

    // window::run blocks for the lifetime of the message pump, so
    // it has to live on its own OS thread. The Lisp thread keeps
    // calling next-event, open-child, etc. against the still-alive
    // FRAME_HWND.
    let handle = std::thread::spawn(|| {
        match super::run(None::<fn()>) {
            Ok(code) => {
                if code != 0 {
                    eprintln!("[igui] message pump exited with code {code}");
                }
            }
            Err(e) => {
                eprintln!("[igui] startup failed: {e}");
            }
        }
    });
    {
        let mut guard = slot.lock().unwrap();
        *guard = Some(handle);
    }

    // Wait up to ~5s for FRAME_HWND to appear. window::run sets it
    // shortly after the CreateWindowExW for the frame returns.
    let deadline = Instant::now() + Duration::from_secs(5);
    while FRAME_HWND.get().is_none() {
        if Instant::now() >= deadline {
            return Word::NIL.raw();
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    Word::T.raw()
}

/// `(igui-wait)` — block the calling Lisp thread until the GUI
/// thread exits (i.e. the message pump returned after WM_QUIT).
/// Returns T on a clean shutdown, NIL if the GUI thread was never
/// started.
///
/// This is what one-shot demos via `--eval` use: `(igui-start)`
/// kicks the GUI off, then `(igui-wait)` keeps the process alive
/// until the user closes the frame. For a long-running REPL or
/// app, you'd typically call `(next-event ...)` in a loop instead
/// and never block on this.
pub extern "C-unwind" fn igui_wait_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    let Some(slot) = GUI_THREAD.get() else {
        return Word::NIL.raw();
    };
    let handle = {
        let mut guard = slot.lock().unwrap();
        guard.take()
    };
    match handle {
        Some(h) => {
            let _ = h.join();
            Word::T.raw()
        }
        None => Word::NIL.raw(),
    }
}

/// `(igui-quit)` — post WM_CLOSE to the frame. The GUI thread
/// tears down at its own pace; subsequent next-event calls drain
/// any pending events before EvFrameClose arrives. Returns T.
pub extern "C-unwind" fn igui_quit_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    if let Some(&hwnd_raw) = FRAME_HWND.get() {
        let hwnd = HWND(hwnd_raw as *mut _);
        unsafe {
            let _ = PostMessageW(Some(hwnd), WM_CLOSE, WPARAM(0), LPARAM(0));
        }
    }
    Word::T.raw()
}

/// `(open-child title)` — open a new MDI child with the given
/// title; returns its child-id as a fixnum, or NIL on failure.
/// Synchronous: blocks the caller while the GUI thread creates
/// the window via SendMessageW.
pub extern "C-unwind" fn open_child_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("open-child: expected 1 arg (title), got {n_args}");
    }
    let Some(title) = arg_string(args, 0) else {
        panic!("open-child: title must be a string");
    };
    match window::open_child(&title) {
        Some(id) => Word::fixnum(id).raw(),
        None => Word::NIL.raw(),
    }
}

/// `(close-child child-id)` — close the named child. Returns T on
/// success, NIL if the child id is unknown.
pub extern "C-unwind" fn close_child_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("close-child: expected 1 arg (child-id), got {n_args}");
    }
    let Some(id) = arg_fixnum(args, 0) else {
        panic!("close-child: child-id must be a fixnum");
    };
    if window::close_child(id) {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

/// `(set-title child-id title)` — change a child's window title.
/// Returns T on success, NIL if the child id is unknown.
pub extern "C-unwind" fn set_title_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("set-title: expected 2 args (child-id, title), got {n_args}");
    }
    let Some(id) = arg_fixnum(args, 0) else {
        panic!("set-title: child-id must be a fixnum");
    };
    let Some(title) = arg_string(args, 1) else {
        panic!("set-title: title must be a string");
    };
    window::set_child_title(id, &title);
    Word::T.raw()
}

/// `(next-event timeout-ms)` — pull the next event off the global
/// mailbox. `timeout-ms` is a fixnum: negative blocks forever,
/// 0 polls without blocking, positive waits up to N ms. Returns
/// a property list `(:kind ... :child-id ... ...)` describing the
/// event, or NIL on timeout.
///
/// Each event variant maps to a fixed set of plist keys. The
/// `:kind` key is always present and is a keyword identifying the
/// event family (`:KEY`, `:MOUSE`, `:RESIZE`, `:CLOSE`, etc.).
pub extern "C-unwind" fn next_event_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("next-event: expected 1 arg (timeout-ms), got {n_args}");
    }
    let Some(timeout) = arg_fixnum(args, 0) else {
        panic!("next-event: timeout-ms must be a fixnum");
    };
    let Some(ev) = channels::next_event(timeout) else {
        return Word::NIL.raw();
    };
    let m = unsafe { &mut *mutator };
    let coord = std::sync::Arc::clone(m.coord());
    event_to_plist(m, &coord, ev).raw()
}

// -- event → plist conversion ------------------------------------------------

fn event_to_plist(
    m: &mut crate::mutator::MutatorState,
    coord: &GcCoordinator,
    ev: IGuiEvent,
) -> Word {
    let mut pairs: Vec<(Word, Word)> = Vec::new();
    match ev {
        IGuiEvent::Key {
            child_id, vkey, scancode, mods, repeat, down, time_ms,
        } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "KEY")));
            pairs.push((kw(coord, "CHILD-ID"), Word::fixnum(child_id)));
            pairs.push((kw(coord, "VKEY"), Word::fixnum(vkey)));
            pairs.push((kw(coord, "SCANCODE"), Word::fixnum(scancode)));
            pairs.push((kw(coord, "MODS"), Word::fixnum(mods)));
            pairs.push((kw(coord, "REPEAT"), Word::fixnum(repeat)));
            pairs.push((kw(coord, "DOWN"), if down { Word::T } else { Word::NIL }));
            pairs.push((kw(coord, "TIME-MS"), Word::fixnum(time_ms)));
        }
        IGuiEvent::Char {
            child_id, codepoint, mods, time_ms,
        } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "CHAR")));
            pairs.push((kw(coord, "CHILD-ID"), Word::fixnum(child_id)));
            let ch = char::from_u32(codepoint as u32)
                .map(Word::char)
                .unwrap_or(Word::NIL);
            pairs.push((kw(coord, "CHAR"), ch));
            pairs.push((kw(coord, "CODEPOINT"), Word::fixnum(codepoint)));
            pairs.push((kw(coord, "MODS"), Word::fixnum(mods)));
            pairs.push((kw(coord, "TIME-MS"), Word::fixnum(time_ms)));
        }
        IGuiEvent::Mouse {
            child_id, x, y, op, button, mods, wheel_delta, wheel_lines, time_ms,
        } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "MOUSE")));
            pairs.push((kw(coord, "CHILD-ID"), Word::fixnum(child_id)));
            pairs.push((kw(coord, "X"), Word::fixnum(x)));
            pairs.push((kw(coord, "Y"), Word::fixnum(y)));
            pairs.push((kw(coord, "OP"), kw(coord, mouse_op_name(op))));
            pairs.push((kw(coord, "BUTTON"), Word::fixnum(button)));
            pairs.push((kw(coord, "MODS"), Word::fixnum(mods)));
            pairs.push((kw(coord, "WHEEL-DELTA"), Word::fixnum(wheel_delta)));
            pairs.push((kw(coord, "WHEEL-LINES"), Word::fixnum(wheel_lines)));
            pairs.push((kw(coord, "TIME-MS"), Word::fixnum(time_ms)));
        }
        IGuiEvent::Focus { child_id, gained } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "FOCUS")));
            pairs.push((kw(coord, "CHILD-ID"), Word::fixnum(child_id)));
            pairs.push((kw(coord, "GAINED"), if gained { Word::T } else { Word::NIL }));
        }
        IGuiEvent::Resize { child_id, width, height } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "RESIZE")));
            pairs.push((kw(coord, "CHILD-ID"), Word::fixnum(child_id)));
            pairs.push((kw(coord, "WIDTH"), Word::fixnum(width)));
            pairs.push((kw(coord, "HEIGHT"), Word::fixnum(height)));
        }
        IGuiEvent::Close { child_id } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "CLOSE")));
            pairs.push((kw(coord, "CHILD-ID"), Word::fixnum(child_id)));
        }
        IGuiEvent::FrameClose => {
            pairs.push((kw(coord, "KIND"), kw(coord, "FRAME-CLOSE")));
        }
        IGuiEvent::ThemeChange => {
            pairs.push((kw(coord, "KIND"), kw(coord, "THEME-CHANGE")));
        }
        IGuiEvent::DpiChange { child_id, dpi_x, dpi_y } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "DPI-CHANGE")));
            pairs.push((kw(coord, "CHILD-ID"), Word::fixnum(child_id)));
            pairs.push((kw(coord, "DPI-X"), Word::fixnum(dpi_x)));
            pairs.push((kw(coord, "DPI-Y"), Word::fixnum(dpi_y)));
        }
        IGuiEvent::Menu { menu_id, item_id } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "MENU")));
            pairs.push((kw(coord, "MENU-ID"), Word::fixnum(menu_id)));
            pairs.push((kw(coord, "ITEM-ID"), Word::fixnum(item_id)));
        }
        IGuiEvent::Tick { child_id, time_ms } => {
            pairs.push((kw(coord, "KIND"), kw(coord, "TICK")));
            pairs.push((kw(coord, "CHILD-ID"), Word::fixnum(child_id)));
            pairs.push((kw(coord, "TIME-MS"), Word::fixnum(time_ms)));
        }
    }
    // Build (k1 v1 k2 v2 ...) right-to-left.
    let mut acc = Word::NIL;
    for (k, v) in pairs.into_iter().rev() {
        acc = m.alloc_cons(v, acc);
        acc = m.alloc_cons(k, acc);
    }
    acc
}

// -- Log view ----------------------------------------------------------------

/// `(log-write s)` — append S as a new line to the iGui log
/// overlay's ring buffer. The log_view child shows the buffer
/// (Tools → Log, or Ctrl+Shift+L). Repeated identical lines
/// coalesce: the count next to the line ticks up rather than
/// pushing a fresh entry.
pub extern "C-unwind" fn log_write_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("log-write: expected 1 arg (string), got {n_args}");
    }
    let Some(s) = arg_string(args, 0) else {
        panic!("log-write: argument must be a string");
    };
    log_view::append(&s);
    Word::T.raw()
}

// -- Drawing -----------------------------------------------------------------
//
// Each call site looks like
//
//   (with-batch CHILD-ID
//     (clear (rgb 30 30 50))
//     (fill-rect 10 10 100 50 (rgb 200 50 50)))
//
// where `with-batch` is a macro wrapping `(%begin-batch ...)` ...
// `(%submit-batch)`. The drawing primitives below assume an active
// batch (thread-local) and just push commands onto it.
//
// Color is a packed fixnum: 0xRRGGBBAA, one byte per channel.
// `(rgb r g b)` and `(rgba r g b a)` are user-Lisp constructors.

fn unpack_rgba(packed: i64) -> Rgba {
    let bits = packed as u64;
    let r = ((bits >> 24) & 0xFF) as f32 / 255.0;
    let g = ((bits >> 16) & 0xFF) as f32 / 255.0;
    let b = ((bits >> 8) & 0xFF) as f32 / 255.0;
    let a = (bits & 0xFF) as f32 / 255.0;
    Rgba { r, g, b, a }
}

/// `(%begin-batch child-id)` — start building a new batch for the
/// given child. Subsequent emit calls push onto it; `(%submit-batch)`
/// hands it to the GUI thread.
pub extern "C-unwind" fn begin_batch_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("%begin-batch: expected 1 arg (child-id), got {n_args}");
    }
    let Some(id) = arg_fixnum(args, 0) else {
        panic!("%begin-batch: child-id must be a fixnum");
    };
    batch_mod::begin(id);
    Word::T.raw()
}

/// `(%submit-batch)` — hand the in-flight batch to the GUI thread.
/// Returns T on success, NIL if no batch is in flight or the
/// child-id no longer exists.
pub extern "C-unwind" fn submit_batch_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    let Some(batch) = batch_mod::finish() else {
        return Word::NIL.raw();
    };
    if batch_mod::submit(batch) {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

/// `(%emit-clear color)` — fill the entire client area with `color`.
pub extern "C-unwind" fn emit_clear_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("%emit-clear: expected 1 arg (color), got {n_args}");
    }
    let Some(c) = arg_fixnum(args, 0) else {
        panic!("%emit-clear: color must be a fixnum");
    };
    batch_mod::push(SurfaceCmd::Clear { color: unpack_rgba(c) });
    Word::T.raw()
}

/// `(%emit-fill-rect x y w h color)` — filled rectangle. Coords
/// are fixnums (pixel-precise; sub-pixel waits on float support).
pub extern "C-unwind" fn emit_fill_rect_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 5 {
        panic!("%emit-fill-rect: expected 5 args (x y w h color), got {n_args}");
    }
    let x = arg_fixnum(args, 0).expect("x") as f32;
    let y = arg_fixnum(args, 1).expect("y") as f32;
    let w = arg_fixnum(args, 2).expect("w") as f32;
    let h = arg_fixnum(args, 3).expect("h") as f32;
    let c = arg_fixnum(args, 4).expect("color");
    batch_mod::push(SurfaceCmd::FillRect {
        rect: Rect { x0: x, y0: y, x1: x + w, y1: y + h },
        corner_radius: 0.0,
        color: unpack_rgba(c),
    });
    Word::T.raw()
}

/// `(%emit-stroke-rect x y w h thickness color)` — outlined rectangle.
pub extern "C-unwind" fn emit_stroke_rect_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 6 {
        panic!("%emit-stroke-rect: expected 6 args, got {n_args}");
    }
    let x = arg_fixnum(args, 0).expect("x") as f32;
    let y = arg_fixnum(args, 1).expect("y") as f32;
    let w = arg_fixnum(args, 2).expect("w") as f32;
    let h = arg_fixnum(args, 3).expect("h") as f32;
    let t = arg_fixnum(args, 4).expect("thickness") as f32;
    let c = arg_fixnum(args, 5).expect("color");
    batch_mod::push(SurfaceCmd::StrokeRect {
        rect: Rect { x0: x, y0: y, x1: x + w, y1: y + h },
        corner_radius: 0.0,
        half_thickness: t * 0.5,
        color: unpack_rgba(c),
    });
    Word::T.raw()
}

/// `(%emit-draw-line x1 y1 x2 y2 thickness color)` — line segment.
pub extern "C-unwind" fn emit_draw_line_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 6 {
        panic!("%emit-draw-line: expected 6 args, got {n_args}");
    }
    let x1 = arg_fixnum(args, 0).expect("x1") as f32;
    let y1 = arg_fixnum(args, 1).expect("y1") as f32;
    let x2 = arg_fixnum(args, 2).expect("x2") as f32;
    let y2 = arg_fixnum(args, 3).expect("y2") as f32;
    let t = arg_fixnum(args, 4).expect("thickness") as f32;
    let c = arg_fixnum(args, 5).expect("color");
    batch_mod::push(SurfaceCmd::DrawLine {
        p0: Point { x: x1, y: y1 },
        p1: Point { x: x2, y: y2 },
        half_thickness: t * 0.5,
        color: unpack_rgba(c),
    });
    Word::T.raw()
}

fn mouse_op_name(op: i64) -> &'static str {
    use super::channels::mouse_op;
    match op {
        x if x == mouse_op::MOVE => "MOVE",
        x if x == mouse_op::LEFT_DOWN => "LEFT-DOWN",
        x if x == mouse_op::LEFT_UP => "LEFT-UP",
        x if x == mouse_op::RIGHT_DOWN => "RIGHT-DOWN",
        x if x == mouse_op::RIGHT_UP => "RIGHT-UP",
        x if x == mouse_op::MIDDLE_DOWN => "MIDDLE-DOWN",
        x if x == mouse_op::MIDDLE_UP => "MIDDLE-UP",
        x if x == mouse_op::WHEEL => "WHEEL",
        _ => "UNKNOWN",
    }
}

// Suppress an unused-import warning in builds where `kind::*` is
// only consulted via mouse_op::* constants (i.e. always).
#[allow(dead_code)]
const _: i64 = kind::NONE;
