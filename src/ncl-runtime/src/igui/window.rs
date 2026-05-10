//! MDI frame window, MDI client child, message pump, and the
//! cross-thread helpers used by `iGui.OpenChild` / `CloseChild` /
//! `SetTitle`.
//!
//! Window-creation operations issued by the language thread are
//! marshalled to the GUI thread via private `WM_USER` messages and
//! `SendMessageW`, which blocks until the WndProc returns. This
//! preserves the iGui rule that all HWND ownership lives on the GUI
//! thread without forcing a typed RPC between the two.

#![cfg(windows)]

use std::ptr;
use std::sync::OnceLock;
use std::sync::Mutex;

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::HiDpi::{
    SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetKeyState, VK_CAPITAL, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefFrameProcW, DispatchMessageW, GetClientRect, GetMessageTime,
    GetMessageW, LoadCursorW, PostMessageW, PostQuitMessage, RegisterClassExW, SendMessageW,
    ShowWindow,
    TranslateAcceleratorW, TranslateMessage, CLIENTCREATESTRUCT, CW_USEDEFAULT, HACCEL,
    IDC_ARROW, MDICREATESTRUCTW, MSG, SW_SHOW, WHEEL_DELTA, WM_CHAR, WM_CLOSE, WM_COMMAND,
    WM_DESTROY, WM_KEYDOWN, WM_KEYUP, WM_KILLFOCUS, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MDICREATE, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_RBUTTONDOWN,
    WM_RBUTTONUP, WM_SETFOCUS, WM_SIZE, WM_SYSCOLORCHANGE, WM_SYSKEYDOWN, WM_SYSKEYUP,
    WM_THEMECHANGED, WM_USER, WNDCLASSEXW, WNDCLASS_STYLES, WS_CHILD, WS_CLIPCHILDREN,
    WS_EX_APPWINDOW, WS_HSCROLL, WS_OVERLAPPEDWINDOW, WS_VISIBLE, WS_VSCROLL,
};

use super::channels::{self, modifier, mouse_op, IGuiEvent};
use super::child::{self, MdiBootstrap, MDI_CHILD_CLASS};
use super::cp_exports::FRAME_HWND;
use super::registry;
use super::renderer;
use super::IGuiError;

const FRAME_CHILD_ID: i64 = 1;
const FRAME_CLASS: PCWSTR = w!("NewCL.iGui.Frame");

// Private messages used to marshal language-thread calls onto the GUI
// thread. lparam is the address of the corresponding *Request struct,
// which the WndProc reads, mutates, and returns 0; the SendMessageW
// caller reads its own request struct on return.
const WM_IGUI_OPEN_CHILD: u32 = WM_USER + 1;
const WM_IGUI_CLOSE_CHILD: u32 = WM_USER + 2;
const WM_IGUI_SET_TITLE: u32 = WM_USER + 3;
const WM_IGUI_SET_MENU: u32 = WM_USER + 4;
const WM_IGUI_MDI_VERB: u32 = WM_USER + 5;
/// Open a built-in text-view MDI child. Like WM_IGUI_OPEN_CHILD but
/// the child class is `text_view`'s, with its own WndProc + grid
/// state. Routed through the frame so the WM_MDICREATE call lands
/// on the GUI thread.
const WM_IGUI_OPEN_TEXT: u32 = WM_USER + 7;
/// Drain the pending text-command queue for a text-view child onto
/// its grid, then invalidate. wparam carries the child_id. Both
/// queue-drain and InvalidateRect run on the GUI thread inside the
/// frame's WndProc — the language thread sees nothing past `child_id`
/// as an opaque token. Posted (not sent) so a tight write loop
/// doesn't block on the GUI thread.
const WM_IGUI_TEXT_FLUSH: u32 = WM_USER + 8;
/// Sent from the language thread to a render-host HWND to install
/// or clear a Win32 timer driving `EvTick` events.
/// `wparam` carries the interval in ms (0 = clear), `lparam` is unused.
pub(crate) const WM_IGUI_SET_TIMER: u32 = WM_USER + 6;
/// Win32 timer id used by the redraw-rate ticker. One timer per
/// render host; reusing the same id replaces the previous one.
pub(crate) const TICK_TIMER_ID: usize = 0xA1;

/// HWND of the MDI client. Set by `run` after `CreateWindowExW`.
static MDI_CLIENT: Mutex<Option<isize>> = Mutex::new(None);
static GUI_THREAD_ID: OnceLock<u32> = OnceLock::new();

fn mdi_client_hwnd() -> Option<HWND> {
    let raw = MDI_CLIENT.lock().ok()?;
    raw.map(|r| HWND(r as *mut _))
}

pub(crate) fn gui_thread_id() -> Option<u32> {
    GUI_THREAD_ID.get().copied()
}

/// Public entry point. Opens the iGui frame, sets up the MDI client,
/// runs the Win32 message pump until `WM_QUIT`, and returns the quit
/// code. If `worker` is provided, it is spawned on a background
/// thread once the frame is up.
pub fn run<F>(worker: Option<F>) -> Result<i32, IGuiError>
where
    F: FnOnce() + Send + 'static,
{
    unsafe {
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
    let _ = GUI_THREAD_ID.set(unsafe { GetCurrentThreadId() });

    let h_instance = unsafe { GetModuleHandleW(None) }
        .map_err(|e| IGuiError::Win32(format!("GetModuleHandleW failed: {e}")))?
        .into();
    let cursor = unsafe { LoadCursorW(None, IDC_ARROW) }
        .map_err(|e| IGuiError::Win32(format!("LoadCursorW failed: {e}")))?;

    // Frame class.
    let frame_class = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: WNDCLASS_STYLES(0),
        lpfnWndProc: Some(frame_wnd_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: h_instance,
        hIcon: Default::default(),
        hCursor: cursor,
        hbrBackground: windows::Win32::Graphics::Gdi::HBRUSH(ptr::null_mut()),
        lpszMenuName: PCWSTR::null(),
        lpszClassName: FRAME_CLASS,
        hIconSm: Default::default(),
    };
    if unsafe { RegisterClassExW(&frame_class) } == 0 {
        return Err(IGuiError::Win32("RegisterClassExW (frame) returned 0".into()));
    }
    child::register_classes()?;

    // Renderer comes up before any window so child WM_NCCREATE can build
    // its swap chain immediately.
    renderer::install()?;

    let hwnd = unsafe {
        CreateWindowExW(
            WS_EX_APPWINDOW,
            FRAME_CLASS,
            w!("NewCL"),
            WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            1024,
            720,
            None,
            None,
            Some(h_instance),
            None,
        )
    }
    .map_err(|e| IGuiError::Win32(format!("CreateWindowExW (frame) failed: {e}")))?;
    let _ = FRAME_HWND.set(hwnd.0 as isize);

    // MDI client occupies the whole frame body for now (no toolbar /
    // status bar yet).
    let mut frame_rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut frame_rect) }
        .map_err(|e| IGuiError::Win32(format!("GetClientRect (frame) failed: {e}")))?;
    let mut create = CLIENTCREATESTRUCT {
        hWindowMenu: Default::default(),
        idFirstChild: 0xCC00,
    };
    let mdi = unsafe {
        CreateWindowExW(
            windows::Win32::UI::WindowsAndMessaging::WINDOW_EX_STYLE(0),
            w!("MDICLIENT"),
            PCWSTR::null(),
            WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN | WS_HSCROLL | WS_VSCROLL,
            0,
            0,
            frame_rect.right - frame_rect.left,
            frame_rect.bottom - frame_rect.top,
            Some(hwnd),
            None,
            Some(h_instance),
            Some(&mut create as *mut _ as *mut _),
        )
    }
    .map_err(|e| IGuiError::Win32(format!("CreateWindowExW (MDICLIENT) failed: {e}")))?;
    {
        let mut slot = MDI_CLIENT.lock().expect("MDI_CLIENT mutex poisoned");
        *slot = Some(mdi.0 as isize);
    }

    channels::install();
    super::system_colors::sample();

    // Install a default Tools menu so the built-in editor and log
    // view are reachable even before any language-thread code runs.
    // `iGui.SetMenu` from CP will replace this, but
    // `menu::install_for_frame` always re-appends the tools so they
    // stay available.
    if let Some(default_menu) = super::tools_menu::build_default_menu_bar() {
        let _ = unsafe {
            windows::Win32::UI::WindowsAndMessaging::SetMenu(hwnd, Some(default_menu))
        };
        let _ = unsafe { windows::Win32::UI::WindowsAndMessaging::DrawMenuBar(hwnd) };
    }

    let _ = unsafe { ShowWindow(hwnd, SW_SHOW) };

    if let Some(worker) = worker {
        std::thread::Builder::new()
            .name("igui-language".into())
            .spawn(worker)
            .map_err(|e| IGuiError::Win32(format!("spawn language thread: {e}")))?;
    }

    // Frame-level accelerator table for the built-in tools:
    // Ctrl+Shift+E opens ledit, Ctrl+Shift+L opens the log view,
    // both regardless of which child has focus.
    let accel: Option<HACCEL> = super::tools_menu::build_accelerator_table();

    let mut msg = MSG::default();
    let exit_code = unsafe {
        loop {
            let r = GetMessageW(&mut msg, None, 0, 0);
            if r.0 == 0 {
                break msg.wParam.0 as i32;
            }
            if r.0 == -1 {
                break 1;
            }
            // Frame accelerators run before MDI accel and TranslateMessage:
            // they own the highest-priority shortcuts (Ctrl+Shift+E to
            // open ledit) regardless of which child has focus.
            if let Some(h) = accel {
                if TranslateAcceleratorW(hwnd, h, &mut msg) != 0 {
                    continue;
                }
            }
            // MDI requires TranslateMDISysAccel before TranslateMessage
            // for system MDI shortcuts (Ctrl+F4, Ctrl+F6, etc.).
            if windows::Win32::UI::WindowsAndMessaging::TranslateMDISysAccel(mdi, &msg).as_bool() {
                continue;
            }
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    };

    Ok(exit_code)
}

unsafe extern "system" fn frame_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let mdi = mdi_client_hwnd().unwrap_or_default();

    match msg {
        WM_IGUI_OPEN_CHILD => {
            let req_ptr = lparam.0 as *mut OpenChildRequest;
            if !req_ptr.is_null() {
                let req = unsafe { &mut *req_ptr };
                req.out = handle_open_child(req);
            }
            LRESULT(0)
        }
        WM_IGUI_OPEN_TEXT => {
            let req_ptr = lparam.0 as *mut OpenTextRequest;
            if !req_ptr.is_null() {
                let req = unsafe { &mut *req_ptr };
                if let Some(mdi_client) = mdi_client_hwnd() {
                    req.out = super::text_view::create_on_gui_thread(mdi_client, &req.title);
                }
            }
            LRESULT(0)
        }
        WM_IGUI_TEXT_FLUSH => {
            let child_id = wparam.0 as i64;
            super::text_view::flush_on_gui_thread(child_id);
            LRESULT(0)
        }
        WM_IGUI_CLOSE_CHILD => {
            let req_ptr = lparam.0 as *mut CloseChildRequest;
            if !req_ptr.is_null() {
                let req = unsafe { &mut *req_ptr };
                if let Some(mdi_child) = registry::mdi_hwnd_of(req.child_id) {
                    if mdi.0 as isize != 0 {
                        child::close_via_mdi(mdi, mdi_child);
                        req.ok = true;
                    }
                }
            }
            LRESULT(0)
        }
        WM_IGUI_SET_TITLE => {
            let req_ptr = lparam.0 as *mut SetTitleRequest;
            if !req_ptr.is_null() {
                let req = unsafe { &*req_ptr };
                if let Some(mdi_child) = registry::mdi_hwnd_of(req.child_id) {
                    child::set_title(mdi_child, &req.title);
                }
            }
            LRESULT(0)
        }
        WM_IGUI_SET_MENU => {
            let req_ptr = lparam.0 as *mut SetMenuRequest;
            if !req_ptr.is_null() {
                let req = unsafe { &mut *req_ptr };
                req.ok = super::menu::install_for_frame(hwnd, mdi, &req.spec);
            }
            LRESULT(0)
        }
        WM_IGUI_MDI_VERB => {
            // wparam high byte = verb tag (avoid having to allocate
            // a request struct).
            let tag = wparam.0 as u8;
            if let Some(verb) = mdi_verb_from_tag(tag) {
                if mdi.0 as isize != 0 {
                    if matches!(verb, super::menu::MdiVerb::CloseAll) {
                        for (_id, mdi_child) in registry::snapshot() {
                            child::close_via_mdi(mdi, mdi_child);
                        }
                    } else {
                        super::menu::dispatch_mdi(mdi, verb);
                    }
                }
            }
            LRESULT(0)
        }
        WM_COMMAND => {
            let cmd_id = (wparam.0 & 0xFFFF) as u16;
            // Built-in tools (ledit, log view) are wired before the
            // user menu so they work even if no language-thread spec
            // has been installed.
            if cmd_id == super::ledit::MENU_CMD_ID {
                if mdi.0 as isize != 0 {
                    super::ledit::open(hwnd, mdi);
                }
                return LRESULT(0);
            }
            if cmd_id == super::log_view::MENU_CMD_ID {
                if mdi.0 as isize != 0 {
                    super::log_view::open(hwnd, mdi);
                }
                return LRESULT(0);
            }
            // MDI verbs auto-allocated in install_for_frame.
            if let Some(verb) = super::menu::lookup_mdi_verb(cmd_id) {
                if mdi.0 as isize != 0 {
                    if matches!(verb, super::menu::MdiVerb::CloseAll) {
                        for (_id, mdi_child) in registry::snapshot() {
                            child::close_via_mdi(mdi, mdi_child);
                        }
                    } else {
                        super::menu::dispatch_mdi(mdi, verb);
                    }
                }
                return LRESULT(0);
            }
            // User menu items: push EvMenu so the language thread can
            // dispatch on item_id.
            channels::push(IGuiEvent::Menu {
                menu_id: 0,
                item_id: cmd_id as i64,
            });
            LRESULT(0)
        }
        WM_SIZE => {
            // MDI client sizes itself via DefFrameProcW.
            channels::push(IGuiEvent::Resize {
                child_id: FRAME_CHILD_ID,
                width: (lparam.0 & 0xFFFF) as i64,
                height: ((lparam.0 >> 16) & 0xFFFF) as i64,
            });
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            push_key(FRAME_CHILD_ID, true, wparam, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_KEYUP | WM_SYSKEYUP => {
            push_key(FRAME_CHILD_ID, false, wparam, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_CHAR => {
            channels::push(IGuiEvent::Char {
                child_id: FRAME_CHILD_ID,
                codepoint: wparam.0 as i64,
                mods: current_modifiers(),
                time_ms: msg_time(),
            });
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_MOUSEMOVE => {
            push_mouse(FRAME_CHILD_ID, mouse_op::MOVE, 0, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_LBUTTONDOWN => {
            push_mouse(FRAME_CHILD_ID, mouse_op::LEFT_DOWN, 1, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_LBUTTONUP => {
            push_mouse(FRAME_CHILD_ID, mouse_op::LEFT_UP, 1, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_RBUTTONDOWN => {
            push_mouse(FRAME_CHILD_ID, mouse_op::RIGHT_DOWN, 2, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_RBUTTONUP => {
            push_mouse(FRAME_CHILD_ID, mouse_op::RIGHT_UP, 2, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_MBUTTONDOWN => {
            push_mouse(FRAME_CHILD_ID, mouse_op::MIDDLE_DOWN, 3, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_MBUTTONUP => {
            push_mouse(FRAME_CHILD_ID, mouse_op::MIDDLE_UP, 3, lparam);
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_MOUSEWHEEL => {
            let raw = ((wparam.0 >> 16) & 0xFFFF) as i16;
            let delta = raw as i64;
            let lines = if WHEEL_DELTA != 0 {
                delta / (WHEEL_DELTA as i64)
            } else {
                0
            };
            channels::push(IGuiEvent::Mouse {
                child_id: FRAME_CHILD_ID,
                x: (lparam.0 & 0xFFFF) as i16 as i64,
                y: ((lparam.0 >> 16) & 0xFFFF) as i16 as i64,
                op: mouse_op::WHEEL,
                button: 0,
                mods: current_modifiers(),
                wheel_delta: delta,
                wheel_lines: lines,
                time_ms: msg_time(),
            });
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_SETFOCUS => {
            channels::push(IGuiEvent::Focus {
                child_id: FRAME_CHILD_ID,
                gained: true,
            });
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_KILLFOCUS => {
            channels::push(IGuiEvent::Focus {
                child_id: FRAME_CHILD_ID,
                gained: false,
            });
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_SYSCOLORCHANGE | WM_THEMECHANGED => {
            super::system_colors::refresh_and_notify();
            unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) }
        }
        WM_CLOSE => {
            channels::push(IGuiEvent::FrameClose);
            // Close every registered MDI child, then destroy the frame.
            if mdi.0 as isize != 0 {
                for (_id, child_hwnd) in registry::snapshot() {
                    child::close_via_mdi(mdi, child_hwnd);
                }
            }
            let _ = unsafe { windows::Win32::UI::WindowsAndMessaging::DestroyWindow(hwnd) };
            LRESULT(0)
        }
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }
        _ => unsafe { DefFrameProcW(hwnd, Some(mdi), msg, wparam, lparam) },
    }
}

fn handle_open_child(req: &OpenChildRequest) -> Option<i64> {
    let mdi = mdi_client_hwnd()?;
    let child_id = registry::allocate_child_id();
    let bootstrap = Box::into_raw(Box::new(MdiBootstrap { child_id }));
    let h_module = unsafe { GetModuleHandleW(None) }.ok()?;
    let h_owner = windows::Win32::Foundation::HANDLE(h_module.0);

    let mdi_create = MDICREATESTRUCTW {
        szClass: MDI_CHILD_CLASS,
        szTitle: PCWSTR::from_raw(req.title.as_ptr()),
        hOwner: h_owner,
        x: CW_USEDEFAULT,
        y: CW_USEDEFAULT,
        cx: CW_USEDEFAULT,
        cy: CW_USEDEFAULT,
        style: WS_VISIBLE | WS_OVERLAPPEDWINDOW,
        lParam: LPARAM(bootstrap as isize),
    };
    let result = unsafe {
        SendMessageW(
            mdi,
            WM_MDICREATE,
            Some(WPARAM(0)),
            Some(LPARAM(&mdi_create as *const _ as isize)),
        )
    };
    let new_hwnd = HWND(result.0 as *mut _);
    if new_hwnd.0.is_null() {
        // WM_MDICREATE failed; reclaim the bootstrap to avoid leaking.
        let _ = unsafe { Box::from_raw(bootstrap) };
        return None;
    }
    Some(child_id)
}

pub(crate) fn msg_time() -> i64 {
    unsafe { GetMessageTime() as i64 }
}

pub(crate) fn current_modifiers() -> i64 {
    let mut m = 0i64;
    unsafe {
        if (GetKeyState(VK_SHIFT.0 as i32) as i16) < 0 {
            m |= modifier::SHIFT;
        }
        if (GetKeyState(VK_CONTROL.0 as i32) as i16) < 0 {
            m |= modifier::CONTROL;
        }
        if (GetKeyState(VK_MENU.0 as i32) as i16) < 0 {
            m |= modifier::ALT;
        }
        if (GetKeyState(VK_LWIN.0 as i32) as i16) < 0
            || (GetKeyState(VK_RWIN.0 as i32) as i16) < 0
        {
            m |= modifier::WIN;
        }
        if (GetKeyState(VK_CAPITAL.0 as i32) & 1) != 0 {
            m |= modifier::CAPS;
        }
    }
    m
}

pub(crate) fn push_key(child_id: i64, down: bool, wparam: WPARAM, lparam: LPARAM) {
    let scancode = ((lparam.0 >> 16) & 0xFF) as i64;
    let repeat = (lparam.0 & 0xFFFF) as i64;
    channels::push(IGuiEvent::Key {
        child_id,
        vkey: wparam.0 as i64,
        scancode,
        mods: current_modifiers(),
        repeat,
        down,
        time_ms: msg_time(),
    });
}

pub(crate) fn push_mouse(child_id: i64, op: i64, button: i64, lparam: LPARAM) {
    let x = (lparam.0 & 0xFFFF) as i16 as i64;
    let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as i64;
    channels::push(IGuiEvent::Mouse {
        child_id,
        x,
        y,
        op,
        button,
        mods: current_modifiers(),
        wheel_delta: 0,
        wheel_lines: 0,
        time_ms: msg_time(),
    });
}

// ─── Cross-thread request structures ─────────────────────────────────

pub(crate) struct OpenChildRequest {
    pub title: Vec<u16>,
    pub out: Option<i64>,
}

pub(crate) struct OpenTextRequest {
    pub title: Vec<u16>,
    pub out: Option<i64>,
}

pub(crate) struct CloseChildRequest {
    pub child_id: i64,
    pub ok: bool,
}

pub(crate) struct SetTitleRequest {
    pub child_id: i64,
    pub title: Vec<u16>,
}

pub(crate) struct SetMenuRequest {
    pub spec: String,
    pub ok: bool,
}

fn mdi_verb_from_tag(tag: u8) -> Option<super::menu::MdiVerb> {
    use super::menu::MdiVerb;
    match tag {
        1 => Some(MdiVerb::Cascade),
        2 => Some(MdiVerb::TileH),
        3 => Some(MdiVerb::TileV),
        4 => Some(MdiVerb::CloseAll),
        5 => Some(MdiVerb::ArrangeIcons),
        _ => None,
    }
}

fn mdi_verb_to_tag(verb: super::menu::MdiVerb) -> u8 {
    use super::menu::MdiVerb;
    match verb {
        MdiVerb::Cascade => 1,
        MdiVerb::TileH => 2,
        MdiVerb::TileV => 3,
        MdiVerb::CloseAll => 4,
        MdiVerb::ArrangeIcons => 5,
    }
}

/// Called from the language thread. Marshals to the GUI thread via
/// SendMessageW; blocks until the child has been created.
pub fn open_child(title: &str) -> Option<i64> {
    let frame_raw = *FRAME_HWND.get()?;
    let frame = HWND(frame_raw as *mut _);
    let mut title_w: Vec<u16> = title.encode_utf16().collect();
    title_w.push(0);
    let mut req = OpenChildRequest {
        title: title_w,
        out: None,
    };
    unsafe {
        SendMessageW(
            frame,
            WM_IGUI_OPEN_CHILD,
            Some(WPARAM(0)),
            Some(LPARAM(&mut req as *mut _ as isize)),
        )
    };
    req.out
}

/// Called from the language thread. Same SendMessageW marshalling
/// as `open_child`, but routes to the text-view class on the GUI
/// thread (where state allocation + WM_MDICREATE happen safely).
pub fn open_text_child(title: &str) -> Option<i64> {
    let frame_raw = *FRAME_HWND.get()?;
    let frame = HWND(frame_raw as *mut _);
    let mut title_w: Vec<u16> = title.encode_utf16().collect();
    title_w.push(0);
    let mut req = OpenTextRequest {
        title: title_w,
        out: None,
    };
    unsafe {
        SendMessageW(
            frame,
            WM_IGUI_OPEN_TEXT,
            Some(WPARAM(0)),
            Some(LPARAM(&mut req as *mut _ as isize)),
        )
    };
    req.out
}

pub fn close_child(child_id: i64) -> bool {
    let Some(frame_raw) = FRAME_HWND.get() else {
        return false;
    };
    let frame = HWND(*frame_raw as *mut _);
    let mut req = CloseChildRequest {
        child_id,
        ok: false,
    };
    unsafe {
        SendMessageW(
            frame,
            WM_IGUI_CLOSE_CHILD,
            Some(WPARAM(0)),
            Some(LPARAM(&mut req as *mut _ as isize)),
        )
    };
    req.ok
}

/// Marshal `spec` to the GUI thread, where it's parsed and installed
/// as the frame's menu bar. Returns true on success.
pub fn set_menu(spec: &str) -> bool {
    let Some(frame_raw) = FRAME_HWND.get() else {
        return false;
    };
    let frame = HWND(*frame_raw as *mut _);
    let mut req = SetMenuRequest {
        spec: spec.to_owned(),
        ok: false,
    };
    unsafe {
        SendMessageW(
            frame,
            WM_IGUI_SET_MENU,
            Some(WPARAM(0)),
            Some(LPARAM(&mut req as *mut _ as isize)),
        )
    };
    req.ok
}

/// Install or clear the per-child redraw timer. `interval_ms <= 0`
/// clears the timer; otherwise WM_TIMER fires every `interval_ms`
/// milliseconds and the render host pushes an `EvTick` event.
pub fn set_redraw_rate(child_id: i64, interval_ms: i64) -> bool {
    let Some(render_hwnd) = registry::render_hwnd_of(child_id) else {
        return false;
    };
    let interval = if interval_ms <= 0 { 0 } else { interval_ms as usize };
    unsafe {
        SendMessageW(
            render_hwnd,
            WM_IGUI_SET_TIMER,
            Some(WPARAM(interval)),
            Some(LPARAM(0)),
        )
    };
    true
}

/// Post a "drain the text-view command queue and repaint" message
/// at the frame. Frame WndProc dispatches to text_view's flush
/// handler on the GUI thread, which applies queued commands to the
/// child's grid and then InvalidateRects the child window. The
/// language thread never touches a child HWND. Posted (not sent)
/// so a tight write loop doesn't block on the GUI thread.
pub(crate) fn post_text_flush(child_id: i64) {
    let Some(frame_raw) = FRAME_HWND.get() else {
        return;
    };
    let frame = HWND(*frame_raw as *mut _);
    let _ = unsafe {
        PostMessageW(
            Some(frame),
            WM_IGUI_TEXT_FLUSH,
            WPARAM(child_id as usize),
            LPARAM(0),
        )
    };
}

/// Marshal an MDI verb to the GUI thread for execution.
pub fn dispatch_mdi_verb(verb: super::menu::MdiVerb) {
    let Some(frame_raw) = FRAME_HWND.get() else {
        return;
    };
    let frame = HWND(*frame_raw as *mut _);
    let tag = mdi_verb_to_tag(verb) as usize;
    unsafe {
        SendMessageW(
            frame,
            WM_IGUI_MDI_VERB,
            Some(WPARAM(tag)),
            Some(LPARAM(0)),
        )
    };
}

pub fn set_child_title(child_id: i64, title: &str) {
    let Some(frame_raw) = FRAME_HWND.get() else {
        return;
    };
    let frame = HWND(*frame_raw as *mut _);
    let mut title_w: Vec<u16> = title.encode_utf16().collect();
    title_w.push(0);
    let req = SetTitleRequest {
        child_id,
        title: title_w,
    };
    unsafe {
        SendMessageW(
            frame,
            WM_IGUI_SET_TITLE,
            Some(WPARAM(0)),
            Some(LPARAM(&req as *const _ as isize)),
        )
    };
}

