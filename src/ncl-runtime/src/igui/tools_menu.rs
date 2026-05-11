//! Frame-level "Tools" menu and the keyboard accelerator table for
//! ledit's built-in tool windows.
//!
//! Both `ledit` and `log_view` are always-available editor tools
//! that hang off a `Tools` submenu on the frame. Keeping their
//! menu/accelerator wiring together here means the one-and-only
//! Tools popup carries every entry, regardless of whether the
//! language thread has installed a custom menu.

#![cfg(windows)]

use windows::core::PCWSTR;
use windows::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreateAcceleratorTableW, CreateMenu, CreatePopupMenu, ACCEL, FCONTROL, FSHIFT,
    FVIRTKEY, HACCEL, HMENU, MF_POPUP, MF_STRING,
};

use super::log_view;
use super::ledit;
use windows::Win32::UI::WindowsAndMessaging::MF_SEPARATOR;

/// Append an `Edit` submenu to `bar`. The items are routed to the
/// active MDI child via WM_COMMAND forwarding from the frame
/// WndProc; ledit recognises the IDs in its own WM_COMMAND handler
/// and dispatches to the matching method.
///
/// Keeping these on a top-level menu makes the paredit ops
/// discoverable — most users have not got the muscle memory for
/// Ctrl-Shift-Right or Alt-W and would never find them otherwise.
pub fn append_edit_menu(bar: HMENU) {
    let popup = match unsafe { CreatePopupMenu() } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[edit-menu] CreatePopupMenu failed: {e}");
            return;
        }
    };

    let items: &[(u16, &str)] = &[
        (ledit::EDIT_CMD_UNDO, "&Undo\tCtrl+Z"),
        (ledit::EDIT_CMD_REDO, "&Redo\tCtrl+Y"),
        (0, "SEP"),
        (ledit::EDIT_CMD_CUT, "Cu&t\tCtrl+X"),
        (ledit::EDIT_CMD_COPY, "&Copy\tCtrl+C"),
        (ledit::EDIT_CMD_PASTE, "&Paste\tCtrl+V"),
        (ledit::EDIT_CMD_SELECT_ALL, "Select &All\tCtrl+A"),
        (0, "SEP"),
        (ledit::EDIT_CMD_FORWARD_SEXP, "Forward S-expression\tCtrl+\u{2192}"),
        (ledit::EDIT_CMD_BACKWARD_SEXP, "Backward S-expression\tCtrl+\u{2190}"),
        (0, "SEP"),
        (ledit::EDIT_CMD_SLURP_FORWARD, "&Slurp Forward\tCtrl+Shift+\u{2192}"),
        (ledit::EDIT_CMD_BARF_FORWARD, "&Barf Forward\tCtrl+Shift+\u{2190}"),
        (ledit::EDIT_CMD_WRAP, "&Wrap with ( )\tAlt+W"),
        (ledit::EDIT_CMD_SPLICE, "Spli&ce / Unwrap\tAlt+S"),
        (ledit::EDIT_CMD_RAISE, "&Raise\tAlt+R"),
        (0, "SEP"),
        (ledit::EDIT_CMD_RUN_FORM, "Run Form at &Point\tCtrl+Enter"),
        (ledit::EDIT_CMD_RUN_BUFFER, "R&un Buffer\tF5"),
    ];

    for &(id, label) in items {
        if label == "SEP" {
            let _ = unsafe { AppendMenuW(popup, MF_SEPARATOR, 0, PCWSTR::null()) };
            continue;
        }
        let mut w: Vec<u16> = label.encode_utf16().collect();
        w.push(0);
        if let Err(e) = unsafe {
            AppendMenuW(popup, MF_STRING, id as usize, PCWSTR(w.as_ptr()))
        } {
            eprintln!("[edit-menu] append {label:?}: {e}");
        }
    }

    let title: Vec<u16> = "&Edit\0".encode_utf16().collect();
    if let Err(e) = unsafe {
        AppendMenuW(bar, MF_POPUP, popup.0 as usize, PCWSTR(title.as_ptr()))
    } {
        eprintln!("[edit-menu] append popup: {e}");
    }
}

/// Append a `Tools` submenu to `bar` containing every built-in tool
/// (currently ledit and the log view). Called both from
/// `build_default_menu_bar` and from `menu::install_for_frame` so
/// the tools stay reachable whatever the language thread does.
pub fn append_tools_menu(bar: HMENU) {
    let popup = match unsafe { CreatePopupMenu() } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[tools-menu] CreatePopupMenu failed: {e}");
            return;
        }
    };

    let ledit_item: Vec<u16> = "ledit\tCtrl+Shift+E\0".encode_utf16().collect();
    if let Err(e) = unsafe {
        AppendMenuW(
            popup,
            MF_STRING,
            ledit::MENU_CMD_ID as usize,
            PCWSTR(ledit_item.as_ptr()),
        )
    } {
        eprintln!("[tools-menu] append ledit: {e}");
    }

    let log_item: Vec<u16> = "Log\tCtrl+Shift+L\0".encode_utf16().collect();
    if let Err(e) = unsafe {
        AppendMenuW(
            popup,
            MF_STRING,
            log_view::MENU_CMD_ID as usize,
            PCWSTR(log_item.as_ptr()),
        )
    } {
        eprintln!("[tools-menu] append log: {e}");
    }

    let title: Vec<u16> = "&Tools\0".encode_utf16().collect();
    if let Err(e) = unsafe {
        AppendMenuW(
            bar,
            MF_POPUP,
            popup.0 as usize,
            PCWSTR(title.as_ptr()),
        )
    } {
        eprintln!("[tools-menu] append popup: {e}");
    }
}

/// Build a stand-alone menu bar containing the Edit and Tools
/// submenus. Used at frame startup when no language-thread menu has
/// been set.
pub fn build_default_menu_bar() -> Option<HMENU> {
    let bar = unsafe { CreateMenu() }.ok()?;
    append_edit_menu(bar);
    append_tools_menu(bar);
    Some(bar)
}

/// Frame-level accelerator table:
///   Ctrl+Shift+E → ledit
///   Ctrl+Shift+L → log view
/// Both dispatch via `WM_COMMAND` to their respective MENU_CMD_IDs,
/// which the frame WndProc routes to the right `open` function.
pub fn build_accelerator_table() -> Option<HACCEL> {
    let entries = [
        ACCEL {
            fVirt: FCONTROL | FSHIFT | FVIRTKEY,
            key: b'E' as u16,
            cmd: ledit::MENU_CMD_ID,
        },
        ACCEL {
            fVirt: FCONTROL | FSHIFT | FVIRTKEY,
            key: b'L' as u16,
            cmd: log_view::MENU_CMD_ID,
        },
    ];
    unsafe { CreateAcceleratorTableW(&entries) }
        .ok()
        .filter(|h| !h.is_invalid())
}
