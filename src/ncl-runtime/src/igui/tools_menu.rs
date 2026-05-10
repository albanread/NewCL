//! Frame-level "Tools" menu and the keyboard accelerator table for
//! redit's built-in tool windows.
//!
//! Both `redit` and `log_view` are always-available editor tools
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
use super::redit;

/// Append a `Tools` submenu to `bar` containing every built-in tool
/// (currently redit and the log view). Called both from
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

    let redit_item: Vec<u16> = "redit\tCtrl+Shift+E\0".encode_utf16().collect();
    if let Err(e) = unsafe {
        AppendMenuW(
            popup,
            MF_STRING,
            redit::MENU_CMD_ID as usize,
            PCWSTR(redit_item.as_ptr()),
        )
    } {
        eprintln!("[tools-menu] append redit: {e}");
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

/// Build a stand-alone menu bar containing only the Tools submenu.
/// Used at frame startup when no language-thread menu has been set.
pub fn build_default_menu_bar() -> Option<HMENU> {
    let bar = unsafe { CreateMenu() }.ok()?;
    append_tools_menu(bar);
    Some(bar)
}

/// Frame-level accelerator table:
///   Ctrl+Shift+E → redit
///   Ctrl+Shift+L → log view
/// Both dispatch via `WM_COMMAND` to their respective MENU_CMD_IDs,
/// which the frame WndProc routes to the right `open` function.
pub fn build_accelerator_table() -> Option<HACCEL> {
    let entries = [
        ACCEL {
            fVirt: FCONTROL | FSHIFT | FVIRTKEY,
            key: b'E' as u16,
            cmd: redit::MENU_CMD_ID,
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
