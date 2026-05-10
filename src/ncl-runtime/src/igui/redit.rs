//! redit — fail-safe Rust-based editor.
//!
//! A minimal text editor that lives entirely on the UI thread of the
//! iGui frame. It does not consume `SurfaceCmd` batches, does not
//! touch the language-thread mailbox, and does not depend on any CP
//! code being loaded. The point is that even when the rest of the
//! NewCP environment has a fault, the editor remains responsive so
//! the user can fix source files and reload them.
//!
//! Architecture: a single-instance MDI child with its own WndProc
//! that handles WM_PAINT (Direct2D + DirectWrite, fixed grid) and
//! all input directly. State is heap-allocated on first
//! `WM_NCCREATE` and stored in `GWLP_USERDATA`.
//!
//! R1 scope:
//!   - open / save (Win32 common dialogs)
//!   - basic editing keys (arrows, Home/End, PgUp/PgDn, Enter,
//!     Backspace, Delete, Tab, printable chars)
//!   - mouse click to position cursor
//!   - vertical wheel scroll
//!   - line numbers in a left gutter
//!   - status line at bottom
//!   - no selection, no clipboard, no undo, no syntax colour, no
//!     compiler hookup (those land in R2/R3/R4)

#![cfg(windows)]

use std::ffi::OsString;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::PathBuf;
use std::sync::Mutex;

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct2D::Common::{
    D2D1_ALPHA_MODE_IGNORE, D2D1_COLOR_F, D2D1_PIXEL_FORMAT, D2D_RECT_F, D2D_SIZE_U,
};
use windows::Win32::Graphics::Direct2D::{
    ID2D1HwndRenderTarget, ID2D1SolidColorBrush, D2D1_BRUSH_PROPERTIES,
    D2D1_DRAW_TEXT_OPTIONS_CLIP, D2D1_FEATURE_LEVEL_DEFAULT,
    D2D1_HWND_RENDER_TARGET_PROPERTIES, D2D1_PRESENT_OPTIONS_NONE,
    D2D1_RENDER_TARGET_PROPERTIES, D2D1_RENDER_TARGET_TYPE_DEFAULT,
    D2D1_RENDER_TARGET_USAGE_NONE,
};
use windows::Win32::Graphics::DirectWrite::{
    IDWriteTextFormat, IDWriteTextLayout, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL,
    DWRITE_FONT_WEIGHT, DWRITE_TEXT_METRICS, DWRITE_TEXT_RANGE, DWRITE_WORD_WRAPPING_NO_WRAP,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;
use windows::Win32::Graphics::Gdi::InvalidateRect;
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, OpenClipboard, SetClipboardData,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE};
use windows::Win32::System::Ole::CF_UNICODETEXT;
use windows::Win32::UI::Controls::Dialogs::{
    GetOpenFileNameW, GetSaveFileNameW, OFN_EXPLORER, OFN_FILEMUSTEXIST, OFN_HIDEREADONLY,
    OFN_OVERWRITEPROMPT, OFN_PATHMUSTEXIST, OPENFILENAMEW,
};
use windows::Win32::UI::HiDpi::GetDpiForWindow;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetKeyState, ReleaseCapture, SetCapture, SetFocus, VK_DELETE, VK_DOWN, VK_END, VK_F7, VK_F8,
    VK_HOME, VK_LEFT, VK_NEXT, VK_PRIOR, VK_RIGHT, VK_SHIFT, VK_UP,
};

/// `WM_MOUSEMOVE`'s `wparam` low word; bit 0 = left button held. The
/// windows-rs constant lives in different modules across versions, so
/// we use the well-known winuser.h value directly.
const MK_LBUTTON: u32 = 0x0001;
use windows::Win32::UI::WindowsAndMessaging::{
    BringWindowToTop, DefMDIChildProcW, GetClientRect, GetWindowLongPtrW, IsWindow, LoadCursorW,
    RegisterClassExW, SendMessageW, SetWindowLongPtrW, CW_USEDEFAULT, GWLP_USERDATA, IDC_IBEAM,
    MDICREATESTRUCTW, WHEEL_DELTA, WM_CHAR, WM_DPICHANGED_AFTERPARENT, WM_KEYDOWN,
    WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MDIACTIVATE, WM_MDICREATE, WM_MOUSEMOVE, WM_MOUSEWHEEL,
    WM_NCCREATE, WM_NCDESTROY, WM_PAINT, WM_SETFOCUS, WM_SIZE, WNDCLASSEXW, WNDCLASS_STYLES,
    WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

use super::renderer;

/// WM_COMMAND id for the "Tools > redit" frame menu entry. Outside
/// the user range (0x1000..=0x1FFF) and the MDI verb range
/// (0x2000..=0x2010) so it can never collide with a language-thread
/// menu spec.
pub const MENU_CMD_ID: u16 = 0x3000;

const REDIT_CLASS: PCWSTR = w!("NewCP.iGui.Redit");
const TITLE_NEW: PCWSTR = w!("redit — untitled");

/// HWND of the singleton redit MDI child, if one exists. Used to
/// activate the existing instance instead of creating a second one.
static REDIT_HWND: Mutex<Option<isize>> = Mutex::new(None);

// ─── Compile-check injection point ──────────────────────────────────
//
// The runtime crate sits below `newcp-parser` and `newcp-sema` in the
// dependency graph, so it cannot import them directly. Instead the
// driver (which already depends on both) hands redit a closure that
// runs a check and returns diagnostics. This keeps the layering
// clean and lets us swap in different checkers (e.g. a fast
// parse-only check vs full semantic) later.

/// One diagnostic from the compile-check pass. Lines and columns are
/// 1-indexed to match what shows up in the status bar.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

type CheckFn = Box<dyn Fn(&str) -> Vec<Diagnostic> + Send + Sync + 'static>;

static CHECKER: Mutex<Option<CheckFn>> = Mutex::new(None);

/// Install a closure that takes the editor's full text and returns
/// diagnostics. Call once at startup, before `iGui::run`.
pub fn install_checker<F>(f: F)
where
    F: Fn(&str) -> Vec<Diagnostic> + Send + Sync + 'static,
{
    *CHECKER.lock().expect("CHECKER poisoned") = Some(Box::new(f));
}

fn run_checker(source: &str) -> Option<Vec<Diagnostic>> {
    let guard = CHECKER.lock().expect("CHECKER poisoned");
    let f = guard.as_ref()?;
    Some(f(source))
}

// ─── Public API ──────────────────────────────────────────────────────

/// Register the redit MDI child WndClass. Called from
/// `child::register_classes`.
pub fn register_class() -> Result<(), super::IGuiError> {
    let h_instance = unsafe { GetModuleHandleW(None) }
        .map_err(|e| super::IGuiError::Win32(format!("GetModuleHandleW (redit): {e}")))?
        .into();
    let cursor = unsafe { LoadCursorW(None, IDC_IBEAM) }
        .map_err(|e| super::IGuiError::Win32(format!("LoadCursorW (redit): {e}")))?;
    let cls = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: WNDCLASS_STYLES(0),
        lpfnWndProc: Some(redit_wnd_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: h_instance,
        hIcon: Default::default(),
        hCursor: cursor,
        hbrBackground: windows::Win32::Graphics::Gdi::HBRUSH(std::ptr::null_mut()),
        lpszMenuName: PCWSTR::null(),
        lpszClassName: REDIT_CLASS,
        hIconSm: Default::default(),
    };
    let _ = unsafe { RegisterClassExW(&cls) };
    Ok(())
}

// The Tools menu and frame accelerator table now live in
// `tools_menu`, which knows about both redit and the log view.

/// Open the redit child, or activate it if already open. Called from
/// the frame WndProc when the user picks the menu item or hits the
/// shortcut. UI-thread only.
pub fn open(frame: HWND, mdi_client: HWND) {
    if let Some(raw) = *REDIT_HWND.lock().expect("REDIT_HWND poisoned") {
        let hwnd = HWND(raw as *mut _);
        if unsafe { IsWindow(Some(hwnd)) }.as_bool() {
            unsafe {
                SendMessageW(
                    mdi_client,
                    windows::Win32::UI::WindowsAndMessaging::WM_MDIACTIVATE,
                    Some(WPARAM(hwnd.0 as usize)),
                    Some(LPARAM(0)),
                )
            };
            let _ = unsafe { BringWindowToTop(hwnd) };
            return;
        }
    }

    let h_instance = match unsafe { GetModuleHandleW(None) } {
        Ok(h) => windows::Win32::Foundation::HANDLE(h.0),
        Err(e) => {
            eprintln!("[redit] GetModuleHandleW: {e}");
            return;
        }
    };
    let create = MDICREATESTRUCTW {
        szClass: REDIT_CLASS,
        szTitle: TITLE_NEW,
        hOwner: h_instance,
        x: CW_USEDEFAULT,
        y: CW_USEDEFAULT,
        cx: CW_USEDEFAULT,
        cy: CW_USEDEFAULT,
        style: WS_VISIBLE | WS_OVERLAPPEDWINDOW,
        lParam: LPARAM(0),
    };
    let result = unsafe {
        SendMessageW(
            mdi_client,
            WM_MDICREATE,
            Some(WPARAM(0)),
            Some(LPARAM(&create as *const _ as isize)),
        )
    };
    if result.0 == 0 {
        eprintln!("[redit] WM_MDICREATE returned 0");
        return;
    }
    let _ = frame; // reserved for future use
}

// ─── State ───────────────────────────────────────────────────────────

type Pos = (usize, usize);

/// Edit operation for undo/redo. Stored after the edit has been
/// applied; `cursor_after` is the cursor at that point and
/// `cursor_before` is what it was before.
#[derive(Clone, Debug)]
enum UndoOp {
    /// Inserted `text` at `start`, ending at `end` (cursor lands at
    /// `cursor_after` which is normally `end`). Reverse: delete the
    /// range `[start, end]`.
    Inserted {
        start: Pos,
        end: Pos,
        text: String,
        cursor_before: Pos,
        cursor_after: Pos,
    },
    /// Deleted `text` (which was at `start`). Reverse: insert `text`
    /// at `start`. Used for backspace, delete-forward, and selection
    /// deletion (cut, replace-on-typing).
    Deleted {
        start: Pos,
        text: String,
        cursor_before: Pos,
        cursor_after: Pos,
    },
}

/// Coalescing hint set after a single-char edit so the next edit of
/// the same kind at the contiguous position can extend the previous
/// undo entry instead of pushing a new one. Cleared on movement,
/// click, paste, undo, redo, or any non-coalescing edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoalesceKind {
    Insert,
    Backspace,
}

const UNDO_CAP: usize = 1024;

/// Visual width of a tab stop in cells. The buffer keeps tabs as-is
/// (so files round-trip cleanly on save); the renderer and the
/// click-mapping use this constant to translate between buffer char
/// positions and on-screen cell columns. CP / BlackBox convention
/// is tab-width 2 (a 15-tab continuation indent in `Strings.cp`
/// aligns to ~30 cells, matching the opening paren of the line
/// above), so that's what we default to.
const TAB_WIDTH: usize = 2;

struct ReditState {
    hwnd: HWND,
    target: Option<ID2D1HwndRenderTarget>,
    text_format: Option<IDWriteTextFormat>,
    cell_w: f32,
    cell_h: f32,
    ascent: f32,

    buffer: Vec<String>,
    cursor_row: usize,
    cursor_col: usize,
    /// Selection anchor. When `(anchor_row, anchor_col) ==
    /// (cursor_row, cursor_col)` there is no active selection.
    anchor_row: usize,
    anchor_col: usize,
    pref_col: usize,
    scroll_top: usize,

    file_path: Option<PathBuf>,
    dirty: bool,

    client_w: u32,
    client_h: u32,
    /// Per-monitor DPI of the current monitor. Cached so we can avoid
    /// asking Win32 every paint, refreshed on `WM_DPICHANGED_AFTERPARENT`.
    dpi: u32,

    /// True while the user is dragging the mouse with the left button
    /// held. We capture the mouse so drags that leave the client area
    /// still extend the selection cleanly.
    selecting_drag: bool,

    undo: Vec<UndoOp>,
    redo: Vec<UndoOp>,
    coalesce: Option<CoalesceKind>,

    /// Per-line tokens for syntax highlighting. Lazily refreshed in
    /// paint() when `tokens_dirty` is set. Re-tokenizing the whole
    /// buffer on every edit is fine for the sub-MB files redit is
    /// designed to handle.
    tokens: Vec<Vec<Token>>,
    tokens_dirty: bool,

    /// Diagnostics from the most recent compile check. Cleared when
    /// the buffer is edited (so stale errors don't lie to the user)
    /// and refreshed on F7 / after-save.
    diagnostics: Vec<Diagnostic>,
    /// True when the buffer has changed since the last check, so the
    /// status bar can show "(stale)" instead of pretending the
    /// diagnostics still apply.
    diagnostics_stale: bool,
}

impl ReditState {
    fn new(hwnd: HWND) -> Self {
        let dpi = unsafe { GetDpiForWindow(hwnd) };
        let dpi = if dpi == 0 { 96 } else { dpi };
        Self {
            hwnd,
            target: None,
            text_format: None,
            cell_w: 8.0,
            cell_h: 16.0,
            ascent: 12.0,
            buffer: vec![String::new()],
            cursor_row: 0,
            cursor_col: 0,
            anchor_row: 0,
            anchor_col: 0,
            pref_col: 0,
            scroll_top: 0,
            file_path: None,
            dirty: false,
            client_w: 0,
            client_h: 0,
            dpi,
            selecting_drag: false,
            undo: Vec::new(),
            redo: Vec::new(),
            coalesce: None,
            tokens: Vec::new(),
            tokens_dirty: true,
            diagnostics: Vec::new(),
            diagnostics_stale: true,
        }
    }

    fn ensure_resources(&mut self, w: u32, h: u32) {
        if self.text_format.is_none() {
            self.text_format = create_text_format();
            if let Some(fmt) = self.text_format.as_ref() {
                if let Some((cw, ch, asc)) = measure_cell(fmt) {
                    self.cell_w = cw;
                    self.cell_h = ch;
                    self.ascent = asc;
                }
            }
        }
        if let Some(target) = self.target.as_ref() {
            let cur = unsafe { target.GetPixelSize() };
            if cur.width != w || cur.height != h {
                let _ = unsafe { target.Resize(&D2D_SIZE_U { width: w, height: h }) };
            }
            return;
        }
        let dpi = self.dpi as f32;
        let factory = &renderer::ctx().d2d.factory;
        let target = unsafe {
            factory.CreateHwndRenderTarget(
                &D2D1_RENDER_TARGET_PROPERTIES {
                    r#type: D2D1_RENDER_TARGET_TYPE_DEFAULT,
                    pixelFormat: D2D1_PIXEL_FORMAT {
                        format: DXGI_FORMAT_B8G8R8A8_UNORM,
                        alphaMode: D2D1_ALPHA_MODE_IGNORE,
                    },
                    dpiX: dpi,
                    dpiY: dpi,
                    usage: D2D1_RENDER_TARGET_USAGE_NONE,
                    minLevel: D2D1_FEATURE_LEVEL_DEFAULT,
                },
                &D2D1_HWND_RENDER_TARGET_PROPERTIES {
                    hwnd: self.hwnd,
                    pixelSize: D2D_SIZE_U { width: w, height: h },
                    presentOptions: D2D1_PRESENT_OPTIONS_NONE,
                },
            )
        };
        match target {
            Ok(t) => self.target = Some(t),
            Err(e) => eprintln!("[redit] CreateHwndRenderTarget failed: {e}"),
        }
    }

    /// Apply a new monitor DPI: drop the current render target so the
    /// next paint recreates it at the new DPI, and re-measure the
    /// cell. Called from `WM_DPICHANGED_AFTERPARENT`.
    fn set_dpi(&mut self, dpi: u32) {
        if dpi == 0 || dpi == self.dpi {
            return;
        }
        self.dpi = dpi;
        // Drop the render target so it gets rebuilt with new dpiX/Y.
        // The text format is dpi-independent (sizes are DIPs) but the
        // measured cell rounds against pixel boundaries, so refresh
        // it on the next paint.
        self.target = None;
        if let Some(fmt) = self.text_format.as_ref() {
            if let Some((cw, ch, asc)) = measure_cell(fmt) {
                self.cell_w = cw;
                self.cell_h = ch;
                self.ascent = asc;
            }
        }
        self.invalidate();
    }

    fn invalidate(&self) {
        let _ = unsafe { InvalidateRect(Some(self.hwnd), None, false) };
    }

    /// Pixels-to-DIPs scale factor for the current monitor. The
    /// render target is configured with the monitor's DPI, so all
    /// drawing math runs in DIPs while Win32 hands us pixel
    /// dimensions and pixel-space mouse coordinates. This converts
    /// at the boundary.
    fn dip_scale(&self) -> f32 {
        if self.dpi == 0 {
            1.0
        } else {
            96.0 / (self.dpi as f32)
        }
    }

    fn px_to_dip(&self, px: i32) -> f32 {
        (px as f32) * self.dip_scale()
    }

    // ─── Selection ───────────────────────────────────────────────

    fn cursor_pos(&self) -> Pos {
        (self.cursor_row, self.cursor_col)
    }

    fn anchor_pos(&self) -> Pos {
        (self.anchor_row, self.anchor_col)
    }

    /// `Some((start, end))` if there's an active selection, else
    /// `None`. Tuples compare lexicographically, so `start <= end`.
    fn selection_range(&self) -> Option<(Pos, Pos)> {
        let cur = self.cursor_pos();
        let anc = self.anchor_pos();
        if cur == anc {
            None
        } else if anc < cur {
            Some((anc, cur))
        } else {
            Some((cur, anc))
        }
    }

    /// Move the cursor and either extend the selection (anchor stays)
    /// or collapse it (anchor follows). All keyboard movement and
    /// mouse-driven cursor placement go through this so the selection
    /// model stays consistent.
    fn set_cursor(&mut self, row: usize, col: usize, extend: bool) {
        self.cursor_row = row;
        self.cursor_col = col;
        if !extend {
            self.anchor_row = row;
            self.anchor_col = col;
        }
        self.coalesce = None;
        self.ensure_cursor_visible();
        self.invalidate();
    }

    /// Extract the text inside the current selection, line by line.
    /// Returns an empty string when the selection is empty.
    fn selected_text(&self) -> String {
        let Some((start, end)) = self.selection_range() else {
            return String::new();
        };
        let (sr, sc) = start;
        let (er, ec) = end;
        if sr == er {
            let line = &self.buffer[sr];
            let from = char_to_byte(line, sc);
            let to = char_to_byte(line, ec);
            return line[from..to].to_string();
        }
        let mut out = String::new();
        let first = &self.buffer[sr];
        let from = char_to_byte(first, sc);
        out.push_str(&first[from..]);
        out.push('\n');
        for row in (sr + 1)..er {
            out.push_str(&self.buffer[row]);
            out.push('\n');
        }
        let last = &self.buffer[er];
        let to = char_to_byte(last, ec);
        out.push_str(&last[..to]);
        out
    }

    // ─── Mutation primitives (no undo bookkeeping) ───────────────

    /// Splice `text` into `buffer` at `pos`, returning the position
    /// just after the inserted text. `text` may contain `\n`s —
    /// `\r\n` is normalized to `\n` first.
    fn splice_in(&mut self, pos: Pos, text: &str) -> Pos {
        let normalized = text.replace("\r\n", "\n");
        let (mut row, mut col) = pos;
        if normalized.is_empty() {
            return pos;
        }
        self.tokens_dirty = true;
        self.diagnostics_stale = true;
        let line = &mut self.buffer[row];
        let byte_idx = char_to_byte(line, col);
        let tail = line.split_off(byte_idx);
        // Pieces of `normalized` separated by '\n'.
        let mut pieces = normalized.split('\n');
        // First piece appends to the current line.
        let first = pieces.next().unwrap_or("");
        self.buffer[row].push_str(first);
        col += first.chars().count();
        // Remaining pieces become new lines after `row`.
        let rest: Vec<&str> = pieces.collect();
        if rest.is_empty() {
            // Single-line insert: re-attach the tail.
            self.buffer[row].push_str(&tail);
        } else {
            for (i, piece) in rest.iter().enumerate() {
                row += 1;
                let mut new_line = piece.to_string();
                if i + 1 == rest.len() {
                    // Last piece carries the original tail.
                    new_line.push_str(&tail);
                    col = piece.chars().count();
                }
                self.buffer.insert(row, new_line);
            }
        }
        (row, col)
    }

    /// Remove the text in `[start, end]` and return it. Lines are
    /// joined with `\n`.
    fn splice_out(&mut self, start: Pos, end: Pos) -> String {
        let (sr, sc) = start;
        let (er, ec) = end;
        if start == end {
            return String::new();
        }
        self.tokens_dirty = true;
        self.diagnostics_stale = true;
        if sr == er {
            let line = &mut self.buffer[sr];
            let from = char_to_byte(line, sc);
            let to = char_to_byte(line, ec);
            let removed = line[from..to].to_string();
            line.replace_range(from..to, "");
            return removed;
        }
        // Multi-line: keep the prefix of `sr` and the suffix of `er`,
        // drop everything in between.
        let first = std::mem::take(&mut self.buffer[sr]);
        let from = char_to_byte(&first, sc);
        let (first_keep, first_drop) = first.split_at(from);
        self.buffer[sr] = first_keep.to_string();

        let mut removed = String::new();
        removed.push_str(first_drop);
        removed.push('\n');

        // Drain rows (sr+1)..er, then the prefix of row er.
        for _ in 0..(er - sr - 1) {
            removed.push_str(&self.buffer.remove(sr + 1));
            removed.push('\n');
        }
        let last = self.buffer.remove(sr + 1);
        let to = char_to_byte(&last, ec);
        let (last_drop, last_keep) = last.split_at(to);
        removed.push_str(last_drop);
        // Stitch the suffix of `er` onto the prefix of `sr`.
        self.buffer[sr].push_str(last_keep);
        removed
    }

    /// Apply an insert and push a coalescible-or-fresh `Inserted`
    /// undo entry. `kind` controls coalescing (Insert = single-char
    /// typing, None = paste/newline/etc).
    fn do_insert(&mut self, text: &str, coalesce: Option<CoalesceKind>) {
        // If there's an active selection, replace it (one combined
        // history entry: a Deleted undo plus an Inserted undo).
        if self.selection_range().is_some() {
            self.delete_selection_to_undo();
        }
        let cursor_before = self.cursor_pos();
        let start = cursor_before;
        let end = self.splice_in(start, text);
        self.cursor_row = end.0;
        self.cursor_col = end.1;
        self.anchor_row = end.0;
        self.anchor_col = end.1;
        self.pref_col = self.cursor_col;
        self.dirty = true;
        self.redo.clear();

        let extend_last = coalesce == Some(CoalesceKind::Insert)
            && self.coalesce == Some(CoalesceKind::Insert)
            && matches!(
                self.undo.last(),
                Some(UndoOp::Inserted { end: prev_end, .. }) if *prev_end == start
            );
        if extend_last {
            if let Some(UndoOp::Inserted {
                end: prev_end,
                text: prev_text,
                cursor_after,
                ..
            }) = self.undo.last_mut()
            {
                prev_text.push_str(text);
                *prev_end = end;
                *cursor_after = end;
            }
        } else {
            self.push_undo(UndoOp::Inserted {
                start,
                end,
                text: text.to_string(),
                cursor_before,
                cursor_after: end,
            });
        }
        self.coalesce = coalesce;
        self.ensure_cursor_visible();
        self.invalidate();
    }

    /// Delete the current selection. Pushes a Deleted entry. Caller
    /// is responsible for clearing the redo stack if appropriate.
    fn delete_selection_to_undo(&mut self) -> bool {
        let Some((start, end)) = self.selection_range() else {
            return false;
        };
        let cursor_before = self.cursor_pos();
        let removed = self.splice_out(start, end);
        self.cursor_row = start.0;
        self.cursor_col = start.1;
        self.anchor_row = start.0;
        self.anchor_col = start.1;
        self.pref_col = self.cursor_col;
        self.dirty = true;
        self.push_undo(UndoOp::Deleted {
            start,
            text: removed,
            cursor_before,
            cursor_after: start,
        });
        self.coalesce = None;
        true
    }

    fn push_undo(&mut self, op: UndoOp) {
        self.undo.push(op);
        if self.undo.len() > UNDO_CAP {
            self.undo.remove(0);
        }
    }

    fn undo(&mut self) {
        let Some(op) = self.undo.pop() else { return };
        let restore_cursor: Pos;
        let mirror: UndoOp;
        match op {
            UndoOp::Inserted {
                start,
                end,
                text,
                cursor_before,
                cursor_after,
            } => {
                self.splice_out(start, end);
                restore_cursor = cursor_before;
                mirror = UndoOp::Inserted {
                    start,
                    end,
                    text,
                    cursor_before,
                    cursor_after,
                };
            }
            UndoOp::Deleted {
                start,
                text,
                cursor_before,
                cursor_after,
            } => {
                self.splice_in(start, &text);
                restore_cursor = cursor_before;
                mirror = UndoOp::Deleted {
                    start,
                    text,
                    cursor_before,
                    cursor_after,
                };
            }
        }
        self.cursor_row = restore_cursor.0;
        self.cursor_col = restore_cursor.1;
        self.anchor_row = self.cursor_row;
        self.anchor_col = self.cursor_col;
        self.pref_col = self.cursor_col;
        self.dirty = true;
        self.coalesce = None;
        self.redo.push(mirror);
        self.ensure_cursor_visible();
        self.invalidate();
    }

    // ─── Compile check ───────────────────────────────────────────

    /// Run the installed checker (if any) against the current
    /// buffer. No-op when no checker is installed — keeps redit
    /// useful as a plain editor in environments where the compiler
    /// hasn't been linked in.
    fn run_check(&mut self) {
        let mut text = String::new();
        for (i, line) in self.buffer.iter().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            text.push_str(line);
        }
        match run_checker(&text) {
            Some(diags) => {
                self.diagnostics = diags;
                self.diagnostics_stale = false;
            }
            None => {
                // No checker installed; clear so we don't show stale.
                self.diagnostics.clear();
                self.diagnostics_stale = false;
            }
        }
        self.invalidate();
    }

    /// Move the cursor to the next diagnostic after the current row,
    /// wrapping to the first if we're past the last. F8 binding.
    fn jump_to_next_diagnostic(&mut self) {
        if self.diagnostics.is_empty() {
            return;
        }
        // Diagnostics may be unsorted; find the smallest line > cursor_row+1
        // (1-indexed), else fall back to the smallest overall.
        let cur = self.cursor_row + 1;
        let next = self
            .diagnostics
            .iter()
            .filter(|d| d.line > cur)
            .min_by_key(|d| (d.line, d.column))
            .or_else(|| {
                self.diagnostics
                    .iter()
                    .min_by_key(|d| (d.line, d.column))
            });
        if let Some(d) = next {
            let last = self.buffer.len().saturating_sub(1);
            let r = d.line.saturating_sub(1).min(last);
            let line_chars = self.buffer[r].chars().count();
            let c = d.column.saturating_sub(1).min(line_chars);
            self.set_cursor(r, c, false);
            self.pref_col = self.cursor_col;
        }
    }

    /// First diagnostic on `line_1based`, if any. Used for the
    /// status line and the gutter mark.
    fn diagnostic_on_line(&self, line_1based: usize) -> Option<&Diagnostic> {
        self.diagnostics.iter().find(|d| d.line == line_1based)
    }

    fn redo(&mut self) {
        let Some(op) = self.redo.pop() else { return };
        let after: Pos;
        let mirror: UndoOp;
        match op {
            UndoOp::Inserted {
                start,
                end,
                text,
                cursor_before,
                cursor_after,
            } => {
                self.splice_in(start, &text);
                after = cursor_after;
                mirror = UndoOp::Inserted {
                    start,
                    end,
                    text,
                    cursor_before,
                    cursor_after,
                };
            }
            UndoOp::Deleted {
                start,
                text,
                cursor_before,
                cursor_after,
            } => {
                let n_chars = text.chars().filter(|c| *c != '\n').count();
                let n_lines = text.bytes().filter(|c| *c == b'\n').count();
                let _ = (n_chars, n_lines); // not needed; we compute end from splice_in
                let _end = self.splice_in(start, &text);
                after = cursor_after;
                mirror = UndoOp::Deleted {
                    start,
                    text,
                    cursor_before,
                    cursor_after,
                };
            }
        }
        self.cursor_row = after.0;
        self.cursor_col = after.1;
        self.anchor_row = self.cursor_row;
        self.anchor_col = self.cursor_col;
        self.pref_col = self.cursor_col;
        self.dirty = true;
        self.coalesce = None;
        self.undo.push(mirror);
        self.ensure_cursor_visible();
        self.invalidate();
    }

    fn paint(&mut self) {
        let mut rect = RECT::default();
        if unsafe { GetClientRect(self.hwnd, &mut rect) }.is_err() {
            return;
        }
        let w = (rect.right - rect.left) as u32;
        let h = (rect.bottom - rect.top) as u32;
        if w == 0 || h == 0 {
            return;
        }
        self.client_w = w;
        self.client_h = h;
        self.ensure_resources(w, h);

        let target = match self.target.clone() {
            Some(t) => t,
            None => return,
        };
        let format = match self.text_format.clone() {
            Some(f) => f,
            None => return,
        };

        // The render target's drawing space is in DIPs; convert the
        // pixel dimensions before doing layout math.
        let scale = self.dip_scale();
        let w_dip = (w as f32) * scale;
        let h_dip = (h as f32) * scale;

        unsafe { target.BeginDraw() };

        // Background.
        unsafe {
            target.Clear(Some(&D2D1_COLOR_F {
                r: 0.10,
                g: 0.11,
                b: 0.13,
                a: 1.0,
            }));
        }

        let fg = solid_brush(&target, 0.85, 0.88, 0.85, 1.0);
        let gutter_fg = solid_brush(&target, 0.45, 0.50, 0.55, 1.0);
        let gutter_bg = solid_brush(&target, 0.06, 0.07, 0.09, 1.0);
        let cursor_brush = solid_brush(&target, 0.95, 0.85, 0.40, 1.0);
        let status_bg = solid_brush(&target, 0.16, 0.18, 0.22, 1.0);
        let status_fg = solid_brush(&target, 0.80, 0.83, 0.88, 1.0);
        let sel_brush = solid_brush(&target, 0.20, 0.30, 0.55, 1.0);
        // Syntax-highlighting brushes. Order matches `TokenKind`.
        let kw_brush = solid_brush(&target, 0.55, 0.78, 1.00, 1.0);
        let num_brush = solid_brush(&target, 0.95, 0.70, 0.30, 1.0);
        let str_brush = solid_brush(&target, 0.65, 0.85, 0.55, 1.0);
        let cmt_brush = solid_brush(&target, 0.50, 0.55, 0.60, 1.0);
        // Error gutter mark — bright red. Greyed when stale (the
        // buffer has been edited since the last check).
        let err_brush = if self.diagnostics_stale {
            solid_brush(&target, 0.55, 0.30, 0.30, 1.0)
        } else {
            solid_brush(&target, 0.95, 0.30, 0.25, 1.0)
        };

        // Refresh tokens lazily, before any line is laid out.
        if self.tokens_dirty {
            self.tokens = tokenize_buffer(&self.buffer);
            self.tokens_dirty = false;
        }

        let gutter_chars: f32 = 6.0;
        let gutter_w = gutter_chars * self.cell_w;
        let status_h = self.cell_h + 2.0;
        let content_top = 0.0;
        let content_bottom = h_dip - status_h;
        let visible_rows = ((content_bottom - content_top) / self.cell_h).floor() as usize;

        // Gutter background.
        if let (Some(target), Some(b)) = (Some(&target), gutter_bg.as_ref()) {
            unsafe {
                target.FillRectangle(
                    &D2D_RECT_F {
                        left: 0.0,
                        top: 0.0,
                        right: gutter_w,
                        bottom: content_bottom,
                    },
                    b,
                )
            };
        }

        // Selection rects, drawn under the text glyphs so the text
        // stays fully readable on top. Buffer columns are translated
        // through `buffer_col_to_display` so tabs in the indent line
        // up with the rendered glyphs.
        if let (Some((s_start, s_end)), Some(brush)) =
            (self.selection_range(), sel_brush.as_ref())
        {
            for screen_row in 0..visible_rows {
                let line_idx = self.scroll_top + screen_row;
                if line_idx >= self.buffer.len() {
                    break;
                }
                if line_idx < s_start.0 || line_idx > s_end.0 {
                    continue;
                }
                let line_text = &self.buffer[line_idx];
                let line_chars = line_text.chars().count();
                let from_col = if line_idx == s_start.0 { s_start.1 } else { 0 };
                let to_col = if line_idx == s_end.0 {
                    s_end.1
                } else {
                    // Multi-line selection: paint past end-of-line
                    // out by one cell so the user sees the newline
                    // is part of the selection.
                    line_chars + 1
                };
                let from_display = buffer_col_to_display(line_text, from_col);
                let to_display = if to_col > line_chars {
                    buffer_col_to_display(line_text, line_chars) + 1
                } else {
                    buffer_col_to_display(line_text, to_col)
                };
                let y = content_top + (screen_row as f32) * self.cell_h;
                let x0 = gutter_w + (from_display as f32) * self.cell_w;
                let x1 = gutter_w + (to_display as f32) * self.cell_w;
                unsafe {
                    target.FillRectangle(
                        &D2D_RECT_F {
                            left: x0,
                            top: y,
                            right: x1,
                            bottom: y + self.cell_h,
                        },
                        brush,
                    )
                };
            }
        }

        // Lines.
        for screen_row in 0..visible_rows {
            let line_idx = self.scroll_top + screen_row;
            if line_idx >= self.buffer.len() {
                break;
            }
            let y = content_top + (screen_row as f32) * self.cell_h;

            // Gutter line number.
            let gutter_text = format!("{:>5} ", line_idx + 1);
            if let (Some(brush), Ok(layout)) = (
                gutter_fg.as_ref(),
                build_layout(&format, &gutter_text, gutter_w, self.cell_h),
            ) {
                unsafe {
                    target.DrawTextLayout(
                        windows_numerics::Vector2 { X: 0.0, Y: y },
                        &layout,
                        brush,
                        D2D1_DRAW_TEXT_OPTIONS_CLIP,
                    );
                }
            }

            // Error mark — a small red bar painted at the right edge
            // of the gutter on lines with diagnostics. Position it
            // inside the gutter so it doesn't overlap with text.
            if self.diagnostic_on_line(line_idx + 1).is_some() {
                if let Some(brush) = err_brush.as_ref() {
                    let bar_w = (self.cell_w * 0.4).max(2.0);
                    unsafe {
                        target.FillRectangle(
                            &D2D_RECT_F {
                                left: gutter_w - bar_w - 1.0,
                                top: y + 2.0,
                                right: gutter_w - 1.0,
                                bottom: y + self.cell_h - 2.0,
                            },
                            brush,
                        )
                    };
                }
            }

            // Line content. The layout is built from the
            // tab-expanded form so the cell grid matches the buffer's
            // visual columns; tokens recorded in buffer-char indices
            // are mapped through `buffer_col_to_display` before being
            // applied as drawing effects.
            let line = &self.buffer[line_idx];
            if !line.is_empty() {
                let expanded = expand_line(line);
                let max_w = w_dip - gutter_w;
                if let (Some(brush), Ok(layout)) = (
                    fg.as_ref(),
                    build_layout(&format, &expanded, max_w, self.cell_h),
                ) {
                    if let Some(line_tokens) = self.tokens.get(line_idx) {
                        for tok in line_tokens {
                            let kind_brush = match tok.kind {
                                TokenKind::Keyword => kw_brush.as_ref(),
                                TokenKind::Number => num_brush.as_ref(),
                                TokenKind::StringLit => str_brush.as_ref(),
                                TokenKind::Comment => cmt_brush.as_ref(),
                            };
                            let Some(b) = kind_brush else { continue };
                            let disp_start = buffer_col_to_display(line, tok.start);
                            let disp_end = buffer_col_to_display(line, tok.end);
                            let range = DWRITE_TEXT_RANGE {
                                startPosition: disp_start as u32,
                                length: (disp_end - disp_start) as u32,
                            };
                            let _ = unsafe { layout.SetDrawingEffect(b, range) };
                        }
                    }
                    unsafe {
                        target.DrawTextLayout(
                            windows_numerics::Vector2 { X: gutter_w, Y: y },
                            &layout,
                            brush,
                            D2D1_DRAW_TEXT_OPTIONS_CLIP,
                        );
                    }
                }
            }
        }

        // Cursor. cursor_col is a buffer char index — translate
        // through tab expansion so the bar lines up with the rendered
        // glyph the cursor is sitting before.
        if self.cursor_row >= self.scroll_top
            && self.cursor_row < self.scroll_top + visible_rows
        {
            let screen_row = self.cursor_row - self.scroll_top;
            let line = &self.buffer[self.cursor_row];
            let display_col = buffer_col_to_display(line, self.cursor_col);
            let cx = gutter_w + (display_col as f32) * self.cell_w;
            let cy = content_top + (screen_row as f32) * self.cell_h;
            if let Some(brush) = cursor_brush.as_ref() {
                unsafe {
                    target.FillRectangle(
                        &D2D_RECT_F {
                            left: cx,
                            top: cy,
                            right: cx + 2.0,
                            bottom: cy + self.cell_h,
                        },
                        brush,
                    )
                };
            }
        }

        // Status line.
        if let Some(brush) = status_bg.as_ref() {
            unsafe {
                target.FillRectangle(
                    &D2D_RECT_F {
                        left: 0.0,
                        top: content_bottom,
                        right: w_dip,
                        bottom: h_dip,
                    },
                    brush,
                )
            };
        }
        let path_str = match self.file_path.as_ref() {
            Some(p) => p.display().to_string(),
            None => "<untitled>".to_string(),
        };
        let dirty_mark = if self.dirty { "*" } else { " " };

        // Diagnostic block: if the cursor is sitting on an errored
        // line, prefer that error's message; otherwise show the
        // count. "(stale)" annotates the count when the buffer has
        // changed since the last check.
        let here = self.diagnostic_on_line(self.cursor_row + 1);
        let stale = if self.diagnostics_stale && !self.diagnostics.is_empty() {
            " (stale)"
        } else {
            ""
        };
        let diag_segment = match here {
            Some(d) => format!("⛔ {} ", d.message),
            None => match self.diagnostics.len() {
                0 => "F7 check  ".to_string(),
                1 => format!("1 error{stale}  F8 next  "),
                n => format!("{n} errors{stale}  F8 next  "),
            },
        };

        let status = format!(
            " {dirty} {path}   Ln {row:4}, Col {col:2}   {nlines} lines   {diag}",
            dirty = dirty_mark,
            path = path_str,
            row = self.cursor_row + 1,
            col = self.cursor_col + 1,
            nlines = self.buffer.len(),
            diag = diag_segment,
        );
        if let (Some(brush), Ok(layout)) = (
            status_fg.as_ref(),
            build_layout(&format, &status, w_dip, status_h),
        ) {
            unsafe {
                target.DrawTextLayout(
                    windows_numerics::Vector2 {
                        X: 0.0,
                        Y: content_bottom + 1.0,
                    },
                    &layout,
                    brush,
                    D2D1_DRAW_TEXT_OPTIONS_CLIP,
                );
            }
        }

        let _ = unsafe { target.EndDraw(None, None) };
    }

    // ─── Editing ─────────────────────────────────────────────────

    fn current_line_len(&self) -> usize {
        self.buffer
            .get(self.cursor_row)
            .map(|s| s.chars().count())
            .unwrap_or(0)
    }

    fn ensure_cursor_visible(&mut self) {
        // Recompute visible rows based on last known size. paint()
        // updates client_h on every frame, so this is good after at
        // least one paint. cell_h is in DIPs, so convert client_h
        // from pixels first.
        if self.client_h == 0 || self.cell_h <= 0.0 {
            return;
        }
        let status_h = self.cell_h + 2.0;
        let content_h_dip = (self.client_h as f32) * self.dip_scale() - status_h;
        let visible = (content_h_dip / self.cell_h).floor() as usize;
        if visible == 0 {
            return;
        }
        if self.cursor_row < self.scroll_top {
            self.scroll_top = self.cursor_row;
        } else if self.cursor_row >= self.scroll_top + visible {
            self.scroll_top = self.cursor_row + 1 - visible;
        }
    }

    fn insert_char(&mut self, c: char) {
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        self.do_insert(s, Some(CoalesceKind::Insert));
    }

    fn insert_newline(&mut self) {
        // Newline breaks the typing-coalesce chain so a subsequent
        // single-char insert starts a fresh undo entry.
        self.do_insert("\n", None);
    }

    fn insert_str(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.do_insert(text, None);
    }

    fn backspace(&mut self) {
        // If there's a selection, delete it (single Deleted entry).
        if self.delete_selection_to_undo() {
            self.redo.clear();
            self.ensure_cursor_visible();
            self.invalidate();
            return;
        }
        if self.cursor_col == 0 && self.cursor_row == 0 {
            return;
        }
        let cursor_before = self.cursor_pos();
        let start: Pos = if self.cursor_col > 0 {
            (self.cursor_row, self.cursor_col - 1)
        } else {
            let prev_len = self.buffer[self.cursor_row - 1].chars().count();
            (self.cursor_row - 1, prev_len)
        };
        let end = cursor_before;
        let removed = self.splice_out(start, end);
        self.cursor_row = start.0;
        self.cursor_col = start.1;
        self.anchor_row = start.0;
        self.anchor_col = start.1;
        self.pref_col = self.cursor_col;
        self.dirty = true;
        self.redo.clear();

        let extend_last = self.coalesce == Some(CoalesceKind::Backspace)
            && matches!(
                self.undo.last(),
                Some(UndoOp::Deleted { start: prev_start, .. }) if *prev_start == end
            );
        if extend_last {
            if let Some(UndoOp::Deleted {
                start: prev_start,
                text: prev_text,
                cursor_after,
                ..
            }) = self.undo.last_mut()
            {
                let mut combined = removed;
                combined.push_str(prev_text);
                *prev_text = combined;
                *prev_start = start;
                *cursor_after = start;
            }
        } else {
            self.push_undo(UndoOp::Deleted {
                start,
                text: removed,
                cursor_before,
                cursor_after: start,
            });
        }
        self.coalesce = Some(CoalesceKind::Backspace);
        self.ensure_cursor_visible();
        self.invalidate();
    }

    fn delete_forward(&mut self) {
        if self.delete_selection_to_undo() {
            self.redo.clear();
            self.ensure_cursor_visible();
            self.invalidate();
            return;
        }
        let line_chars = self.current_line_len();
        let start = self.cursor_pos();
        let end: Pos = if self.cursor_col < line_chars {
            (self.cursor_row, self.cursor_col + 1)
        } else if self.cursor_row + 1 < self.buffer.len() {
            (self.cursor_row + 1, 0)
        } else {
            return;
        };
        let cursor_before = start;
        let removed = self.splice_out(start, end);
        self.dirty = true;
        self.redo.clear();
        self.coalesce = None;
        self.push_undo(UndoOp::Deleted {
            start,
            text: removed,
            cursor_before,
            cursor_after: start,
        });
        self.invalidate();
    }

    fn move_left(&mut self, extend: bool) {
        // If there's a selection and we're not extending, collapse to
        // the start (this is what most editors do — left arrow moves
        // to the beginning of the selection).
        if !extend {
            if let Some((start, _)) = self.selection_range() {
                self.set_cursor(start.0, start.1, false);
                self.pref_col = self.cursor_col;
                return;
            }
        }
        let (mut r, mut c) = self.cursor_pos();
        if c > 0 {
            c -= 1;
        } else if r > 0 {
            r -= 1;
            c = self.buffer[r].chars().count();
        }
        self.set_cursor(r, c, extend);
        self.pref_col = self.cursor_col;
    }

    fn move_right(&mut self, extend: bool) {
        if !extend {
            if let Some((_, end)) = self.selection_range() {
                self.set_cursor(end.0, end.1, false);
                self.pref_col = self.cursor_col;
                return;
            }
        }
        let (mut r, mut c) = self.cursor_pos();
        let n = self.buffer[r].chars().count();
        if c < n {
            c += 1;
        } else if r + 1 < self.buffer.len() {
            r += 1;
            c = 0;
        }
        self.set_cursor(r, c, extend);
        self.pref_col = self.cursor_col;
    }

    fn move_up(&mut self, extend: bool) {
        let mut r = self.cursor_row;
        if r == 0 {
            return;
        }
        r -= 1;
        let n = self.buffer[r].chars().count();
        let c = self.pref_col.min(n);
        // Don't reset pref_col across vertical moves — that's the
        // whole point of remembering it.
        let pref = self.pref_col;
        self.set_cursor(r, c, extend);
        self.pref_col = pref;
    }

    fn move_down(&mut self, extend: bool) {
        let mut r = self.cursor_row;
        if r + 1 >= self.buffer.len() {
            return;
        }
        r += 1;
        let n = self.buffer[r].chars().count();
        let c = self.pref_col.min(n);
        let pref = self.pref_col;
        self.set_cursor(r, c, extend);
        self.pref_col = pref;
    }

    fn move_home(&mut self, extend: bool) {
        let r = self.cursor_row;
        self.set_cursor(r, 0, extend);
        self.pref_col = 0;
    }

    fn move_end(&mut self, extend: bool) {
        let r = self.cursor_row;
        let n = self.buffer[r].chars().count();
        self.set_cursor(r, n, extend);
        self.pref_col = self.cursor_col;
    }

    fn page_up(&mut self, extend: bool) {
        let visible = self.visible_rows().max(1);
        let r = self.cursor_row.saturating_sub(visible);
        let n = self.buffer[r].chars().count();
        let c = self.pref_col.min(n);
        let pref = self.pref_col;
        self.set_cursor(r, c, extend);
        self.pref_col = pref;
    }

    fn page_down(&mut self, extend: bool) {
        let visible = self.visible_rows().max(1);
        let last = self.buffer.len().saturating_sub(1);
        let r = (self.cursor_row + visible).min(last);
        let n = self.buffer[r].chars().count();
        let c = self.pref_col.min(n);
        let pref = self.pref_col;
        self.set_cursor(r, c, extend);
        self.pref_col = pref;
    }

    fn select_all(&mut self) {
        let last_row = self.buffer.len().saturating_sub(1);
        let last_col = self.buffer[last_row].chars().count();
        self.anchor_row = 0;
        self.anchor_col = 0;
        self.cursor_row = last_row;
        self.cursor_col = last_col;
        self.pref_col = self.cursor_col;
        self.coalesce = None;
        self.ensure_cursor_visible();
        self.invalidate();
    }

    // ─── Clipboard ───────────────────────────────────────────────

    fn cut(&mut self) {
        if self.selection_range().is_none() {
            return;
        }
        let text = self.selected_text();
        if !clipboard_set(self.hwnd, &text) {
            // If the clipboard write failed, leave the selection
            // alone — the user can retry. Don't push an undo for a
            // half-completed operation.
            return;
        }
        self.delete_selection_to_undo();
        self.redo.clear();
        self.coalesce = None;
        self.ensure_cursor_visible();
        self.invalidate();
    }

    fn copy(&self) {
        if self.selection_range().is_none() {
            return;
        }
        let _ = clipboard_set(self.hwnd, &self.selected_text());
    }

    fn paste(&mut self) {
        let Some(text) = clipboard_get(self.hwnd) else {
            return;
        };
        if text.is_empty() {
            return;
        }
        // Replace selection (handled inside do_insert) with the
        // pasted text. No coalescing for paste.
        self.do_insert(&text, None);
    }

    fn visible_rows(&self) -> usize {
        if self.client_h == 0 || self.cell_h <= 0.0 {
            return 0;
        }
        let status_h = self.cell_h + 2.0;
        let content_h_dip = (self.client_h as f32) * self.dip_scale() - status_h;
        (content_h_dip / self.cell_h).floor() as usize
    }

    /// Translate a client-area pixel point to a logical buffer
    /// position, clamping to the buffer bounds. Mouse coordinates
    /// arrive from Win32 in physical pixels; cell metrics live in
    /// DIPs, so convert before doing the cell math. The display
    /// column the click lands on is then mapped back to a buffer
    /// char index via `display_col_to_buffer`, which handles tab
    /// expansion.
    fn pos_at_pixel(&self, x: i32, y: i32) -> Pos {
        let gutter_w = 6.0 * self.cell_w;
        let x_dip = self.px_to_dip(x);
        let y_dip = self.px_to_dip(y);
        let row_f = if self.cell_h > 0.0 {
            (y_dip / self.cell_h).floor()
        } else {
            0.0
        };
        let row = if row_f < 0.0 {
            self.scroll_top
        } else {
            self.scroll_top + row_f as usize
        };
        let row = row.min(self.buffer.len().saturating_sub(1));
        let cx = x_dip - gutter_w;
        let display_col = if cx <= 0.0 || self.cell_w <= 0.0 {
            0
        } else {
            (cx / self.cell_w).round() as usize
        };
        let line = &self.buffer[row];
        let buf_col = display_col_to_buffer(line, display_col);
        let n = line.chars().count();
        (row, buf_col.min(n))
    }

    fn click(&mut self, x: i32, y: i32, extend: bool) {
        let (r, c) = self.pos_at_pixel(x, y);
        self.set_cursor(r, c, extend);
        self.pref_col = self.cursor_col;
    }

    fn drag_to(&mut self, x: i32, y: i32) {
        if !self.selecting_drag {
            return;
        }
        let (r, c) = self.pos_at_pixel(x, y);
        // While dragging, anchor stays put (it was set on
        // mouse-down), cursor follows the pointer.
        self.cursor_row = r;
        self.cursor_col = c;
        self.pref_col = self.cursor_col;
        self.coalesce = None;
        self.ensure_cursor_visible();
        self.invalidate();
    }

    fn wheel(&mut self, delta: i32) {
        let lines = if WHEEL_DELTA != 0 {
            -(delta / WHEEL_DELTA as i32) * 3
        } else {
            0
        };
        if lines == 0 {
            return;
        }
        let max_top = self.buffer.len().saturating_sub(1);
        let new_top = (self.scroll_top as i32 + lines).max(0) as usize;
        self.scroll_top = new_top.min(max_top);
        self.invalidate();
    }

    // ─── File I/O ────────────────────────────────────────────────

    fn load_from(&mut self, path: PathBuf) {
        match std::fs::read(&path) {
            Ok(bytes) => {
                let text = decode_utf8_lossy_with_bom(&bytes);
                let lines: Vec<String> = if text.is_empty() {
                    vec![String::new()]
                } else {
                    let mut v: Vec<String> =
                        text.split('\n').map(|s| s.trim_end_matches('\r').to_string()).collect();
                    if v.is_empty() {
                        v.push(String::new());
                    }
                    v
                };
                self.buffer = lines;
                self.cursor_row = 0;
                self.cursor_col = 0;
                self.anchor_row = 0;
                self.anchor_col = 0;
                self.pref_col = 0;
                self.scroll_top = 0;
                self.dirty = false;
                self.undo.clear();
                self.redo.clear();
                self.coalesce = None;
                self.tokens_dirty = true;
                self.diagnostics.clear();
                self.diagnostics_stale = true;
                self.file_path = Some(path);
                self.update_title();
                self.invalidate();
            }
            Err(e) => eprintln!("[redit] read {path:?} failed: {e}", path = self.file_path),
        }
    }

    fn save_to(&mut self, path: PathBuf) -> bool {
        let mut text = String::new();
        for (i, line) in self.buffer.iter().enumerate() {
            if i > 0 {
                text.push_str("\r\n");
            }
            text.push_str(line);
        }
        match std::fs::write(&path, text.as_bytes()) {
            Ok(()) => {
                self.file_path = Some(path);
                self.dirty = false;
                self.update_title();
                // Saving is a natural moment to refresh diagnostics:
                // the file the compiler will see now matches the
                // editor buffer, so checker output is meaningful.
                self.run_check();
                self.invalidate();
                true
            }
            Err(e) => {
                eprintln!("[redit] save failed: {e}");
                false
            }
        }
    }

    fn update_title(&self) {
        let name = match self.file_path.as_ref() {
            Some(p) => p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "<untitled>".to_string()),
            None => "<untitled>".to_string(),
        };
        let title = format!("redit — {name}{star}", star = if self.dirty { " *" } else { "" });
        let mut w: Vec<u16> = title.encode_utf16().collect();
        w.push(0);
        unsafe {
            SendMessageW(
                self.hwnd,
                windows::Win32::UI::WindowsAndMessaging::WM_SETTEXT,
                Some(WPARAM(0)),
                Some(LPARAM(w.as_ptr() as isize)),
            )
        };
    }
}

// ─── Win32 plumbing ──────────────────────────────────────────────────

unsafe extern "system" fn redit_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_NCCREATE {
        let state = Box::new(ReditState::new(hwnd));
        let raw = Box::into_raw(state) as isize;
        unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, raw) };
        if let Ok(mut slot) = REDIT_HWND.lock() {
            *slot = Some(hwnd.0 as isize);
        }
        return unsafe { DefMDIChildProcW(hwnd, msg, wparam, lparam) };
    }

    let state_ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut ReditState;
    if state_ptr.is_null() {
        return unsafe { DefMDIChildProcW(hwnd, msg, wparam, lparam) };
    }
    let state = unsafe { &mut *state_ptr };

    match msg {
        WM_PAINT => {
            // Use BeginPaint/EndPaint to satisfy the paint
            // notification, but draw via Direct2D to our HWND target.
            let mut ps = windows::Win32::Graphics::Gdi::PAINTSTRUCT::default();
            let _ = unsafe { windows::Win32::Graphics::Gdi::BeginPaint(hwnd, &mut ps) };
            state.paint();
            let _ = unsafe { windows::Win32::Graphics::Gdi::EndPaint(hwnd, &ps) };
            LRESULT(0)
        }
        WM_SIZE => {
            let w = (lparam.0 & 0xFFFF) as u32;
            let h = ((lparam.0 >> 16) & 0xFFFF) as u32;
            state.client_w = w;
            state.client_h = h;
            if let Some(target) = state.target.as_ref() {
                let _ = unsafe { target.Resize(&D2D_SIZE_U { width: w, height: h }) };
            }
            state.invalidate();
            unsafe { DefMDIChildProcW(hwnd, msg, wparam, lparam) }
        }
        WM_LBUTTONDOWN => {
            let x = (lparam.0 & 0xFFFF) as i16 as i32;
            let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as i32;
            let _ = unsafe { SetFocus(Some(hwnd)) };
            // Shift+click extends, plain click collapses to point.
            state.click(x, y, shift_down());
            // Begin drag selection. Capture so we keep getting moves
            // even if the cursor leaves the client area.
            state.selecting_drag = true;
            unsafe { SetCapture(hwnd) };
            LRESULT(0)
        }
        WM_MOUSEMOVE => {
            // wparam low word holds button state. We only care about
            // dragging while LBUTTON is held; if the user released
            // outside our window we'd otherwise stay in drag mode.
            let buttons = (wparam.0 & 0xFFFF) as u32;
            let lbutton_down = (buttons & MK_LBUTTON) != 0;
            if state.selecting_drag && lbutton_down {
                let x = (lparam.0 & 0xFFFF) as i16 as i32;
                let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as i32;
                state.drag_to(x, y);
            } else if state.selecting_drag && !lbutton_down {
                // We missed the up edge somehow — release capture.
                let _ = unsafe { ReleaseCapture() };
                state.selecting_drag = false;
            }
            LRESULT(0)
        }
        WM_LBUTTONUP => {
            if state.selecting_drag {
                let _ = unsafe { ReleaseCapture() };
                state.selecting_drag = false;
            }
            LRESULT(0)
        }
        WM_SETFOCUS | WM_MDIACTIVATE => {
            state.invalidate();
            unsafe { DefMDIChildProcW(hwnd, msg, wparam, lparam) }
        }
        WM_MOUSEWHEEL => {
            let raw = ((wparam.0 >> 16) & 0xFFFF) as i16;
            state.wheel(raw as i32);
            LRESULT(0)
        }
        WM_KEYDOWN => {
            handle_key(state, wparam.0 as u32);
            LRESULT(0)
        }
        WM_DPICHANGED_AFTERPARENT => {
            let dpi = unsafe { GetDpiForWindow(hwnd) };
            if dpi != 0 {
                state.set_dpi(dpi);
            }
            unsafe { DefMDIChildProcW(hwnd, msg, wparam, lparam) }
        }
        WM_CHAR => {
            let cp = wparam.0 as u32;
            if let Some(c) = char::from_u32(cp) {
                handle_char(state, c);
            }
            LRESULT(0)
        }
        WM_NCDESTROY => {
            // Drop the heap state. Clear singleton slot if it matches.
            let _ = unsafe { Box::from_raw(state_ptr) };
            unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0) };
            if let Ok(mut slot) = REDIT_HWND.lock() {
                if matches!(*slot, Some(h) if h == hwnd.0 as isize) {
                    *slot = None;
                }
            }
            unsafe { DefMDIChildProcW(hwnd, msg, wparam, lparam) }
        }
        _ => unsafe { DefMDIChildProcW(hwnd, msg, wparam, lparam) },
    }
}

fn shift_down() -> bool {
    (unsafe { GetKeyState(VK_SHIFT.0 as i32) } as i16) < 0
}

fn handle_key(state: &mut ReditState, vk: u32) {
    let vk16 = vk as u16;
    let extend = shift_down();
    if vk16 == VK_LEFT.0 {
        state.move_left(extend);
    } else if vk16 == VK_RIGHT.0 {
        state.move_right(extend);
    } else if vk16 == VK_UP.0 {
        state.move_up(extend);
    } else if vk16 == VK_DOWN.0 {
        state.move_down(extend);
    } else if vk16 == VK_HOME.0 {
        state.move_home(extend);
    } else if vk16 == VK_END.0 {
        state.move_end(extend);
    } else if vk16 == VK_PRIOR.0 {
        state.page_up(extend);
    } else if vk16 == VK_NEXT.0 {
        state.page_down(extend);
    } else if vk16 == VK_DELETE.0 {
        // Shift+Delete is a Windows "cut to clipboard" alias. Honor
        // it because users from other editors expect it.
        if shift_down() && state.selection_range().is_some() {
            state.cut();
        } else {
            state.delete_forward();
        }
    } else if vk16 == VK_F7.0 {
        // F7 — run compile check on the current buffer.
        state.run_check();
    } else if vk16 == VK_F8.0 {
        // F8 — jump to next diagnostic.
        state.jump_to_next_diagnostic();
    }
}

fn handle_char(state: &mut ReditState, c: char) {
    // WM_CHAR delivers control codes (Ctrl+A = 0x01, Ctrl+C = 0x03,
    // Ctrl+V = 0x16, ...) before any further processing. We dispatch
    // shortcuts here because by this point Win32 has already mapped
    // modifier state through.
    match c as u32 {
        0x01 => {
            // Ctrl+A — select all.
            state.select_all();
            return;
        }
        0x03 => {
            // Ctrl+C — copy.
            state.copy();
            return;
        }
        0x0F => {
            // Ctrl+O — open.
            if let Some(p) = open_file_dialog(state.hwnd) {
                state.load_from(p);
            }
            return;
        }
        0x13 => {
            // Ctrl+S — save (Shift+S = save as).
            if shift_down() || state.file_path.is_none() {
                if let Some(p) = save_file_dialog(state.hwnd, state.file_path.as_deref()) {
                    state.save_to(p);
                }
            } else if let Some(p) = state.file_path.clone() {
                state.save_to(p);
            }
            return;
        }
        0x16 => {
            // Ctrl+V — paste.
            state.paste();
            return;
        }
        0x18 => {
            // Ctrl+X — cut.
            state.cut();
            return;
        }
        0x19 => {
            // Ctrl+Y — redo.
            state.redo();
            return;
        }
        0x1A => {
            // Ctrl+Z — undo.
            state.undo();
            return;
        }
        _ => {}
    }

    if c == '\r' {
        state.insert_newline();
        return;
    }
    if c == '\n' {
        return;
    }
    if c == '\t' {
        // Soft tab: insert spaces up to the next display tab stop so
        // typed indentation lines up with rendered tabs from
        // existing files. Single insert => one undo entry.
        let line = &state.buffer[state.cursor_row];
        let display_col = buffer_col_to_display(line, state.cursor_col);
        let pad = TAB_WIDTH - (display_col % TAB_WIDTH);
        let spaces: String = std::iter::repeat(' ').take(pad).collect();
        state.insert_str(&spaces);
        return;
    }
    if c == '\u{0008}' {
        state.backspace();
        return;
    }
    if (c as u32) < 0x20 {
        // Suppress other control characters not handled above.
        return;
    }
    state.insert_char(c);
}

// ─── File dialogs ───────────────────────────────────────────────────

fn cp_filter() -> Vec<u16> {
    // Each filter is a pair of NUL-terminated strings, terminated by
    // an extra NUL.
    let raw = "Component Pascal (*.cp)\0*.cp\0Text files (*.txt)\0*.txt\0All files (*.*)\0*.*\0\0";
    raw.encode_utf16().collect()
}

fn open_file_dialog(owner: HWND) -> Option<PathBuf> {
    let mut buf = vec![0u16; 1024];
    let filter = cp_filter();
    let mut ofn = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        hwndOwner: owner,
        lpstrFilter: PCWSTR(filter.as_ptr()),
        nFilterIndex: 1,
        lpstrFile: windows::core::PWSTR(buf.as_mut_ptr()),
        nMaxFile: buf.len() as u32,
        Flags: OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY,
        ..Default::default()
    };
    let ok = unsafe { GetOpenFileNameW(&mut ofn) }.as_bool();
    if !ok {
        return None;
    }
    let n = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    Some(PathBuf::from(OsString::from_wide(&buf[..n])))
}

fn save_file_dialog(owner: HWND, suggested: Option<&std::path::Path>) -> Option<PathBuf> {
    let mut buf = vec![0u16; 1024];
    if let Some(p) = suggested {
        let s: Vec<u16> = p.as_os_str().encode_wide().collect();
        let n = s.len().min(buf.len() - 1);
        buf[..n].copy_from_slice(&s[..n]);
    }
    let filter = cp_filter();
    let mut ofn = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        hwndOwner: owner,
        lpstrFilter: PCWSTR(filter.as_ptr()),
        nFilterIndex: 1,
        lpstrFile: windows::core::PWSTR(buf.as_mut_ptr()),
        nMaxFile: buf.len() as u32,
        Flags: OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT,
        ..Default::default()
    };
    let ok = unsafe { GetSaveFileNameW(&mut ofn) }.as_bool();
    if !ok {
        return None;
    }
    let n = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    Some(PathBuf::from(OsString::from_wide(&buf[..n])))
}

// ─── Helpers ─────────────────────────────────────────────────────────

fn create_text_format() -> Option<IDWriteTextFormat> {
    let factory = &renderer::ctx().dwrite.factory;
    for family in ["Cascadia Mono", "Consolas", "Lucida Console", "Courier New"] {
        let family_w: Vec<u16> = family.encode_utf16().chain(std::iter::once(0)).collect();
        let locale_w: Vec<u16> = "en-us".encode_utf16().chain(std::iter::once(0)).collect();
        let result = unsafe {
            factory.CreateTextFormat(
                PCWSTR(family_w.as_ptr()),
                None,
                DWRITE_FONT_WEIGHT(400),
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                14.0,
                PCWSTR(locale_w.as_ptr()),
            )
        };
        if let Ok(f) = result {
            return Some(f);
        }
    }
    None
}

fn measure_cell(format: &IDWriteTextFormat) -> Option<(f32, f32, f32)> {
    // Lay out a single "M" to learn the cell metrics. Monospaced fonts
    // give equal advance for every character, so this is enough.
    let factory = &renderer::ctx().dwrite.factory;
    let text: Vec<u16> = "M".encode_utf16().collect();
    let layout = unsafe {
        factory.CreateTextLayout(&text, format, 1024.0, 1024.0)
    }
    .ok()?;
    let mut metrics = DWRITE_TEXT_METRICS::default();
    if unsafe { layout.GetMetrics(&mut metrics) }.is_err() {
        return None;
    }
    let mut line_metrics =
        [windows::Win32::Graphics::DirectWrite::DWRITE_LINE_METRICS::default(); 1];
    let mut actual: u32 = 0;
    let _ = unsafe { layout.GetLineMetrics(Some(&mut line_metrics), &mut actual) };
    let ascent = if actual > 0 {
        line_metrics[0].baseline
    } else {
        metrics.height * 0.8
    };
    Some((metrics.widthIncludingTrailingWhitespace, metrics.height, ascent))
}

fn build_layout(
    format: &IDWriteTextFormat,
    text: &str,
    max_w: f32,
    max_h: f32,
) -> Result<IDWriteTextLayout, windows::core::Error> {
    let factory = &renderer::ctx().dwrite.factory;
    let text_w: Vec<u16> = text.encode_utf16().collect();
    let layout =
        unsafe { factory.CreateTextLayout(&text_w, format, max_w.max(1.0), max_h.max(1.0)) }?;
    // For an editor we never want lines to wrap — long content should
    // be horizontally clipped at the right edge of the content area,
    // not wrap into the next row's slot. The DrawTextLayout call site
    // pairs this with `D2D1_DRAW_TEXT_OPTIONS_CLIP` so overflow gets
    // glyph-clipped rather than bleeding past `max_w`.
    unsafe { layout.SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP) }?;
    Ok(layout)
}

fn solid_brush(
    target: &ID2D1HwndRenderTarget,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) -> Option<ID2D1SolidColorBrush> {
    let color = D2D1_COLOR_F { r, g, b, a };
    let props = D2D1_BRUSH_PROPERTIES {
        opacity: 1.0,
        transform: windows_numerics::Matrix3x2 {
            M11: 1.0,
            M12: 0.0,
            M21: 0.0,
            M22: 1.0,
            M31: 0.0,
            M32: 0.0,
        },
    };
    unsafe { target.CreateSolidColorBrush(&color, Some(&props)) }.ok()
}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or_else(|| s.len())
}

/// Expand `\t` characters in `line` to spaces, padding to the next
/// `TAB_WIDTH` boundary on each tab. Returns the visual line we feed
/// into DirectWrite so the fixed cell grid actually lines up.
fn expand_line(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut col = 0usize;
    for c in line.chars() {
        if c == '\t' {
            let pad = TAB_WIDTH - (col % TAB_WIDTH);
            for _ in 0..pad {
                out.push(' ');
            }
            col += pad;
        } else {
            out.push(c);
            col += 1;
        }
    }
    out
}

/// Translate a buffer char index into its on-screen cell column,
/// expanding tabs the same way `expand_line` does. Used for cursor
/// drawing, selection rect endpoints, and token range mapping.
fn buffer_col_to_display(line: &str, char_col: usize) -> usize {
    let mut col = 0usize;
    for (i, c) in line.chars().enumerate() {
        if i == char_col {
            return col;
        }
        if c == '\t' {
            let pad = TAB_WIDTH - (col % TAB_WIDTH);
            col += pad;
        } else {
            col += 1;
        }
    }
    col
}

/// Inverse of `buffer_col_to_display`: given a display column (e.g.
/// from a mouse click), find the buffer char index that lands
/// closest. Used when translating mouse events into cursor moves.
fn display_col_to_buffer(line: &str, display_col: usize) -> usize {
    let mut col = 0usize;
    for (i, c) in line.chars().enumerate() {
        if col >= display_col {
            return i;
        }
        if c == '\t' {
            let pad = TAB_WIDTH - (col % TAB_WIDTH);
            // If the click landed inside a tab, snap to the closer
            // edge — keeps clicks on indented lines feel natural.
            if col + pad > display_col {
                let mid = col + pad / 2;
                return if display_col <= mid { i } else { i + 1 };
            }
            col += pad;
        } else {
            col += 1;
        }
    }
    line.chars().count()
}

fn decode_utf8_lossy_with_bom(bytes: &[u8]) -> String {
    let stripped = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    String::from_utf8_lossy(stripped).into_owned()
}

// ─── Component Pascal tokenizer (R3 syntax highlighting) ───────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum TokenKind {
    Keyword,
    Number,
    StringLit,
    Comment,
}

#[derive(Clone, Debug)]
struct Token {
    /// Inclusive char index within the line.
    start: usize,
    /// Exclusive char index within the line.
    end: usize,
    kind: TokenKind,
}

/// All Component Pascal reserved words plus the standard built-in
/// type names. We treat them all the same — distinguishing types
/// (`INTEGER`, `REAL`, etc.) from keywords would mean baking the
/// full standard prelude in here for marginal aesthetic gain.
const CP_KEYWORDS: &[&str] = &[
    "ABSTRACT", "ARRAY", "BEGIN", "BY", "CASE", "CONST", "DIV", "DO", "ELSE", "ELSIF", "EMPTY",
    "END", "EXIT", "EXTENSIBLE", "FALSE", "FOR", "IF", "IMPORT", "IN", "IS", "LIMITED", "LOOP",
    "MOD", "MODULE", "NEW", "NIL", "OF", "OR", "OUT", "POINTER", "PROCEDURE", "RECORD", "REPEAT",
    "RETURN", "THEN", "TO", "TRUE", "TYPE", "UNTIL", "VAR", "WHILE", "WITH",
    // Built-in types / common pseudo-keywords
    "ANYPTR", "ANYREC", "BOOLEAN", "BYTE", "CHAR", "INTEGER", "INTSHORT", "LONGINT", "REAL",
    "SET", "SHORTCHAR", "SHORTINT", "SHORTREAL",
];

fn is_cp_keyword(word: &str) -> bool {
    // Linear scan — array is small (~50) and this runs once per
    // identifier. Fine for the file sizes we care about.
    CP_KEYWORDS.iter().any(|k| *k == word)
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_cont(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Tokenize one line, given the comment-nesting depth carried in
/// from the previous line. Returns the produced tokens and the
/// depth to feed into the next line.
fn tokenize_line(line: &str, depth_in: u32) -> (Vec<Token>, u32) {
    let chars: Vec<char> = line.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let mut depth = depth_in;

    // If we entered this line already inside a comment, scan for
    // matching `*)` first.
    if depth > 0 {
        let start = 0;
        while i < chars.len() && depth > 0 {
            if i + 1 < chars.len() && chars[i] == '(' && chars[i + 1] == '*' {
                depth += 1;
                i += 2;
            } else if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == ')' {
                depth -= 1;
                i += 2;
            } else {
                i += 1;
            }
        }
        tokens.push(Token {
            start,
            end: i,
            kind: TokenKind::Comment,
        });
    }

    while i < chars.len() {
        let c = chars[i];

        // Comment opener.
        if i + 1 < chars.len() && c == '(' && chars[i + 1] == '*' {
            let start = i;
            depth = 1;
            i += 2;
            while i < chars.len() && depth > 0 {
                if i + 1 < chars.len() && chars[i] == '(' && chars[i + 1] == '*' {
                    depth += 1;
                    i += 2;
                } else if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == ')' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            tokens.push(Token {
                start,
                end: i,
                kind: TokenKind::Comment,
            });
            continue;
        }

        // String / char literal — single line, no escapes (CP syntax).
        if c == '"' || c == '\'' {
            let quote = c;
            let start = i;
            i += 1;
            while i < chars.len() && chars[i] != quote {
                i += 1;
            }
            if i < chars.len() {
                i += 1; // consume the closing quote
            }
            tokens.push(Token {
                start,
                end: i,
                kind: TokenKind::StringLit,
            });
            continue;
        }

        // Number — decimal, hex (with H suffix), real (with E exponent),
        // char literal (with X suffix). CP-style: 0FFH, 1.5E2, 41X.
        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || chars[i] == '.' || chars[i] == '+'
                    || chars[i] == '-')
            {
                // Stop +/- only if it follows an E or D (the exponent
                // sign); otherwise it's an operator and we should
                // leave it for the next iteration.
                if (chars[i] == '+' || chars[i] == '-') && i > start {
                    let prev = chars[i - 1];
                    if !(prev == 'E' || prev == 'D' || prev == 'e' || prev == 'd') {
                        break;
                    }
                }
                i += 1;
            }
            tokens.push(Token {
                start,
                end: i,
                kind: TokenKind::Number,
            });
            continue;
        }

        // Identifier or keyword.
        if is_ident_start(c) {
            let start = i;
            i += 1;
            while i < chars.len() && is_ident_cont(chars[i]) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            if is_cp_keyword(&word) {
                tokens.push(Token {
                    start,
                    end: i,
                    kind: TokenKind::Keyword,
                });
            }
            // Plain identifiers stay default-colored — no token emitted.
            continue;
        }

        // Anything else (operators, whitespace, punctuation) is left
        // unstyled.
        i += 1;
    }

    (tokens, depth)
}

/// Tokenize the whole buffer. Comment depth is threaded through
/// lines so a `(* ... \n ... *)` block is one comment.
fn tokenize_buffer(buffer: &[String]) -> Vec<Vec<Token>> {
    let mut out = Vec::with_capacity(buffer.len());
    let mut depth: u32 = 0;
    for line in buffer {
        let (tokens, next_depth) = tokenize_line(line, depth);
        out.push(tokens);
        depth = next_depth;
    }
    out
}

// ─── Clipboard helpers ──────────────────────────────────────────────

/// Write `text` to the system clipboard as `CF_UNICODETEXT`. Returns
/// `true` on success. Each line is normalized to `\r\n` per Win32
/// convention so other apps see expected line endings.
fn clipboard_set(owner: HWND, text: &str) -> bool {
    let normalized = text.replace("\r\n", "\n").replace('\n', "\r\n");
    let mut wide: Vec<u16> = normalized.encode_utf16().collect();
    wide.push(0);
    let bytes = wide.len() * std::mem::size_of::<u16>();

    if unsafe { OpenClipboard(Some(owner)) }.is_err() {
        return false;
    }
    let mut ok = false;
    unsafe {
        let _ = EmptyClipboard();
        match GlobalAlloc(GMEM_MOVEABLE, bytes) {
            Ok(handle) => {
                let p = GlobalLock(handle) as *mut u16;
                if !p.is_null() {
                    std::ptr::copy_nonoverlapping(wide.as_ptr(), p, wide.len());
                    let _ = GlobalUnlock(handle);
                    let h = windows::Win32::Foundation::HANDLE(handle.0);
                    if SetClipboardData(CF_UNICODETEXT.0 as u32, Some(h)).is_ok() {
                        ok = true;
                    }
                }
            }
            Err(e) => eprintln!("[redit] GlobalAlloc failed: {e}"),
        }
        let _ = CloseClipboard();
    }
    ok
}

/// Read `CF_UNICODETEXT` from the clipboard. Returns `None` if no
/// such format is available or any step fails.
fn clipboard_get(owner: HWND) -> Option<String> {
    if unsafe { OpenClipboard(Some(owner)) }.is_err() {
        return None;
    }
    let result = unsafe {
        match GetClipboardData(CF_UNICODETEXT.0 as u32) {
            Ok(handle) => {
                let g = windows::Win32::Foundation::HGLOBAL(handle.0);
                let p = GlobalLock(g) as *const u16;
                if p.is_null() {
                    None
                } else {
                    // Walk to the NUL terminator (max 16 MiB to be
                    // defensive against malformed clipboard data).
                    let mut len = 0usize;
                    let cap = 16 * 1024 * 1024;
                    while len < cap && *p.add(len) != 0 {
                        len += 1;
                    }
                    let slice = std::slice::from_raw_parts(p, len);
                    let s = String::from_utf16_lossy(slice);
                    let _ = GlobalUnlock(g);
                    Some(s.replace("\r\n", "\n"))
                }
            }
            Err(_) => None,
        }
    };
    unsafe {
        let _ = CloseClipboard();
    }
    result
}
