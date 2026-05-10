//! Event mailbox: GUI thread → language thread.
//!
//! A bounded MPSC queue carrying typed `IGuiEvent` values. Producers
//! are Win32 message handlers on the GUI thread (and, later, the
//! surface executor when it answers synchronous queries). Consumer
//! is the language thread, which calls `next_event` from
//! `iGui.NextEvent`.

#![cfg(windows)]

use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Stable enum tags exported to CP as `iGui.Ev*` constants.
pub mod kind {
    pub const NONE: i64 = 0;
    pub const KEY: i64 = 1;
    pub const CHAR: i64 = 2;
    pub const MOUSE: i64 = 3;
    pub const FOCUS: i64 = 4;
    pub const RESIZE: i64 = 5;
    pub const PAINT: i64 = 6;
    pub const CLOSE: i64 = 7;
    pub const FRAME_CLOSE: i64 = 8;
    pub const MENU: i64 = 9;
    pub const THEME_CHANGE: i64 = 10;
    pub const DPI_CHANGE: i64 = 11;
    pub const SURFACE_REPLY: i64 = 12;
    pub const TICK: i64 = 13;
}

/// Mouse-event sub-kinds packed into the `mouse_op` field. Each is a
/// distinct value (not a bitmask) so the language side can match
/// directly.
pub mod mouse_op {
    pub const MOVE: i64 = 0;
    pub const LEFT_DOWN: i64 = 1;
    pub const LEFT_UP: i64 = 2;
    pub const RIGHT_DOWN: i64 = 3;
    pub const RIGHT_UP: i64 = 4;
    pub const MIDDLE_DOWN: i64 = 5;
    pub const MIDDLE_UP: i64 = 6;
    pub const WHEEL: i64 = 7;
}

/// Modifier-key bits as a packed `i64`. Matches Win32 GetKeyState bit
/// layout where convenient; CP code reads the named bits via
/// `iGui.Mod*` constants.
pub mod modifier {
    pub const SHIFT: i64 = 1 << 0;
    pub const CONTROL: i64 = 1 << 1;
    pub const ALT: i64 = 1 << 2;
    pub const WIN: i64 = 1 << 3;
    pub const CAPS: i64 = 1 << 4;
}

/// All input and lifecycle events flow as one of these structs.
/// Specialised carriers per kind keep the variant fields self-describing
/// without a tagged-union ABI on the wire.
#[derive(Debug, Clone)]
pub enum IGuiEvent {
    Key {
        child_id: i64,
        vkey: i64,
        scancode: i64,
        mods: i64,
        repeat: i64,
        down: bool,
        time_ms: i64,
    },
    Char {
        child_id: i64,
        codepoint: i64,
        mods: i64,
        time_ms: i64,
    },
    Mouse {
        child_id: i64,
        x: i64,
        y: i64,
        op: i64, // mouse_op::*
        button: i64,
        mods: i64,
        wheel_delta: i64,
        wheel_lines: i64,
        time_ms: i64,
    },
    Focus {
        child_id: i64,
        gained: bool,
    },
    Resize {
        child_id: i64,
        width: i64,
        height: i64,
    },
    Close {
        child_id: i64,
    },
    FrameClose,
    ThemeChange,
    DpiChange {
        child_id: i64,
        dpi_x: i64, // ×100 (e.g. 192 means 192 dpi; ×100 reserves room for fractional later)
        dpi_y: i64,
    },
    Menu {
        menu_id: i64,
        item_id: i64,
    },
    /// Animation tick. Fires from a Win32 timer running on a child's
    /// render host; Win32 auto-coalesces queued WM_TIMERs so the
    /// language thread sees at most one tick per child per drain
    /// cycle even if it lags.
    Tick {
        child_id: i64,
        time_ms: i64,
    },
}

struct Mailbox {
    tx: SyncSender<IGuiEvent>,
    rx: Mutex<Receiver<IGuiEvent>>,
}

const CAPACITY: usize = 1024;

static MAILBOX: OnceLock<Mailbox> = OnceLock::new();

pub fn install() {
    MAILBOX.get_or_init(|| {
        let (tx, rx) = sync_channel(CAPACITY);
        Mailbox {
            tx,
            rx: Mutex::new(rx),
        }
    });
}

/// Push from the GUI thread. If the queue is full, drop the new event
/// and log; spamming during a wedged language thread should not block
/// the message pump.
pub fn push(ev: IGuiEvent) {
    let Some(mb) = MAILBOX.get() else {
        return;
    };
    match mb.tx.try_send(ev) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            // Dropping is correct: the GUI thread cannot block on the
            // language thread, and a stalled consumer means whatever
            // we just lost is the least of the user's problems.
            eprintln!("[igui] event mailbox full, dropping event");
        }
        Err(TrySendError::Disconnected(_)) => {
            // Receiver gone; mailbox is being torn down. Silently ignore.
        }
    }
}

/// Pop from the language thread. `timeout_ms < 0` blocks indefinitely.
pub fn next_event(timeout_ms: i64) -> Option<IGuiEvent> {
    let mb = MAILBOX.get()?;
    let rx = mb.rx.lock().ok()?;
    if timeout_ms < 0 {
        rx.recv().ok()
    } else {
        rx.recv_timeout(Duration::from_millis(timeout_ms as u64))
            .ok()
    }
}
