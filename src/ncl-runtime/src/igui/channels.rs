//! Event mailbox: GUI thread → language thread.
//!
//! A bounded MPSC queue carrying typed `IGuiEvent` values. Producers
//! are Win32 message handlers on the GUI thread (and, later, the
//! surface executor when it answers synchronous queries). Consumer
//! is the language thread, which calls `next_event` from
//! `iGui.NextEvent`.

#![cfg(windows)]

use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

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
    pub const EVAL_BUFFER: i64 = 14;
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
    /// "Evaluate this Lisp source." Fired when the user hits Ctrl+R
    /// inside the ledit (Lisp editor) pane. The pane snapshots its
    /// buffer (or current selection) and pushes the text as
    /// `source`. The language thread evaluates it via the
    /// active session and, if the printed result fits on one
    /// line, writes that line to the iGui log overlay. The
    /// event-loop macros in Library/events.lisp dispatch this
    /// automatically so every iGui app gets the shortcut.
    EvalBuffer {
        source: String,
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

// ─── Per-window event filtering / stash ────────────────────────────────
//
// Pattern ported from NewBCPL. Lisp programs typically open one child
// window and only care about its events plus globals like FrameClose.
// Without filtering, events for OTHER children (e.g. an embedded log
// view or REPL pane) pollute the loop and force every clause to write
// (= (getf ev :child-id) win).
//
// The runtime now offers two filtering shapes:
//
//   * `next_event_for(target, timeout)` — one-shot. Block until an
//     event matches target (or is a global). Non-matches park in
//     the stash so they're not lost.
//
//   * `filter_on_window(child_id)` / `unfilter_window` /
//     `clear_filter` — persistent. Any `next_event` calls while the
//     filter set is non-empty only return matching events; the rest
//     stash. The language thread typically sets up the filter on
//     start and clears it on shutdown.

/// Stash of events that arrived but didn't match the current
/// consumer's interest. Drained ahead of the channel by every
/// consumer.
static EVENT_STASH: Mutex<VecDeque<IGuiEvent>> = Mutex::new(VecDeque::new());

/// Persistent set of windows the language thread is interested in.
/// Initialised on first access.
static EVENT_FILTER: OnceLock<Mutex<HashSet<i64>>> = OnceLock::new();

fn filter_lock() -> &'static Mutex<HashSet<i64>> {
    EVENT_FILTER.get_or_init(|| Mutex::new(HashSet::new()))
}

pub fn filter_on_window(child_id: i64) {
    if let Ok(mut filter) = filter_lock().lock() {
        filter.insert(child_id);
    }
}

pub fn unfilter_window(child_id: i64) {
    if let Ok(mut filter) = filter_lock().lock() {
        filter.remove(&child_id);
    }
}

pub fn clear_filter() {
    if let Ok(mut filter) = filter_lock().lock() {
        filter.clear();
    }
}

pub fn discard_stashed_events() {
    if let Ok(mut stash) = EVENT_STASH.lock() {
        stash.clear();
    }
}

/// Does `ev` match the registered-interest set? Used by `next_event`
/// when the filter is non-empty.
fn matches_filter(ev: &IGuiEvent, filter: &HashSet<i64>) -> bool {
    match ev {
        IGuiEvent::FrameClose | IGuiEvent::ThemeChange | IGuiEvent::EvalBuffer { .. } => true,
        IGuiEvent::Menu { .. } => true,
        IGuiEvent::Key { child_id, .. }
        | IGuiEvent::Char { child_id, .. }
        | IGuiEvent::Mouse { child_id, .. }
        | IGuiEvent::Focus { child_id, .. }
        | IGuiEvent::Resize { child_id, .. }
        | IGuiEvent::Close { child_id }
        | IGuiEvent::DpiChange { child_id, .. }
        | IGuiEvent::Tick { child_id, .. } => filter.contains(child_id),
    }
}

/// Does `ev` belong to the consumer that asked for `target`?
/// Per-window events match when their `child_id` == `target`. Global
/// events (FrameClose, ThemeChange, Menu) match every target.
fn matches_target(ev: &IGuiEvent, target: i64) -> bool {
    match ev {
        IGuiEvent::FrameClose | IGuiEvent::ThemeChange | IGuiEvent::EvalBuffer { .. } => true,
        IGuiEvent::Menu { .. } => true,
        IGuiEvent::Key { child_id, .. }
        | IGuiEvent::Char { child_id, .. }
        | IGuiEvent::Mouse { child_id, .. }
        | IGuiEvent::Focus { child_id, .. }
        | IGuiEvent::Resize { child_id, .. }
        | IGuiEvent::Close { child_id }
        | IGuiEvent::DpiChange { child_id, .. }
        | IGuiEvent::Tick { child_id, .. } => *child_id == target,
    }
}

/// Pop from the language thread. `timeout_ms < 0` blocks indefinitely.
///
/// Semantics depend on the EVENT_FILTER:
///   * filter empty:     return next event (stash, then channel)
///   * filter non-empty: return next event matching the set; non-
///                       matches park in stash.
pub fn next_event(timeout_ms: i64) -> Option<IGuiEvent> {
    let filter_snapshot: Option<HashSet<i64>> = filter_lock()
        .lock()
        .ok()
        .map(|f| f.clone())
        .filter(|f| !f.is_empty());

    // Drain stash first.
    {
        let mut stash = EVENT_STASH.lock().expect("EVENT_STASH poisoned");
        match &filter_snapshot {
            None => {
                if let Some(ev) = stash.pop_front() {
                    return Some(ev);
                }
            }
            Some(filter) => {
                for i in 0..stash.len() {
                    if matches_filter(&stash[i], filter) {
                        return stash.remove(i);
                    }
                }
            }
        }
    }

    let mb = MAILBOX.get()?;
    let rx = mb.rx.lock().ok()?;
    let deadline = if timeout_ms < 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_millis(timeout_ms as u64))
    };
    loop {
        let ev = match deadline {
            None => rx.recv().ok()?,
            Some(deadline) => {
                let now = Instant::now();
                if now >= deadline {
                    return None;
                }
                rx.recv_timeout(deadline - now).ok()?
            }
        };
        match &filter_snapshot {
            None => {
                return Some(ev);
            }
            Some(filter) => {
                if matches_filter(&ev, filter) {
                    return Some(ev);
                }
                if let Ok(mut stash) = EVENT_STASH.lock() {
                    stash.push_back(ev);
                }
            }
        }
    }
}

/// Block until an event matches `target` (or is a global). Non-
/// matches park into EVENT_STASH for later consumers.
/// `timeout_ms < 0` blocks indefinitely; the deadline is wall-clock,
/// not reset by non-matches.
pub fn next_event_for(target: i64, timeout_ms: i64) -> Option<IGuiEvent> {
    if let Ok(mut stash) = EVENT_STASH.lock() {
        for i in 0..stash.len() {
            if matches_target(&stash[i], target) {
                return stash.remove(i);
            }
        }
    }
    let mb = MAILBOX.get()?;
    let rx = mb.rx.lock().ok()?;
    let deadline = if timeout_ms < 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_millis(timeout_ms as u64))
    };
    loop {
        let ev = match deadline {
            None => rx.recv().ok()?,
            Some(deadline) => {
                let now = Instant::now();
                if now >= deadline {
                    return None;
                }
                rx.recv_timeout(deadline - now).ok()?
            }
        };
        if matches_target(&ev, target) {
            return Some(ev);
        }
        if let Ok(mut stash) = EVENT_STASH.lock() {
            stash.push_back(ev);
        }
    }
}
