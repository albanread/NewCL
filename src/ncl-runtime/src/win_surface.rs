//! Windows surface bootstrap. Phase 1 of `docs/WINDOWS_FFI.md`.
//!
//! The driver decides at startup whether the Windows surface is on
//! (the `--windows` flag). If it is:
//!
//!   * thread 0 (the process main) becomes the UI thread; it runs a
//!     Win32 message pump via `run_message_pump`
//!   * a worker thread runs the Lisp eval / REPL — what `main()`
//!     does today
//!   * `(windows-enabled-p)` returns T from Lisp, opening up the
//!     conditional `(when (windows-enabled-p) (require 'win32-threading))`
//!     branch in init.lisp
//!
//! If it isn't:
//!
//!   * thread 0 runs Lisp directly (today's behaviour)
//!   * `(windows-enabled-p)` returns NIL
//!   * the UI-thread predicates return NIL too
//!
//! All state here is process-global `OnceLock`. The driver sets the
//! flag once, very early, before any worker can race; everything
//! else reads.
//!
//! Phase 2 will add the hidden HWND_MESSAGE dispatcher window, the
//! `WM_NCL_EXECUTE` arm, and `(on-ui-thread …)`. Phase 3 layers the
//! libffi-based `%ffi-call` on top of the same dispatcher.

use std::sync::OnceLock;

use crate::mutator::MutatorState;
use crate::word::Word;

#[cfg(windows)]
use windows::Win32::Foundation::{LPARAM, WPARAM};
#[cfg(windows)]
use windows::Win32::System::Threading::GetCurrentThreadId;
#[cfg(windows)]
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetMessageW, PostThreadMessageW, TranslateMessage, MSG, WM_QUIT,
};

// ─── Global flags ─────────────────────────────────────────────────────

/// True iff the driver was started with `--windows`. Set once by
/// `set_windows_enabled` before the Lisp worker boots. Defaults to
/// false (i.e. unset OnceLock reads as false) so embedders and tests
/// that don't go through the driver get the conservative answer.
static WINDOWS_ENABLED: OnceLock<bool> = OnceLock::new();

/// The OS thread ID of the UI thread (thread 0 in `--windows` mode).
/// Set once by `register_ui_thread` from thread 0 itself. None
/// outside `--windows` mode.
static UI_THREAD_ID: OnceLock<u32> = OnceLock::new();

/// Idempotently set the "Windows surface enabled" flag. Called once
/// by the driver, before any worker thread is spawned. Subsequent
/// calls are silently ignored (OnceLock semantics).
pub fn set_windows_enabled(enabled: bool) {
    let _ = WINDOWS_ENABLED.set(enabled);
}

/// True iff the Windows surface is active for this process.
pub fn windows_enabled() -> bool {
    WINDOWS_ENABLED.get().copied().unwrap_or(false)
}

// ─── UI thread identity ───────────────────────────────────────────────

#[cfg(windows)]
mod platform {
    use super::*;

    /// Record the calling thread as the UI (dispatcher) thread.
    /// Driver-only; called from thread 0 immediately after deciding
    /// `--windows` is on, before spawning the Lisp worker.
    pub fn register_ui_thread() {
        let tid = unsafe { GetCurrentThreadId() };
        let _ = UI_THREAD_ID.set(tid);
    }

    /// The OS thread ID of the UI thread, or None if not registered.
    pub fn ui_thread_id() -> Option<u32> {
        UI_THREAD_ID.get().copied()
    }

    /// True iff the calling thread IS the UI thread. False if no UI
    /// thread is registered (i.e. `--windows` wasn't passed).
    pub fn am_i_ui_thread() -> bool {
        match ui_thread_id() {
            Some(want) => (unsafe { GetCurrentThreadId() }) == want,
            None => false,
        }
    }

    /// Run the Win32 message pump on the calling thread. Blocks
    /// until a `WM_QUIT` arrives (typically posted by the worker
    /// thread via `post_quit_to_ui_thread` when Lisp finishes).
    ///
    /// `GetMessageW(hWnd=NULL)` retrieves thread messages too (those
    /// posted by `PostThreadMessageW`), so we don't need a dispatch
    /// HWND for Phase 1 — Phase 2 adds one.
    pub fn run_message_pump() {
        let mut msg = MSG::default();
        unsafe {
            loop {
                let r = GetMessageW(&mut msg, None, 0, 0);
                if r.0 <= 0 {
                    // 0 = WM_QUIT, -1 = error. Either way we're done.
                    break;
                }
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    /// Post WM_QUIT to the UI thread, unblocking its message pump.
    /// Worker thread calls this when it's done with all Lisp work.
    /// No-op if no UI thread is registered.
    pub fn post_quit_to_ui_thread() {
        if let Some(tid) = ui_thread_id() {
            unsafe {
                let _ = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0));
            }
        }
    }
}

// On non-Windows platforms (we're Windows-only today, but the
// runtime crate still has to build on other hosts for tooling) the
// surface is a no-op.
#[cfg(not(windows))]
mod platform {
    pub fn register_ui_thread() {}
    pub fn ui_thread_id() -> Option<u32> { None }
    pub fn am_i_ui_thread() -> bool { false }
    pub fn run_message_pump() {
        panic!("run_message_pump called on a non-Windows platform");
    }
    pub fn post_quit_to_ui_thread() {}
}

pub use platform::{
    am_i_ui_thread, post_quit_to_ui_thread, register_ui_thread, run_message_pump, ui_thread_id,
};

// ─── Lisp shims ───────────────────────────────────────────────────────
//
// All three are installed unconditionally — they're tiny and they
// return the truth (NIL) when the surface is off. That keeps Lisp
// code uniformly able to ask `(windows-enabled-p)` and branch on it
// in init.lisp without an "is this shim installed?" dance.

/// `(windows-enabled-p) → t / nil`
pub extern "C-unwind" fn windows_enabled_p_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    if windows_enabled() {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

/// `(ui-thread-id) → fixnum / nil`
pub extern "C-unwind" fn ui_thread_id_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    match ui_thread_id() {
        Some(tid) => Word::fixnum(tid as i64).raw(),
        None => Word::NIL.raw(),
    }
}

/// `(ui-thread-p) → t / nil`
pub extern "C-unwind" fn ui_thread_p_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    if am_i_ui_thread() {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_enabled_defaults_false() {
        // OnceLock — if any earlier test in this binary set it true,
        // we can't reset. So this only checks the unset case is
        // false. The actual reset semantics are tested by running
        // `cargo test --test ...` in a fresh process.
        if WINDOWS_ENABLED.get().is_none() {
            assert!(!windows_enabled());
        }
    }

    #[test]
    fn ui_thread_id_unset_means_none() {
        if UI_THREAD_ID.get().is_none() {
            assert_eq!(ui_thread_id(), None);
            assert!(!am_i_ui_thread());
        }
    }
}
