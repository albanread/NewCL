//! Roger Corman's THREADS package, ported to a cross-platform Rust
//! foundation. Every Lisp thread here is an OS thread spawned by
//! `std::thread::spawn`; we hand out monotonically increasing
//! integer IDs and stash the `JoinHandle` in a global registry.
//!
//! Cross-platform notes:
//!
//!  * `create-thread` uses `std::thread::Builder` — works on every
//!    platform Rust supports.
//!  * `suspend-thread` / `resume-thread` are **cooperative**, not
//!    `SuspendThread`-style hard pre-emption. The target thread
//!    parks at its next safepoint. The driver emits safepoints at
//!    function-call boundaries (eventually — for v1 the user can
//!    insert `(thread-safepoint)` calls explicitly).
//!  * `terminate-thread` is also cooperative: it sets a flag that
//!    the target thread observes at a safepoint, and that thread
//!    then unwinds by raising the same panic-sentinel as
//!    `exit-thread`.
//!  * `exit-thread` panics with a sentinel payload that the
//!    spawn-thunk's `catch_unwind` recognises and swallows.
//!  * `critical-section` is implemented as a reentrant mutex
//!    (a `Mutex<(owner, count)>` + `Condvar` pair), keyed by an
//!    integer handle. It's not a Win32 CRITICAL_SECTION; matches
//!    Corman's semantics (reentrance, same-thread re-enter is OK).
//!
//! What we deliberately don't expose:
//!
//!  * Per-thread dynamic binding stacks. Corman's docs say
//!    rebinding `*print-base*` in a child thread doesn't bleed into
//!    the parent. NewCormanLisp's per-thread binding stack hasn't
//!    landed yet (Tier 3 in the porting plan); for v1, `setq` on a
//!    special variable is process-global. Don't rebind specials
//!    across threads expecting Corman semantics until that lands.
//!  * Raw Win32 HANDLEs. `thread-handle` returns the same integer
//!    ID; `*current-thread-handle*` / `*current-process-handle*`
//!    are the same integers as their `*-id*` counterparts. Keeping
//!    the surface uniform across platforms.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

use crate::mutator::MutatorState;
use crate::word::Word;

// ─── Per-thread Lisp ID ─────────────────────────────────────────────────

thread_local! {
    /// Lisp thread id of the current OS thread. 0 means "unassigned"
    /// — the very first thread to call `current_thread_id` gets id 1
    /// retroactively (typically the Lisp primary thread).
    static CURRENT_THREAD_ID: Cell<i64> = const { Cell::new(0) };
}

/// First-call-wins flag: the very first thread to reach
/// `current_thread_id` gets id 1 (the Lisp primary / "main"
/// thread). Subsequent unassigned threads get fresh ids from
/// the registry's next_id counter — they're "implicit" threads
/// (e.g. the GUI thread, hot-reload watcher, test runner pool)
/// that observe the THREADS API without going through
/// `create-thread`.
static MAIN_THREAD_INIT: OnceLock<()> = OnceLock::new();

/// Currently-executing Lisp thread id. The first thread to ask
/// gets id 1; every other unassigned OS thread gets a fresh id
/// auto-registered in the thread table. A thread launched by
/// `create_thread` always has its id set explicitly before this
/// function ever runs there.
pub fn current_thread_id() -> i64 {
    let cached = CURRENT_THREAD_ID.with(|c| c.get());
    if cached != 0 {
        return cached;
    }
    // Race: pick "main" for the first caller, fresh ids for the
    // rest. `get_or_init` returning to its initialiser tells us
    // we're the first caller.
    let mut was_first = false;
    MAIN_THREAD_INIT.get_or_init(|| {
        was_first = true;
    });
    let id = if was_first { 1 } else { alloc_thread_id() };
    let entry = new_entry(id, &format!("implicit-{id}"));
    registry().lock().unwrap().threads.insert(id, entry);
    CURRENT_THREAD_ID.with(|c| c.set(id));
    id
}

/// Force-resolve a Lisp id for the main thread. Called from
/// `create_thread` so the caller has an id BEFORE the new thread
/// races with it.
fn ensure_main_thread_record() {
    let _ = current_thread_id();
}

// ─── Thread registry ────────────────────────────────────────────────────

struct ThreadEntry {
    #[allow(dead_code)]
    id: i64,
    /// Set to None after `join` is taken. For threads not spawned
    /// via `create-thread` (e.g. the main thread, the iGui thread)
    /// this stays None forever.
    handle: Mutex<Option<JoinHandle<()>>>,
    /// Cooperative termination request. Checked at safepoints.
    terminate_requested: AtomicBool,
    /// Cooperative suspend request. Checked at safepoints.
    suspend_requested: AtomicBool,
    /// CV/mutex pair the suspended thread parks on.
    suspend_mu: Mutex<()>,
    suspend_cv: Condvar,
    /// If true, print a line to stderr when this thread terminates.
    report_when_finished: AtomicBool,
    #[allow(dead_code)]
    name: String,
}

fn new_entry(id: i64, name: &str) -> Arc<ThreadEntry> {
    Arc::new(ThreadEntry {
        id,
        handle: Mutex::new(None),
        terminate_requested: AtomicBool::new(false),
        suspend_requested: AtomicBool::new(false),
        suspend_mu: Mutex::new(()),
        suspend_cv: Condvar::new(),
        report_when_finished: AtomicBool::new(false),
        name: name.to_string(),
    })
}

struct ThreadRegistry {
    next_id: i64,
    threads: HashMap<i64, Arc<ThreadEntry>>,
}

fn registry() -> &'static Mutex<ThreadRegistry> {
    static R: OnceLock<Mutex<ThreadRegistry>> = OnceLock::new();
    R.get_or_init(|| {
        Mutex::new(ThreadRegistry {
            // Id 1 is reserved for the main thread (assigned on
            // first ensure_main_thread_record). User threads from 2.
            next_id: 2,
            threads: HashMap::new(),
        })
    })
}

fn alloc_thread_id() -> i64 {
    let mut r = registry().lock().unwrap();
    let id = r.next_id;
    r.next_id += 1;
    id
}

fn get_entry(id: i64) -> Option<Arc<ThreadEntry>> {
    registry().lock().unwrap().threads.get(&id).cloned()
}

fn unregister(id: i64) {
    registry().lock().unwrap().threads.remove(&id);
}

// ─── exit-thread sentinel ───────────────────────────────────────────────

// (We do NOT unwind through JIT'd Lisp frames.) Rust panics on
// Windows are implemented over SEH; our JIT (LLVM) doesn't yet
// emit the unwind tables that would let a panic propagate cleanly
// past a JIT'd frame, so an exit-thread / terminate-thread that
// panicked from deep inside a Lisp call would take the whole
// process down with E06D7363. v1 therefore keeps both calls
// purely cooperative: they set a flag, and the target thread
// observes it the next time it calls (thread-safepoint).

// ─── create-thread ──────────────────────────────────────────────────────

/// Spawn a new OS thread that runs `fn_word` (a Function-tagged
/// Lisp value) with zero arguments. Returns the new thread's
/// Lisp id (≥ 2). The caller's mutator is only used to fetch the
/// shared `GcCoordinator`; we register a fresh mutator on the new
/// thread so allocations there use a per-thread TLAB.
///
/// `fn_word`'s raw bits are passed across the thread boundary.
/// This is safe because:
///   * Function objects live in the pinned static area (see
///     `gc_function::alloc_function_in_static`) and are never
///     moved by the GC.
///   * The Function's env pointer is read fresh from inside the
///     Function object on each invocation (acquire load), so if
///     the captured environment moves during a GC, the new
///     thread observes the updated location automatically.
pub fn create_thread(
    caller: &MutatorState,
    fn_word: Word,
    report_when_finished: bool,
) -> i64 {
    ensure_main_thread_record();
    let coord = Arc::clone(caller.coord());
    let id = alloc_thread_id();
    let entry = new_entry(id, &format!("ncl-thread-{id}"));
    entry
        .report_when_finished
        .store(report_when_finished, Ordering::Release);
    registry().lock().unwrap().threads.insert(id, Arc::clone(&entry));

    let fn_raw = fn_word.raw();
    let entry_for_thread = Arc::clone(&entry);

    let join = thread::Builder::new()
        .name(format!("ncl-thread-{id}"))
        .spawn(move || {
            CURRENT_THREAD_ID.with(|c| c.set(id));

            // Each Lisp thread needs its own MutatorState.
            let mut m = Box::new(coord.register_mutator());
            let mptr: *mut MutatorState = &mut *m;

            // The user-supplied closure is called via the standard
            // Lisp ABI. catch_unwind is still cheap insurance even
            // though v1 doesn't panic through JIT frames — if a
            // Rust shim somewhere panics, we'd rather log it than
            // take the whole process down.
            let result = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| unsafe {
                    crate::abi::ncl_funcall(mptr, fn_raw, std::ptr::null(), 0)
                }),
            );
            if entry_for_thread
                .report_when_finished
                .load(Ordering::Acquire)
            {
                match &result {
                    Ok(_) => eprintln!("[threads] thread {id} finished normally"),
                    Err(_) => eprintln!(
                        "[threads] thread {id} died with an unhandled condition"
                    ),
                }
            }
            drop(m);
            unregister(id);
        })
        .expect("std::thread::spawn failed");

    *entry.handle.lock().unwrap() = Some(join);
    id
}

// ─── exit-thread (cooperative) ──────────────────────────────────────────

/// Request that the current thread exit at its next safepoint.
/// `exit-thread` in Corman is documented to never return — once
/// JIT unwind tables land we'll honour that contract. In v1 it
/// sets a flag and returns NIL; the calling Lisp code is expected
/// to (loop) back to a safepoint check that observes the flag.
pub fn exit_thread(_condition: Word) {
    let tid = current_thread_id();
    if let Some(entry) = get_entry(tid) {
        entry.terminate_requested.store(true, Ordering::Release);
    }
}

// ─── suspend / resume / terminate ───────────────────────────────────────

pub fn suspend_thread(id: i64) -> bool {
    match get_entry(id) {
        Some(e) => {
            e.suspend_requested.store(true, Ordering::Release);
            true
        }
        None => false,
    }
}

pub fn resume_thread(id: i64) -> bool {
    match get_entry(id) {
        Some(e) => {
            e.suspend_requested.store(false, Ordering::Release);
            let _g = e.suspend_mu.lock().unwrap();
            e.suspend_cv.notify_all();
            true
        }
        None => false,
    }
}

pub fn terminate_thread(id: i64) -> bool {
    match get_entry(id) {
        Some(e) => {
            e.terminate_requested.store(true, Ordering::Release);
            // Also clear suspend so the target wakes and observes the
            // terminate flag instead of staying parked forever.
            e.suspend_requested.store(false, Ordering::Release);
            let _g = e.suspend_mu.lock().unwrap();
            e.suspend_cv.notify_all();
            true
        }
        None => false,
    }
}

/// Cooperative safepoint. ONE call covers both concerns a Lisp
/// thread has to poll for at quiet moments:
///
///   1. GC stop-the-world — `MutatorState::safepoint` parks if
///      another mutator is about to collect.
///   2. THREADS-package suspend / terminate — read the per-thread
///      registry entry's flags and either park (for suspend) or
///      return TRUE (for terminate / exit-thread).
///
/// Tight Lisp loops that don't allocate (and therefore never hit
/// the GC's TLAB-refill safepoint by accident) MUST call this
/// periodically, or peers can't trigger a GC and the THREADS API's
/// suspend/terminate become ineffective. Once the compiler emits
/// safepoints automatically at loop back-edges this becomes a
/// belt-and-braces hook.
///
/// Returns `true` if a termination has been requested for this
/// thread (via `terminate-thread` or `exit-thread`). The caller's
/// Lisp code should observe the return value and `(return)` from
/// its own loop. We deliberately do NOT panic-unwind on terminate
/// because v1's JIT doesn't emit Windows SEH unwind tables, so a
/// panic through Lisp frames takes down the process. Cooperative
/// only.
pub fn thread_safepoint(m: &mut MutatorState) -> bool {
    m.safepoint();

    let tid = CURRENT_THREAD_ID.with(|c| c.get());
    if tid == 0 {
        return false;
    }
    let Some(entry) = get_entry(tid) else { return false };

    if entry.suspend_requested.load(Ordering::Acquire) {
        let mut g = entry.suspend_mu.lock().unwrap();
        while entry.suspend_requested.load(Ordering::Acquire) {
            g = entry.suspend_cv.wait(g).unwrap();
        }
    }

    entry.terminate_requested.load(Ordering::Acquire)
}

// ─── Critical sections (reentrant) ──────────────────────────────────────

struct CriticalSection {
    state: Mutex<CsState>,
    cv: Condvar,
}

struct CsState {
    /// 0 = unowned, else the Lisp thread id of the current holder.
    owner: i64,
    /// Reentrance count. Each `enter` from the owner increments, each
    /// `leave` decrements; when this hits 0 the section is released.
    count: u32,
}

struct CsRegistry {
    next_id: i64,
    sections: HashMap<i64, Arc<CriticalSection>>,
}

fn cs_registry() -> &'static Mutex<CsRegistry> {
    static R: OnceLock<Mutex<CsRegistry>> = OnceLock::new();
    R.get_or_init(|| {
        Mutex::new(CsRegistry {
            next_id: 1,
            sections: HashMap::new(),
        })
    })
}

pub fn allocate_critical_section() -> i64 {
    let mut reg = cs_registry().lock().unwrap();
    let id = reg.next_id;
    reg.next_id += 1;
    reg.sections.insert(
        id,
        Arc::new(CriticalSection {
            state: Mutex::new(CsState { owner: 0, count: 0 }),
            cv: Condvar::new(),
        }),
    );
    id
}

pub fn deallocate_critical_section(id: i64) -> bool {
    cs_registry().lock().unwrap().sections.remove(&id).is_some()
}

pub fn enter_critical_section(id: i64) -> bool {
    let cs = match cs_registry().lock().unwrap().sections.get(&id) {
        Some(cs) => Arc::clone(cs),
        None => return false,
    };
    let me = current_thread_id();
    let mut s = cs.state.lock().unwrap();
    loop {
        if s.owner == 0 {
            s.owner = me;
            s.count = 1;
            return true;
        }
        if s.owner == me {
            s.count += 1;
            return true;
        }
        s = cs.cv.wait(s).unwrap();
    }
}

pub fn leave_critical_section(id: i64) -> bool {
    let cs = match cs_registry().lock().unwrap().sections.get(&id) {
        Some(cs) => Arc::clone(cs),
        None => return false,
    };
    let me = current_thread_id();
    let mut s = cs.state.lock().unwrap();
    if s.owner != me {
        return false;
    }
    s.count -= 1;
    if s.count == 0 {
        s.owner = 0;
        cs.cv.notify_one();
    }
    true
}

// ─── Shim entry points (extern "C-unwind") ──────────────────────────────

unsafe fn arg(args: *const u64, i: u64) -> Word {
    Word::from_raw(unsafe { *args.add(i as usize) })
}

unsafe fn arg_fixnum_or(args: *const u64, i: u64, default: i64) -> i64 {
    unsafe { arg(args, i) }.as_fixnum().unwrap_or(default)
}

/// `(create-thread function)` — spawn a new thread running FUNCTION
/// with zero arguments. Returns the new thread's integer id.
///
/// Corman's signature includes `&key (report-when-finished t)`;
/// the runtime always returns the id, and the Lisp wrapper in
/// Library/threads.lisp handles the keyword argument.
pub extern "C-unwind" fn create_thread_shim(
    mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args < 1 {
        panic!("create-thread: expected at least 1 arg (function), got {n_args}");
    }
    let fn_word = unsafe { arg(args, 0) };
    let report = if n_args >= 2 {
        unsafe { arg(args, 1) }.raw() != Word::NIL.raw()
    } else {
        true
    };
    let m = unsafe { &*mutator };
    let id = create_thread(m, fn_word, report);
    Word::fixnum(id).raw()
}

/// `(exit-thread &optional condition)` — unwind the current thread.
/// Calling from the main thread is undefined behaviour (we don't
/// install a catch_unwind there); caller's responsibility.
pub extern "C-unwind" fn exit_thread_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    let cond = if n_args >= 1 {
        unsafe { arg(args, 0) }
    } else {
        Word::NIL
    };
    exit_thread(cond);
    Word::NIL.raw()
}

/// `(thread-handle thread-id)` — Corman's API returns a Win32
/// HANDLE. We return the same integer id (no separate handle), or
/// NIL if the id isn't a live registered thread.
pub extern "C-unwind" fn thread_handle_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    _n_args: u64,
) -> u64 {
    let tid = unsafe { arg_fixnum_or(args, 0, -1) };
    if get_entry(tid).is_some() {
        Word::fixnum(tid).raw()
    } else {
        Word::NIL.raw()
    }
}

pub extern "C-unwind" fn suspend_thread_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    _n_args: u64,
) -> u64 {
    let tid = unsafe { arg_fixnum_or(args, 0, -1) };
    if suspend_thread(tid) { Word::T.raw() } else { Word::NIL.raw() }
}

pub extern "C-unwind" fn resume_thread_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    _n_args: u64,
) -> u64 {
    let tid = unsafe { arg_fixnum_or(args, 0, -1) };
    if resume_thread(tid) { Word::T.raw() } else { Word::NIL.raw() }
}

pub extern "C-unwind" fn terminate_thread_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    _n_args: u64,
) -> u64 {
    let tid = unsafe { arg_fixnum_or(args, 0, -1) };
    if terminate_thread(tid) { Word::T.raw() } else { Word::NIL.raw() }
}

pub extern "C-unwind" fn current_thread_id_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    Word::fixnum(current_thread_id()).raw()
}

pub extern "C-unwind" fn current_process_id_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    Word::fixnum(std::process::id() as i64).raw()
}

pub extern "C-unwind" fn allocate_critical_section_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    Word::fixnum(allocate_critical_section()).raw()
}

pub extern "C-unwind" fn deallocate_critical_section_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    _n_args: u64,
) -> u64 {
    let id = unsafe { arg_fixnum_or(args, 0, -1) };
    if deallocate_critical_section(id) { Word::T.raw() } else { Word::NIL.raw() }
}

pub extern "C-unwind" fn enter_critical_section_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    _n_args: u64,
) -> u64 {
    let id = unsafe { arg_fixnum_or(args, 0, -1) };
    if enter_critical_section(id) { Word::T.raw() } else { Word::NIL.raw() }
}

pub extern "C-unwind" fn leave_critical_section_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    _n_args: u64,
) -> u64 {
    let id = unsafe { arg_fixnum_or(args, 0, -1) };
    if leave_critical_section(id) { Word::T.raw() } else { Word::NIL.raw() }
}

/// `(thread-safepoint)` — returns T if a termination has been
/// requested for the current thread (caller should bail out of its
/// loop), NIL otherwise. Always performs the GC + suspend poll
/// regardless of return value.
pub extern "C-unwind" fn thread_safepoint_shim(
    mutator: *mut MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    if thread_safepoint(m) { Word::T.raw() } else { Word::NIL.raw() }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_section_serialises() {
        // Two threads each increment a shared counter 1000 times,
        // inside a critical section. Final value must be 2000.
        let cs = allocate_critical_section();
        let counter = Arc::new(Mutex::new(0i64));
        let n = 1000;

        let c1 = Arc::clone(&counter);
        let c2 = Arc::clone(&counter);
        let t1 = thread::spawn(move || {
            // Each thread gets a fresh CURRENT_THREAD_ID via the
            // critical section's enter/leave reentrance check (which
            // uses current_thread_id internally).
            for _ in 0..n {
                enter_critical_section(cs);
                let mut g = c1.lock().unwrap();
                *g += 1;
                drop(g);
                leave_critical_section(cs);
            }
        });
        let t2 = thread::spawn(move || {
            for _ in 0..n {
                enter_critical_section(cs);
                let mut g = c2.lock().unwrap();
                *g += 1;
                drop(g);
                leave_critical_section(cs);
            }
        });
        t1.join().unwrap();
        t2.join().unwrap();
        assert_eq!(*counter.lock().unwrap(), 2 * n);
        deallocate_critical_section(cs);
    }

    #[test]
    fn critical_section_is_reentrant_same_thread() {
        let cs = allocate_critical_section();
        assert!(enter_critical_section(cs));
        assert!(enter_critical_section(cs)); // reentrant
        assert!(enter_critical_section(cs));
        assert!(leave_critical_section(cs));
        assert!(leave_critical_section(cs));
        assert!(leave_critical_section(cs));
        // Now unowned — a fresh thread can grab it.
        let cs_id = cs;
        let h = thread::spawn(move || enter_critical_section(cs_id));
        assert!(h.join().unwrap());
        // Leave from that thread is implicit on thread exit? No —
        // we never call leave there. The CS stays owned by t2.
        // Test cleanup: just drop the registry entry.
        deallocate_critical_section(cs);
    }

    #[test]
    fn leave_from_non_owner_returns_false() {
        let cs = allocate_critical_section();
        // Acquired by the main thread.
        assert!(enter_critical_section(cs));
        // A different thread tries to leave — must fail without
        // releasing the lock.
        let cs_id = cs;
        let h = thread::spawn(move || leave_critical_section(cs_id));
        assert!(!h.join().unwrap());
        assert!(leave_critical_section(cs));
        deallocate_critical_section(cs);
    }

    #[test]
    fn unknown_cs_id_is_rejected() {
        assert!(!enter_critical_section(99999));
        assert!(!leave_critical_section(99999));
        assert!(!deallocate_critical_section(99999));
    }

    #[test]
    fn unknown_thread_id_is_rejected() {
        assert!(!suspend_thread(99999));
        assert!(!resume_thread(99999));
        assert!(!terminate_thread(99999));
    }

    #[test]
    fn current_thread_id_is_stable_and_unique() {
        // The same OS thread sees the same Lisp id every call.
        let a = current_thread_id();
        let b = current_thread_id();
        assert_eq!(a, b);
        assert!(a > 0);
        // A spawned thread sees a *different* id from the parent.
        let h = thread::spawn(|| current_thread_id());
        let child = h.join().unwrap();
        assert_ne!(a, child, "spawned thread shouldn't share parent id");
    }
}
