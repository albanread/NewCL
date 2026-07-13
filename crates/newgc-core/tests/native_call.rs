//! MM-6: native-call boundary convention (design §4.6).
//!
//! A thread blocked in foreign code runs no managed code and hits no
//! safepoint poll, sometimes for seconds. `enter_native` publishes the
//! thread's roots + flushes its TLABs and marks it `InNative`; a driver
//! then *collects around* it (skips it in the wait loop) instead of
//! holding every GC hostage until the 10 s timeout. The collector still
//! visits the native thread's published roots and forwards them in
//! place; `leave_native` blocks until any in-flight cycle resumes, then
//! copies the forwarded roots back.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use newgc_core::{GcCoordinator, Generation, LispLayout, PAYLOAD_MASK, Tag, Word};

type Coord = GcCoordinator<LispLayout>;

fn car_fixnum(root: Word) -> Option<i64> {
    let addr = (root.raw() & PAYLOAD_MASK) as *const u64;
    unsafe { Word::from_raw(*addr).as_fixnum() }
}

fn addr_of(w: Word) -> usize {
    (w.raw() & PAYLOAD_MASK) as usize
}

// ---------------------------------------------------------------------------
// A thread blocked InNative must NOT be waited on: a driver collects
// while the thread is parked in foreign code, and the thread's root is
// forwarded in place. If the InNative skip were broken, each collect
// would burn the full 10 s timeout.
// ---------------------------------------------------------------------------

#[test]
fn driver_does_not_stall_on_native_thread() {
    let coord = Coord::with_reservation(16 * 64 * 1024);
    let ready = Arc::new(Barrier::new(2));
    let release = Arc::new(AtomicBool::new(false));

    let worker = thread::spawn({
        let c = coord.clone();
        let ready = Arc::clone(&ready);
        let release = Arc::clone(&release);
        move || {
            let mut m = c.register_mutator();
            let p = m.try_alloc_cons_in(Generation::G0).unwrap();
            unsafe {
                *p.as_ptr() = Word::fixnum(0xBEEF).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];

            // Enter foreign code (publishes the root) and stay blocked.
            m.enter_native(&roots);
            ready.wait(); // tell the driver we're now InNative
            while !release.load(Ordering::Acquire) {
                std::hint::spin_loop(); // "blocking" foreign call — no heap access
            }
            m.leave_native(&mut roots);
            car_fixnum(roots[0]) // forwarded value
        }
    });

    let mut driver = coord.register_mutator();
    ready.wait(); // worker is InNative

    // None of these wait on the native worker; if the skip were broken,
    // the first collect alone would block for the 10 s timeout.
    let mut dr: [Word; 0] = [];
    let t0 = Instant::now();
    for _ in 0..5 {
        driver.collect_minor(&mut dr, |_| {});
    }
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "driver stalled on the native thread: {elapsed:?}"
    );

    release.store(true, Ordering::Release);
    let survived = worker.join().expect("worker panicked");
    assert_eq!(survived, Some(0xBEEF), "native thread's root survived + forwarded");
}

// ---------------------------------------------------------------------------
// The common case: a native call with no collection around it. enter +
// leave must round-trip roots unchanged, leave_native must not block
// (world is running), and the thread re-enters at the current epoch so a
// subsequent poll is a no-op (it must not park — there is no driver to
// resume it; if it parked this test would hang).
// ---------------------------------------------------------------------------

#[test]
fn enter_leave_native_without_gc_roundtrips_roots() {
    let coord = Coord::with_reservation(8 * 64 * 1024);
    let mut m = coord.register_mutator();
    let p = m.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *p.as_ptr() = Word::fixnum(7).raw();
        *p.as_ptr().add(1) = Word::NIL.raw();
    }
    let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];

    m.enter_native(&roots);
    // ... short foreign call, no GC ...
    m.leave_native(&mut roots);
    assert_eq!(car_fixnum(roots[0]), Some(7), "root unchanged with no GC");

    for _ in 0..100 {
        m.poll_safepoint(&mut roots); // must be a no-op (epoch already current)
    }
    assert_eq!(car_fixnum(roots[0]), Some(7));
}

// ---------------------------------------------------------------------------
// §4.6 ⚠ FFI pinning (ties MM-0 + MM-6). enter_native updates the
// thread's *own* root slots, but cannot reach a raw address copy the
// foreign code holds. An object whose address is passed into a blocking
// call must therefore be pinned (MM-0) so a concurrent GC keeps it fixed.
// Here the driver collects repeatedly while the thread is InNative; the
// pinned object's address must never change.
// ---------------------------------------------------------------------------

#[test]
fn ffi_object_pinned_across_native_call_keeps_address() {
    let coord = Coord::with_reservation(32 * 64 * 1024);
    let ready = Arc::new(Barrier::new(2));
    let release = Arc::new(AtomicBool::new(false));

    let worker = thread::spawn({
        let c = coord.clone();
        let ready = Arc::clone(&ready);
        let release = Arc::clone(&release);
        move || {
            let mut m = c.register_mutator();
            let p = m.try_alloc_cons_in(Generation::G0).unwrap();
            unsafe {
                *p.as_ptr() = Word::fixnum(0xF00D).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            let w = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
            let addr_before = addr_of(w);

            // FFI pin BEFORE the native call: the foreign code will hold a
            // raw copy of this address that in-place root updates can't
            // reach, so the object must not be relocated while we block.
            let pin = m.pin(w);

            let mut roots = [w];
            m.enter_native(&roots);
            ready.wait();
            while !release.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            m.leave_native(&mut roots);

            // Pinned: the address survived every cycle unchanged, so the
            // raw copy the foreign call held is still valid.
            assert_eq!(addr_of(roots[0]), addr_before, "pinned FFI object moved!");
            assert_eq!(car_fixnum(roots[0]), Some(0xF00D), "pinned value intact");
            m.unpin(pin);
        }
    });

    let mut driver = coord.register_mutator();
    ready.wait(); // worker pinned its object and is now InNative

    let mut dr: [Word; 0] = [];
    for _ in 0..4 {
        driver.collect_minor(&mut dr, |_| {});
    }
    release.store(true, Ordering::Release);
    worker.join().expect("worker panicked");
}

// ---------------------------------------------------------------------------
// Integration: a native thread coexists with normally-polling workers
// under a driver running many cycles. The native thread's root survives
// every cycle (visited but never waited on); the pollers participate in
// STW as usual. Proves the §4.6 skip composes with the §4.4 park path.
// ---------------------------------------------------------------------------

#[test]
fn native_and_polling_workers_survive_concurrent_gc() {
    let coord = Coord::with_reservation(64 * 64 * 1024);
    let n_poll = 2usize;
    let ready = Arc::new(Barrier::new(n_poll + 2)); // pollers + native + main
    let stop = Arc::new(AtomicBool::new(false));
    let release_native = Arc::new(AtomicBool::new(false));

    let pollers: Vec<_> = (0..n_poll)
        .map(|t| {
            let c = coord.clone();
            let ready = Arc::clone(&ready);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut m = c.register_mutator();
                let val = 0xC000i64 + t as i64;
                let p = m.try_alloc_cons_in(Generation::G0).unwrap();
                unsafe {
                    *p.as_ptr() = Word::fixnum(val).raw();
                    *p.as_ptr().add(1) = Word::NIL.raw();
                }
                let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];
                ready.wait();
                while !stop.load(Ordering::Acquire) {
                    m.poll_safepoint(&mut roots);
                    assert_eq!(car_fixnum(roots[0]), Some(val), "poller {t} corrupted");
                    std::hint::spin_loop();
                }
                m.poll_safepoint(&mut roots);
                assert_eq!(car_fixnum(roots[0]), Some(val));
                val
            })
        })
        .collect();

    let native = thread::spawn({
        let c = coord.clone();
        let ready = Arc::clone(&ready);
        let release_native = Arc::clone(&release_native);
        move || {
            let mut m = c.register_mutator();
            let val = 0xD00Di64;
            let p = m.try_alloc_cons_in(Generation::G0).unwrap();
            unsafe {
                *p.as_ptr() = Word::fixnum(val).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];
            m.enter_native(&roots);
            ready.wait();
            // Blocked in foreign code across every collection below.
            while !release_native.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            m.leave_native(&mut roots);
            assert_eq!(car_fixnum(roots[0]), Some(val), "native worker corrupted");
            val
        }
    });

    let mut driver = coord.register_mutator();
    ready.wait(); // pollers in their loop, native worker InNative

    let mut dr: [Word; 0] = [];
    for _ in 0..20 {
        driver.collect_minor(&mut dr, |_| {});
        thread::yield_now();
    }
    stop.store(true, Ordering::Release);
    driver.collect_minor(&mut dr, |_| {}); // release any poller blocked at a poll
    release_native.store(true, Ordering::Release);

    let mut seen = Vec::new();
    for h in pollers {
        seen.push(h.join().expect("poller panicked"));
    }
    seen.sort_unstable();
    assert_eq!(seen, vec![0xC000, 0xC001]);
    assert_eq!(native.join().expect("native panicked"), 0xD00D);
}
