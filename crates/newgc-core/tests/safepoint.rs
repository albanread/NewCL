//! MM-4 + MM-5: cooperative safepoint protocol and per-mutator roots.
//!
//! A collection is *driven* by a mutator: it self-parks (publishing its
//! roots, flushing its TLABs, marking itself the coordinator), requests
//! the safepoint, waits for every other active mutator to park at the
//! same epoch, then collects with **all** mutators' published roots
//! visited in place, and resumes the world. This is the first point at
//! which multi-mutator stop-the-world GC is sound (every mutator's roots
//! are seen; no mutator runs while the heap is being collected).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use newgc_core::{GcCoordinator, Generation, LispLayout, PAYLOAD_MASK, Tag, Word};

type Coord = GcCoordinator<LispLayout>;

fn car_fixnum(root: Word) -> Option<i64> {
    let addr = (root.raw() & PAYLOAD_MASK) as *const u64;
    unsafe { Word::from_raw(*addr).as_fixnum() }
}

// ---------------------------------------------------------------------------
// B-2 regression: a lone mutator that drives a collection must not
// deadlock waiting for *itself* to park.
// ---------------------------------------------------------------------------

#[test]
fn driver_does_not_wait_on_itself() {
    let coord = Coord::with_reservation(16 * 64 * 1024);
    let mut m = coord.register_mutator();

    let p = m.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *p.as_ptr() = Word::fixnum(42).raw();
        *p.as_ptr().add(1) = Word::NIL.raw();
    }
    let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];

    // Drives the cycle from this single mutator. If the wait loop didn't
    // skip the acting coordinator, this would hang.
    m.collect_minor(&mut roots, |_| {});

    assert_eq!(car_fixnum(roots[0]), Some(42), "driver's own root survived + updated");
}

// ---------------------------------------------------------------------------
// poll_safepoint is a no-op when no collection is pending (if it parked,
// with no driver to resume it, this test would hang).
// ---------------------------------------------------------------------------

#[test]
fn poll_safepoint_noop_when_no_gc() {
    let coord = Coord::with_reservation(8 * 64 * 1024);
    let mut m = coord.register_mutator();
    let p = m.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *p.as_ptr() = Word::fixnum(1).raw();
        *p.as_ptr().add(1) = Word::NIL.raw();
    }
    let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];
    for _ in 0..1000 {
        m.poll_safepoint(&mut roots); // no driver -> must return immediately
    }
    assert_eq!(car_fixnum(roots[0]), Some(1));
}

// ---------------------------------------------------------------------------
// A dropped mutator is not waited on by a subsequent driver.
// ---------------------------------------------------------------------------

#[test]
fn dropped_mutator_not_waited_on() {
    let coord = Coord::with_reservation(16 * 64 * 1024);
    let m1 = coord.register_mutator();
    let mut driver = coord.register_mutator();
    assert_eq!(coord.mutator_count(), 2);

    drop(m1); // deregisters + marks inactive
    assert_eq!(coord.mutator_count(), 1);

    // Driver collects; must not wait for the departed m1 (it would
    // otherwise hit the 10 s timeout, but never complete the wait).
    let mut roots: [Word; 0] = [];
    driver.collect_minor(&mut roots, |_| {});
}

// ---------------------------------------------------------------------------
// MM-5 core: N worker mutators each hold a distinct rooted cons and poll
// in a loop; a driver collects repeatedly. Every worker's cons must
// survive every cycle with its value intact and its root followed to the
// new location — proving per-mutator roots are enumerated and updated,
// and that no worker runs while the heap is collected.
// ---------------------------------------------------------------------------

#[test]
fn multi_worker_rooted_survival_under_concurrent_gc() {
    let coord = Coord::with_reservation(64 * 64 * 1024);
    let n_workers = 3usize;
    let ready = Arc::new(Barrier::new(n_workers + 1));
    let stop = Arc::new(AtomicBool::new(false));

    let workers: Vec<_> = (0..n_workers)
        .map(|t| {
            let c = coord.clone();
            let ready = Arc::clone(&ready);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut m = c.register_mutator();
                let val = 0xA000i64 + t as i64;
                let p = m.try_alloc_cons_in(Generation::G0).unwrap();
                unsafe {
                    *p.as_ptr() = Word::fixnum(val).raw();
                    *p.as_ptr().add(1) = Word::NIL.raw();
                }
                let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];

                ready.wait(); // all workers allocated + rooted

                while !stop.load(Ordering::Acquire) {
                    // Publish-and-park happens inside poll if a cycle is
                    // pending; on return our root has been forwarded.
                    m.poll_safepoint(&mut roots);
                    assert_eq!(
                        car_fixnum(roots[0]),
                        Some(val),
                        "worker {t}: cons value corrupted across GC"
                    );
                    std::hint::spin_loop();
                }
                // Final check after the driver stops.
                m.poll_safepoint(&mut roots);
                assert_eq!(car_fixnum(roots[0]), Some(val));
                val
            })
        })
        .collect();

    let mut driver = coord.register_mutator();
    ready.wait(); // workers are in their poll loop

    let mut driver_roots: [Word; 0] = [];
    for _ in 0..25 {
        driver.collect_minor(&mut driver_roots, |_| {});
        thread::yield_now(); // let workers run + re-poll between cycles
    }

    stop.store(true, Ordering::Release);
    // One more cycle so any worker blocked at a poll gets released.
    driver.collect_minor(&mut driver_roots, |_| {});

    let mut seen = Vec::new();
    for h in workers {
        seen.push(h.join().expect("worker panicked"));
    }
    seen.sort_unstable();
    assert_eq!(seen, vec![0xA000, 0xA001, 0xA002]);
}

// ---------------------------------------------------------------------------
// MM-8: the safepoint wait-timeout (a diagnostic re-check backstop) is
// configurable. It round-trips, clamps to >= 1 ms, and a normal
// cooperative collection still works under a short value (a worker parks
// well before the timeout fires; its root survives + is forwarded).
// ---------------------------------------------------------------------------

#[test]
fn safepoint_timeout_is_configurable() {
    let coord = Coord::with_reservation(16 * 64 * 1024);

    assert_eq!(coord.safepoint_timeout(), Duration::from_secs(10), "default");
    coord.set_safepoint_timeout(Duration::from_millis(250));
    assert_eq!(coord.safepoint_timeout(), Duration::from_millis(250));
    coord.set_safepoint_timeout(Duration::from_millis(0));
    assert_eq!(
        coord.safepoint_timeout(),
        Duration::from_millis(1),
        "clamped to >= 1 ms so the driver's wait can't busy-spin"
    );

    // Functional under a short timeout: a cooperating worker parks long
    // before 100 ms, so collection proceeds normally.
    coord.set_safepoint_timeout(Duration::from_millis(100));
    let ready = Arc::new(Barrier::new(2));
    let stop = Arc::new(AtomicBool::new(false));
    let worker = thread::spawn({
        let c = coord.clone();
        let ready = Arc::clone(&ready);
        let stop = Arc::clone(&stop);
        move || {
            let mut m = c.register_mutator();
            let p = m.try_alloc_cons_in(Generation::G0).unwrap();
            unsafe {
                *p.as_ptr() = Word::fixnum(0x9090).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];
            ready.wait();
            while !stop.load(Ordering::Acquire) {
                m.poll_safepoint(&mut roots);
                assert_eq!(car_fixnum(roots[0]), Some(0x9090));
                std::hint::spin_loop();
            }
            m.poll_safepoint(&mut roots);
            car_fixnum(roots[0])
        }
    });

    let mut driver = coord.register_mutator();
    ready.wait();
    let mut dr: [Word; 0] = [];
    for _ in 0..5 {
        driver.collect_minor(&mut dr, |_| {});
        thread::yield_now();
    }
    stop.store(true, Ordering::Release);
    driver.collect_minor(&mut dr, |_| {});
    assert_eq!(worker.join().expect("worker panicked"), Some(0x9090));
}
