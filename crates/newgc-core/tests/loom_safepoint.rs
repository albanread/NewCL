//! MM-8: loom model of the cooperative safepoint handshake.
//!
//! Loom exhaustively explores thread interleavings and memory orderings,
//! so it is the right tool for the *safety / happens-before* properties
//! the protocol rests on. Each model below is a small standalone replica
//! of one ordering rule in `mutator.rs`, tied to a concrete fix:
//!
//!   1. MM-5 root publication — a parked mutator publishes its root
//!      *values* then announces arrival via `last_epoch` (Release); the
//!      driver that observes arrival (Acquire) must see those values.
//!   2. MM-4 "Fix B" — the driver publishes the stop (epoch bump +
//!      `world_running = 0`) under `park_mutex`, so a worker reading both
//!      under the same lock can never observe a torn (stale-epoch,
//!      fresh-stop) state and re-park at the wrong target.
//!   3. MM-4 resume — `ResumeGuard` sets `world_running = 1` under
//!      `park_mutex`; a worker that observes the resume under the lock
//!      must see the collector's in-place root forwarding from that cycle.
//!
//! Liveness (the cross-cycle straggler deadlock) is a progress property
//! loom does not check directly; it is covered by the targeted analysis
//! and `tests/stress_mt.rs`. Run these models with:
//!   RUSTFLAGS="--cfg loom" cargo test -p newgc-core --test loom_safepoint
//!
//! The whole file is `#![cfg(loom)]`, so a normal build compiles it to an
//! empty (0-test) binary and never pulls in the `loom` dependency.
#![cfg(loom)]

use loom::sync::atomic::AtomicUsize;
use loom::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use loom::sync::{Arc, Mutex};
use loom::thread;

// 1. Published roots are visible once the driver observes arrival.
#[test]
fn loom_published_roots_visible_after_arrival() {
    loom::model(|| {
        let last_epoch = Arc::new(AtomicUsize::new(0)); // 0 = not yet arrived
        let published = Arc::new(AtomicUsize::new(0)); // the root snapshot value

        let w_last = last_epoch.clone();
        let w_pub = published.clone();
        let worker = thread::spawn(move || {
            // park(): publish roots BEFORE announcing arrival.
            w_pub.store(0xCAFE, Relaxed);
            w_last.store(1, Release);
        });

        // Driver wait predicate: if it sees last_epoch >= target, the
        // Release/Acquire pair makes the published snapshot visible.
        if last_epoch.load(Acquire) >= 1 {
            assert_eq!(
                published.load(Relaxed),
                0xCAFE,
                "published roots not visible after observing arrival"
            );
        }
        worker.join().unwrap();
    });
}

// 2. Stop published under park_mutex => no torn (stale-epoch, fresh-stop).
#[test]
fn loom_stop_under_lock_has_no_torn_read() {
    loom::model(|| {
        let lock = Arc::new(Mutex::new(()));
        let epoch = Arc::new(AtomicUsize::new(0)); // old epoch
        let world_running = Arc::new(AtomicUsize::new(1)); // 1 = running

        let d_lock = lock.clone();
        let d_epoch = epoch.clone();
        let d_wr = world_running.clone();
        let driver = thread::spawn(move || {
            // drive_collect: bump epoch + stop the world UNDER park_mutex.
            let _g = d_lock.lock().unwrap();
            d_epoch.store(1, Release);
            d_wr.store(0, Release);
        });

        // park(): a worker reads epoch THEN world_running under the lock —
        // the same order as the real park loop. Without the lock this order
        // admits a torn read: epoch.load sees the stale value while a later
        // world_running.load(Acquire) sees the fresh stop (the Acquire's
        // synchronizes-with edge cannot retroactively constrain the earlier
        // epoch.load). The shared lock forbids interleaving the two stores
        // between the two loads, eliminating it. (Verified: removing the
        // lock here makes loom report this exact torn read.)
        {
            let _g = lock.lock().unwrap();
            let ep = epoch.load(Acquire);
            let wr = world_running.load(Acquire);
            // Observing the stop must imply observing the *new* epoch, so a
            // straggler re-arms to the live target, never the stale one.
            if wr == 0 {
                assert_eq!(ep, 1, "torn read: world stopped but epoch stale");
            }
        }
        driver.join().unwrap();
    });
}

// 3. Resume published under park_mutex => forwarded roots are visible.
#[test]
fn loom_resume_under_lock_publishes_forwarded_roots() {
    loom::model(|| {
        let lock = Arc::new(Mutex::new(()));
        let world_running = Arc::new(AtomicUsize::new(0)); // 0 = cycle running
        let forwarded = Arc::new(AtomicUsize::new(0)); // collector's in-place update

        let d_lock = lock.clone();
        let d_wr = world_running.clone();
        let d_fwd = forwarded.clone();
        let driver = thread::spawn(move || {
            // Collector forwards the root in place, THEN ResumeGuard sets
            // world_running = 1 under park_mutex.
            d_fwd.store(0xF00D, Relaxed);
            let _g = d_lock.lock().unwrap();
            d_wr.store(1, Release);
        });

        // park() tail: observe the resume under the lock, then copy the
        // (forwarded) snapshot back. The mutex release/acquire orders the
        // forwarding store before the worker's read.
        let resumed = {
            let _g = lock.lock().unwrap();
            world_running.load(Acquire) == 1
        };
        if resumed {
            assert_eq!(
                forwarded.load(Relaxed),
                0xF00D,
                "forwarded root not visible after observing resume"
            );
        }
        driver.join().unwrap();
    });
}
