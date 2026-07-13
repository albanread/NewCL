//! MM-8: multi-mutator torture test.
//!
//! `N` worker threads concurrently allocate (churning garbage to build
//! GC pressure), hold a fixed set of rooted conses with known sentinels,
//! poll safepoints, take native excursions (`enter_native`/`leave_native`),
//! and pin/unpin objects across collections — while one driver thread runs
//! minor (and occasional full) collections in a tight loop. Every worker
//! asserts, on every iteration, that each of its rooted conses still holds
//! its sentinel: this catches a lost object, a mis-forwarded root, a torn
//! cell, or a double-free anywhere in the MM-4..MM-7 machinery.
//!
//! Iteration count is tunable via `NEWGC_STRESS_ITERS` (default modest so
//! the suite stays fast); set it high for a real torture run, e.g.
//! `NEWGC_STRESS_ITERS=100000 cargo test --release -p newgc-core --test stress_mt`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use newgc_core::{GcCoordinator, Generation, LispLayout, PAYLOAD_MASK, Tag, Word};

type Coord = GcCoordinator<LispLayout>;

const N_WORKERS: usize = 6;
const ROOTS_PER_WORKER: usize = 4;

fn car_fixnum(root: Word) -> Option<i64> {
    let addr = (root.raw() & PAYLOAD_MASK) as *const u64;
    unsafe { Word::from_raw(*addr).as_fixnum() }
}

/// Distinct, recognizable sentinel for worker `t`'s root slot `s`.
fn sentinel(t: usize, s: usize) -> i64 {
    0x0010_0000 | ((t as i64) << 8) | (s as i64)
}

fn alloc_cons(m: &mut newgc_core::Mutator<LispLayout>, car: i64) -> Option<Word> {
    let p = m.try_alloc_cons_in(Generation::G0)?;
    unsafe {
        *p.as_ptr() = Word::fixnum(car).raw();
        *p.as_ptr().add(1) = Word::NIL.raw();
    }
    Some(Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons))
}

#[test]
fn stress_multi_mutator_alloc_gc_native_pin() {
    let iters: usize = std::env::var("NEWGC_STRESS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    // Generous reservation: peak live is tiny (N*K rooted conses + a few
    // in-flight churn conses); the rest is headroom for garbage between
    // the driver's collections.
    let coord = Coord::with_reservation(512 * 64 * 1024);
    let ready = Arc::new(Barrier::new(N_WORKERS + 1));
    let stop = Arc::new(AtomicBool::new(false));
    let workers_done = Arc::new(AtomicUsize::new(0));

    let workers: Vec<_> = (0..N_WORKERS)
        .map(|t| {
            let c = coord.clone();
            let ready = Arc::clone(&ready);
            let stop = Arc::clone(&stop);
            let workers_done = Arc::clone(&workers_done);
            thread::spawn(move || {
                let mut m = c.register_mutator();

                // Fixed rooted set, each with a distinct sentinel.
                let mut roots = [Word::NIL; ROOTS_PER_WORKER];
                for s in 0..ROOTS_PER_WORKER {
                    roots[s] = alloc_cons(&mut m, sentinel(t, s))
                        .expect("startup root alloc");
                }

                let check = |roots: &[Word; ROOTS_PER_WORKER]| {
                    for s in 0..ROOTS_PER_WORKER {
                        assert_eq!(
                            car_fixnum(roots[s]),
                            Some(sentinel(t, s)),
                            "worker {t} root {s} corrupted across GC"
                        );
                    }
                };

                ready.wait();

                // -- Work phase: churn + safepoints + native + pins. --
                for i in 0..iters {
                    // Churn: a throwaway cons builds G0 pressure. Not
                    // rooted, so the next collection reclaims it. Under
                    // pressure the bump may miss; that's fine, we poll and
                    // let the driver free space.
                    let _ = alloc_cons(&mut m, 0x7777);

                    if i % 17 == 0 {
                        // Native excursion: publish roots, "block", return.
                        m.enter_native(&roots);
                        std::hint::spin_loop();
                        m.leave_native(&mut roots);
                    } else {
                        m.poll_safepoint(&mut roots);
                    }

                    if i % 23 == 0 {
                        // Pin a root across a safepoint, then release it.
                        let h = m.pin(roots[i % ROOTS_PER_WORKER]);
                        m.poll_safepoint(&mut roots);
                        m.unpin(h);
                    }

                    check(&roots);
                }

                // Done working; keep cooperating with the driver until it
                // tells everyone to stop (mirrors the safepoint suite).
                workers_done.fetch_add(1, Ordering::AcqRel);
                while !stop.load(Ordering::Acquire) {
                    m.poll_safepoint(&mut roots);
                    check(&roots);
                    std::hint::spin_loop();
                }
                m.poll_safepoint(&mut roots);
                check(&roots);
                t
            })
        })
        .collect();

    let mut driver = coord.register_mutator();
    ready.wait();

    // Drive collections until every worker has finished its work phase.
    // Hard cap on cycles so a bug surfaces as a failure, not a hang (the
    // outer test harness also runs under a wall-clock timeout).
    let mut dr: [Word; 0] = [];
    let mut cycle = 0usize;
    let cap = 50 + iters * 4;
    while workers_done.load(Ordering::Acquire) < N_WORKERS {
        if cycle % 10 == 9 {
            driver.collect_full(&mut dr, |_| {});
        } else {
            driver.collect_minor(&mut dr, |_| {});
        }
        cycle += 1;
        thread::yield_now();
        assert!(cycle < cap, "driver exceeded cycle cap — workers not progressing");
    }

    // Release any worker parked at a poll, then let them all exit.
    stop.store(true, Ordering::Release);
    driver.collect_minor(&mut dr, |_| {});

    let mut seen: Vec<usize> = workers
        .into_iter()
        .map(|h| h.join().expect("worker panicked"))
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..N_WORKERS).collect::<Vec<_>>());
}
