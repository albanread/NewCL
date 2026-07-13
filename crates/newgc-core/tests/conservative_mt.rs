//! MM-7: conservative stack pins across mutators + precise-only build.
//!
//! NCL's JIT spills tagged Lisp `Word`s and raw native values onto one
//! stack with no compiler-enforced separation, so its primary root
//! source is a *conservative* scan: anything pointer-shaped on the stack
//! pins its target against movement (`conservative-pin` feature). The
//! multi-mutator extension is per-mutator stack windows (`set_stack_range`)
//! that the driver unions for one combined `pin_pointers_in_ranges`.
//!
//! A precise-only build (`--no-default-features`) compiles that scan out;
//! published `Snapshot` roots alone keep objects alive (and forward them).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use newgc_core::{GcCoordinator, Generation, LispLayout, PAYLOAD_MASK, Tag, Word};

type Coord = GcCoordinator<LispLayout>;

fn car_fixnum(root: Word) -> Option<i64> {
    let addr = (root.raw() & PAYLOAD_MASK) as *const u64;
    unsafe { Word::from_raw(*addr).as_fixnum() }
}

// ---------------------------------------------------------------------------
// Two mutators each publish a stack window holding the sole reference to a
// cons (NOT a precise root). One driver cycle must union both windows and
// pin both conses — each survives at its address on the strength of the
// conservative pin alone. Proves per-mutator windows combine (§5.3).
// ---------------------------------------------------------------------------

#[cfg(feature = "conservative-pin")]
#[test]
fn conservative_pins_combine_across_mutators() {
    let coord = Coord::with_reservation(32 * 64 * 1024);
    let ready = Arc::new(Barrier::new(3)); // 2 workers + driver
    let release = Arc::new(AtomicBool::new(false));

    let workers: Vec<_> = (0..2)
        .map(|t| {
            let c = coord.clone();
            let ready = Arc::clone(&ready);
            let release = Arc::clone(&release);
            thread::spawn(move || {
                let mut m = c.register_mutator();
                let val = 0xE000i64 + t as i64;
                let p = m.try_alloc_cons_in(Generation::G0).unwrap();
                unsafe {
                    *p.as_ptr() = Word::fixnum(val).raw();
                    *p.as_ptr().add(1) = Word::NIL.raw();
                }
                let w = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
                let cons_addr = (w.raw() & PAYLOAD_MASK) as usize;

                // Fake stack window holding ONLY a copy of the cons pointer.
                // The cons is never published as a precise root, so its sole
                // retention is the conservative pin from this window.
                let stack: Box<[u64]> = vec![w.raw()].into_boxed_slice();
                let lo = stack.as_ptr() as usize;
                let hi = unsafe { stack.as_ptr().add(stack.len()) } as usize;
                m.set_stack_range(lo, hi);

                // Go InNative so the driver collects around us (no poll
                // loop) yet still scans our published window. Publish empty
                // precise roots: the cons must survive on the pin alone.
                let empty: [Word; 0] = [];
                m.enter_native(&empty);
                ready.wait();
                while !release.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let mut none: [Word; 0] = [];
                m.leave_native(&mut none);

                // Read the cons at its ORIGINAL address. If it had moved
                // (unpinned) or been reclaimed, this would not hold `val`.
                let car = unsafe { Word::from_raw(*(cons_addr as *const u64)).as_fixnum() };
                let result = (cons_addr, val, car);
                drop(stack); // keep the window alive until the driver scanned it
                result
            })
        })
        .collect();

    let mut driver = coord.register_mutator();
    ready.wait(); // both workers pinned-via-window and InNative

    let mut dr: [Word; 0] = [];
    for _ in 0..3 {
        driver.collect_minor(&mut dr, |_| {});
    }
    release.store(true, Ordering::Release);

    let mut seen = Vec::new();
    for h in workers {
        let (addr, val, car) = h.join().expect("worker panicked");
        assert_eq!(
            car,
            Some(val),
            "conservatively-pinned cons @ {addr:#x} did not survive in place"
        );
        seen.push(val);
    }
    seen.sort_unstable();
    assert_eq!(seen, vec![0xE000, 0xE001]);
}

// ---------------------------------------------------------------------------
// Precise snapshot roots alone keep an object alive and forward it — with
// NO conservative stack window. This is the only root path in a
// precise-only (`--no-default-features`) build, so the test is unconditional.
// ---------------------------------------------------------------------------

#[test]
fn precise_roots_only_keeps_objects_alive() {
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
                *p.as_ptr() = Word::fixnum(0x5151).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            // Precise root, NO stack window published.
            let mut roots = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];
            m.enter_native(&roots);
            ready.wait();
            while !release.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            m.leave_native(&mut roots);
            // The collector forwarded the (moved) object; the snapshot
            // round-trip put the new address back into `roots`.
            car_fixnum(roots[0])
        }
    });

    let mut driver = coord.register_mutator();
    ready.wait();
    let mut dr: [Word; 0] = [];
    for _ in 0..3 {
        driver.collect_minor(&mut dr, |_| {});
    }
    release.store(true, Ordering::Release);
    assert_eq!(worker.join().expect("worker panicked"), Some(0x5151));
}
