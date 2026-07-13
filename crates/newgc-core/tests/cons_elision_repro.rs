//! Repros for cons-chain corruption first seen through NCL, pure
//! newgc-core (no NCL/JIT). Permanent regression coverage for two
//! pinning bugs found and fixed in this crate.
//!
//! - `pinned_partial_cons_chain_keeps_integrity_under_churn`:
//!   head-only conservative pin under heavy churn; guards the
//!   extension-mark fix in `apply_pins_and_extend_mark`.
//! - `pinned_chain_survives_repeated_minor_cycles` (#[ignore]): the
//!   pin-only-across-cascade edge (separate, deeper bug — see
//!   project memory).
//! - `random_interior_pin_debug`: mimics NCL's real pin set — its
//!   conservative scan covers the whole JIT stack, whose stale slots
//!   randomly pin *interior* chain nodes. This caught the cascade
//!   pin-loss bug where `evacuate_with_roots`'s `clear_all_pins`
//!   between G0→G1 and the G1→Tenured cascade dropped conservatively
//!   pinned G1 objects (now: pin cleanup moved to logical-cycle
//!   boundary in `cycle.rs`).

#![cfg(feature = "conservative-pin")]

use newgc_core::{GcCoordinator, Generation, LispLayout, PAYLOAD_MASK, Tag, Word};

type Coord = GcCoordinator<LispLayout>;

/// Walk a supposed `(n-1 … 1 0)` chain; return (count, null_at, gap).
fn check_chain(head: u64, n: i64) -> (i64, i64, (i64, i64)) {
    let mut cur = head;
    let mut prev = n;
    let mut count = 0i64;
    let mut gap = (-1i64, -1i64);
    let mut null_at = -1i64;
    while cur != Word::NIL.raw() {
        let addr = (cur & PAYLOAD_MASK) as *const u64;
        if addr.is_null() {
            null_at = count;
            break;
        }
        let car = unsafe { Word::from_raw(*addr) }.as_fixnum().unwrap_or(-999);
        if car != prev - 1 && gap.0 == -1 {
            gap = (prev, car);
        }
        prev = car;
        cur = unsafe { *addr.add(1) };
        count += 1;
        if count > n + 5 {
            break;
        }
    }
    (count, null_at, gap)
}

#[test]
#[ignore = "separate deeper edge: pin-ONLY object across a G1->Tenured cascade; see project memory"]
fn pinned_chain_survives_repeated_minor_cycles() {
    let coord = Coord::new(8 * 1024 * 1024, 256 * 1024 * 1024);
    let mut m = coord.register_mutator();
    let mut slot: [u64; 1] = [Word::NIL.raw()];
    m.set_stack_range(slot.as_ptr() as usize, unsafe { slot.as_ptr().add(1) } as usize);
    const N: i64 = 50;
    for i in 0..N {
        let p = loop {
            match m.try_alloc_cons_in(Generation::G0) {
                Some(p) => break p,
                None => { m.collect_minor(&mut [], |_| {}); }
            }
        };
        unsafe { *p.as_ptr() = Word::fixnum(i).raw(); *p.as_ptr().add(1) = slot[0]; }
        slot[0] = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons).raw();
    }
    for cycle in 0..40 {
        m.collect_minor(&mut [], |_| {});
        let (count, null_at, gap) = check_chain(slot[0], N);
        assert!(count == N && null_at == -1, "corrupt after cycle {cycle}: count={count} null_at={null_at} gap=({},{})", gap.0, gap.1);
    }
}

#[test]
fn pinned_partial_cons_chain_keeps_integrity_under_churn() {
    let coord = Coord::new(8 * 1024 * 1024, 256 * 1024 * 1024);
    let mut m = coord.register_mutator();
    let mut slot: [u64; 1] = [Word::NIL.raw()];
    m.set_stack_range(slot.as_ptr() as usize, unsafe { slot.as_ptr().add(1) } as usize);
    const N: i64 = 50;
    const ITERS: usize = 100_000;
    let mut bad = 0usize;
    for _ in 0..ITERS {
        slot[0] = Word::NIL.raw();
        for i in 0..N {
            let p = loop {
                match m.try_alloc_cons_in(Generation::G0) {
                    Some(p) => break p,
                    None => { m.collect_minor(&mut [], |_| {}); }
                }
            };
            unsafe { *p.as_ptr() = Word::fixnum(i).raw(); *p.as_ptr().add(1) = slot[0]; }
            slot[0] = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons).raw();
        }
        let (count, null_at, _gap) = check_chain(slot[0], N);
        if count != N || null_at != -1 { bad += 1; }
    }
    assert_eq!(bad, 0, "{bad} corrupted chains");
}

#[test]
fn random_interior_pin_debug() {
    let coord = Coord::new(8 * 1024 * 1024, 256 * 1024 * 1024);
    let mut m = coord.register_mutator();

    const MAXPIN: usize = 64;
    let mut pins: [u64; MAXPIN] = [Word::NIL.raw(); MAXPIN];
    m.set_stack_range(pins.as_ptr() as usize, unsafe { pins.as_ptr().add(MAXPIN) } as usize);

    const N: i64 = 50;
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;

    for iter in 0..200_000usize {
        for p in pins.iter_mut() {
            *p = Word::NIL.raw();
        }
        let mut head = Word::NIL.raw();
        let mut np = 1usize; // pins[0] = head
        let mut pinned: Vec<i64> = Vec::new();
        for i in 0..N {
            let p = loop {
                match m.try_alloc_cons_in(Generation::G0) {
                    Some(p) => break p,
                    None => {
                        // A mid-build collection just evacuated the
                        // partial chain (i nodes: values i-1 .. 0,
                        // head = current `head`). Verify it survived.
                        m.collect_minor(&mut [], |_| {});
                        let (c, na, g) = check_chain(head, i);
                        assert!(
                            c == i && na == -1,
                            "BREAK iter={iter} before node i={i}: expected_len={i} count={c} null_at={na} gap=({},{}) pinned_vals={pinned:?}",
                            g.0, g.1
                        );
                    }
                }
            };
            unsafe {
                *p.as_ptr() = Word::fixnum(i).raw();
                *p.as_ptr().add(1) = head;
            }
            head = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons).raw();
            pins[0] = head;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if np < MAXPIN && rng % 4 == 0 {
                pins[np] = head;
                np += 1;
                pinned.push(i);
            }
        }
        // Final whole-chain check too.
        let (c, na, g) = check_chain(head, N);
        assert!(
            c == N && na == -1,
            "BREAK iter={iter} final: count={c} null_at={na} gap=({},{}) pinned_vals={pinned:?}",
            g.0, g.1
        );
    }
    println!("random_interior_pin_debug: no break in 200k iters");
}
