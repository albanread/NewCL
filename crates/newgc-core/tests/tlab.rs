//! MM-3: per-mutator TLABs with a lock-free bump.
//!
//! Each `Mutator` carves a thread-local allocation buffer (TLAB) from
//! the heap under the refill lock, then bumps within it **without**
//! locking — only the start-bit bitmap, alloc counter, and poison flag
//! (all in `SharedHeap`) are touched atomically on the fast path.
//!
//! Soundness scope (MM-3): a mutator must not hold a live TLAB across a
//! collection (its cursor would dangle if GC moved the page). These
//! tests therefore either don't collect, or `flush_tlabs()` / drop the
//! mutator first. MM-4's safepoint protocol automates the flush.

use std::thread;

use newgc_core::{
    GcCoordinator, Generation, LispLayout, PAYLOAD_MASK, Tag, Word, PAGE_SIZE_CELLS,
};

type Coord = GcCoordinator<LispLayout>;

// ---------------------------------------------------------------------------
// The bump fast path amortizes the refill lock: thousands of allocs take
// the heap lock only a handful of times (once per TLAB refill).
// ---------------------------------------------------------------------------

#[test]
fn tlab_bump_amortizes_heap_lock() {
    let coord = Coord::with_reservation(16 * 64 * 1024);
    let mut m = coord.register_mutator();

    for i in 0..4096i64 {
        let p = m.try_alloc_cons_in(Generation::G0).expect("alloc");
        unsafe {
            *p.as_ptr() = Word::fixnum(i).raw();
            *p.as_ptr().add(1) = Word::NIL.raw();
        }
    }

    // 4096 conses = 8192 cells. With 4 KB→64 KB doubling TLABs
    // (512,1024,2048,4096,…) refills are single digits, well under 16.
    let refills = m.tlab_refill_count();
    assert!(
        refills < 16,
        "expected < 16 refills for 4096 conses, got {refills}"
    );
    assert!(refills > 0, "some refills must have happened");
}

// ---------------------------------------------------------------------------
// Concurrent allocation across threads produces no torn pointers. Each
// cons stores (i, hash(i)); afterward every cons we recorded must still
// read back its exact pair — proving no two threads' bumps overlapped.
// ---------------------------------------------------------------------------

fn hash64(x: u64) -> u64 {
    // splitmix64 finalizer — cheap, good mixing.
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn concurrent_cons_alloc_no_torn_pointers() {
    let coord = Coord::with_reservation(64 * 64 * 1024);
    let n_threads = 4;
    let per_thread = 10_000u64;

    let handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let c = coord.clone();
            thread::spawn(move || {
                let mut m = c.register_mutator();
                let mut addrs = Vec::with_capacity(per_thread as usize);
                for i in 0..per_thread {
                    // Encode a thread-unique key so collisions across
                    // threads are detectable.
                    let key = (t as u64) << 40 | i;
                    let p = m.try_alloc_cons_in(Generation::G0).expect("alloc");
                    unsafe {
                        // car = key as fixnum-ish raw, cdr = hash(key).
                        // Store raw u64s; we only compare the bit
                        // patterns, not interpret them as tagged Words.
                        *p.as_ptr() = key;
                        *p.as_ptr().add(1) = hash64(key);
                    }
                    addrs.push((p.as_ptr() as usize, key));
                }
                addrs
            })
        })
        .collect();

    let mut total = 0usize;
    for h in handles {
        let addrs = h.join().expect("thread panicked");
        for (addr, key) in addrs {
            total += 1;
            let car = unsafe { *(addr as *const u64) };
            let cdr = unsafe { *((addr as *const u64).add(1)) };
            assert_eq!(car, key, "car torn at {addr:#x}");
            assert_eq!(cdr, hash64(key), "cdr torn at {addr:#x}");
        }
    }
    assert_eq!(total, (n_threads as u64 * per_thread) as usize);
}

// ---------------------------------------------------------------------------
// TLAB refill honors young_page_cap: a capped G0 stops handing out slabs.
// ---------------------------------------------------------------------------

#[test]
fn tlab_refill_respects_young_page_cap() {
    // young = 2 pages (cap 2), old = 6. Allocate cons cells until refill
    // is refused; G0 must never exceed the cap.
    let coord = GcCoordinator::<LispLayout>::new(2 * 64 * 1024, 6 * 64 * 1024);
    let mut m = coord.register_mutator();

    let mut count = 0u64;
    loop {
        match m.try_alloc_cons_in(Generation::G0) {
            Some(p) => {
                unsafe {
                    *p.as_ptr() = Word::fixnum(0).raw();
                    *p.as_ptr().add(1) = Word::NIL.raw();
                }
                count += 1;
                if count > 100_000 {
                    break; // safety valve
                }
            }
            None => break, // cap hit (refill refused)
        }
    }

    let g0 = coord.with_heap(|h| h.count_pages_in_gen(Generation::G0));
    assert!(g0 <= 2, "G0 pages {g0} exceeded the young cap of 2");
    assert!(count > 0, "allocated at least some cells before the cap");
    // 2 pages of cons cells ≈ 2 * 4096 = 8192 conses (minus slop).
    assert!(count >= 4096, "expected ~2 pages of conses, got {count}");
}

// ---------------------------------------------------------------------------
// Every TLAB-allocated cons has its start bit set (and the heap's view of
// G0 is consistent: collecting with no roots reclaims everything).
// ---------------------------------------------------------------------------

#[test]
fn start_bits_set_correctly_under_concurrent_alloc() {
    let coord = Coord::with_reservation(64 * 64 * 1024);
    let n_threads = 4;
    let per_thread = 5_000u64;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let c = coord.clone();
            thread::spawn(move || {
                let mut m = c.register_mutator();
                for i in 0..per_thread {
                    let p = m.try_alloc_cons_in(Generation::G0).expect("alloc");
                    unsafe {
                        *p.as_ptr() = Word::fixnum(i as i64).raw();
                        *p.as_ptr().add(1) = Word::NIL.raw();
                    }
                }
                // Mutator drops here (TLAB abandoned — safe, no live cursor).
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    // All mutators gone; safe to collect. No roots -> all reclaimed,
    // which exercises the start-bit-driven walkers over every allocated
    // cons (a torn/un-bitted cell would corrupt the walk or leak).
    let g0_before = coord.with_heap(|h| h.count_pages_in_gen(Generation::G0));
    assert!(g0_before > 0);
    coord.collect_minor(|_| {});
    let g0_after = coord.with_heap(|h| h.count_pages_in_gen(Generation::G0));
    assert_eq!(g0_after, 0, "no roots -> G0 fully reclaimed after concurrent alloc");
}

// ---------------------------------------------------------------------------
// Dropping a mutator abandons its TLAB tail safely: a later collection
// runs without panic and reclaims the (unrooted) objects.
// ---------------------------------------------------------------------------

#[test]
fn tlab_drop_abandons_tail_safely() {
    let coord = Coord::with_reservation(32 * 64 * 1024);
    {
        let mut m = coord.register_mutator();
        for i in 0..1000i64 {
            let p = m.try_alloc_cons_in(Generation::G0).expect("alloc");
            unsafe {
                *p.as_ptr() = Word::fixnum(i).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
        }
        // m has a partially-used TLAB here.
    } // <- drop abandons the tail (no reconcile; over-stated words_used
      //    is harmless because the tail has no start bits set).

    assert_eq!(coord.mutator_count(), 0);
    // A collection after the drop must be safe and reclaim everything.
    coord.collect_minor(|_| {});
    assert_eq!(
        coord.with_heap(|h| h.count_pages_in_gen(Generation::G0)),
        0,
        "unrooted conses reclaimed after the mutator dropped"
    );
}

// ---------------------------------------------------------------------------
// flush_tlabs lets a single mutator collect safely and keep allocating.
// ---------------------------------------------------------------------------

#[test]
fn flush_then_collect_then_realloc() {
    let coord = Coord::with_reservation(32 * 64 * 1024);
    let mut m = coord.register_mutator();

    // Allocate + root one cons.
    let p = m.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *p.as_ptr() = Word::fixnum(0xF00D).raw();
        *p.as_ptr().add(1) = Word::NIL.raw();
    }
    let mut root = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];

    // Flush before collecting (clears the live TLAB), then collect.
    m.flush_tlabs();
    coord.collect_minor(|evac| evac.visit(&mut root[0]));

    // Root survived with value intact.
    let car = unsafe { Word::from_raw(*((root[0].raw() & PAYLOAD_MASK) as *const u64)) };
    assert_eq!(car.as_fixnum(), Some(0xF00D));

    // The mutator refills a fresh TLAB and keeps allocating safely.
    for i in 0..1000i64 {
        let q = m.try_alloc_cons_in(Generation::G0).expect("realloc after collect");
        unsafe {
            *q.as_ptr() = Word::fixnum(i).raw();
            *q.as_ptr().add(1) = Word::NIL.raw();
        }
    }
    let _ = PAGE_SIZE_CELLS; // (imported for documentation parity)
}
