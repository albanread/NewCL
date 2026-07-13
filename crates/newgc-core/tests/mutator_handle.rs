//! MM-1: `Mutator<L>` handle + `GcCoordinator`, serialized allocation.
//!
//! Introduces the handle API shape. Allocation is serialized by the
//! heap mutex (no TLABs/safepoints yet — those are MM-3/MM-4), so these
//! tests assert *correctness*, not throughput. The existing
//! `tests/threading.rs` (which drives `PageHeap` directly) stays green
//! unchanged.

use std::thread;

use newgc_core::{GcCoordinator, Generation, LispLayout, PAYLOAD_MASK, Tag, Word};

type Coord = GcCoordinator<LispLayout>;

#[test]
fn coordinator_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Coord>();
}

// ---------------------------------------------------------------------------
// Two threads each hold a handle on one shared heap. Allocation is
// serialized but correct; total count is exact; a no-root GC reclaims it.
// ---------------------------------------------------------------------------

#[test]
fn two_mutators_share_one_heap_via_handle() {
    let coord = Coord::with_reservation(32 * 64 * 1024);
    let per_thread: i64 = 1000;

    let handles: Vec<_> = (0..2)
        .map(|_| {
            let c = coord.clone();
            thread::spawn(move || {
                let mut m = c.register_mutator();
                let mut count = 0usize;
                for i in 0..per_thread {
                    if let Some(p) = m.try_alloc_cons_in(Generation::G0) {
                        unsafe {
                            *p.as_ptr() = Word::fixnum(i).raw();
                            *p.as_ptr().add(1) = Word::NIL.raw();
                        }
                        count += 1;
                    }
                }
                count
            })
        })
        .collect();

    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total, 2 * per_thread as usize, "every alloc succeeded");

    // Both mutators dropped at thread exit.
    assert_eq!(coord.mutator_count(), 0);

    // Allocations happened; a no-root minor reclaims them all.
    let g0_before = coord.with_heap(|h| h.count_pages_in_gen(Generation::G0));
    assert!(g0_before > 0, "allocations opened G0 pages");
    coord.collect_minor(|_evac| { /* no roots */ });
    let g0_after = coord.with_heap(|h| h.count_pages_in_gen(Generation::G0));
    assert_eq!(g0_after, 0, "no roots -> G0 fully reclaimed");
}

// ---------------------------------------------------------------------------
// A rooted allocation made through a handle survives a collection driven
// through the coordinator.
// ---------------------------------------------------------------------------

#[test]
fn rooted_alloc_via_mutator_survives_collect() {
    let coord = Coord::with_reservation(16 * 64 * 1024);
    let mut m = coord.register_mutator();
    let p = m.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *p.as_ptr() = Word::fixnum(42).raw();
        *p.as_ptr().add(1) = Word::NIL.raw();
    }
    let mut root = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];

    // MM-3: the mutator holds a live TLAB; flush it before collecting so
    // its cursor can't dangle if GC moves the TLAB's page. (MM-4's
    // safepoint protocol will do this automatically.)
    m.flush_tlabs();
    coord.collect_minor(|evac| evac.visit(&mut root[0]));

    let addr = (root[0].raw() & PAYLOAD_MASK) as *const u64;
    let car = unsafe { Word::from_raw(*addr) };
    assert_eq!(car.as_fixnum(), Some(42), "rooted cons survived with value intact");
}

// ---------------------------------------------------------------------------
// Dropping a mutator releases its registry slot.
// ---------------------------------------------------------------------------

#[test]
fn mutator_drop_releases_slot() {
    let coord = Coord::with_reservation(8 * 64 * 1024);
    assert_eq!(coord.mutator_count(), 0);

    let m1 = coord.register_mutator();
    let m2 = coord.register_mutator();
    assert_eq!(coord.mutator_count(), 2);

    drop(m1);
    assert_eq!(coord.mutator_count(), 1);

    // A fresh registration reuses the freed slot (dense ids).
    let m3 = coord.register_mutator();
    assert_eq!(coord.mutator_count(), 2);

    drop(m2);
    drop(m3);
    assert_eq!(coord.mutator_count(), 0);
}

// ---------------------------------------------------------------------------
// A poisoned heap (mid-evac OOM via the coordinator) refuses further
// allocation through the mutator handle.
// ---------------------------------------------------------------------------

#[test]
fn mutator_alloc_returns_none_when_poisoned() {
    newgc_core::page_heap::evac::install_quiet_gc_stall_panic_hook();
    let coord = Coord::with_reservation(2 * 64 * 1024); // tight: 2 pages
    let mut m = coord.register_mutator();

    // Fill the heap, retaining everything as a chain of roots.
    let mut roots: Vec<Word> = Vec::new();
    while let Some(p) = m.try_alloc_cons_in(Generation::G0) {
        unsafe {
            *p.as_ptr() = Word::fixnum(0).raw();
            *p.as_ptr().add(1) =
                roots.last().map(|w| w.raw()).unwrap_or(Word::NIL.raw());
        }
        m.mark_card_at(p.as_ptr() as *const u8);
        roots.push(Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons));
        if roots.len() > 100_000 {
            break;
        }
    }

    let result = coord.try_collect_minor(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });

    match result {
        Err(_) => {
            assert!(coord.is_poisoned(), "Err -> heap poisoned");
            assert!(
                m.try_alloc_cons_in(Generation::G0).is_none(),
                "poisoned heap refuses alloc through the mutator handle"
            );
        }
        Ok(_) => {
            eprintln!("heap was big enough to not OOM — poison path not exercised");
        }
    }
}
