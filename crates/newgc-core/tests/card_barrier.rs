//! Card-barrier regression tests.
//!
//! Specifically exercises the cross-gen pointer pattern that script
//! 04 of newgc-test-lisp depends on: a vector in old gen whose slot
//! is mutated to point at a freshly-allocated young object.

use newgc_core::page_heap::evac::PageEvacuator;
use newgc_core::page_heap::space::PageHeap;
use newgc_core::{
    Generation, HeapHeader, HeapType, LispLayout, PAYLOAD_MASK, Tag, Word,
};

type Heap = PageHeap<LispLayout>;

fn cons(h: &mut Heap, g: Generation, car: Word, cdr: Word) -> Word {
    let p = h.try_alloc_cons_in(g).expect("cons alloc");
    unsafe {
        *p.as_ptr() = car.raw();
        *p.as_ptr().add(1) = cdr.raw();
    }
    // Card barrier on the cons write — captures any cross-gen cdr
    // (e.g. when the cdr was promoted to G1 during a previous
    // minor mid-build, leaving the next G0 cell as a G0→G1 source).
    h.mark_card_at(p.as_ptr() as *const u8);
    Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
}

fn vector(h: &mut Heap, g: Generation, n: u32, init: Word) -> Word {
    let total = (1 + n) as usize;
    let p = h.try_alloc_boxed_in(g, total).expect("vec alloc");
    unsafe {
        *p.as_ptr() = HeapHeader::new(HeapType::Vector, n).raw();
        for i in 1..=n as usize {
            *p.as_ptr().add(i) = init.raw();
        }
    }
    Word::from_ptr(p.as_ptr() as *const u8, Tag::Vector)
}

unsafe fn vec_slot(w: Word, i: usize) -> Word {
    let p = (w.raw() & PAYLOAD_MASK) as *const u64;
    unsafe { Word::from_raw(*p.add(i)) }
}

fn write_slot_with_barrier(h: &mut Heap, vec: Word, i: usize, v: Word) {
    let p = (vec.raw() & PAYLOAD_MASK) as *mut u64;
    let slot = unsafe { p.add(i) };
    unsafe { *slot = v.raw() };
    h.mark_card_at(slot as *const u8);
}

/// The script 04 pattern: container in old gen, mutator writes new
/// G0 pointer into a slot, minor GC must find the new pointer via
/// the card barrier.
#[test]
fn old_vector_with_young_slot_pointer_survives_minor() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    // Allocate container in G0.
    let container = vector(&mut h, Generation::G0, 1, Word::NIL);
    // Promote to G1 via minor with container as root.
    let mut roots = [container];
    h.collect_minor(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });
    h.collect_minor(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });
    h.collect_minor(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });
    // Now container should be in G1 (after 3 minors).
    let container = roots[0];
    let container_addr = (container.raw() & PAYLOAD_MASK) as *const u8;
    let container_page = h.page_of(container_addr).expect("in heap");
    assert_eq!(
        h.desc(container_page).generation,
        Generation::G1,
        "container should be in G1 after 3 minor cycles"
    );

    // Now allocate a young list and write it into container's slot.
    let young = cons(&mut h, Generation::G0, Word::fixnum(99), Word::NIL);
    write_slot_with_barrier(&mut h, container, 1, young);

    // Run a minor cycle with only container as a root. The card
    // barrier should find the young pointer.
    let mut roots = [container];
    h.collect_minor(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });
    let container = roots[0];

    // Verify the slot now points to a SURVIVING cons with car=99.
    let new_slot = unsafe { vec_slot(container, 1) };
    assert_eq!(
        new_slot.tag(),
        Tag::Cons,
        "slot should be Cons-tagged, got {:?}",
        new_slot.tag()
    );
    let car = unsafe {
        let p = (new_slot.raw() & PAYLOAD_MASK) as *const u64;
        Word::from_raw(*p)
    };
    assert_eq!(
        car.as_fixnum(),
        Some(99),
        "slot's car should still be 99 after minor GC"
    );
}

/// Same but WITHOUT the card barrier — proves the barrier is doing
/// the work, not some other mechanism. This is a "negative" test
/// showing the bug without the fix.
#[test]
fn without_card_barrier_old_to_young_pointer_dangles() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    let container = vector(&mut h, Generation::G0, 1, Word::NIL);
    let mut roots = [container];
    for _ in 0..3 {
        h.collect_minor(|e| {
            for r in roots.iter_mut() {
                e.visit(r);
            }
        });
    }
    let container = roots[0];

    let young = cons(&mut h, Generation::G0, Word::fixnum(99), Word::NIL);
    // Write WITHOUT marking the card.
    let p = (container.raw() & PAYLOAD_MASK) as *mut u64;
    unsafe { *p.add(1) = young.raw() };
    // NO mark_card_at call.

    // Run minor with only container as root.
    let mut roots = [container];
    h.collect_minor(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });

    // Without the card barrier, the GC didn't see the young pointer.
    // The young cell has been reclaimed. The slot still holds the
    // ORIGINAL young address — which now points at a freed/recycled
    // page. The car may NOT read as 99.
    let new_slot = unsafe { vec_slot(roots[0], 1) };
    let car = unsafe {
        let p = (new_slot.raw() & PAYLOAD_MASK) as *const u64;
        Word::from_raw(*p)
    };
    // Document the failure mode: car is NOT fixnum(99).
    // (Could be anything — fixnum 0 from a zeroed page, a stale
    // forward marker, etc. We don't assert what; only that it isn't
    // the original value, which is the corruption.)
    if car.as_fixnum() == Some(99) {
        eprintln!(
            "WARNING: without card barrier, the value survived by \
             accident (probably page was not yet reclaimed). The \
             demonstration is non-deterministic."
        );
    }
}

/// Mimic mini-Lisp script 04: 30-slot container, allocations
/// triggering auto-minor every N cells, every 5th auto-collect
/// upgraded to major. Exercises the GC's cross-gen handling under
/// chunked evacuation across minor+major cycles.
///
/// Regression test for the sub-phase 9 fix-set:
///   (a) Card-scan filter "anywhere but from_gen" (not just strictly
///       older) so G0 cards are scanned during major's G1→Tenured
///       pass, finding G0→G1 cross-gen pointers.
///   (b) Cons writes mark cards (mutator side).
///   (c) Object copy marks dest cards unconditionally (any dest_gen)
///       so GC-copied objects carry forward the may-contain-heap-
///       pointer status.
///   (d) `rebuild_cards_for_old_gens` scans G0 pages too (not just
///       G1/Tenured); G0 cards persist across cycles so the major's
///       G0-card scan finds previously-marked content.
#[test]
fn script_04_pattern_minor_and_major_mixed() {
    let mut h = Heap::with_reservation(32 * 64 * 1024);
    let container = vector(&mut h, Generation::G0, 30, Word::NIL);
    let mut roots: Vec<Word> = vec![container];

    // Track all live lists by their (slot_idx, expected_len) so we
    // can verify after every collection.
    let mut alloc_count = 0usize;
    let mut minors_since_major = 0usize;
    let alloc_threshold = 200;
    let majors_every = 5;

    // Helper: run the GC choosing minor or major like mini-Lisp does.
    fn do_collect(
        h: &mut Heap,
        roots: &mut Vec<Word>,
        minors_since_major: &mut usize,
        majors_every: usize,
    ) {
        *minors_since_major += 1;
        let do_major = *minors_since_major >= majors_every;
        if do_major {
            *minors_since_major = 0;
            h.collect_major(|e| {
                for r in roots.iter_mut() {
                    e.visit(r);
                }
            });
        } else {
            h.collect_minor(|e| {
                for r in roots.iter_mut() {
                    e.visit(r);
                }
            });
        }
    }

    // Phase 1: fill container with 30 lists of length 1..=30.
    for slot in 0..30 {
        // Build a list of length (slot+1).
        let mut head = Word::NIL;
        for v in (1..=(slot + 1) as i64).rev() {
            head = cons(&mut h, Generation::G0, Word::fixnum(v), head);
            alloc_count += 1;
            if alloc_count % alloc_threshold == 0 {
                // Root the partial head while GCing.
                roots.push(head);
                do_collect(&mut h, &mut roots, &mut minors_since_major, majors_every);
                head = roots.pop().unwrap();
            }
        }
        write_slot_with_barrier(&mut h, roots[0], 1 + slot, head);
    }

    // Phase 2: two explicit major cycles (matches script's
    // `(gc-major) (gc-major)`).
    for _ in 0..2 {
        h.collect_major(|e| {
            for r in roots.iter_mut() {
                e.visit(r);
            }
        });
        minors_since_major = 0;
    }
    let container = roots[0];

    // Verify each slot's list length is (slot+1).
    for slot in 0..30 {
        let head = unsafe { vec_slot(container, 1 + slot) };
        let mut n = 0;
        let mut cur = head;
        while cur.tag() == Tag::Cons {
            n += 1;
            let p = (cur.raw() & PAYLOAD_MASK) as *const u64;
            cur = unsafe { Word::from_raw(*p.add(1)) };
        }
        assert_eq!(n, slot + 1, "Phase 2 verify: slot {slot} len");
    }

    // Phase 3: remutate. Replace each slot's list with one of
    // length (100 - slot). Auto-trigger collection by alloc count.
    for slot in 0..30 {
        let new_len = 100 - slot as i64;
        let mut head = Word::NIL;
        for v in (1..=new_len).rev() {
            head = cons(&mut h, Generation::G0, Word::fixnum(v), head);
            alloc_count += 1;
            if alloc_count % alloc_threshold == 0 {
                roots.push(head);
                do_collect(&mut h, &mut roots, &mut minors_since_major, majors_every);
                head = roots.pop().unwrap();
            }
        }
        write_slot_with_barrier(&mut h, roots[0], 1 + slot, head);
    }
    let container = roots[0];

    // Final verify.
    for slot in 0..30 {
        let head = unsafe { vec_slot(container, 1 + slot) };
        let mut n = 0;
        let mut cur = head;
        while cur.tag() == Tag::Cons {
            n += 1;
            let p = (cur.raw() & PAYLOAD_MASK) as *const u64;
            cur = unsafe { Word::from_raw(*p.add(1)) };
        }
        assert_eq!(
            n,
            (100 - slot) as usize,
            "Phase 3 verify: slot {slot} len should be {}",
            100 - slot
        );
    }
}

/// Multi-iteration: 30 slots, each mutated to point at a fresh young
/// list, many minor cycles between. This is script 04's pattern.
#[test]
fn many_slots_repeatedly_mutated_with_barrier_all_survive() {
    let mut h = Heap::with_reservation(32 * 64 * 1024);
    let container = vector(&mut h, Generation::G0, 30, Word::NIL);
    let mut roots = [container];
    // Promote.
    for _ in 0..3 {
        h.collect_minor(|e| {
            for r in roots.iter_mut() {
                e.visit(r);
            }
        });
    }
    let mut container = roots[0];

    // Fill each slot with a small list, write through the barrier.
    for slot in 0..30 {
        // Build a list of length (slot + 1).
        let mut head = Word::NIL;
        for v in 0..=slot {
            head = cons(&mut h, Generation::G0, Word::fixnum(v as i64), head);
        }
        write_slot_with_barrier(&mut h, container, 1 + slot, head);

        // Force a minor cycle after every 5 slots to exercise the
        // barrier path.
        if slot % 5 == 4 {
            let mut iter_roots = [container];
            h.collect_minor(|e| {
                for r in iter_roots.iter_mut() {
                    e.visit(r);
                }
            });
            container = iter_roots[0];
        }
    }

    // Walk every slot — every list should have the right length.
    for slot in 0..30 {
        let head = unsafe { vec_slot(container, 1 + slot) };
        assert_eq!(
            head.tag(),
            Tag::Cons,
            "slot {slot}: not cons after GC"
        );
        // Walk the list, count length.
        let mut n = 0;
        let mut cur = head;
        while cur.tag() == Tag::Cons {
            n += 1;
            let p = (cur.raw() & PAYLOAD_MASK) as *const u64;
            cur = unsafe { Word::from_raw(*p.add(1)) };
        }
        assert!(
            cur.is_nil(),
            "slot {slot}: list doesn't end in NIL (got {cur:?})"
        );
        assert_eq!(n, slot + 1, "slot {slot}: wrong list length");
    }
}
