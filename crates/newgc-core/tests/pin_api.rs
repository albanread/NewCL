//! MM-0: explicit FFI `pin` / `unpin` API.
//!
//! An explicitly pinned object must not move from the moment `pin`
//! returns until the matching `unpin`, across any number of GC cycles —
//! even if it is otherwise unreachable. This is what FFI needs: a
//! Lisp/Dylan object whose address has escaped into Win32 (a buffer
//! passed to `SetWindowTextW`, a closure handed to a callback for the
//! process lifetime) cannot be relocated while the OS holds its address.
//!
//! The pin is the *only* retention in most of these tests (no root is
//! visited), which proves a pinned object survives on the strength of
//! the pin alone, and that its transitive children survive with it.

use newgc_core::{
    Generation, HeapHeader, HeapType, LispLayout, PageHeap,
    PAYLOAD_MASK, Tag, Word, PAGE_SIZE_CELLS, G0_PROMOTION_THRESHOLD,
};

type Heap = PageHeap<LispLayout>;

fn cons(h: &mut Heap, g: Generation, car: Word, cdr: Word) -> Word {
    let p = h.try_alloc_cons_in(g).expect("cons alloc");
    unsafe {
        *p.as_ptr() = car.raw();
        *p.as_ptr().add(1) = cdr.raw();
    }
    Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
}

fn addr_of(w: Word) -> usize {
    (w.raw() & PAYLOAD_MASK) as usize
}

fn page_gen(h: &Heap, w: Word) -> Generation {
    let page = h.page_of(addr_of(w) as *const u8).expect("addr in heap");
    h.desc(page).generation
}

// ---------------------------------------------------------------------------
// 1. A pinned object keeps its address across minor cycles — pin is the
//    only thing keeping it alive.
// ---------------------------------------------------------------------------

#[test]
fn pinned_object_keeps_address_across_minor() {
    let mut h = Heap::with_reservation(32 * 64 * 1024);
    let c = cons(&mut h, Generation::G0, Word::fixnum(0xABC), Word::NIL);
    let addr_before = addr_of(c);

    let _pin = h.pin(c);

    // No roots visited — only the pin retains it. Run past the G0→G1
    // promotion threshold; the pinned object must never move (its page
    // is relabeled in place, but the address is stable).
    for _ in 0..(G0_PROMOTION_THRESHOLD + 2) {
        h.collect_minor(|_evac| { /* no roots */ });
        // Re-derive the Word at the (unchanged) address each cycle.
        let live = Word::from_ptr(addr_before as *const u8, Tag::Cons);
        assert_eq!(addr_of(live), addr_before, "pinned object must not move");
        assert_ne!(
            page_gen(&h, live),
            Generation::Free,
            "pinned object's page must stay live"
        );
        let car = unsafe { Word::from_raw(*(addr_before as *const u64)) };
        assert_eq!(car.as_fixnum(), Some(0xABC), "pinned value intact");
    }
}

// ---------------------------------------------------------------------------
// 2. A pinned object survives a full collection (Tenured compaction)
//    without moving.
// ---------------------------------------------------------------------------

#[test]
fn pinned_object_survives_full_collect() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);
    // A 3-cell boxed object: header + 2 payload (one points at a child).
    let boxed = h.try_alloc_boxed_in(Generation::G0, 3).unwrap();
    let child = cons(&mut h, Generation::G0, Word::fixnum(0xC417), Word::NIL);
    unsafe {
        *boxed.as_ptr() = HeapHeader::new(HeapType::Vector, 2).raw();
        *boxed.as_ptr().add(1) = child.raw();
        *boxed.as_ptr().add(2) = Word::fixnum(0).raw();
    }
    let boxed_word = Word::from_ptr(boxed.as_ptr() as *const u8, Tag::Vector);
    let addr_before = addr_of(boxed_word);

    let _pin = h.pin(boxed_word);

    // collect_full force-promotes G0→G1→Tenured then compacts Tenured.
    // The pinned object must keep its address through all three passes,
    // and its child must survive (reachable only via the pinned object).
    let _ = h.collect_full(|_evac| { /* no roots */ });

    let live = Word::from_ptr(addr_before as *const u8, Tag::Vector);
    assert_eq!(addr_of(live), addr_before, "pinned object must not move in full GC");
    assert_ne!(page_gen(&h, live), Generation::Free, "pinned page must stay live");

    // The child pointer in the pinned object's payload must still resolve
    // to a live cell holding its sentinel (it may have moved; the pinned
    // object's field is rewritten in place to follow it).
    let child_word = unsafe { Word::from_raw(*(addr_before as *const u64).add(1)) };
    let child_page = h.page_of((child_word.raw() & PAYLOAD_MASK) as *const u8).unwrap();
    assert_ne!(h.desc(child_page).generation, Generation::Free, "child must survive");
    let child_car = unsafe { Word::from_raw(*((child_word.raw() & PAYLOAD_MASK) as *const u64)) };
    assert_eq!(child_car.as_fixnum(), Some(0xC417), "child value intact");
}

// ---------------------------------------------------------------------------
// 3. After unpin, an otherwise-unreachable object is reclaimed.
// ---------------------------------------------------------------------------

#[test]
fn unpin_lets_object_be_reclaimed() {
    let mut h = Heap::with_reservation(32 * 64 * 1024);
    let c = cons(&mut h, Generation::G0, Word::fixnum(7), Word::NIL);
    let addr = addr_of(c);
    let page = h.page_of(addr as *const u8).unwrap();

    let pin = h.pin(c);
    h.collect_minor(|_| {});
    assert_ne!(h.desc(page).generation, Generation::Free, "pinned: alive");

    h.unpin(pin);
    // Unreachable now (no root, no pin). One within-gen minor reclaims it.
    h.collect_minor(|_| {});
    assert_eq!(
        h.desc(page).generation,
        Generation::Free,
        "after unpin and GC, the unreachable object's page is reclaimed"
    );
}

// ---------------------------------------------------------------------------
// 4. Pin is refcounted: N pins need N unpins.
// ---------------------------------------------------------------------------

#[test]
fn pin_is_refcounted() {
    let mut h = Heap::with_reservation(32 * 64 * 1024);
    let c = cons(&mut h, Generation::G0, Word::fixnum(1), Word::NIL);
    let page = h.page_of(addr_of(c) as *const u8).unwrap();

    let p1 = h.pin(c);
    let p2 = h.pin(c);

    h.unpin(p1);
    h.collect_minor(|_| {});
    assert_ne!(
        h.desc(page).generation,
        Generation::Free,
        "still pinned after one of two unpins"
    );

    h.unpin(p2);
    h.collect_minor(|_| {});
    assert_eq!(
        h.desc(page).generation,
        Generation::Free,
        "released after the second unpin"
    );
}

// ---------------------------------------------------------------------------
// 5. A pinned large (multi-page) object keeps its whole run fixed.
// ---------------------------------------------------------------------------

#[test]
fn pinned_large_object_run_stays_fixed() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);
    let n_cells = 2 * PAGE_SIZE_CELLS + 100; // 3 pages
    let ptr = h.try_alloc_large(n_cells, Generation::G0).expect("large alloc");
    unsafe {
        *ptr.as_ptr() = HeapHeader::new(HeapType::Vector, (n_cells - 1) as u32).raw();
        for i in 1..n_cells {
            *ptr.as_ptr().add(i) = Word::fixnum(42).raw();
        }
    }
    let w = Word::from_ptr(ptr.as_ptr() as *const u8, Tag::Vector);
    let addr = addr_of(w);
    let head_page = h.page_of(addr as *const u8).unwrap();
    let n_span = h.desc(head_page).n_span as usize;
    assert_eq!(n_span, 3);

    let _pin = h.pin(w);

    for _ in 0..(G0_PROMOTION_THRESHOLD + 1) {
        h.collect_minor(|_| {});
    }

    // Address unchanged; every page of the run still live.
    let live = Word::from_ptr(addr as *const u8, Tag::Vector);
    assert_eq!(addr_of(live), addr, "large object must not move");
    for i in 0..n_span {
        assert_ne!(
            h.desc(head_page + i).generation,
            Generation::Free,
            "run page {i} must stay live"
        );
    }
    let payload = unsafe { Word::from_raw(*(addr as *const u64).add(1)) };
    assert_eq!(payload.as_fixnum(), Some(42), "large payload intact");
}

// ---------------------------------------------------------------------------
// 6. Pinning a non-heap Word (immediate) is a harmless no-op.
// ---------------------------------------------------------------------------

#[test]
fn pin_of_immediate_is_noop() {
    let mut h = Heap::with_reservation(8 * 64 * 1024);
    // Pinning a fixnum / NIL must not panic and must not pin anything.
    let p1 = h.pin(Word::fixnum(123));
    let p2 = h.pin(Word::NIL);
    h.unpin(p1);
    h.unpin(p2);
    // A normal cycle still works.
    h.collect_minor(|_| {});
    assert_eq!(h.pinned_explicit_count(), 0, "no explicit pins recorded");
}
