//! Sprint VM-1: large-object allocation and GC tests.
//!
//! Large objects span one or more whole 64 KB pages. They are allocated
//! via `try_alloc_large`, never copied during evacuation, and released
//! when they become unreachable.

use newgc_core::{
    Generation, HeapHeader, HeapType, LispLayout, PageHeap, PageKind,
    PAYLOAD_MASK, Tag, Word, PAGE_SIZE_CELLS,
    G0_PROMOTION_THRESHOLD, G1_PROMOTION_THRESHOLD,
};

type Heap = PageHeap<LispLayout>;

/// Allocate a large object: write a Vector header at cell 0 and fill
/// payload cells with `fixnum(42)`. Returns a tagged `Word` pointing
/// at the object.
///
/// `n_cells` is the TOTAL cells to allocate (1 header + (n_cells-1) payload).
fn alloc_large_object(
    h: &mut Heap,
    generation: Generation,
    n_cells: usize,
) -> Word {
    assert!(n_cells >= 1);
    let n_payload = (n_cells - 1) as u32;
    let ptr = h
        .try_alloc_large(n_cells, generation)
        .expect("try_alloc_large must succeed");
    unsafe {
        // Write a Vector header covering n_payload payload cells.
        let header = HeapHeader::new(HeapType::Vector, n_payload);
        (ptr.as_ptr() as *mut u64).write(header.raw());
        // Fill payload with fixnum(42).
        for i in 1..n_cells {
            ptr.as_ptr().add(i).write(Word::fixnum(42).raw());
        }
    }
    Word::from_ptr(ptr.as_ptr() as *const u8, Tag::Vector)
}

// ---------------------------------------------------------------------------
// Test 1: Large object spans more than one page
// ---------------------------------------------------------------------------

#[test]
fn vm1_large_object_allocates_across_page_boundary() {
    // n_cells = PAGE_SIZE_CELLS + 1 → needs 2 pages.
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    let n_cells = PAGE_SIZE_CELLS + 1;
    let ptr = h.try_alloc_large(n_cells, Generation::G0);
    assert!(ptr.is_some(), "alloc of PAGE_SIZE_CELLS+1 cells must succeed");
    let ptr = ptr.unwrap();
    let page_idx = h.page_of(ptr.as_ptr() as *const u8).unwrap();
    let head_desc = h.desc(page_idx);
    assert!(head_desc.is_large_head(), "first page must be a large head");
    assert_eq!(head_desc.n_span, 2, "n_span must be 2 for a 2-page object");
    assert_eq!(head_desc.kind, PageKind::Large);
    let cont_desc = h.desc(page_idx + 1);
    assert!(cont_desc.is_large_cont(), "second page must be a large cont");
    assert_eq!(cont_desc.kind, PageKind::Large);
}

// ---------------------------------------------------------------------------
// Test 2: Large object survives minor GC without moving
// ---------------------------------------------------------------------------

#[test]
fn vm1_large_object_survives_minor_gc_as_pinned() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);
    // Allocate large object spanning 3 pages.
    let n_cells = 2 * PAGE_SIZE_CELLS + 100;
    let obj = alloc_large_object(&mut h, Generation::G0, n_cells);
    let addr_before = obj.raw() & PAYLOAD_MASK;
    let mut root = [obj];

    // Run G0_PROMOTION_THRESHOLD minor GC cycles so the object promotes to G1.
    for _ in 0..G0_PROMOTION_THRESHOLD {
        h.collect_minor(|evac| evac.visit(&mut root[0]));
    }

    // Large object must still be at the same address (never moved).
    let addr_after = root[0].raw() & PAYLOAD_MASK;
    assert_eq!(
        addr_before, addr_after,
        "large object address must not change across GC"
    );

    // All 3 pages must now be in G1 (promoted in-place).
    let page_idx = h.page_of(addr_after as *const u8).unwrap();
    for i in 0..3 {
        assert_eq!(
            h.desc(page_idx + i).generation,
            Generation::G1,
            "page {} of large object must be G1 after promotion",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: Unrooted large object is reclaimed
// ---------------------------------------------------------------------------

#[test]
fn vm1_large_object_reclaimed_when_unrooted() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);
    // Allocate a 3-page large object.
    let n_cells = 2 * PAGE_SIZE_CELLS + 1;
    let obj = alloc_large_object(&mut h, Generation::G0, n_cells);
    let addr = obj.raw() & PAYLOAD_MASK;
    let mut root = [obj];

    // One minor cycle WITH the root → object survives.
    h.collect_minor(|evac| evac.visit(&mut root[0]));
    let page_idx = h.page_of(addr as *const u8).unwrap();
    assert_ne!(
        h.desc(page_idx).generation,
        Generation::Free,
        "object must be live after first GC with root"
    );

    // Drop the root and run another minor cycle → object is unreachable → reclaimed.
    h.collect_minor(|_| {});

    // All 3 pages of the run must now be Free.
    for i in 0..3 {
        assert_eq!(
            h.desc(page_idx + i).generation,
            Generation::Free,
            "page {} must be reclaimed after large object becomes unreachable",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Large and small objects coexist across GC cycles
// ---------------------------------------------------------------------------

#[test]
fn vm1_large_and_small_objects_coexist() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);
    let n_cells = PAGE_SIZE_CELLS + 500;
    let large_obj = alloc_large_object(&mut h, Generation::G0, n_cells);
    let mut large_root = [large_obj];

    // Also allocate a small cons in G0.
    let small_ptr = h.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *small_ptr.as_ptr() = Word::fixnum(7).raw();
        *small_ptr.as_ptr().add(1) = Word::NIL.raw();
    }
    let mut small_root = [Word::from_ptr(
        small_ptr.as_ptr() as *const u8,
        Tag::Cons,
    )];

    // Run a minor GC rooting both objects.
    h.collect_minor(|evac| {
        evac.visit(&mut large_root[0]);
        evac.visit(&mut small_root[0]);
    });

    // Both survive.

    // Check large object payload cell 1 still holds fixnum(42).
    let large_val = unsafe {
        let addr = (large_root[0].raw() & PAYLOAD_MASK) as *const u64;
        Word::from_raw(*addr.add(1)).as_fixnum()
    };
    assert_eq!(large_val, Some(42), "large object payload must be intact");

    // Check small cons car is still fixnum(7).
    let small_val = unsafe {
        let addr = (small_root[0].raw() & PAYLOAD_MASK) as *const u64;
        Word::from_raw(*addr).as_fixnum()
    };
    assert_eq!(small_val, Some(7), "small cons car must be intact");
}

// ---------------------------------------------------------------------------
// Test 5: Simulate a Dylan table bucket array (~24000 cells = 3 pages)
// ---------------------------------------------------------------------------

#[test]
fn vm1_table_buckets_size_24000_cells() {
    // Simulate a 5000-key Dylan table bucket array: ~24000 cells = 3 pages.
    let mut h = Heap::with_reservation(64 * 64 * 1024);
    let n_cells = 24_000;
    let obj = alloc_large_object(&mut h, Generation::G0, n_cells);
    let mut root = [obj];

    // Run enough cycles to promote to G1.
    for _ in 0..G0_PROMOTION_THRESHOLD {
        h.collect_minor(|evac| evac.visit(&mut root[0]));
    }

    let addr = root[0].raw() & PAYLOAD_MASK;
    let page = h.page_of(addr as *const u8).unwrap();
    let head = h.desc(page);
    assert_eq!(
        head.generation,
        Generation::G1,
        "large object must promote to G1"
    );
    assert!(head.is_large_head(), "promoted page must still be a large head");
    assert_eq!(head.n_span, 3, "24000 cells spans 3 pages");
}

// ---------------------------------------------------------------------------
// Test 6: collect_full reclaims dead Tenured large object
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Test 7: Large object's heap-pointer payload keeps small children alive
// ---------------------------------------------------------------------------
//
// Regression for the gap surfaced by the May-2026 review: `mark_scan_object`
// returned immediately for `PageKind::Large`, so a small G0 cell reached
// ONLY via a Large object's payload was never marked. A minor cycle then
// released the small cell's page to Free and the Large object's payload
// pointer dangled (pointing into a zeroed Free page).

#[test]
fn vm1_large_payload_keeps_small_g0_child_alive_across_minor() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);

    // Allocate a small G0 cons; remember its raw value so we can verify
    // it's still readable after GC.
    let small_ptr = h.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *small_ptr.as_ptr() = Word::fixnum(0xdeed).raw();
        *small_ptr.as_ptr().add(1) = Word::NIL.raw();
    }
    let small_word = Word::from_ptr(small_ptr.as_ptr() as *const u8, Tag::Cons);

    // Allocate a 2-page large Vector and write the small cons pointer into
    // its first payload cell. Treat the rest of the payload as fixnum
    // padding so the layout walker only sees the one pointer.
    let n_cells = PAGE_SIZE_CELLS + 1;
    let large_ptr = h
        .try_alloc_large(n_cells, Generation::G0)
        .expect("large alloc");
    unsafe {
        let n_payload = (n_cells - 1) as u32;
        (large_ptr.as_ptr() as *mut u64)
            .write(HeapHeader::new(HeapType::Vector, n_payload).raw());
        large_ptr.as_ptr().add(1).write(small_word.raw());
        for i in 2..n_cells {
            large_ptr.as_ptr().add(i).write(Word::fixnum(0).raw());
        }
        // Card barrier: the mutator just stored a heap pointer into a
        // Large page. Tell the card table so any future card-only scan
        // would see it. (Doesn't matter for an in-G0 reference today
        // since the card scan skips G0, but it documents the contract.)
        h.mark_card_at(large_ptr.as_ptr().add(1) as *const u8);
    }
    let large_word =
        Word::from_ptr(large_ptr.as_ptr() as *const u8, Tag::Vector);
    let mut root = [large_word];

    // One minor cycle. Only the large object is rooted; the small cons is
    // reachable only via the large object's payload pointer. In-gen G0
    // evac will MOVE the small cons to a fresh G0 page; the test below
    // dereferences the (possibly rewritten) payload pointer to find it
    // wherever it ended up.
    h.collect_minor(|evac| evac.visit(&mut root[0]));

    // The Large object's payload pointer must still resolve to a live
    // (non-Free) cell. If the mark BFS skipped the Large payload, the
    // small cons would have been reclaimed and the payload pointer would
    // either dangle (still referencing the old now-Free page) or — once
    // rewriting also gets confused — point at garbage.
    let payload_word = unsafe {
        let large_addr =
            (root[0].raw() & PAYLOAD_MASK) as *const u64;
        Word::from_raw(*large_addr.add(1))
    };
    let payload_addr = payload_word.raw() & PAYLOAD_MASK;
    let payload_page = h.page_of(payload_addr as *const u8).unwrap();
    assert_ne!(
        h.desc(payload_page).generation,
        Generation::Free,
        "Large object's payload pointer must reference a live (non-Free) \
         page after minor GC"
    );

    // And the value at that cell must be the original sentinel (proves no
    // dangle into garbage-zeroed memory).
    let car = unsafe { Word::from_raw(*(payload_addr as *const u64)) };
    assert_eq!(
        car.as_fixnum(),
        Some(0xdeed),
        "child cons car must still hold its sentinel value"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Large payload keeps small G0 child alive across cascade promotion
// ---------------------------------------------------------------------------
//
// Same shape as Test 7, but runs enough minor cycles to drive the Large
// object through G0 → G1 promotion AND the small child through whatever
// cohort it ends up in. The Large object's payload pointer must follow
// the (possibly moved) small child.

#[test]
fn vm1_large_payload_keeps_small_child_alive_across_promotion() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);

    let small_ptr = h.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *small_ptr.as_ptr() = Word::fixnum(0xbeef).raw();
        *small_ptr.as_ptr().add(1) = Word::NIL.raw();
    }
    let small_word = Word::from_ptr(small_ptr.as_ptr() as *const u8, Tag::Cons);

    let n_cells = PAGE_SIZE_CELLS + 1;
    let large_ptr = h
        .try_alloc_large(n_cells, Generation::G0)
        .expect("large alloc");
    unsafe {
        let n_payload = (n_cells - 1) as u32;
        (large_ptr.as_ptr() as *mut u64)
            .write(HeapHeader::new(HeapType::Vector, n_payload).raw());
        large_ptr.as_ptr().add(1).write(small_word.raw());
        for i in 2..n_cells {
            large_ptr.as_ptr().add(i).write(Word::fixnum(0).raw());
        }
        h.mark_card_at(large_ptr.as_ptr().add(1) as *const u8);
    }
    let large_word =
        Word::from_ptr(large_ptr.as_ptr() as *const u8, Tag::Vector);
    let mut root = [large_word];

    for _ in 0..G0_PROMOTION_THRESHOLD * G1_PROMOTION_THRESHOLD {
        h.collect_minor(|evac| evac.visit(&mut root[0]));
    }

    // Verify the Large is still readable and its payload pointer still
    // resolves to a live cell holding the sentinel.
    let payload_word = unsafe {
        let large_addr =
            (root[0].raw() & PAYLOAD_MASK) as *const u64;
        Word::from_raw(*large_addr.add(1))
    };
    let payload_addr = payload_word.raw() & PAYLOAD_MASK;
    let payload_page = h.page_of(payload_addr as *const u8).unwrap();
    assert_ne!(
        h.desc(payload_page).generation,
        Generation::Free,
        "child cell's page must be live after cascade promotion"
    );
    let car = unsafe { Word::from_raw(*(payload_addr as *const u64)) };
    assert_eq!(
        car.as_fixnum(),
        Some(0xbeef),
        "child cell value must persist through promotion"
    );
}

// ---------------------------------------------------------------------------
// Test 10: collect_full preserves pre-pinned Tenured across all three passes
// ---------------------------------------------------------------------------
//
// `collect_full` runs three sub-evacs (G0→G1, G1→Tenured, Tenured→Tenured)
// inside one logical cycle. The conservative pin scan runs ONCE before the
// cycle (or, here, the caller pre-pins via `pin_pointers_in_ranges`); the
// per-cycle pin set must survive every sub-evac so the Tenured compact in
// pass 3 honors the pin and doesn't reclaim the pre-pinned object.
//
// This was previously a bug: each sub-evac's `clear_all_pins` wiped the
// pre-pin between passes, and pass 3's Tenured compact ran with an empty
// pin set, releasing the page. The fix moved pin cleanup from per-evac to
// per-logical-cycle (cycle.rs `collect_minor`/`collect_major`/`collect_full`
// clear pins after all their sub-evacs complete). The same fix also closes
// the cascade pin-loss in `collect_minor` (G0→G1 wiping G1 pins before the
// G1→Tenured cascade) — see `tests/cons_elision_repro.rs`.

#[cfg(feature = "conservative-pin")]
#[test]
fn vm1_collect_full_preserves_pre_pinned_tenured() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);
    // Allocate a small object in G0 and promote it to Tenured.
    let p = h.try_alloc_cons_in(Generation::G0).unwrap();
    unsafe {
        *p.as_ptr() = Word::fixnum(0xfeed).raw();
        *p.as_ptr().add(1) = Word::NIL.raw();
    }
    let mut root = [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];
    let total_minors =
        G0_PROMOTION_THRESHOLD * G1_PROMOTION_THRESHOLD;
    for _ in 0..total_minors {
        h.collect_minor(|evac| evac.visit(&mut root[0]));
    }
    let tenured_addr = root[0].raw() & PAYLOAD_MASK;
    let tenured_page = h.page_of(tenured_addr as *const u8).unwrap();
    assert_eq!(h.desc(tenured_page).generation, Generation::Tenured);

    // Build a fake "stack" containing the Tenured pointer and pre-pin.
    let stack: Box<[u64]> = vec![root[0].raw()].into_boxed_slice();
    let lo = stack.as_ptr() as usize;
    let hi = unsafe { stack.as_ptr().add(stack.len()) } as usize;
    let pin_result = h.pin_pointers_in_ranges(Generation::Tenured, &[(lo, hi)]);
    assert_eq!(pin_result.n_objects, 1, "pre-pin must register the object");

    // collect_full with NO explicit roots. The pre-pin must survive all
    // three sub-evacs (logical-cycle pin lifetime), so pass 3's Tenured
    // compact keeps the pinned object in place.
    let _ = h.collect_full(|_| {});

    // The Tenured object's page should still be Tenured (pinned object
    // stays in place, page FLIPs/keeps gen). Critically, NOT Free.
    assert_ne!(
        h.desc(tenured_page).generation,
        Generation::Free,
        "pre-pinned Tenured object must survive collect_full"
    );
    // The original payload must still be intact.
    unsafe {
        assert_eq!(
            Word::from_raw(*(tenured_addr as *const u64)).as_fixnum(),
            Some(0xfeed),
            "pre-pinned Tenured object's payload must be preserved"
        );
    }

    // Keep `stack` alive to the end of the test so the OS doesn't reuse
    // its pages and confuse the pin scanner on the next run.
    drop(stack);
}

// ---------------------------------------------------------------------------
// Test 9: Tight young_page_cap does not break in-flight evacuation
// ---------------------------------------------------------------------------
//
// young_page_cap caps mutator-driven G0 growth between collections, but the
// allocator must let GC-internal copies grow G0 beyond the cap during an
// in-flight G0 → G0 evac. This test fills G0 right to the cap with rooted
// data, then runs a minor cycle. If the cap were enforced for GC-internal
// allocs, phase1_copy_chunk would panic via `panic_any(GcStallError)`.

#[test]
fn vm1_minor_collect_survives_tight_young_page_cap() {
    use newgc_core::PAGE_SIZE_CELLS as CELLS;
    // 2 young pages + 6 old pages. The cap is 2.
    let mut h = Heap::new(2 * 64 * 1024, 6 * 64 * 1024);

    // Fill G0 to the cap with rooted cons cells. We pack ~2 pages
    // worth (cons cells use 2 cells each → 4096 conses per page).
    let mut roots: Vec<Word> = Vec::new();
    for i in 0..(2 * CELLS / 2) {
        let p = h.try_alloc_cons_in(Generation::G0).expect("under cap");
        unsafe {
            *p.as_ptr() = Word::fixnum(i as i64).raw();
            *p.as_ptr().add(1) = Word::NIL.raw();
        }
        roots.push(Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons));
    }
    assert!(
        h.try_alloc_cons_in(Generation::G0).is_none(),
        "mutator must hit the young cap before GC"
    );

    // Minor cycle. With everything rooted, the survivors must go
    // somewhere. The cap-bypass for GC-internal allocs lets that
    // happen; without the bypass, this call would panic.
    let n_before = roots.len();
    h.collect_minor(|evac| {
        for r in roots.iter_mut() {
            evac.visit(r);
        }
    });
    assert_eq!(roots.len(), n_before);

    // After GC, every rooted cons must still be in a live (non-Free) page.
    for r in &roots {
        let addr = (r.raw() & PAYLOAD_MASK) as *const u8;
        let page = h.page_of(addr).unwrap();
        assert_ne!(
            h.desc(page).generation,
            Generation::Free,
            "rooted cons must survive the tight-cap minor cycle"
        );
    }
}

#[test]
fn vm1_collect_full_reclaims_dead_large_object_in_tenured() {
    // Promote a large object all the way to Tenured, then reclaim it.
    let mut h = Heap::with_reservation(256 * 64 * 1024);
    let n_cells = PAGE_SIZE_CELLS + 1; // 2 pages
    let obj = alloc_large_object(&mut h, Generation::G0, n_cells);
    let mut root = [obj];

    // Promote to G1 via minor GCs.
    let total_minors =
        G0_PROMOTION_THRESHOLD * G1_PROMOTION_THRESHOLD;
    for _ in 0..total_minors {
        h.collect_minor(|evac| evac.visit(&mut root[0]));
    }

    let addr = root[0].raw() & PAYLOAD_MASK;
    let page = h.page_of(addr as *const u8).unwrap();
    let page_gen = h.desc(page).generation;
    assert_eq!(page_gen, Generation::Tenured, "large object must be Tenured");

    // Now collect_full with no root → object unreachable → reclaimed.
    let result = h.collect_full(|_| {});
    assert!(
        result.tenured_freed_bytes > 0 || result.tenured_evac.pages_freed > 0,
        "collect_full must reclaim the dead large Tenured object: {:?}",
        result
    );
    assert_eq!(
        h.desc(page).generation,
        Generation::Free,
        "large object pages must be Free after collect_full"
    );
}
