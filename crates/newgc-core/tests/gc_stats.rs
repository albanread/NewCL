//! Tests for the `GcStats` snapshot API.

use newgc_core::page_heap::space::PageHeap;
use newgc_core::{Generation, GcStats, LispLayout, Tag, Word};

type Heap = PageHeap<LispLayout>;

fn cons(h: &mut Heap, g: Generation, car: Word, cdr: Word) -> Word {
    let p = h.try_alloc_cons_in(g).expect("cons alloc");
    unsafe {
        *p.as_ptr() = car.raw();
        *p.as_ptr().add(1) = cdr.raw();
    }
    h.mark_card_at(p.as_ptr() as *const u8);
    Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
}

#[test]
fn fresh_heap_has_zero_used_bytes_in_all_gens() {
    let h = Heap::with_reservation(16 * 64 * 1024);
    let s = h.stats();
    assert_eq!(s.g0_used_bytes, 0);
    assert_eq!(s.g1_used_bytes, 0);
    assert_eq!(s.tenured_used_bytes, 0);
    assert_eq!(s.total_used_bytes, 0);
}

#[test]
fn capacity_fields_reflect_reservation() {
    let h = Heap::with_reservation(16 * 64 * 1024);
    let s = h.stats();
    assert_eq!(s.reserved_bytes, 16 * 64 * 1024);
    assert_eq!(s.page_count, 16);
}

#[test]
fn alloc_updates_g0_pages_and_bytes() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    for _ in 0..100 {
        let _ = cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL);
    }
    let s = h.stats();
    assert!(s.g0_pages > 0, "G0 should hold at least one page");
    // 100 cons = 1600 bytes
    assert_eq!(s.g0_used_bytes, 1600);
    assert_eq!(s.total_used_bytes, 1600);
    assert_eq!(s.bytes_alloc_since_gc, 1600);
}

#[test]
fn stats_consistency_total_equals_sum() {
    let mut h = Heap::with_reservation(32 * 64 * 1024);
    let head = cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL);
    let mut roots = [head];
    for _ in 0..3 {
        h.collect_minor(|e| {
            for r in roots.iter_mut() {
                e.visit(r);
            }
        });
    }
    let s = h.stats();
    assert_eq!(
        s.total_used_bytes,
        s.g0_used_bytes + s.g1_used_bytes + s.tenured_used_bytes
    );
    let total_pages = s.g0_pages + s.g1_pages + s.tenured_pages + s.free_pages;
    assert_eq!(total_pages, s.page_count);
}

#[test]
fn stats_after_promotion_shows_g1_occupancy() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    // Build a 50-cell list.
    let mut head = Word::NIL;
    for i in 0..50 {
        head = cons(&mut h, Generation::G0, Word::fixnum(i), head);
    }
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    let s = h.stats();
    assert!(s.g1_used_bytes > 0, "G1 should hold the survivors");
    assert_eq!(s.g0_used_bytes, 0, "G0 emptied by evac");
}

#[test]
fn collect_resets_bytes_alloc_since_gc() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    for _ in 0..100 {
        let _ = cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL);
    }
    assert!(h.stats().bytes_alloc_since_gc > 0);
    h.collect_minor(|_| {});
    h.recompute_auto_trigger();
    assert_eq!(h.stats().bytes_alloc_since_gc, 0);
}

#[test]
fn stats_render_includes_all_sections() {
    let mut h = Heap::with_reservation(8 * 64 * 1024);
    for _ in 0..10 {
        let _ = cons(&mut h, Generation::G0, Word::NIL, Word::NIL);
    }
    let s = h.stats();
    let line = s.render();
    // Each section's leading key should appear.
    for key in [
        "reserved=", "committed=", "pages=", "g0=", "g1=", "tenured=",
        "free=", "alloc_since_gc=", "trigger=", "budget_min=",
        "tenured_thresh_bps=", "last_mark=", "zero_live_released=",
        "pin_objs=", "pin_cells=", "minors_since_promote=",
    ] {
        assert!(line.contains(key), "render missing key `{key}`: {line}");
    }
}

#[test]
fn stats_struct_is_clone_copy() {
    let h = Heap::with_reservation(8 * 64 * 1024);
    let s1: GcStats = h.stats();
    let s2 = s1;  // Copy
    let _s3 = s1.clone();
    assert_eq!(s1, s2);
}

#[test]
fn default_gc_stats_is_zero() {
    let s = GcStats::default();
    assert_eq!(s.reserved_bytes, 0);
    assert_eq!(s.total_used_bytes, 0);
}
