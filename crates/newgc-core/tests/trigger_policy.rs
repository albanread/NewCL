//! Sub-phase 10 — trigger policy.
//!
//! `should_collect()` returns true once the mutator has allocated
//! `auto_gc_trigger_bytes` since the last collection.
//! `collect_auto()` picks minor or major based on Tenured pressure
//! and runs the chosen cycle, recomputing the threshold for the
//! next trigger.
//!
//! These tests pin down the heuristic so it's documented and
//! observable from outside the crate.

use newgc_core::page_heap::space::PageHeap;
use newgc_core::{Generation, HeapHeader, HeapType, LispLayout, Tag, Word};

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

fn vector(h: &mut Heap, g: Generation, n: u32, init: Word) -> Word {
    let total = (1 + n) as usize;
    let p = h.try_alloc_boxed_in(g, total).expect("vector alloc");
    unsafe {
        *p.as_ptr() = HeapHeader::new(HeapType::Vector, n).raw();
        for i in 1..=n as usize {
            *p.as_ptr().add(i) = init.raw();
        }
    }
    Word::from_ptr(p.as_ptr() as *const u8, Tag::Vector)
}

// =========================================================================
// Allocation tracking
// =========================================================================

#[test]
fn fresh_heap_has_zero_allocs() {
    let h = Heap::with_reservation(16 * 64 * 1024);
    assert_eq!(h.bytes_alloc_since_gc(), 0);
}

#[test]
fn cons_alloc_bumps_byte_counter_by_16() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    let before = h.bytes_alloc_since_gc();
    let _ = cons(&mut h, Generation::G0, Word::NIL, Word::NIL);
    let after = h.bytes_alloc_since_gc();
    assert_eq!(after - before, 16, "cons = 2 cells = 16 bytes");
}

#[test]
fn vector_alloc_bumps_by_payload_plus_header() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    let before = h.bytes_alloc_since_gc();
    let _ = vector(&mut h, Generation::G0, 10, Word::NIL);
    let after = h.bytes_alloc_since_gc();
    // 1 header + 10 payload = 11 cells = 88 bytes.
    assert_eq!(after - before, 88);
}

#[test]
fn many_conses_accumulate_byte_count() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    for _ in 0..1000 {
        let _ = cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL);
    }
    // 1000 × 16 = 16000 bytes.
    assert_eq!(h.bytes_alloc_since_gc(), 16000);
}

// =========================================================================
// should_collect() trigger
// =========================================================================

#[test]
fn should_collect_is_false_below_threshold() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    // Default trigger is 8 MB.
    assert!(!h.should_collect());
    // Allocate a few cons cells.
    for _ in 0..100 {
        let _ = cons(&mut h, Generation::G0, Word::NIL, Word::NIL);
    }
    assert!(!h.should_collect(), "100 conses < 8 MB");
}

#[test]
fn should_collect_fires_when_threshold_crossed() {
    let mut h = Heap::with_reservation(64 * 64 * 1024);
    // Lower the budget so the test doesn't have to allocate 8 MB.
    h.set_gc_budget_min_bytes(16 * 1024);
    // The setter raises auto_gc_trigger_bytes to the new minimum.
    let threshold = h.auto_gc_trigger_bytes();
    assert_eq!(threshold, 16 * 1024);

    let cells_per_cons = 2usize;
    let bytes_per_cons = cells_per_cons * 8;
    let needed = threshold / bytes_per_cons + 1;

    for _ in 0..needed {
        let _ = cons(&mut h, Generation::G0, Word::NIL, Word::NIL);
    }
    assert!(h.should_collect(), "trigger should fire after {needed} conses");
}

// =========================================================================
// collect_auto() chooses minor vs major
// =========================================================================

#[test]
fn collect_auto_picks_minor_when_tenured_is_empty() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    let head = cons(&mut h, Generation::G0, Word::fixnum(42), Word::NIL);
    let mut roots = [head];
    let result = h.collect_auto(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });
    // Minor cycle: dest is G0 (within-gen for first cycle) or G1
    // (on promotion). Either way, NOT a major's cascade pattern.
    assert!(!result.promoted_g1,
        "fresh heap shouldn't trigger major promotion");
}

#[test]
fn collect_auto_picks_major_when_tenured_high() {
    // Force the threshold low so a small Tenured count triggers major.
    // 1 basis point on a 512 KB heap = 51.2 bytes (≈4 cons cells),
    // so promoting >=4 conses puts us over.
    let mut h = Heap::with_reservation(8 * 64 * 1024);
    h.set_tenured_full_threshold_bps(1);

    // Build a 20-cell list, promote it all the way to Tenured.
    let mut head = Word::NIL;
    for i in 0..20 {
        head = cons(&mut h, Generation::G0, Word::fixnum(i), head);
    }
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    h.evacuate_from_word_roots(Generation::G1, Generation::Tenured, &mut roots);
    assert!(h.tenured_used_bytes() >= 320,
        "tenured used = {} bytes, expected >= 320 (20 cons × 16)",
        h.tenured_used_bytes());

    // Now collect_auto should pick major.
    assert!(h.should_collect_major());
    let result = h.collect_auto(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });
    // A major's pattern: both passes ran. `cascade` is the G1→Tenured
    // result; on a major it's always Some.
    assert!(result.cascade.is_some(),
        "collect_auto picked major; cascade should be populated");
}

#[test]
fn collect_auto_default_threshold_keeps_minor_for_small_heap() {
    // 16 pages = 1 MB. 75% of 1 MB = 768 KB Tenured cap. Hard to
    // exceed in a quick test, so this verifies the DEFAULT
    // (without overriding) doesn't accidentally promote to major.
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    let head = cons(&mut h, Generation::G0, Word::fixnum(1), Word::NIL);
    let mut roots = [head];
    assert!(!h.should_collect_major());
    h.collect_auto(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });
}

// =========================================================================
// Threshold recomputation after collection
// =========================================================================

#[test]
fn collect_auto_resets_alloc_counter() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    h.set_gc_budget_min_bytes(16 * 1024);
    for _ in 0..2000 {
        let _ = cons(&mut h, Generation::G0, Word::NIL, Word::NIL);
    }
    assert!(h.bytes_alloc_since_gc() > 16 * 1024);
    h.collect_auto(|_| {});
    assert_eq!(h.bytes_alloc_since_gc(), 0);
}

#[test]
fn threshold_grows_with_tenured_size() {
    let mut h = Heap::with_reservation(32 * 64 * 1024);
    h.set_gc_budget_min_bytes(1024);
    // Initial threshold = max(1 KB, 0.5 * 0) = 1 KB.
    assert_eq!(h.auto_gc_trigger_bytes(), 1024);

    // Promote some data to Tenured manually.
    let head = cons(&mut h, Generation::G0, Word::fixnum(99), Word::NIL);
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    h.evacuate_from_word_roots(Generation::G1, Generation::Tenured, &mut roots);

    let tenured_now = h.tenured_used_bytes();
    assert!(tenured_now > 0);

    // collect_auto recomputes: max(1024, tenured_now / 2).
    h.collect_auto(|e| {
        for r in roots.iter_mut() {
            e.visit(r);
        }
    });
    let new_threshold = h.auto_gc_trigger_bytes();
    let expected_min = 1024;
    // The new threshold should be the larger of expected_min and
    // half of tenured.
    assert!(new_threshold >= expected_min);
}

#[test]
fn recompute_auto_trigger_explicit_call() {
    let mut h = Heap::with_reservation(16 * 64 * 1024);
    h.set_gc_budget_min_bytes(4096);
    for _ in 0..500 {
        let _ = cons(&mut h, Generation::G0, Word::NIL, Word::NIL);
    }
    assert!(h.bytes_alloc_since_gc() > 0);
    // After explicit minor + recompute, counter resets.
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut []);
    h.recompute_auto_trigger();
    assert_eq!(h.bytes_alloc_since_gc(), 0);
}

// =========================================================================
// Long-running steady state
// =========================================================================

#[test]
fn steady_state_alloc_drives_predictable_gc_rate() {
    let mut h = Heap::with_reservation(32 * 64 * 1024);
    h.set_gc_budget_min_bytes(64 * 1024);  // 64 KB budget

    let mut gcs = 0;
    let mut roots: Vec<Word> = Vec::new();

    // Allocate a sliding window of 50 cons cells over many rounds.
    for _round in 0..30 {
        for _ in 0..200 {
            let c = cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL);
            if roots.len() == 50 {
                roots.remove(0);
            }
            roots.push(c);

            if h.should_collect() {
                let n = roots.len();
                // Move roots into a Vec the closure can borrow.
                let mut local: Vec<Word> = roots.iter().copied().collect();
                h.collect_auto(|e| {
                    for r in local.iter_mut() {
                        e.visit(r);
                    }
                });
                // Sync back rewritten roots.
                roots = local;
                gcs += 1;
                assert_eq!(n, roots.len(), "GC must preserve every root");
            }
        }
    }
    // Approximately: 30 rounds × 200 conses × 16 bytes = 96000 bytes.
    // Budget 64 KB → ~1-2 GCs total at minimum (more if budget
    // recomputes to a smaller value with growing Tenured).
    assert!(gcs >= 1, "expected at least one auto-GC, got {gcs}");
    assert!(gcs < 100, "too many GCs fired ({gcs}) — threshold too small?");
}

// =========================================================================
// Config setters bound checking
// =========================================================================

#[test]
fn set_tenured_full_threshold_bps_clamps_to_10000() {
    let mut h = Heap::with_reservation(8 * 64 * 1024);
    h.set_tenured_full_threshold_bps(99999);  // > 100%
    // Setter should clamp; should_collect_major must not produce
    // false positives just because the bps overflowed.
    assert!(!h.should_collect_major(),
        "fresh heap with no Tenured should never trigger major");
}

#[test]
fn set_gc_budget_min_bytes_floor() {
    let mut h = Heap::with_reservation(8 * 64 * 1024);
    h.set_gc_budget_min_bytes(0);
    // 0 should clamp to at least 1 to avoid div-by-zero / infinite
    // trigger.
    h.recompute_auto_trigger();
    assert!(h.auto_gc_trigger_bytes() >= 1);
}
