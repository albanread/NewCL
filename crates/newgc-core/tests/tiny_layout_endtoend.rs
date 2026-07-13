//! End-to-end tests for `PageHeap<TinyLayout>`.
//!
//! These exercise the GC engine through a completely different
//! layout than `LispLayout` — 2-bit tag, no NCL Word, no NCL header
//! type tag. If these pass, the `HeapLayout` trait surface is
//! genuinely polymorphic at runtime, not just at type-level.
//!
//! What this catches that pure-trait-shape tests don't:
//!   - The GC's tag-vs-page-kind gates work for any 2-state
//!     (cons / header) classification.
//!   - Forwarding markers are correctly distinguished from pointers.
//!   - `rewrite_pointer_addr` preserves the language's tag bits
//!     across evacuation.
//!   - `header_layout` is called once per object and its
//!     `total_cells` and pointer-range answers are honored.

use newgc_core::page_heap::space::PageHeap;
use newgc_core::tiny_layout::{
    tiny_cons_ptr, tiny_fixnum, tiny_header, tiny_header_ptr, TinyLayout,
    TINY_PAYLOAD_MASK,
};
use newgc_core::traits::WordKind;
use newgc_core::word::Word;
use newgc_core::{Generation, HeapLayout};

type TinyHeap = PageHeap<TinyLayout>;

fn small_heap() -> TinyHeap {
    TinyHeap::with_reservation(8 * 64 * 1024)
}

fn medium_heap() -> TinyHeap {
    TinyHeap::with_reservation(32 * 64 * 1024)
}

/// Allocate a cons `(car . cdr)` returning the TinyLayout-tagged Word.
fn cons(h: &mut TinyHeap, g: Generation, car: u64, cdr: u64) -> Word {
    let p = h.try_alloc_cons_in(g).expect("cons alloc");
    unsafe {
        *p.as_ptr() = car;
        *p.as_ptr().add(1) = cdr;
    }
    Word::from_raw(tiny_cons_ptr(p.as_ptr() as *const u8))
}

/// Allocate a header-bearing object with `n_payload` pointer-typed
/// slots, all initialised to `init`. Returns the TinyLayout-tagged
/// Word pointer to the header cell.
fn boxed(h: &mut TinyHeap, g: Generation, n_payload: u32, init: u64) -> Word {
    let total = (1 + n_payload) as usize;
    let p = h.try_alloc_boxed_in(g, total).expect("boxed alloc");
    unsafe {
        *p.as_ptr() = tiny_header(n_payload);
        for i in 1..=n_payload as usize {
            *p.as_ptr().add(i) = init;
        }
    }
    Word::from_raw(tiny_header_ptr(p.as_ptr() as *const u8))
}

unsafe fn read_slot(w: Word, idx: usize) -> u64 {
    let base = (w.raw() & TINY_PAYLOAD_MASK) as *const u64;
    unsafe { *base.add(idx) }
}

unsafe fn write_slot(w: Word, idx: usize, value: u64) {
    let base = (w.raw() & TINY_PAYLOAD_MASK) as *mut u64;
    unsafe { *base.add(idx) = value }
}

/// Build a (0 1 2 … n-1) linked list. Returns the head.
fn list(h: &mut TinyHeap, g: Generation, n: usize) -> Word {
    let mut head = tiny_fixnum(0);
    for i in (0..n).rev() {
        let c = cons(h, g, tiny_fixnum(i as i64), head);
        head = c.raw();
    }
    Word::from_raw(head)
}

unsafe fn list_to_vec(head: Word) -> Vec<i64> {
    let mut v = Vec::new();
    let mut cur = head.raw();
    while let WordKind::PointerCons(addr) = TinyLayout::classify(cur) {
        let ptr = addr as *const u64;
        let car = unsafe { *ptr };
        let cdr = unsafe { *ptr.add(1) };
        // car must be a fixnum
        match TinyLayout::classify(car) {
            WordKind::Immediate => v.push((car as i64) >> 2),
            other => panic!("car not fixnum: {other:?}"),
        }
        cur = cdr;
    }
    // Tail is the terminating fixnum 0 (set up by list()).
    assert!(matches!(
        TinyLayout::classify(cur),
        WordKind::Immediate
    ));
    v
}

// -- Tests ------------------------------------------------------------------

#[test]
fn allocates_and_evacuates_a_single_cons() {
    let mut h = small_heap();
    let c = cons(&mut h, Generation::G0, tiny_fixnum(42), tiny_fixnum(0));
    let mut roots = [c];
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    assert_eq!(result.objects_copied, 1);
    assert_eq!(result.cells_copied, 2);

    // New root points into G1.
    let new = roots[0];
    assert!(matches!(
        TinyLayout::classify(new.raw()),
        WordKind::PointerCons(_)
    ));

    // Cell contents preserved.
    let car = unsafe { read_slot(new, 0) };
    let cdr = unsafe { read_slot(new, 1) };
    assert_eq!(car, tiny_fixnum(42));
    assert_eq!(cdr, tiny_fixnum(0));
}

#[test]
fn deep_chain_survives_evacuation() {
    let mut h = medium_heap();
    let head = list(&mut h, Generation::G0, 500);
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    let elems = unsafe { list_to_vec(roots[0]) };
    assert_eq!(elems.len(), 500);
    for (i, v) in elems.iter().enumerate() {
        assert_eq!(*v, i as i64);
    }
}

#[test]
fn unrooted_objects_reclaimed() {
    let mut h = small_heap();
    let _ = list(&mut h, Generation::G0, 100);
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut [],
    );
    assert_eq!(result.objects_copied, 0);
    assert_eq!(h.count_pages_in_gen(Generation::G0), 0);
}

#[test]
fn boxed_object_survives_with_pointer_payload() {
    let mut h = small_heap();
    let leaf = cons(&mut h, Generation::G0, tiny_fixnum(7), tiny_fixnum(0));
    let vec = boxed(&mut h, Generation::G0, 3, tiny_fixnum(0));
    unsafe {
        write_slot(vec, 1, leaf.raw());
        write_slot(vec, 2, tiny_fixnum(99));
        write_slot(vec, 3, leaf.raw()); // shared with slot 1
    }
    let mut roots = [vec];
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    // 2 unique objects: boxed + leaf.
    assert_eq!(result.objects_copied, 2);

    // Verify payload.
    let new = roots[0];
    let slot1 = unsafe { read_slot(new, 1) };
    let slot2 = unsafe { read_slot(new, 2) };
    let slot3 = unsafe { read_slot(new, 3) };
    assert!(matches!(
        TinyLayout::classify(slot1),
        WordKind::PointerCons(_)
    ));
    assert_eq!(slot2, tiny_fixnum(99));
    assert_eq!(slot1, slot3, "shared leaf was duplicated");
}

#[test]
fn diamond_dag_not_duplicated() {
    let mut h = small_heap();
    let leaf = cons(&mut h, Generation::G0, tiny_fixnum(7), tiny_fixnum(0));
    let left = cons(&mut h, Generation::G0, leaf.raw(), tiny_fixnum(0));
    let right = cons(&mut h, Generation::G0, leaf.raw(), tiny_fixnum(0));
    let root = cons(&mut h, Generation::G0, left.raw(), right.raw());

    let mut roots = [root];
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    assert_eq!(result.objects_copied, 4);

    // Verify both paths reach the same leaf.
    let new_root = roots[0];
    let new_left_raw = unsafe { read_slot(new_root, 0) };
    let new_right_raw = unsafe { read_slot(new_root, 1) };
    let new_left = Word::from_raw(new_left_raw);
    let new_right = Word::from_raw(new_right_raw);
    let left_leaf = unsafe { read_slot(new_left, 0) };
    let right_leaf = unsafe { read_slot(new_right, 0) };
    assert_eq!(left_leaf, right_leaf, "diamond leaf duplicated");
}

#[test]
fn self_referential_cons_terminates() {
    let mut h = small_heap();
    let p = h.try_alloc_cons_in(Generation::G0).unwrap();
    let w_raw = tiny_cons_ptr(p.as_ptr() as *const u8);
    unsafe {
        *p.as_ptr() = w_raw;
        *p.as_ptr().add(1) = w_raw;
    }
    let mut roots = [Word::from_raw(w_raw)];
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    assert_eq!(result.objects_copied, 1);
    let new = roots[0].raw();
    let car = unsafe { read_slot(roots[0], 0) };
    let cdr = unsafe { read_slot(roots[0], 1) };
    assert_eq!(car, new);
    assert_eq!(cdr, new);
}

#[test]
fn random_graph_reachability_preserved() {
    let mut h = medium_heap();
    let mut state: u64 = 0xc0ffee;
    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }

    let n = 300usize;
    let mut all: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        let cdr = if i == 0 || (lcg(&mut state) & 3) == 0 {
            tiny_fixnum(0)
        } else {
            all[(lcg(&mut state) as usize) % i]
        };
        let c = cons(&mut h, Generation::G0, tiny_fixnum(i as i64), cdr);
        all.push(c.raw());
    }

    // Pick 10 random roots.
    let mut roots: Vec<Word> = (0..10)
        .map(|_| Word::from_raw(all[(lcg(&mut state) as usize) % n]))
        .collect();

    // Pre-cycle reachable set.
    let mut reachable_pre = std::collections::HashSet::new();
    for &r in &roots {
        let mut cur = r.raw();
        while let WordKind::PointerCons(addr) = TinyLayout::classify(cur) {
            let car = unsafe { *(addr as *const u64) };
            let cdr = unsafe { *((addr as *const u64).add(1)) };
            match TinyLayout::classify(car) {
                WordKind::Immediate => {
                    reachable_pre.insert((car as i64) >> 2);
                }
                _ => panic!("car not immediate"),
            }
            cur = cdr;
        }
    }

    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    assert_eq!(result.objects_copied, reachable_pre.len());

    // Post-cycle reachable from new roots.
    let mut reachable_post = std::collections::HashSet::new();
    for &r in &roots {
        let mut cur = r.raw();
        while let WordKind::PointerCons(addr) = TinyLayout::classify(cur) {
            let car = unsafe { *(addr as *const u64) };
            let cdr = unsafe { *((addr as *const u64).add(1)) };
            if let WordKind::Immediate = TinyLayout::classify(car) {
                reachable_post.insert((car as i64) >> 2);
            }
            cur = cdr;
        }
    }
    assert_eq!(reachable_post, reachable_pre);
}

#[test]
fn allocate_until_oom_returns_none() {
    let mut h = TinyHeap::with_reservation(4 * 64 * 1024);
    let mut n = 0usize;
    while h.try_alloc_cons_in(Generation::G0).is_some() {
        n += 1;
        if n > 100_000 {
            panic!("alloc runaway");
        }
    }
    assert!(n > 0);
    // Reclaim and re-alloc.
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut []);
    assert!(h.try_alloc_cons_in(Generation::G0).is_some());
}

#[test]
fn major_collect_reclaims_and_preserves() {
    let mut h = medium_heap();
    let head = list(&mut h, Generation::G0, 100);
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    assert!(h.count_pages_in_gen(Generation::G1) > 0);

    // Major with the root preserved.
    h.collect_major(|evac| {
        for r in roots.iter_mut() {
            evac.visit(r);
        }
    });
    let elems = unsafe { list_to_vec(roots[0]) };
    assert_eq!(elems.len(), 100);
    for (i, v) in elems.iter().enumerate() {
        assert_eq!(*v, i as i64);
    }
}

#[test]
fn fill_word_is_immediate_so_fresh_pages_are_safe() {
    // After a page is committed and zeroed, every cell reads as
    // TINY_TAG_IMMEDIATE. A conservative scan over a freshly
    // committed page must not pin spurious pointers.
    let mut h = small_heap();
    // Force a page commit by allocating one cons.
    let _ = cons(&mut h, Generation::G0, tiny_fixnum(0), tiny_fixnum(0));
    // Walk the (committed) page's cells; every uninitialised cell
    // should classify as Immediate.
    let base = h.base_ptr() as *const u64;
    let page_cells = 8192;
    let mut non_immediate = 0;
    for i in 0..page_cells {
        let raw = unsafe { *base.add(i) };
        if !matches!(TinyLayout::classify(raw), WordKind::Immediate) {
            non_immediate += 1;
        }
    }
    // The one allocated cons contributes 2 non-immediate cells
    // (its tagged car & cdr both happen to be fixnum 0 → immediate).
    // The cell containing the cons header is set by alloc to nil
    // (FILL_WORD = 0) — also immediate. So technically EVERY cell
    // should be immediate. The cons was set up with car=fixnum(0),
    // cdr=fixnum(0); both immediate. So zero non-immediate cells.
    assert_eq!(non_immediate, 0);
}
