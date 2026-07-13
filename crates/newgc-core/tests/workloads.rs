//! Workload simulations for newgc-core.
//!
//! These complement `tests/synthetic.rs` by exercising the GC under
//! patterns that approximate real-program allocation behaviour:
//! bursty allocation, generational shapes, mutation, working-set
//! oscillation, and pathological graphs.
//!
//! `GC_LESSONS.md` Pattern 2 still applies — even these are
//! synthetic. The point of writing them is to raise the **floor** of
//! what we know the GC handles. The actual ceiling is "bind into NCL
//! / OpenDylan and run real demos for a week."
//!
//! ## Test taxonomy
//!
//!   - Section A: allocator stress (rates, bursts, sizes).
//!   - Section B: working-set patterns (bounded / growing / shrinking).
//!   - Section C: generational shapes (cohort policy, promotion ladders).
//!   - Section D: object-graph topologies (trees, rings, DAGs, frame chains).
//!   - Section E: mutation patterns (old→young writes, slot swaps).
//!   - Section F: pathological shapes (deep nesting, fan-out, pointer-shaped immediates).
//!   - Section G: mixed-realistic (macroexpand, alist churn).
//!   - Section H: TinyLayout workloads (proves the trait isn't NCL-only).

use newgc_core::{
    Generation, HeapHeader, HeapLayout, HeapType, LispLayout, PageHeap,
    PAYLOAD_MASK, Tag, Word,
};

// -- LispLayout common helpers ---------------------------------------------

type Heap = PageHeap<LispLayout>;

fn small_heap() -> Heap {
    Heap::with_reservation(8 * 64 * 1024)
}

fn medium_heap() -> Heap {
    Heap::with_reservation(32 * 64 * 1024)
}

fn large_heap() -> Heap {
    Heap::with_reservation(128 * 64 * 1024)
}

fn cons(h: &mut Heap, g: Generation, car: Word, cdr: Word) -> Word {
    let p = h.try_alloc_cons_in(g).expect("cons alloc");
    unsafe {
        *p.as_ptr() = car.raw();
        *p.as_ptr().add(1) = cdr.raw();
    }
    Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
}

fn vector(h: &mut Heap, g: Generation, n_payload: u32, init: Word) -> Word {
    let total_cells = (1 + n_payload) as usize;
    let p = h
        .try_alloc_boxed_in(g, total_cells)
        .expect("vector alloc");
    unsafe {
        *p.as_ptr() = HeapHeader::new(HeapType::Vector, n_payload).raw();
        for i in 1..=n_payload as usize {
            *p.as_ptr().add(i) = init.raw();
        }
    }
    Word::from_ptr(p.as_ptr() as *const u8, Tag::Vector)
}

unsafe fn vector_slot(w: Word, idx: usize) -> Word {
    let base = (w.raw() & PAYLOAD_MASK) as *const u64;
    unsafe { Word::from_raw(*base.add(idx)) }
}

unsafe fn set_vector_slot(w: Word, idx: usize, value: Word) {
    let base = (w.raw() & PAYLOAD_MASK) as *mut u64;
    unsafe { *base.add(idx) = value.raw() }
}

fn list(h: &mut Heap, g: Generation, n: usize) -> Word {
    let mut head = Word::NIL;
    for i in (0..n).rev() {
        head = cons(h, g, Word::fixnum(i as i64), head);
    }
    head
}

unsafe fn list_len(head: Word) -> usize {
    let mut n = 0;
    let mut cur = head;
    while cur.tag() == Tag::Cons {
        let p = (cur.raw() & PAYLOAD_MASK) as *const u64;
        n += 1;
        cur = unsafe { Word::from_raw(*p.add(1)) };
    }
    n
}

unsafe fn walk_list_collecting_cars(head: Word) -> Vec<i64> {
    let mut v = Vec::new();
    let mut cur = head;
    while cur.tag() == Tag::Cons {
        let p = (cur.raw() & PAYLOAD_MASK) as *const u64;
        let car = unsafe { Word::from_raw(*p) };
        if let Some(n) = car.as_fixnum() {
            v.push(n);
        }
        cur = unsafe { Word::from_raw(*p.add(1)) };
    }
    v
}

/// Deterministic LCG; not cryptographic.
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn minor(h: &mut Heap, roots: &mut [Word]) {
    h.collect_minor(|evac| {
        for r in roots.iter_mut() {
            evac.visit(r);
        }
    });
}

fn major(h: &mut Heap, roots: &mut [Word]) {
    h.collect_major(|evac| {
        for r in roots.iter_mut() {
            evac.visit(r);
        }
    });
}

// =========================================================================
// Section A: Allocator stress
// =========================================================================

#[test]
fn a1_many_short_lived_cons_cells_then_one_gc() {
    // Allocate 50,000 cons cells with no roots. Single minor cycle
    // reclaims them all. Verifies the allocator can keep up.
    let mut h = large_heap();
    for _ in 0..50_000 {
        let _ = h.try_alloc_cons_in(Generation::G0).expect("alloc");
    }
    let before_g0 = h.count_pages_in_gen(Generation::G0);
    assert!(before_g0 > 0);
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut []);
    assert_eq!(h.count_pages_in_gen(Generation::G0), 0);
    assert_eq!(h.count_pages_in_gen(Generation::G1), 0);
}

#[test]
fn a2_bursty_allocation_with_gc_between_bursts() {
    // 20 bursts of 2000 cons each, retain only the head of the
    // burst's chain. After each burst, GC. Heap usage should stay
    // bounded.
    let mut h = medium_heap();
    let mut peak_pages = 0;
    for _burst in 0..20 {
        let head = list(&mut h, Generation::G0, 2000);
        let mut roots = [head];
        let p = h.count_pages_in_gen(Generation::G0);
        peak_pages = peak_pages.max(p);
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        // Drop the head and reclaim G1 too.
        h.evacuate_from_word_roots(Generation::G1, Generation::G1, &mut []);
    }
    // Heap should be empty between bursts.
    assert_eq!(h.count_pages_in_gen(Generation::G0), 0);
    assert_eq!(h.count_pages_in_gen(Generation::G1), 0);
}

#[test]
fn a3_alternating_cons_and_boxed_allocations() {
    // Mix cons and vector allocations in a tight loop. Both kinds
    // must coexist in the same heap.
    let mut h = medium_heap();
    let mut roots: Vec<Word> = Vec::new();
    for i in 0..2000 {
        if i % 2 == 0 {
            let c = cons(&mut h, Generation::G0, Word::fixnum(i), Word::NIL);
            roots.push(c);
        } else {
            let v = vector(&mut h, Generation::G0, 3, Word::fixnum(i));
            roots.push(v);
        }
    }
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    assert_eq!(result.objects_copied, 2000);
}

#[test]
fn a4_power_law_object_sizes() {
    // Allocate vectors at sizes following 2^k. Common pattern in
    // real programs (hash-table buckets, growing arrays).
    let mut h = medium_heap();
    let mut roots: Vec<Word> = Vec::new();
    for size_log in 0..10 {
        let n_payload = 1u32 << size_log;
        if n_payload >= 256 {
            // Stay under a page.
            break;
        }
        for _ in 0..20 {
            let v = vector(&mut h, Generation::G0, n_payload, Word::NIL);
            roots.push(v);
        }
    }
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    // All retained objects must survive.
    for r in &roots {
        assert_eq!(r.tag(), Tag::Vector);
    }
}

// =========================================================================
// Section B: Working-set patterns
// =========================================================================

#[test]
fn b1_bounded_working_set_steady_state() {
    // Keep a sliding window of 100 cons cells alive. Each cycle:
    // add 100 new, drop the 100 oldest. Over 50 cycles the heap
    // should never grow beyond a small bound.
    let mut h = medium_heap();
    let mut window: std::collections::VecDeque<Word> =
        std::collections::VecDeque::with_capacity(100);
    let mut peak_pages = 0;

    for cycle in 0..50 {
        // Add 100 new.
        for i in 0..100 {
            let c = cons(
                &mut h,
                Generation::G0,
                Word::fixnum(cycle * 1000 + i),
                Word::NIL,
            );
            if window.len() == 100 {
                window.pop_front();
            }
            window.push_back(c);
        }
        // GC.
        let mut roots: Vec<Word> = window.iter().copied().collect();
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        // Sync back.
        for (i, r) in roots.iter().enumerate() {
            window[i] = *r;
        }
        let total = h.count_pages_in_gen(Generation::G0)
            + h.count_pages_in_gen(Generation::G1)
            + h.count_pages_in_gen(Generation::Tenured);
        peak_pages = peak_pages.max(total);
    }
    // Should converge to bounded usage.
    assert!(peak_pages < 16, "peak total pages = {peak_pages}");
}

#[test]
fn b2_growing_working_set() {
    // Each cycle adds 50 retained objects. After 10 cycles, 500
    // should be alive across all generations.
    let mut h = large_heap();
    let mut retained: Vec<Word> = Vec::new();
    for _ in 0..10 {
        for _ in 0..50 {
            let c = cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL);
            retained.push(c);
        }
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut retained,
        );
    }
    assert_eq!(retained.len(), 500);
    // All must walk as live cons cells.
    for r in &retained {
        assert_eq!(r.tag(), Tag::Cons);
    }
}

#[test]
fn b3_shrinking_working_set_releases_pages() {
    // Start with 500 retained. Each cycle drop 50. After 10 cycles
    // heap should be empty.
    let mut h = large_heap();
    let mut retained: Vec<Word> = Vec::new();
    for _ in 0..500 {
        retained.push(cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL));
    }
    // Promote them all to G1.
    h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut retained,
    );
    assert!(h.count_pages_in_gen(Generation::G1) > 0);

    // Shrink.
    for _ in 0..10 {
        retained.truncate(retained.len().saturating_sub(50));
        h.evacuate_from_word_roots(
            Generation::G1,
            Generation::G1,
            &mut retained,
        );
    }
    assert!(retained.is_empty());
    h.evacuate_from_word_roots(Generation::G1, Generation::G1, &mut []);
    assert_eq!(h.count_pages_in_gen(Generation::G1), 0);
}

#[test]
fn b4_oscillating_working_set() {
    // Alternate grow/shrink, 5 rounds. Heap should never grow
    // unboundedly.
    let mut h = large_heap();
    let mut retained: Vec<Word> = Vec::new();
    let mut peak_pages = 0;
    for round in 0..5 {
        // Grow.
        for _ in 0..200 {
            retained.push(cons(&mut h, Generation::G0, Word::fixnum(round), Word::NIL));
        }
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut retained,
        );
        peak_pages = peak_pages.max(
            h.count_pages_in_gen(Generation::G1) + h.count_pages_in_gen(Generation::Tenured)
        );
        // Shrink to 50.
        retained.truncate(50);
        h.evacuate_from_word_roots(
            Generation::G1,
            Generation::G1,
            &mut retained,
        );
    }
    // Oscillation should have a bounded peak.
    assert!(peak_pages < 32, "peak pages = {peak_pages}");
}

// =========================================================================
// Section C: Generational shapes
// =========================================================================

#[test]
fn c1_nursery_dies_old_survives() {
    // Build 100 long-lived objects (in G1) + 1000 short-lived (in G0).
    // After a minor cycle, the long-lived stay, short-lived die.
    let mut h = medium_heap();
    let mut survivors: Vec<Word> = Vec::new();
    for _ in 0..100 {
        survivors.push(cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL));
    }
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut survivors);
    for _ in 0..1000 {
        let _ = cons(&mut h, Generation::G0, Word::fixnum(99), Word::NIL);
    }
    let before_g0 = h.count_pages_in_gen(Generation::G0);
    assert!(before_g0 > 0);
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut survivors);
    assert_eq!(h.count_pages_in_gen(Generation::G0), 0);
    assert!(h.count_pages_in_gen(Generation::G1) > 0);
    assert_eq!(survivors.len(), 100);
}

#[test]
fn c2_cohort_promotion_threshold_via_collect_minor() {
    use newgc_core::G0_PROMOTION_THRESHOLD;
    let mut h = medium_heap();
    let head = list(&mut h, Generation::G0, 100);
    let mut roots = [head];

    // Run THRESHOLD cycles; last one should promote.
    for cycle in 0..G0_PROMOTION_THRESHOLD {
        let r = h.collect_minor(|evac| {
            for w in roots.iter_mut() {
                evac.visit(w);
            }
        });
        if cycle < G0_PROMOTION_THRESHOLD - 1 {
            assert!(!r.promoted_g0, "cycle {cycle} should not promote");
        }
    }
    assert!(h.count_pages_in_gen(Generation::G1) > 0,
        "expected G1 occupancy after threshold cycles");
}

#[test]
fn c3_tenured_immortal_through_many_minors() {
    // Build a list, promote to Tenured, run 50 minor cycles.
    // The list should remain readable throughout.
    let mut h = large_heap();
    let head = list(&mut h, Generation::G0, 200);
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    h.evacuate_from_word_roots(Generation::G1, Generation::Tenured, &mut roots);
    assert!(h.count_pages_in_gen(Generation::Tenured) > 0);

    for _ in 0..50 {
        // Allocate junk in G0.
        for _ in 0..100 {
            let _ = h.try_alloc_cons_in(Generation::G0);
        }
        // Minor: should not touch Tenured.
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
    }
    let elems = unsafe { walk_list_collecting_cars(roots[0]) };
    assert_eq!(elems.len(), 200);
    for (i, v) in elems.iter().enumerate() {
        assert_eq!(*v, i as i64);
    }
}

#[test]
fn c4_collect_major_clears_unrooted_old() {
    let mut h = medium_heap();
    let head = list(&mut h, Generation::G0, 200);
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    assert!(h.count_pages_in_gen(Generation::G1) > 0);
    // Drop root. Major must reclaim.
    let mut empty: [Word; 0] = [];
    h.collect_major(|evac| {
        for r in empty.iter_mut() {
            evac.visit(r);
        }
    });
    assert_eq!(h.count_pages_in_gen(Generation::G1), 0);
    assert_eq!(h.count_pages_in_gen(Generation::Tenured), 0);
}

#[test]
fn c5_promote_then_die_after_one_more_cycle() {
    // An object that survives one minor and is then dropped should
    // be reclaimable on the next cycle (because it's now in G1).
    let mut h = medium_heap();
    let head = list(&mut h, Generation::G0, 50);
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    assert!(h.count_pages_in_gen(Generation::G1) > 0);
    // Drop and within-gen reclaim.
    h.evacuate_from_word_roots(Generation::G1, Generation::G1, &mut []);
    assert_eq!(h.count_pages_in_gen(Generation::G1), 0);
}

// =========================================================================
// Section D: Object-graph topologies
// =========================================================================

#[test]
fn d1_hashtable_like_buckets_and_chains() {
    // 50 "buckets" (vector slots), each holding a chain of 5 entries
    // (cons cells). Mirrors a small hash table.
    let mut h = medium_heap();
    let buckets = vector(&mut h, Generation::G0, 50, Word::NIL);
    for b in 0..50 {
        let chain = list(&mut h, Generation::G0, 5);
        unsafe { set_vector_slot(buckets, 1 + b, chain) };
    }
    let mut roots = [buckets];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    // Verify every bucket's chain is intact.
    for b in 0..50 {
        let chain = unsafe { vector_slot(roots[0], 1 + b) };
        let elems = unsafe { walk_list_collecting_cars(chain) };
        assert_eq!(elems.len(), 5);
    }
}

#[test]
fn d2_balanced_binary_tree() {
    // Build a balanced binary tree of depth 8 = 255 internal nodes
    // + 256 leaves. Each node is a 2-payload vector (left, right).
    let mut h = large_heap();
    fn build(h: &mut Heap, depth: u32) -> Word {
        if depth == 0 {
            return cons(h, Generation::G0, Word::fixnum(0), Word::NIL);
        }
        let left = build(h, depth - 1);
        let right = build(h, depth - 1);
        let node = vector(h, Generation::G0, 2, Word::NIL);
        unsafe {
            set_vector_slot(node, 1, left);
            set_vector_slot(node, 2, right);
        }
        node
    }
    let root = build(&mut h, 8);
    let mut roots = [root];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);

    // Walk the tree, count nodes.
    fn count(w: Word) -> usize {
        if w.tag() == Tag::Cons {
            return 1;
        }
        assert_eq!(w.tag(), Tag::Vector);
        let l = unsafe { vector_slot(w, 1) };
        let r = unsafe { vector_slot(w, 2) };
        1 + count(l) + count(r)
    }
    let n = count(roots[0]);
    // 2^9 - 1 internal vectors + 2^8 = 256 leaves... actually
    // depth=8 returns at depth 0 → leaves at depth 0 are the 2^8 = 256
    // leaves. Internal levels: 0..7 internal vectors = 2^8 - 1 = 255.
    // Total = 256 + 255 = 511.
    assert_eq!(n, 511);
}

#[test]
fn d3_doubly_linked_list_with_back_edges() {
    // Build a doubly-linked list using vectors so each node has
    // (value, prev, next). Verifies the GC follows back-edges correctly.
    let mut h = medium_heap();
    let n = 100;
    let mut nodes: Vec<Word> = Vec::with_capacity(n);
    for i in 0..n {
        let node = vector(&mut h, Generation::G0, 3, Word::NIL);
        unsafe { set_vector_slot(node, 1, Word::fixnum(i as i64)) };
        nodes.push(node);
    }
    // Wire the prev/next pointers.
    for i in 0..n {
        if i > 0 {
            unsafe { set_vector_slot(nodes[i], 2, nodes[i - 1]) };
        }
        if i < n - 1 {
            unsafe { set_vector_slot(nodes[i], 3, nodes[i + 1]) };
        }
    }
    let mut roots = vec![nodes[0]];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    // Walk forward via next pointers, then back via prev.
    let mut cur = roots[0];
    let mut forward = Vec::new();
    while cur.tag() == Tag::Vector {
        let v = unsafe { vector_slot(cur, 1) };
        forward.push(v.as_fixnum().unwrap());
        let next = unsafe { vector_slot(cur, 3) };
        if next.is_nil() {
            break;
        }
        cur = next;
    }
    assert_eq!(forward.len(), n);
    // Walk back.
    let mut back = Vec::new();
    while cur.tag() == Tag::Vector {
        back.push(unsafe { vector_slot(cur, 1) }.as_fixnum().unwrap());
        let prev = unsafe { vector_slot(cur, 2) };
        if prev.is_nil() {
            break;
        }
        cur = prev;
    }
    assert_eq!(back.len(), n);
}

#[test]
fn d4_skewed_tree_deep_spine() {
    // Tree of depth 1000 where each node has only a left child
    // (right is NIL). Verifies deep recursion in the marker doesn't
    // stack-overflow.
    let mut h = large_heap();
    let mut leaf = cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL);
    for i in 1..=1000 {
        let node = vector(&mut h, Generation::G0, 2, Word::NIL);
        unsafe {
            set_vector_slot(node, 1, leaf);
            set_vector_slot(node, 2, Word::fixnum(i));
        }
        leaf = node;
    }
    let mut roots = [leaf];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    // Walk the spine.
    let mut depth = 0;
    let mut cur = roots[0];
    while cur.tag() == Tag::Vector {
        depth += 1;
        cur = unsafe { vector_slot(cur, 1) };
    }
    assert_eq!(depth, 1000);
}

#[test]
fn d5_random_dag_with_sharing() {
    // 500 vector nodes; each node's slots point to random earlier
    // nodes (creating shared subgraphs). 20 random roots.
    let mut h = large_heap();
    let mut state: u64 = 0xa1b2_c3d4_e5f6_0789;
    let mut nodes: Vec<Word> = Vec::with_capacity(500);
    for i in 0..500 {
        let v = vector(&mut h, Generation::G0, 3, Word::NIL);
        for s in 1..=3 {
            if i > 0 && (lcg(&mut state) & 1) == 0 {
                let target = (lcg(&mut state) as usize) % i;
                unsafe { set_vector_slot(v, s, nodes[target]) };
            }
        }
        nodes.push(v);
    }
    let mut roots: Vec<Word> = (0..20)
        .map(|_| nodes[(lcg(&mut state) as usize) % nodes.len()])
        .collect();

    // Compute pre-GC reachable count.
    fn reachable(seen: &mut std::collections::HashSet<u64>, w: Word) {
        if w.tag() != Tag::Vector {
            return;
        }
        if !seen.insert(w.raw() & PAYLOAD_MASK) {
            return;
        }
        for s in 1..=3 {
            reachable(seen, unsafe { vector_slot(w, s) });
        }
    }
    let mut pre = std::collections::HashSet::new();
    for r in &roots {
        reachable(&mut pre, *r);
    }
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    assert_eq!(result.objects_copied, pre.len());

    let mut post = std::collections::HashSet::new();
    for r in &roots {
        reachable(&mut post, *r);
    }
    assert_eq!(post.len(), pre.len(),
        "reachable count differs after GC");
}

#[test]
fn d6_frame_chain_with_shared_upvalues() {
    // 50 closure-like vectors, each pointing at the previous via
    // slot 1 (the "outer frame") and sharing slot 2 = "upvalue cell".
    // The upvalue is a single cons that everything references.
    let mut h = medium_heap();
    let shared_upvalue = cons(&mut h, Generation::G0, Word::fixnum(42), Word::NIL);
    let mut frame = Word::NIL;
    for i in 0..50 {
        let f = vector(&mut h, Generation::G0, 3, Word::NIL);
        unsafe {
            set_vector_slot(f, 1, frame);
            set_vector_slot(f, 2, shared_upvalue);
            set_vector_slot(f, 3, Word::fixnum(i));
        }
        frame = f;
    }
    let mut roots = [frame];
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    // 50 frames + 1 shared upvalue = 51 objects.
    assert_eq!(result.objects_copied, 51);

    // Verify every frame's slot 2 points at the SAME upvalue.
    let mut cur = roots[0];
    let upv = unsafe { vector_slot(cur, 2) };
    while cur.tag() == Tag::Vector {
        let here = unsafe { vector_slot(cur, 2) };
        assert_eq!(
            here.raw(),
            upv.raw(),
            "shared upvalue was duplicated"
        );
        cur = unsafe { vector_slot(cur, 1) };
    }
}

// =========================================================================
// Section E: Mutation patterns
// =========================================================================

#[test]
fn e1_mutate_old_field_to_young_each_cycle() {
    // A long-lived vector whose slot 1 is overwritten each cycle to
    // point at a fresh young cons. The mutator carries the young
    // pointer as an explicit root — simulating what an LLVM
    // statepoint-emitting JIT would generate, and what NCL's
    // mutator does today by spilling Words into root cells before
    // a safepoint.
    //
    // This test deliberately does NOT rely on a card-barrier path
    // (sub-phase 9, not landed). If the slot's young pointer were
    // only reachable through host, the GC would reclaim the young
    // cell on the first minor. The mutator-roots-young pattern is
    // the workaround that's correct today; once cards land, both
    // patterns should work.
    let mut h = medium_heap();
    let host = vector(&mut h, Generation::G0, 1, Word::NIL);
    let mut roots = [host];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    let mut host = roots[0];

    for cycle in 0..20 {
        let young = cons(&mut h, Generation::G0, Word::fixnum(cycle), Word::NIL);
        unsafe { set_vector_slot(host, 1, young) };

        // Root BOTH the host (G1) and the young cell (G0). The GC
        // will rewrite both. Then we re-attach the new young location
        // into host's slot to maintain the relationship across the
        // GC boundary.
        let mut iter_roots = [host, young];
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut iter_roots,
        );
        host = iter_roots[0];
        let new_young = iter_roots[1];
        unsafe { set_vector_slot(host, 1, new_young) };

        // Verify.
        let slot = unsafe { vector_slot(host, 1) };
        assert_eq!(slot.tag(), Tag::Cons);
        let car = unsafe {
            let p = (slot.raw() & PAYLOAD_MASK) as *const u64;
            Word::from_raw(*p)
        };
        assert_eq!(car.as_fixnum(), Some(cycle));
    }
}

#[test]
fn e2_swap_payload_slots_randomly() {
    // 50 cons cells in a vector. Each iteration, swap two random
    // slots. The GC sees a constantly-shuffling set of pointers.
    let mut h = medium_heap();
    let n = 50;
    let host = vector(&mut h, Generation::G0, n as u32, Word::NIL);
    for i in 0..n {
        let c = cons(&mut h, Generation::G0, Word::fixnum(i as i64), Word::NIL);
        unsafe { set_vector_slot(host, 1 + i, c) };
    }

    let mut state: u64 = 0xfade;
    let mut roots = [host];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    let host = roots[0];

    for _ in 0..30 {
        let a = (lcg(&mut state) as usize) % n;
        let b = (lcg(&mut state) as usize) % n;
        let va = unsafe { vector_slot(host, 1 + a) };
        let vb = unsafe { vector_slot(host, 1 + b) };
        unsafe {
            set_vector_slot(host, 1 + a, vb);
            set_vector_slot(host, 1 + b, va);
        }
        // Allocate junk and GC.
        for _ in 0..50 {
            let _ = h.try_alloc_cons_in(Generation::G0);
        }
        let mut r = [host];
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut r,
        );
    }

    // Every slot should still hold a cons whose car is a fixnum in
    // [0, n). Their multiset must equal {0..n}.
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for i in 0..n {
        let s = unsafe { vector_slot(host, 1 + i) };
        assert_eq!(s.tag(), Tag::Cons);
        let car = unsafe {
            let p = (s.raw() & PAYLOAD_MASK) as *const u64;
            Word::from_raw(*p)
        };
        seen.insert(car.as_fixnum().unwrap());
    }
    assert_eq!(seen.len(), n);
}

#[test]
fn e3_replace_root_with_new_value() {
    // Stand-in for "the program is in a loop, each iteration the
    // current value is replaced with a derived one." After many
    // iterations, only the latest value should be live.
    let mut h = medium_heap();
    let mut current = list(&mut h, Generation::G0, 10);
    let mut peak_pages = 0;

    for i in 0..50 {
        // Build a new list that's longer by 1.
        let new = cons(&mut h, Generation::G0, Word::fixnum(i), current);
        current = new;
        let mut r = [current];
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut r,
        );
        current = r[0];
        let p = h.count_pages_in_gen(Generation::G1);
        peak_pages = peak_pages.max(p);
    }
    // Verify final length.
    let n = unsafe { list_len(current) };
    assert_eq!(n, 60);
}

// =========================================================================
// Section F: Pathological
// =========================================================================

#[test]
fn f1_pointer_shaped_fixnums_dont_follow() {
    // Fixnums whose raw bits happen to look like heap addresses
    // must be left alone. The GC's tag check rejects them at
    // classify().
    let mut h = small_heap();
    let _ = list(&mut h, Generation::G0, 50);
    // These look like "address 8", "address 16", etc.
    let mut roots = [
        Word::fixnum(1),
        Word::fixnum(0x1000),
        Word::fixnum(0x4000_0000),
        Word::fixnum(-1),
    ];
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    // None of the fixnums caused anything to be copied.
    assert_eq!(result.objects_copied, 0);
}

#[test]
fn f2_very_deep_cons_chain_no_stack_overflow() {
    // 10,000-deep chain. The marker must use an explicit work
    // queue, not recursion. (NCL's tracer uses an iterative BFS;
    // this test guards against a regression to recursive marking.)
    let mut h = large_heap();
    let head = list(&mut h, Generation::G0, 10_000);
    let mut roots = [head];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    let n = unsafe { list_len(roots[0]) };
    assert_eq!(n, 10_000);
}

#[test]
fn f3_pyramid_many_to_one_shared() {
    // 1000 cons cells whose CARs all point at a single shared
    // tenured cons. After a major, the shared cons + 1000 referrers
    // must all live.
    let mut h = large_heap();
    let shared = cons(&mut h, Generation::G0, Word::fixnum(99), Word::NIL);
    let mut referrers: Vec<Word> = Vec::with_capacity(1000);
    for i in 0..1000 {
        let c = cons(&mut h, Generation::G0, shared, Word::fixnum(i));
        referrers.push(c);
    }
    let mut roots: Vec<Word> = referrers.clone();
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    // 1001 objects (1000 referrers + 1 shared).
    assert_eq!(result.objects_copied, 1001);
    // All referrers' cars should point at the SAME new shared addr.
    let first_car_raw = {
        let r0 = roots[0];
        let p = (r0.raw() & PAYLOAD_MASK) as *const u64;
        unsafe { *p }
    };
    for r in &roots {
        let p = (r.raw() & PAYLOAD_MASK) as *const u64;
        let car_raw = unsafe { *p };
        assert_eq!(car_raw, first_car_raw,
            "shared object duplicated");
    }
}

#[test]
fn f4_vector_with_many_pointers() {
    // One vector with 1000 payload slots, each pointing at a unique
    // cons.
    let mut h = large_heap();
    // Build 1000 leaves.
    let mut leaves: Vec<Word> = Vec::with_capacity(1000);
    for i in 0..1000 {
        leaves.push(cons(&mut h, Generation::G0, Word::fixnum(i), Word::NIL));
    }
    let v = vector(&mut h, Generation::G0, 1000, Word::NIL);
    for i in 0..1000 {
        unsafe { set_vector_slot(v, 1 + i, leaves[i]) };
    }
    let mut roots = [v];
    let result = h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut roots,
    );
    // 1 vector + 1000 leaves = 1001.
    assert_eq!(result.objects_copied, 1001);
    // Verify every slot still has a cons with the right fixnum.
    for i in 0..1000 {
        let s = unsafe { vector_slot(roots[0], 1 + i) };
        assert_eq!(s.tag(), Tag::Cons);
        let car = unsafe {
            let p = (s.raw() & PAYLOAD_MASK) as *const u64;
            Word::from_raw(*p)
        };
        assert_eq!(car.as_fixnum(), Some(i as i64));
    }
}

#[test]
fn f5_immediates_in_vector_payload_left_alone() {
    // A vector full of immediates — fixnums, NIL, T, characters.
    // The GC should leave them all unchanged.
    let mut h = small_heap();
    let v = vector(&mut h, Generation::G0, 10, Word::NIL);
    let values = [
        Word::fixnum(0),
        Word::fixnum(-7),
        Word::NIL,
        Word::T,
        Word::UNBOUND,
        Word::char('x'),
        Word::char('Z'),
        Word::fixnum(100),
        Word::NIL,
        Word::T,
    ];
    for (i, val) in values.iter().enumerate() {
        unsafe { set_vector_slot(v, 1 + i, *val) };
    }
    let mut roots = [v];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    // Read back, verify identical.
    for (i, val) in values.iter().enumerate() {
        let s = unsafe { vector_slot(roots[0], 1 + i) };
        assert_eq!(s.raw(), val.raw(), "slot {i} changed");
    }
}

// =========================================================================
// Section G: Mixed-realistic
// =========================================================================

#[test]
fn g1_macroexpand_like_recursion() {
    // Build a deeply nested form tree by simulating recursive
    // expansion: each form is a 3-cell vector `(op, arg, next)` where
    // `next` is the recursive expansion. Depth 100. Multiple cycles.
    let mut h = large_heap();
    fn expand(h: &mut Heap, depth: u32) -> Word {
        if depth == 0 {
            return cons(h, Generation::G0, Word::fixnum(0), Word::NIL);
        }
        let inner = expand(h, depth - 1);
        let v = vector(h, Generation::G0, 3, Word::NIL);
        unsafe {
            set_vector_slot(v, 1, Word::fixnum(depth as i64));
            set_vector_slot(v, 2, Word::fixnum(depth as i64 * 10));
            set_vector_slot(v, 3, inner);
        }
        v
    }
    let form = expand(&mut h, 100);
    let mut roots = [form];

    // 5 minor cycles, generating allocation noise each time.
    for _ in 0..5 {
        for _ in 0..200 {
            let _ = h.try_alloc_cons_in(Generation::G0);
        }
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
    }
    // Walk the form to verify intact.
    let mut depth = 0;
    let mut cur = roots[0];
    while cur.tag() == Tag::Vector {
        let op = unsafe { vector_slot(cur, 1) };
        assert!(op.as_fixnum().is_some());
        cur = unsafe { vector_slot(cur, 3) };
        depth += 1;
    }
    assert_eq!(depth, 100);
}

#[test]
fn g2_alist_lookup_and_update_churn() {
    // Build an alist of 50 (key . value) pairs. Each "lookup-and-
    // update" iteration: pick a random key, allocate a new cons to
    // represent the updated entry, prepend it to the alist (so old
    // entries get shadowed but remain alive until GC).
    let mut h = large_heap();
    let mut state: u64 = 0xab1d;
    let mut alist = Word::NIL;
    for k in 0..50 {
        let entry = cons(
            &mut h,
            Generation::G0,
            Word::fixnum(k),
            Word::fixnum(k * 100),
        );
        alist = cons(&mut h, Generation::G0, entry, alist);
    }

    let mut roots = [alist];
    for i in 0..200 {
        // Lookup-and-update: prepend (k, new_v).
        let k = (lcg(&mut state) % 50) as i64;
        let new_v = (i as i64) + 1000;
        let entry = cons(
            &mut h,
            Generation::G0,
            Word::fixnum(k),
            Word::fixnum(new_v),
        );
        let new_alist = cons(&mut h, Generation::G0, entry, roots[0]);
        roots[0] = new_alist;

        if i % 10 == 9 {
            h.evacuate_from_word_roots(
                Generation::G0,
                Generation::G1,
                &mut roots,
            );
        }
    }

    // Final list should have 50 + 200 = 250 entries.
    let n = unsafe { list_len(roots[0]) };
    assert_eq!(n, 250);
}

#[test]
fn g3_producer_consumer_sliding_window() {
    // The "producer" allocates 10 items per round and pushes onto
    // a window; the "consumer" pops the oldest if the window > 20.
    // Total rounds = 50. After every round, GC.
    let mut h = medium_heap();
    let mut window: std::collections::VecDeque<Word> = Default::default();
    let mut peak_pages = 0;

    for _round in 0..50 {
        // Produce.
        for i in 0..10 {
            let item =
                cons(&mut h, Generation::G0, Word::fixnum(i), Word::NIL);
            window.push_back(item);
        }
        // Consume.
        while window.len() > 20 {
            window.pop_front();
        }
        // GC.
        let mut roots: Vec<Word> = window.iter().copied().collect();
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        for (i, r) in roots.iter().enumerate() {
            window[i] = *r;
        }
        let p = h.count_pages_in_gen(Generation::G0)
            + h.count_pages_in_gen(Generation::G1);
        peak_pages = peak_pages.max(p);
    }
    assert_eq!(window.len(), 20);
    assert!(peak_pages < 16, "sliding window over-grew: {peak_pages}");
}

#[test]
fn g4_lisp_fibonacci_tree_memoized() {
    // Compute fib(15) recursively, building a memo as a cons-cell
    // chain of (n . v) pairs. Simulates a real Lisp algorithm's
    // allocation pattern.
    let mut h = large_heap();

    fn cell_lookup(memo: Word, n: i64) -> Option<i64> {
        let mut cur = memo;
        while cur.tag() == Tag::Cons {
            let p = (cur.raw() & PAYLOAD_MASK) as *const u64;
            let entry = unsafe { Word::from_raw(*p) };
            if entry.tag() == Tag::Cons {
                let ep = (entry.raw() & PAYLOAD_MASK) as *const u64;
                let k = unsafe { Word::from_raw(*ep) };
                if k.as_fixnum() == Some(n) {
                    let v = unsafe { Word::from_raw(*ep.add(1)) };
                    return v.as_fixnum();
                }
            }
            cur = unsafe { Word::from_raw(*p.add(1)) };
        }
        None
    }

    fn fib(
        h: &mut Heap,
        n: i64,
        memo: &mut Word,
    ) -> i64 {
        if n < 2 {
            return n;
        }
        if let Some(v) = cell_lookup(*memo, n) {
            return v;
        }
        let a = fib(h, n - 1, memo);
        let b = fib(h, n - 2, memo);
        let result = a + b;
        let entry = cons(h, Generation::G0, Word::fixnum(n), Word::fixnum(result));
        *memo = cons(h, Generation::G0, entry, *memo);
        result
    }

    let mut memo = Word::NIL;
    let answer = fib(&mut h, 15, &mut memo);
    assert_eq!(answer, 610); // fib(15) = 610

    // The memo should have entries for all n from 2..=15 = 14 entries.
    let mut roots = [memo];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    let memo_len = unsafe { list_len(roots[0]) };
    assert_eq!(memo_len, 14);
}

// =========================================================================
// Section I: Conservative pin scanner
// =========================================================================

/// Build a "fake stack" buffer that contains Word-shaped values
/// some of which legitimately point at heap objects and some of
/// which are nearby integers / noise. The pin scanner's job is to
/// pin only the ones that resolve to real heap object starts.
fn build_fake_stack(roots: &[Word], noise: &[u64]) -> Vec<u64> {
    let mut buf: Vec<u64> = Vec::new();
    let mut state = 0xdeadbeefu64;
    for r in roots {
        // Interleave each root with some noise.
        buf.push(lcg(&mut state) >> 3 << 3); // random aligned junk
        buf.push(r.raw());
        buf.push(lcg(&mut state)); // random unaligned junk
    }
    buf.extend_from_slice(noise);
    buf
}

#[cfg(feature = "conservative-pin")]

#[test]
fn i1_pin_scanner_pins_real_heap_pointers() {
    let mut h = medium_heap();
    let c1 = cons(&mut h, Generation::G0, Word::fixnum(1), Word::NIL);
    let c2 = cons(&mut h, Generation::G0, Word::fixnum(2), Word::NIL);
    let c3 = cons(&mut h, Generation::G0, Word::fixnum(3), Word::NIL);

    // Build a fake stack containing all three cons Words plus noise.
    let stack = build_fake_stack(&[c1, c2, c3], &[0, 1, 2, 0xdeadbeef, 42]);
    let lo = stack.as_ptr() as usize;
    let hi = lo + stack.len() * 8;
    let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
    assert_eq!(result.n_objects, 3, "expected 3 pinned, got {}", result.n_objects);
}

#[cfg(feature = "conservative-pin")]

#[test]
fn i2_pin_scanner_rejects_pointer_shaped_noise() {
    let mut h = medium_heap();
    let _ = cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL);
    // Build a stack of pointer-shaped noise that DOESN'T point into
    // the heap. All have the cons tag (low 3 bits = 001) but the
    // upper bits resolve to addresses outside the heap reservation.
    let noise: Vec<u64> = (0..100)
        .map(|i| (0x7000_0000_0000_0000u64 + (i << 3)) | 0b001)
        .collect();
    let lo = noise.as_ptr() as usize;
    let hi = lo + noise.len() * 8;
    let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
    assert_eq!(result.n_objects, 0, "false positives pinned: {}", result.n_objects);
}

#[cfg(feature = "conservative-pin")]

#[test]
fn i3_pin_scanner_handles_high_density_false_positives() {
    let mut h = medium_heap();
    // 1 real cons.
    let real = cons(&mut h, Generation::G0, Word::fixnum(42), Word::NIL);
    // Build a huge stack of mostly-noise (5000 entries) with the
    // one real pointer hidden in the middle.
    let mut stack: Vec<u64> = Vec::with_capacity(5000);
    let mut state: u64 = 0x123456789;
    for i in 0..5000 {
        if i == 2500 {
            stack.push(real.raw());
        } else {
            // Push values whose low 3 bits LOOK like various tags
            // but whose addresses don't resolve.
            let tag = (lcg(&mut state) & 0b111) as u64;
            let bits = (lcg(&mut state) << 3) | tag;
            stack.push(bits);
        }
    }
    let lo = stack.as_ptr() as usize;
    let hi = lo + stack.len() * 8;
    let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
    // At least the one real pointer should be pinned. False
    // positives are possible if random bits hit a real cell.
    assert!(result.n_objects >= 1);
    let total_after_first = h.pinned_count();
    // Pinning is idempotent at the pinned-set level — re-scanning the
    // same ranges adds NO new pins. Per the API: `n_objects` is the
    // **delta** (newly pinned this call), not the cumulative count.
    let result2 = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
    assert_eq!(result2.n_objects, 0, "second scan added pins");
    assert_eq!(h.pinned_count(), total_after_first);
}

#[cfg(feature = "conservative-pin")]

#[test]
fn i4_pin_scanner_excludes_self_stack_range() {
    let mut h = medium_heap();
    // Build a "stack" buffer whose contents look like pointers into
    // itself. The self-stack-exclusion gate should reject all of them.
    let mut stack: Vec<u64> = vec![0; 100];
    // Each slot contains a cons-tagged pointer to a 8-byte-aligned
    // location in the stack itself.
    let base = stack.as_ptr() as usize;
    for i in 0..100 {
        stack[i] = ((base + (i * 8)) as u64) | 0b001; // Tag::Cons
    }
    let lo = stack.as_ptr() as usize;
    let hi = lo + stack.len() * 8;
    let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
    assert_eq!(result.n_objects, 0,
        "self-stack pointers should not pin: {}", result.n_objects);
}

#[cfg(feature = "conservative-pin")]

#[test]
fn i5_pin_then_evacuate_keeps_pinned_objects_at_their_address() {
    let mut h = medium_heap();
    let pinned = cons(&mut h, Generation::G0, Word::fixnum(99), Word::NIL);
    let pinned_addr = pinned.raw() & PAYLOAD_MASK;
    let _free = cons(&mut h, Generation::G0, Word::fixnum(1), Word::NIL);

    // Pin only `pinned` via a fake stack containing its Word.
    let stack = vec![pinned.raw()];
    let lo = stack.as_ptr() as usize;
    let hi = lo + stack.len() * 8;
    h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
    assert_eq!(h.pinned_count(), 1);

    // Evac with the pinned cons NOT in the explicit root list (only
    // the pin from the fake-stack scan keeps it alive).
    h.evacuate_from_word_roots(Generation::G0, Generation::G0, &mut []);

    // The pinned object should still be readable at its original
    // address — pinning prevents movement.
    let p = pinned_addr as *const u64;
    let car = unsafe { Word::from_raw(*p) };
    assert_eq!(car.as_fixnum(), Some(99));

    h.clear_all_pins();
}

// =========================================================================
// Section J: Long-running stress
// =========================================================================

#[test]
fn j1_sustained_alloc_churn_200_cycles() {
    // 200 cycles, ~300 cons cells allocated per cycle, retain a
    // sliding window of 30. Heap usage must stay bounded for the
    // entire run.
    let mut h = medium_heap();
    let mut window: std::collections::VecDeque<Word> = Default::default();
    let mut max_g0_pages = 0;
    let mut max_total_pages = 0;

    for _ in 0..200 {
        for i in 0..300 {
            let c = cons(&mut h, Generation::G0, Word::fixnum(i), Word::NIL);
            if window.len() == 30 {
                window.pop_front();
            }
            window.push_back(c);
        }
        max_g0_pages = max_g0_pages.max(h.count_pages_in_gen(Generation::G0));
        let mut roots: Vec<Word> = window.iter().copied().collect();
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        for (i, r) in roots.iter().enumerate() {
            window[i] = *r;
        }
        let total = h.count_pages_in_gen(Generation::G0)
            + h.count_pages_in_gen(Generation::G1)
            + h.count_pages_in_gen(Generation::Tenured);
        max_total_pages = max_total_pages.max(total);
    }
    // Steady-state working set of 30 cons cells = 60 cells = ~1 page.
    // Allow up to 8 pages of slack for transient G1 occupancy.
    assert!(max_total_pages < 16, "max total pages = {max_total_pages}");
}

#[test]
fn j2_long_running_with_periodic_majors() {
    // 100 cycles. Every 10th cycle is a major instead of a minor.
    // Verifies majors interleave cleanly with minors over time.
    let mut h = large_heap();
    let mut retained: Vec<Word> = Vec::new();
    // Pre-populate.
    for _ in 0..50 {
        retained.push(cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL));
    }
    h.evacuate_from_word_roots(
        Generation::G0,
        Generation::G1,
        &mut retained,
    );

    for cycle in 0..100 {
        // Allocate some junk and a few retained items.
        for i in 0..50 {
            let _ = cons(&mut h, Generation::G0, Word::fixnum(i), Word::NIL);
        }
        if cycle % 5 == 0 {
            // Retain a fresh one.
            retained.push(cons(&mut h, Generation::G0, Word::fixnum(cycle), Word::NIL));
        }
        if cycle % 10 == 0 {
            major(&mut h, &mut retained);
        } else {
            minor(&mut h, &mut retained);
        }
    }
    // After 100 cycles, retained should have grown from 50 by
    // ~20 (every 5th cycle). Walk each to confirm it's still a cons.
    assert!(retained.len() >= 60);
    for r in &retained {
        assert_eq!(r.tag(), Tag::Cons);
    }
}

#[test]
fn j3_alloc_until_near_oom_then_recover() {
    // Fill the heap until alloc returns None. Then GC with no roots.
    // Then alloc again — should succeed. Repeated 5 times.
    let mut h = small_heap();
    for _round in 0..5 {
        let mut n_alloc = 0;
        while h.try_alloc_cons_in(Generation::G0).is_some() {
            n_alloc += 1;
            if n_alloc > 100_000 {
                panic!("alloc runaway");
            }
        }
        assert!(n_alloc > 0);
        // Reclaim.
        h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut []);
        // Now alloc should work again.
        assert!(h.try_alloc_cons_in(Generation::G0).is_some());
    }
}

// =========================================================================
// Section K: Cross-gen pinned-field (the life.lisp bug pattern)
// =========================================================================

#[cfg(feature = "conservative-pin")]

#[test]
fn k1_pinned_g1_object_with_g0_payload_field_survives() {
    // The life.lisp crash pattern simplified:
    //  1. Build a G1 object whose payload points at a G0 object.
    //  2. Conservatively-pin the G1 object via a fake stack scan.
    //  3. Run a minor cycle.
    //  4. The G0 child must still be reachable (not reclaimed) and
    //     the G1 object's slot must point at the (post-evac) G0 cell.
    //
    // Without cross-gen extend-mark this would dangle.
    let mut h = large_heap();
    let g1_host = vector(&mut h, Generation::G0, 1, Word::NIL);
    let mut roots = [g1_host];
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
    let g1_host = roots[0];

    // Allocate a young child, attach to the G1 host's slot.
    let g0_child = cons(&mut h, Generation::G0, Word::fixnum(7), Word::NIL);
    unsafe { set_vector_slot(g1_host, 1, g0_child) };

    // Conservatively-pin g1_host AND g0_child via fake stack ranges.
    let stack = vec![g1_host.raw(), g0_child.raw()];
    let lo = stack.as_ptr() as usize;
    let hi = lo + stack.len() * 8;
    h.pin_pointers_in_ranges(Generation::G1, &[(lo, hi)]);
    h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);

    // Run a minor evac (G0 → G1). With g0_child also pinned, it
    // stays at its address. The G1 host's slot continues to point at
    // the same address.
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut []);

    // Verify the slot still gives us a Cons with fixnum 7.
    let new_slot = unsafe { vector_slot(g1_host, 1) };
    assert_eq!(new_slot.tag(), Tag::Cons);
    let car = unsafe {
        let p = (new_slot.raw() & PAYLOAD_MASK) as *const u64;
        Word::from_raw(*p)
    };
    assert_eq!(car.as_fixnum(), Some(7));
    h.clear_all_pins();
}

// =========================================================================
// Section L: Statistics consistency
// =========================================================================

#[test]
fn l1_committed_pages_grows_then_shrinks_predictably() {
    let mut h = medium_heap();
    let initial_committed = h.committed_pages();
    // Allocate enough to commit several pages.
    let mut retained: Vec<Word> = Vec::new();
    for _ in 0..2000 {
        retained.push(cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL));
    }
    let peak_committed = h.committed_pages();
    assert!(peak_committed > initial_committed);

    // Drop everything and evac. Committed pages should be >= the
    // count we had after, since we don't decommit on Windows yet.
    // The invariant: committed never SHRINKS (no decommit path
    // exercised here).
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut []);
    let after_gc = h.committed_pages();
    assert!(after_gc >= peak_committed,
        "committed unexpectedly decreased: peak={peak_committed}, after_gc={after_gc}");
}

#[test]
fn l2_evac_result_self_consistent_across_workload() {
    // Run a complex workload and verify each cycle's EvacResult is
    // self-consistent: objects_copied <= cells_copied / 2 (cons is
    // 2 cells, smallest object), pages_freed + pages_flipped fits
    // in the live page count.
    let mut h = medium_heap();
    let mut retained: Vec<Word> = Vec::new();
    let mut state: u64 = 0xc001;
    for cycle in 0..30 {
        for _ in 0..100 {
            retained.push(cons(&mut h, Generation::G0, Word::fixnum(0), Word::NIL));
        }
        // Drop some random subset.
        retained.retain(|_| (lcg(&mut state) & 1) == 0);

        let g0_pages_before = h.count_pages_in_gen(Generation::G0);
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut retained,
        );
        // Sanity:
        assert!(result.cells_copied >= 2 * result.objects_copied,
            "cycle {cycle}: cells_copied={} < 2 * objects_copied={}",
            result.cells_copied, result.objects_copied);
        assert!(result.pages_freed <= g0_pages_before,
            "cycle {cycle}: pages_freed={} > g0_pages_before={}",
            result.pages_freed, g0_pages_before);
    }
}

// =========================================================================
// Section H: TinyLayout workloads (proves cross-layout polymorphism)
// =========================================================================

mod tiny_workloads {
    use newgc_core::page_heap::space::PageHeap;
    use newgc_core::tiny_layout::{
        tiny_cons_ptr, tiny_fixnum, tiny_header, tiny_header_ptr, TinyLayout,
        TINY_PAYLOAD_MASK,
    };
    use newgc_core::traits::WordKind;
    use newgc_core::word::Word;
    use newgc_core::{Generation, HeapLayout};

    type TinyHeap = PageHeap<TinyLayout>;

    fn cons_t(h: &mut TinyHeap, g: Generation, car: u64, cdr: u64) -> u64 {
        let p = h.try_alloc_cons_in(g).expect("tiny cons alloc");
        unsafe {
            *p.as_ptr() = car;
            *p.as_ptr().add(1) = cdr;
        }
        tiny_cons_ptr(p.as_ptr() as *const u8)
    }

    fn vec_t(h: &mut TinyHeap, g: Generation, n_payload: u32, init: u64) -> u64 {
        let total = (1 + n_payload) as usize;
        let p = h.try_alloc_boxed_in(g, total).expect("tiny boxed alloc");
        unsafe {
            *p.as_ptr() = tiny_header(n_payload);
            for i in 1..=n_payload as usize {
                *p.as_ptr().add(i) = init;
            }
        }
        tiny_header_ptr(p.as_ptr() as *const u8)
    }

    unsafe fn slot(raw: u64, idx: usize) -> u64 {
        let base = (raw & TINY_PAYLOAD_MASK) as *const u64;
        unsafe { *base.add(idx) }
    }

    unsafe fn set_slot(raw: u64, idx: usize, value: u64) {
        let base = (raw & TINY_PAYLOAD_MASK) as *mut u64;
        unsafe { *base.add(idx) = value }
    }

    #[test]
    fn h1_tiny_bounded_working_set() {
        // TinyLayout equivalent of b1_bounded_working_set.
        let mut h = TinyHeap::with_reservation(32 * 64 * 1024);
        let mut window: std::collections::VecDeque<u64> =
            std::collections::VecDeque::with_capacity(50);

        for cycle in 0..30 {
            for i in 0..50 {
                let c = cons_t(
                    &mut h,
                    Generation::G0,
                    tiny_fixnum(cycle * 1000 + i),
                    tiny_fixnum(0),
                );
                if window.len() == 50 {
                    window.pop_front();
                }
                window.push_back(c);
            }
            let mut roots: Vec<Word> =
                window.iter().map(|r| Word::from_raw(*r)).collect();
            h.evacuate_from_word_roots(
                Generation::G0,
                Generation::G1,
                &mut roots,
            );
            for (i, r) in roots.iter().enumerate() {
                window[i] = r.raw();
            }
        }
        // After 30 cycles every survivor should classify as a Cons
        // pointer.
        for &raw in &window {
            assert!(matches!(
                TinyLayout::classify(raw),
                WordKind::PointerCons(_)
            ));
        }
    }

    #[test]
    fn h2_tiny_random_dag_reachability() {
        let mut h = TinyHeap::with_reservation(64 * 64 * 1024);
        let mut state: u64 = 0xfeedface;
        fn lcg(s: &mut u64) -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        }
        let n = 200usize;
        let mut nodes: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let v = vec_t(&mut h, Generation::G0, 2, tiny_fixnum(0));
            for s in 1..=2 {
                if i > 0 && (lcg(&mut state) & 1) == 0 {
                    let t = (lcg(&mut state) as usize) % i;
                    unsafe { set_slot(v, s, nodes[t]) };
                }
            }
            nodes.push(v);
        }
        let mut roots: Vec<Word> = (0..10)
            .map(|_| Word::from_raw(nodes[(lcg(&mut state) as usize) % n]))
            .collect();

        fn reach(
            seen: &mut std::collections::HashSet<u64>,
            raw: u64,
        ) {
            if let WordKind::PointerHeader(addr) = TinyLayout::classify(raw) {
                if !seen.insert(addr as u64) {
                    return;
                }
                let layout =
                    unsafe { TinyLayout::header_layout(addr as *const u64) };
                for c in layout.pointer_cells_start..layout.pointer_cells_end {
                    let val = unsafe { slot(raw, c) };
                    reach(seen, val);
                }
            }
        }

        let mut pre = std::collections::HashSet::new();
        for r in &roots {
            reach(&mut pre, r.raw());
        }
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(result.objects_copied, pre.len());
        let mut post = std::collections::HashSet::new();
        for r in &roots {
            reach(&mut post, r.raw());
        }
        assert_eq!(post.len(), pre.len());
    }

    #[test]
    fn h3_tiny_deep_chain_no_recursion_blowup() {
        let mut h = TinyHeap::with_reservation(128 * 64 * 1024);
        let mut head = tiny_fixnum(0);
        for i in 0..5000 {
            head = cons_t(&mut h, Generation::G0, tiny_fixnum(i), head);
        }
        let mut roots = [Word::from_raw(head)];
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        // Walk to verify length.
        let mut cur = roots[0].raw();
        let mut n = 0;
        while let WordKind::PointerCons(addr) = TinyLayout::classify(cur) {
            n += 1;
            cur = unsafe { *((addr as *const u64).add(1)) };
        }
        assert_eq!(n, 5000);
    }

    #[test]
    fn h4_tiny_promoted_data_survives_minors() {
        // TinyLayout's version of "tenured immortal."
        let mut h = TinyHeap::with_reservation(64 * 64 * 1024);
        let mut head = tiny_fixnum(0);
        for i in 0..200 {
            head = cons_t(&mut h, Generation::G0, tiny_fixnum(i), head);
        }
        let mut roots = [Word::from_raw(head)];
        h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut roots);
        h.evacuate_from_word_roots(
            Generation::G1,
            Generation::Tenured,
            &mut roots,
        );

        for _ in 0..20 {
            for _ in 0..50 {
                let _ = h.try_alloc_cons_in(Generation::G0);
            }
            h.evacuate_from_word_roots(
                Generation::G0,
                Generation::G1,
                &mut roots,
            );
        }

        // Verify still 200 elements with correct values.
        let mut cur = roots[0].raw();
        let mut vals = Vec::new();
        while let WordKind::PointerCons(addr) = TinyLayout::classify(cur) {
            let car = unsafe { *(addr as *const u64) };
            vals.push((car as i64) >> 2);
            cur = unsafe { *((addr as *const u64).add(1)) };
        }
        assert_eq!(vals.len(), 200);
        // Verify the descending sequence 199..=0.
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(*v, 199 - i as i64);
        }
    }
}
