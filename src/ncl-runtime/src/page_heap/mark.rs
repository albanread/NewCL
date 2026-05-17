//! Mark pass for the page heap.
//!
//! Sub-phase 5 of the Phase 3 plan in `docs/GC_DESIGN.md`. BFS
//! from a set of root `Word`s, set mark bits on every reachable
//! object's first cell. No movement, no sweeping, no evacuation —
//! marks are an input to sub-phase 7's compaction.
//!
//! ## Algorithm
//!
//! 1. Clear mark bits across all pages in the target generation.
//! 2. For each root `Word`: if it tags as a heap pointer AND its
//!    target falls on a page in the target generation, set the
//!    mark bit on the first cell and push the cell index onto
//!    the work queue.
//! 3. Drain the queue: for each cell `c`, determine the object's
//!    size (cons = 2 cells; boxed = header.length_cells() + 1),
//!    walk every payload cell as a candidate `Word`, recurse via
//!    step 2.
//!
//! The mark bit doubles as the "have I seen this?" predicate, so
//! cycles in the object graph terminate.
//!
//! ## What's marked
//!
//! Sub-phase 5 marks only ONE generation per call — typically
//! `G0` for a minor mark pass. Inter-generational pointers (G1 →
//! G0) are NOT followed automatically; the caller is expected to
//! seed the root set with the targets of dirty cards as well as
//! the mutator root lists. That logic lands in sub-phase 9 (soft
//! cards). For now `mark_from_roots` takes its root set as a
//! plain `&[Word]`.
//!
//! ## Conservative pin candidates
//!
//! Sub-phase 6 adds the conservative-stack scan that pins targets
//! against movement. For sub-phase 5, "marked" and "pinned" are
//! independent: marked = "is alive", pinned = "do not move."
//! Sub-phase 7 evacuation reads both bits.

use crate::heap::HeapHeader;
use crate::word::{Tag, Word};

use super::alloc::{is_cons_start_at, is_start_at};
use super::page_desc::{Generation, PageKind};
use super::space::{PageHeap, PAGE_SIZE_CELLS};

pub(crate) struct PageMarker<'a> {
    heap: &'a mut PageHeap,
    target: Generation,
    queue: Vec<usize>,
}

impl<'a> PageMarker<'a> {
    pub(crate) fn new(heap: &'a mut PageHeap, target: Generation) -> Self {
        heap.clear_mark_bits_in_gen(target);
        Self {
            heap,
            target,
            queue: Vec::new(),
        }
    }

    /// Constructor that does NOT clear existing mark bits. Used by
    /// `extend_mark_from_pinned` to add to the existing mark set
    /// rather than replace it.
    pub(crate) fn new_without_clear(
        heap: &'a mut PageHeap,
        target: Generation,
    ) -> Self {
        Self {
            heap,
            target,
            queue: Vec::new(),
        }
    }

    pub(crate) fn visit(&mut self, slot: &mut Word) {
        self.visit_word(*slot);
    }

    pub(crate) fn visit_word(&mut self, word: Word) {
        self.heap.try_mark_root(word, self.target, &mut self.queue);
    }

    pub(crate) unsafe fn visit_cell(&mut self, cell_ptr: *mut u64) {
        self.visit_word(Word::from_raw(unsafe { *cell_ptr }));
    }

    pub(crate) fn drain(&mut self) {
        while let Some(cell_idx) = self.queue.pop() {
            self.heap
                .scan_marked_object(cell_idx, self.target, &mut self.queue);
        }
    }
}

pub struct MarkScanner<'s, 'a: 's> {
    marker: &'s mut PageMarker<'a>,
}

impl<'s, 'a: 's> MarkScanner<'s, 'a> {
    pub(crate) fn new(marker: &'s mut PageMarker<'a>) -> Self {
        Self { marker }
    }

    pub fn visit(&mut self, slot: &mut Word) {
        self.marker.visit(slot);
    }
}

impl PageHeap {
    /// Mark all objects in `target` generation reachable from
    /// `roots`. Clears existing mark bits on the target generation
    /// first, then runs BFS. Sub-phase 5 — no movement, no
    /// sweeping.
    ///
    /// `roots` is typically the union of: per-mutator explicit
    /// root lists, static-area dirty-card targets, and (later)
    /// conservative stack pin candidates. The caller assembles
    /// them; the mark pass treats them uniformly.
    ///
    /// Words pointing outside the target generation are silently
    /// ignored — useful for minor-only marks where roots may
    /// include G1/Tenured pointers from older code.
    pub fn mark_from_roots(&mut self, target: Generation, roots: &[Word]) {
        let mut marker = PageMarker::new(self, target);
        for &root in roots {
            marker.visit_word(root);
        }
        marker.drain();
    }

    /// If `w` is a heap-pointer Word into the target generation
    /// and its first cell isn't already marked, mark it and push
    /// onto the queue.
    fn try_mark_root(
        &mut self,
        w: Word,
        target: Generation,
        queue: &mut Vec<usize>,
    ) {
        // Fast-reject non-pointer or wrong-tag values.
        let tag = w.tag();
        if !matches!(
            tag,
            Tag::Cons | Tag::Symbol | Tag::Vector | Tag::Function | Tag::String
        ) {
            return;
        }
        // Decode the target address (mask off the tag bits).
        let target_addr =
            (w.raw() & crate::word::PAYLOAD_MASK) as *const u8;
        // Find the page; reject if outside the reservation.
        let page_idx = match self.page_of(target_addr) {
            Some(p) => p,
            None => return,
        };
        // Reject if not in the generation we're marking.
        if self.desc(page_idx).generation != target {
            return;
        }
        // Bug #2 from the code review (docs/GC_DESIGN.md sub-phase
        // 6.5): even with a matching generation and a set start
        // bit, a Free or Large page must never be followed. The
        // start bit could be stale from a prior tenant (the
        // recycle path in `acquire_free_page` zeros it as of
        // bug-fix #1, but defence-in-depth) or belong to a Large
        // object whose payload shape isn't yet defined. Without
        // this gate the next call to `scan_marked_object` would
        // hit the panic arm.
        let kind = self.desc(page_idx).kind;
        if !matches!(kind, PageKind::Cons | PageKind::Boxed) {
            return;
        }
        // Convert to global cell index.
        let cell_idx = (target_addr as usize - self.base_ptr() as usize) / 8;
        // For Boxed pages, the cell must be a header-start.
        // For Cons pages, every even cell within an allocated
        // pair is a cons-start (which the start bitmap encodes
        // with pair `11`). We accept any encoded start.
        if !is_start_at(self.start_bits_slice(), cell_idx) {
            // Sketchy candidate — probably a stale/false pointer.
            // Sub-phase 6 will harden this for the conservative
            // pinner; sub-phase 5 just skips.
            return;
        }
        let is_cons_start = is_cons_start_at(self.start_bits_slice(), cell_idx);
        match tag {
            Tag::Cons if is_cons_start => {}
            Tag::Symbol | Tag::Vector | Tag::Function | Tag::String
                if !is_cons_start => {}
            _ => return,
        }
        // Mark; if already marked, no re-queueing.
        if self.mark_cell(cell_idx) {
            return;
        }
        queue.push(cell_idx);
    }

    /// Scan the payload cells of a marked object and recurse on
    /// any heap-pointer children. `cell_idx` is the object's
    /// first cell (header for boxed, car for cons).
    ///
    /// **Object-kind dispatch**: for `PageKind::Cons` pages, every
    /// cell is a 2-cell cons pair (a small optimization for pages
    /// produced by `try_alloc_cons_in` — fixed stride, no start-
    /// bit lookup). For `PageKind::Boxed` pages, the per-cell
    /// **start-bit pattern** decides: `11` → cons (2 cells), `01`
    /// → header-bearing (`1 + length_cells()`). Mutator TLABs
    /// land on Boxed pages and intermix cons + boxed allocations,
    /// so the start-bit check is the source of truth there.
    fn scan_marked_object(
        &mut self,
        cell_idx: usize,
        target: Generation,
        queue: &mut Vec<usize>,
    ) {
        let page_idx = cell_idx / PAGE_SIZE_CELLS;
        let kind = self.desc(page_idx).kind;
        let is_cons = match kind {
            PageKind::Cons => true,
            PageKind::Boxed => {
                // Dispatch by start-bit pattern. Mutator TLABs
                // intermix conses and boxed objects on the same
                // page; `11` is the cons-start pair, `01` is the
                // boxed-header start.
                is_cons_start_at(self.start_bits_slice(), cell_idx)
            }
            PageKind::Large => {
                // Sub-phase 7 will define large-object marking
                // shape. For now, skip — the test workloads
                // don't allocate Large.
                return;
            }
            PageKind::Free => {
                // Bug #2 from the code review: try_mark_root rejects
                // Free-page candidates up front via the kind gate,
                // so this arm is unreachable in correct operation.
                // Defensive skip rather than panic keeps the GC
                // robust against latent kind-table races / future
                // refactors that might queue a stale cell. A panic
                // here would tear down the whole collector — too
                // expensive for the protective value.
                let _ = (cell_idx, page_idx);
                return;
            }
        };
        let size_cells = if is_cons {
            2
        } else {
            let header_word = self.read_cell(cell_idx);
            let h = HeapHeader::from_raw(header_word);
            1 + h.length_cells() as usize
        };
        // Walk payload. For Cons, both cells are Words. For Boxed,
        // cells 1..size_cells are payload; cell 0 is the header
        // and we skip it. SOME boxed objects have non-Word cells
        // in their payload (Function::code_ptr, Bignum limbs,
        // Float bits) but the conservative scanner treats them
        // as candidate Words and harmlessly rejects non-pointer
        // bit patterns.
        let payload_start = if is_cons { cell_idx } else { cell_idx + 1 };
        let payload_end = cell_idx + size_cells;
        for c in payload_start..payload_end {
            let w = Word::from_raw(self.read_cell(c));
            self.try_mark_root(w, target, queue);
        }
    }

    /// Read the raw u64 at global cell `cell_idx`. Bounds-checked
    /// in debug; trusted in release — the BFS only ever asks for
    /// cells inside marked objects on committed pages.
    fn read_cell(&self, cell_idx: usize) -> u64 {
        debug_assert!(cell_idx < self.total_cells());
        let ptr = unsafe { (self.base_ptr() as *const u64).add(cell_idx) };
        unsafe { *ptr }
    }

    /// Cross-generation extension: seed `target`'s mark BFS from
    /// pinned objects whose own page is in a DIFFERENT generation.
    /// Walks each such pinned object's payload Words and offers them
    /// to `try_mark_root(_, target, _)`; the BFS then propagates
    /// transitively through `target`.
    ///
    /// Closes a heap-walk hole identified in `docs/GC_HEAP_WALK_CLOSURE.md`:
    /// the conservative pin scan retains G1 cells via stack-resident
    /// pointer-shaped values, and those pinned G1 objects' fields may
    /// point at G0 cells. Without this extension, the cross-gen G0
    /// children are never marked → Phase 1 doesn't evacuate them →
    /// Phase 3 releases their pages → the pinned-G1 field dangles.
    ///
    /// The pinned object itself is NOT marked in `target`'s bitmap:
    /// it lives in a different gen and stays put under the pin
    /// contract; only its payload's `target`-gen children get marked.
    pub fn extend_mark_from_cross_gen_pinned(&mut self, target: Generation) -> usize {
        if self.pinned_cells.is_empty() {
            return 0;
        }
        let pinned: Vec<usize> = self.pinned_cells.iter().copied().collect();
        let mut marker = PageMarker::new_without_clear(self, target);
        let mut new_seeds = 0usize;
        for cell_idx in pinned {
            let page_idx = cell_idx / PAGE_SIZE_CELLS;
            let pinned_gen = marker.heap.desc(page_idx).generation;
            // Same-gen pins are covered by `extend_mark_from_pinned(target)`.
            // Free and Large pages aren't legal pin targets.
            if pinned_gen == target {
                continue;
            }
            if !matches!(
                pinned_gen,
                Generation::G0 | Generation::G1 | Generation::Tenured
            ) {
                continue;
            }
            let kind = marker.heap.desc(page_idx).kind;
            if !matches!(kind, PageKind::Cons | PageKind::Boxed) {
                continue;
            }
            if !is_start_at(marker.heap.start_bits_slice(), cell_idx) {
                continue;
            }
            let is_cons =
                is_cons_start_at(marker.heap.start_bits_slice(), cell_idx);
            let size_cells = if is_cons {
                2
            } else {
                let header_word = marker.heap.read_cell(cell_idx);
                let h = HeapHeader::from_raw(header_word);
                1 + h.length_cells() as usize
            };
            let payload_start = if is_cons { cell_idx } else { cell_idx + 1 };
            let payload_end = cell_idx + size_cells;
            for c in payload_start..payload_end {
                let w = Word::from_raw(marker.heap.read_cell(c));
                let before = marker.queue.len();
                marker
                    .heap
                    .try_mark_root(w, target, &mut marker.queue);
                if marker.queue.len() > before {
                    new_seeds += marker.queue.len() - before;
                }
            }
        }
        marker.drain();
        new_seeds
    }

    /// Extend the mark by treating every cell in `pinned_cells` as an
    /// additional root: walk each pinned object's payload, marking
    /// targets transitively. Returns the number of newly-marked
    /// objects.
    ///
    /// Why this is needed: the conservative pin scan runs AFTER the
    /// precise mark pass, so objects that are only reachable via a
    /// conservatively-pinned object's payload (i.e. the precise root
    /// graph doesn't include the pinned object, but it's kept alive
    /// by a register-spilled stack candidate) are missed by the
    /// initial mark. They would then not be copied by evacuation,
    /// their pages would be reclaimed, and any Word inside the
    /// pinned object that points at them would dangle.
    ///
    /// This extension uses the same `PageMarker` walker as
    /// `mark_from_roots`, just seeded from the pinned cells instead
    /// of caller-supplied roots.
    ///
    /// Covers same-gen pinning only — pinned objects whose page is
    /// in `target`. Cross-gen children of pinned objects in OTHER
    /// generations are handled by `extend_mark_from_cross_gen_pinned`.
    pub fn extend_mark_from_pinned(&mut self, target: Generation) -> usize {
        if self.pinned_cells.is_empty() {
            return 0;
        }
        let pinned: Vec<usize> = self.pinned_cells.iter().copied().collect();
        let mut marker = PageMarker::new_without_clear(self, target);
        let mut newly_marked = 0usize;
        for cell_idx in pinned {
            let page_idx = cell_idx / PAGE_SIZE_CELLS;
            if marker.heap.desc(page_idx).generation != target {
                continue;
            }
            let kind = marker.heap.desc(page_idx).kind;
            if !matches!(kind, PageKind::Cons | PageKind::Boxed) {
                continue;
            }
            if !is_start_at(marker.heap.start_bits_slice(), cell_idx) {
                continue;
            }
            // `mark_cell` returns the PREVIOUS state. If it was
            // already marked, the precise mark pass already walked
            // its payload — don't re-queue. If it was not previously
            // marked, queue for traversal.
            if marker.heap.mark_cell(cell_idx) {
                continue;
            }
            newly_marked += 1;
            marker.queue.push(cell_idx);
        }
        marker.drain();
        newly_marked
    }

    /// Diagnostic: walk all marked cells in `target` generation
    /// and return their global cell indices. O(n_cells_in_target).
    /// Test-only.
    pub fn marked_cells_in_gen(&self, target: Generation) -> Vec<usize> {
        let mut out = Vec::new();
        for (page_idx, d) in self.descs().iter().enumerate() {
            if d.generation != target {
                continue;
            }
            let first_cell = page_idx * PAGE_SIZE_CELLS;
            let last_cell = first_cell + PAGE_SIZE_CELLS;
            // Limit by words_used so we don't iterate past the
            // bump pointer (extra zeros, but no need to look).
            let cap = first_cell + d.words_used as usize;
            for c in first_cell..cap.min(last_cell) {
                if self.is_marked(c) {
                    out.push(c);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::word::{Tag, Word};

    fn small_heap() -> PageHeap {
        // 4 pages = 256 KB. Plenty for 1000 cons cells (250 per
        // page).
        PageHeap::with_reservation(4 * 64 * 1024)
    }

    /// Allocate `n` cons cells, chained `next.cdr = prev`. Returns
    /// the head Word (pointer to the last-allocated cons, which
    /// points back to the chain).
    fn alloc_cons_chain(h: &mut PageHeap, n: usize) -> Vec<Word> {
        let mut prev = Word::NIL;
        let mut all = Vec::with_capacity(n);
        for i in 0..n {
            let p = h.try_alloc_cons_in(Generation::G0).unwrap();
            let car = Word::fixnum(i as i64);
            unsafe {
                *p.as_ptr() = car.raw();
                *p.as_ptr().add(1) = prev.raw();
            }
            let w = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
            all.push(w);
            prev = w;
        }
        all
    }

    #[test]
    fn empty_roots_marks_nothing() {
        let mut h = small_heap();
        let _ = alloc_cons_chain(&mut h, 100);
        h.mark_from_roots(Generation::G0, &[]);
        assert_eq!(h.count_marked_in_gen(Generation::G0), 0);
    }

    #[test]
    fn marking_a_single_cons_marks_only_that_cell() {
        let mut h = small_heap();
        let chain = alloc_cons_chain(&mut h, 10);
        // Allocate one more cons that points to nil — no chain.
        let standalone = {
            let p = h.try_alloc_cons_in(Generation::G0).unwrap();
            unsafe {
                *p.as_ptr() = Word::fixnum(99).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
        };
        // Marking the standalone cons should mark exactly one
        // cell (its first cell — cons-start).
        h.mark_from_roots(Generation::G0, &[standalone]);
        let marked = h.marked_cells_in_gen(Generation::G0);
        assert_eq!(marked.len(), 1);
        let _ = chain; // chain wasn't reached from roots.
    }

    #[test]
    fn marking_chain_head_marks_whole_chain() {
        let mut h = small_heap();
        let chain = alloc_cons_chain(&mut h, 100);
        // The head (last allocated) chains backward through every
        // earlier cons cell via cdr.
        let head = *chain.last().unwrap();
        h.mark_from_roots(Generation::G0, &[head]);
        let marked = h.marked_cells_in_gen(Generation::G0);
        assert_eq!(marked.len(), 100, "every chain link marked");
    }

    #[test]
    fn marking_half_of_disjoint_objects() {
        // Acceptance test from the design doc: alloc 1000 conses,
        // mark half via fake roots, verify the mark bitmap.
        let mut h = small_heap();
        // 1000 disjoint cons cells, each pointing to NIL.cdr=NIL.
        let mut all = Vec::with_capacity(1000);
        for i in 0..1000 {
            let p = h.try_alloc_cons_in(Generation::G0).unwrap();
            unsafe {
                *p.as_ptr() = Word::fixnum(i).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            all.push(Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons));
        }
        // Mark every other one (500 total).
        let roots: Vec<Word> = all.iter().step_by(2).copied().collect();
        assert_eq!(roots.len(), 500);
        h.mark_from_roots(Generation::G0, &roots);
        assert_eq!(
            h.count_marked_in_gen(Generation::G0),
            500,
            "exactly 500 marked, no extras, no misses"
        );
    }

    #[test]
    fn marking_is_idempotent() {
        let mut h = small_heap();
        let chain = alloc_cons_chain(&mut h, 50);
        let head = *chain.last().unwrap();
        h.mark_from_roots(Generation::G0, &[head]);
        let first = h.count_marked_in_gen(Generation::G0);
        // Marking the same root again clears + re-marks; same result.
        h.mark_from_roots(Generation::G0, &[head]);
        let second = h.count_marked_in_gen(Generation::G0);
        assert_eq!(first, second);
        assert_eq!(first, 50);
    }

    #[test]
    fn cycles_in_object_graph_terminate() {
        let mut h = small_heap();
        // Build a 5-cycle: cons-a.cdr = cons-b.cdr = cons-c.cdr =
        // cons-d.cdr = cons-e.cdr = cons-a. We do this by
        // allocating 5 conses with placeholder cdrs, then
        // patching them.
        let mut conses = Vec::new();
        for i in 0..5 {
            let p = h.try_alloc_cons_in(Generation::G0).unwrap();
            unsafe {
                *p.as_ptr() = Word::fixnum(i).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            conses.push(p);
        }
        // Patch cdrs to form the cycle.
        for i in 0..5 {
            let next = (i + 1) % 5;
            let next_word =
                Word::from_ptr(conses[next].as_ptr() as *const u8, Tag::Cons);
            unsafe {
                *conses[i].as_ptr().add(1) = next_word.raw();
            }
        }
        // Mark from any one root. BFS must terminate (cycles
        // detected by the mark-bit-already-set check).
        let root = Word::from_ptr(conses[0].as_ptr() as *const u8, Tag::Cons);
        h.mark_from_roots(Generation::G0, &[root]);
        assert_eq!(h.count_marked_in_gen(Generation::G0), 5);
    }

    #[test]
    fn fixnums_and_immediates_are_not_followed() {
        let mut h = small_heap();
        let _ = alloc_cons_chain(&mut h, 10);
        // Fake roots that are all non-pointers.
        let roots = [
            Word::fixnum(0),
            Word::fixnum(42),
            Word::fixnum(-1),
            Word::NIL,
            Word::T,
            Word::char('a'),
        ];
        h.mark_from_roots(Generation::G0, &roots);
        assert_eq!(h.count_marked_in_gen(Generation::G0), 0);
    }

    #[test]
    fn out_of_range_pointer_word_is_ignored() {
        let mut h = small_heap();
        let _ = alloc_cons_chain(&mut h, 10);
        // A "Cons" Word whose target is outside the reservation.
        // We construct it by tagging an arbitrary address.
        let bogus = Word::from_raw(0xdeadbeef_dead0000 | (Tag::Cons as u64));
        h.mark_from_roots(Generation::G0, &[bogus]);
        assert_eq!(h.count_marked_in_gen(Generation::G0), 0);
    }

    #[test]
    fn marking_a_g0_root_doesnt_touch_g1_pages() {
        let mut h = small_heap();
        // Allocate something in G1 by manually transitioning a
        // page. We do this by forcing alloc into G1 directly.
        let g1_cons = h.try_alloc_cons_in(Generation::G1).unwrap();
        let g1_word =
            Word::from_ptr(g1_cons.as_ptr() as *const u8, Tag::Cons);
        // Now allocate a G0 cons that points to the G1 cons via
        // cdr. That's the cross-generational pointer.
        let g0_cons = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *g0_cons.as_ptr() = Word::fixnum(0).raw();
            *g0_cons.as_ptr().add(1) = g1_word.raw();
        }
        let g0_word =
            Word::from_ptr(g0_cons.as_ptr() as *const u8, Tag::Cons);
        // Minor mark — should mark only the G0 cons, not the G1
        // target it references.
        h.mark_from_roots(Generation::G0, &[g0_word]);
        assert_eq!(h.count_marked_in_gen(Generation::G0), 1);
        assert_eq!(
            h.count_marked_in_gen(Generation::G1),
            0,
            "minor mark must not cross generations"
        );
    }

    #[test]
    fn marked_root_on_free_page_does_not_panic() {
        // Regression test for bug #2 from the code review: when
        // the kind table is corrupted (or pre bug-fix #1 left a
        // stale start bit on a freed page), a stack-residual Word
        // could target a Free page and crash `scan_marked_object`
        // with `panic!("... on a Free page ...")`. After the fix,
        // the kind gate in `try_mark_root` rejects the candidate
        // and the Free arm in `scan_marked_object` defensively
        // returns rather than panicking.
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        let base = h.base_ptr() as usize;
        let cell_idx = (p.as_ptr() as usize - base) / 8;
        let page_idx = cell_idx / PAGE_SIZE_CELLS;
        // Corrupt the invariant: flip the page back to Free while
        // its start bit stays set (exactly the dangerous state
        // bug #1 would leave behind without its fix).
        h.desc_mut(page_idx).generation = Generation::Free;
        let root = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
        // Must NOT panic.
        h.mark_from_roots(Generation::G0, &[root]);
        assert_eq!(
            h.count_marked_in_gen(Generation::G0),
            0,
            "no cells in G0 should be marked — the only page is Free now"
        );
    }

    #[test]
    fn marking_a_boxed_object_walks_its_payload() {
        // Mark coverage for Boxed pages, which the previous test
        // set missed entirely (every existing mark test uses cons
        // cells). Exercises `scan_marked_object`'s HeapHeader
        // decoding path.
        use crate::heap::{HeapHeader, HeapType};
        let mut h = small_heap();
        // 3-cell boxed object: header + 2 payload cells.
        let boxed = h.try_alloc_boxed_in(Generation::G0, 3).unwrap();
        let target = h.try_alloc_cons_in(Generation::G0).unwrap();
        let target_word =
            Word::from_ptr(target.as_ptr() as *const u8, Tag::Cons);
        unsafe {
            // Cell 0: HeapHeader. length_cells = 2 (payload only,
            // header excluded). The walker computes size as
            // `1 + length_cells()` = 3.
            *boxed.as_ptr() = HeapHeader::new(HeapType::Vector, 2).raw();
            // Cell 1: a Word pointing at the target cons.
            *boxed.as_ptr().add(1) = target_word.raw();
            // Cell 2: harmless fixnum so the walker has a
            // non-pointer to skip.
            *boxed.as_ptr().add(2) = Word::fixnum(0).raw();
        }
        let root = Word::from_ptr(boxed.as_ptr() as *const u8, Tag::Vector);
        h.mark_from_roots(Generation::G0, &[root]);
        // Two cells marked: the boxed object's header cell + the
        // target cons's first cell.
        assert_eq!(h.count_marked_in_gen(Generation::G0), 2);
    }
}
