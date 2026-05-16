//! Evacuation / compaction pass for the page heap.
//!
//! Sub-phase 7 of the Phase 3 plan in `docs/GC_DESIGN.md`. Cheney-
//! style BFS evacuation, adapted for a page-based heap.
//!
//! ## Algorithm
//!
//! Caller provides a closure that walks roots and presents each
//! mutable Word slot to a `PageEvacuator::visit`. For each slot:
//!
//!   1. Read the current Word.
//!   2. If it's a heap pointer into `from_gen`:
//!      - If the source object is pinned: leave the slot alone
//!        (the pinned object stays at its old address).
//!      - If the source cell already holds a `Tag::Forward`:
//!        follow it; rewrite the slot with the forward target.
//!      - Otherwise: allocate in `dest_gen`, copy the cells,
//!        set the start bit at the destination, write
//!        `Word::forward(dest_ptr)` at the source cell, push a
//!        `CopiedObject` onto the BFS queue, rewrite the slot
//!        with the new tagged Word.
//!   3. Drain the queue: each entry references one freshly-copied
//!      object at its NEW location; walk its payload cells, treat
//!      each as a candidate Word, and recurse via the same rule.
//!      Payload slots get updated in place at the destination so
//!      no caller has to re-walk.
//!
//! After the BFS finishes, the from-pages are reclaimed:
//!   - Pages with no pins → `PageDesc::release()` (back to Free).
//!     Their start bits are cleared.
//!   - Pages with pins → generation flips from `from_gen` to
//!     `dest_gen` in place; the pinned objects "promote for free."
//!     Their non-pinned start bits are cleared so future scanners
//!     can't see the abandoned forwarding markers or dead-but-
//!     allocated cells.
//!
//! Pin set and mark bits for `from_gen` are cleared at the end —
//! the cycle is complete.
//!
//! ## What evacuation does NOT do
//!
//! - It doesn't touch `dest_gen` pages that were already populated
//!   before the cycle: their objects keep their addresses.
//! - It doesn't follow cross-generation pointers automatically.
//!   The caller is responsible for seeding cross-gen roots (dirty
//!   cards, static area). Sub-phase 9 wires this up.
//! - It doesn't run the mark pass. `mark.rs` builds a separate
//!   bitmap that's currently used for diagnostics and will drive
//!   sub-phase 8's age accounting. Evacuation is independent of
//!   that bitmap — Cheney BFS discovers liveness as it goes.

use std::ptr;

use crate::heap::HeapHeader;
use crate::word::{Tag, Word, PAYLOAD_MASK};

use super::alloc::{is_start_at, set_cons_start_bit_at, set_start_bit_at};
use super::page_desc::{Generation, PageKind};
use super::space::{PageHeap, PAGE_SIZE_CELLS};

/// Cells per start-bits word (2 bits per cell, 32 cells per u64).
/// Duplicated here from `alloc.rs` to avoid an extra import for one
/// constant in a tight loop.
const CELLS_PER_STARTS_WORD: usize = 32;
const STARTS_WORDS_PER_PAGE: usize = PAGE_SIZE_CELLS / CELLS_PER_STARTS_WORD;

/// Tally of what happened during one evacuation cycle. Reported
/// back to the GC coordinator for `(gc-stats)` and the trigger
/// policy in sub-phase 10.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvacResult {
    /// Distinct objects copied to `dest_gen`. Pinned objects don't
    /// count — they were never moved.
    pub objects_copied: usize,
    /// Total cells (including headers / cons-pair second cells)
    /// copied. Useful for reporting "data moved" volume.
    pub cells_copied: usize,
    /// `from_gen` pages reclaimed back to Free. These are the
    /// pages with no pins after evacuation.
    pub pages_freed: usize,
    /// `from_gen` pages with at least one pin, generation-flipped
    /// to `dest_gen` in place. The pinned objects "promote for
    /// free."
    pub pages_flipped: usize,
}

/// One queue entry: an object that has just been copied into
/// `dest_gen`. The BFS will visit its payload to find children
/// that also need to be evacuated.
#[derive(Clone, Copy, Debug)]
struct CopiedObject {
    /// Global cell index of this object's first cell in the
    /// DESTINATION page.
    to_cell_idx: usize,
    /// Total cells (header + payload for boxed; 2 for cons).
    size: usize,
    /// True for cons cells (both cells are Words). False for boxed
    /// (cell 0 is HeapHeader, cells 1..size are payload Words).
    is_cons: bool,
}

/// Stateful evacuator handed to the caller's root-walking closure.
/// `visit(&mut Word)` is the only method the caller needs.
pub struct PageEvacuator<'a> {
    heap: &'a mut PageHeap,
    from_gen: Generation,
    dest_gen: Generation,
    queue: Vec<CopiedObject>,
    /// Tally fields — published into `EvacResult` at end of cycle.
    objects_copied: usize,
    cells_copied: usize,
}

impl<'a> PageEvacuator<'a> {
    /// Visit a root slot. Reads the current Word, decides whether
    /// it needs to be copied / forwarded / left alone, and updates
    /// the slot in place.
    ///
    /// Caller pattern (single-threaded under STW):
    ///
    /// ```ignore
    /// heap.evacuate_with_roots(Gen::G0, Gen::G1, |evac| {
    ///     for root_slot in mutator_roots.iter_mut() {
    ///         evac.visit(root_slot);
    ///     }
    /// });
    /// ```
    pub fn visit(&mut self, slot: &mut Word) {
        let w = *slot;
        if let Some(new) = self.maybe_copy(w) {
            *slot = new;
        }
    }

    /// Same as `visit`, but for a raw cell address. Used by the
    /// BFS drain to update payload Words in place at the
    /// destination, and by the dirty-card scanner in
    /// `coordinator_api::collect_minor_with_static` to scan
    /// external regions (the static area, older-generation pages)
    /// for cross-gen pointers into `from_gen`.
    ///
    /// SAFETY: caller asserts `cell_ptr` is a valid `*mut u64`
    /// inside the page heap's reservation OR an externally-supplied
    /// region (e.g., the static area) and points at an 8-byte
    /// aligned cell. The cell content is read as a `Word`; if it
    /// tags as a heap pointer into `from_gen`, evacuation runs and
    /// the cell is rewritten.
    pub unsafe fn visit_cell(&mut self, cell_ptr: *mut u64) {
        let raw = unsafe { *cell_ptr };
        let w = Word::from_raw(raw);
        if let Some(new) = self.maybe_copy(w) {
            unsafe { *cell_ptr = new.raw() };
        }
    }

    /// Core decision: return the post-evacuation Word for `w`, or
    /// `None` if the slot doesn't need rewriting (immediate, or
    /// pointer outside `from_gen`, etc.).
    fn maybe_copy(&mut self, w: Word) -> Option<Word> {
        let tag = w.tag();
        // Fast reject non-heap-pointer tags. Forward is rejected
        // here because a Forward Word in a root slot would mean
        // the GC was re-entered against an already-forwarded slot,
        // which shouldn't happen under STW. Defensive None.
        if !matches!(
            tag,
            Tag::Cons | Tag::Symbol | Tag::Vector | Tag::Function | Tag::String
        ) {
            return None;
        }
        let target_addr = (w.raw() & PAYLOAD_MASK) as usize;
        // Look up the source page.
        let page_idx = self.heap.page_of(target_addr as *const u8)?;
        // Source must be in from_gen.
        if self.heap.desc(page_idx).generation != self.from_gen {
            return None;
        }
        let from_cell_idx = (target_addr - self.heap.base_ptr() as usize) / 8;

        // Start-bit gate: the target cell must be a real object
        // start. Without this, a non-pointer payload word that
        // coincidentally tags as a heap pointer and lands in
        // `from_gen` would be followed as if it were a real
        // reference, corrupting heap state (false copy + spurious
        // forwarding marker + tag-confusion at the destination).
        // The pin and mark passes apply the same gate; doing it
        // here closes the same hole for evacuation.
        if !is_start_at(self.heap.start_bits_slice(), from_cell_idx) {
            return None;
        }

        // Tag-vs-start-bit consistency. A Cons-tagged Word must
        // point at a cons-start cell (pair `11`); Symbol/Vector/
        // Function/String-tagged Words must point at a boxed-
        // header start (pair `01`). Mutator TLABs intermix conses
        // and header-bearing objects on the same `PageKind::Boxed`
        // page, so we can't dispatch on page.kind alone — the
        // start-bit pattern is the source of truth.
        let kind = self.heap.desc(page_idx).kind;
        if matches!(kind, PageKind::Free | PageKind::Large) {
            return None;
        }
        let is_cons_start = super::alloc::is_cons_start_at(
            self.heap.start_bits_slice(),
            from_cell_idx,
        );
        match tag {
            Tag::Cons if is_cons_start => {}
            Tag::Symbol | Tag::Vector | Tag::Function | Tag::String
                if !is_cons_start => {}
            // Tag/start-bit mismatch — bogus pointer or tag-confused
            // root. Refuse.
            _ => return None,
        }

        // Is the source already forwarded? If yes, just re-tag and
        // return the forward target.
        let src_header_raw = self.read_cell(from_cell_idx);
        if Word::from_raw(src_header_raw).is_forward() {
            let new_ptr = Word::from_raw(src_header_raw)
                .forward_target()
                .expect("forward target");
            return Some(Word::from_ptr(new_ptr as *const u8, tag));
        }

        // Pinned? Don't move it — keep the slot pointing at the
        // original address. The pinned object's page stays alive
        // (and flips to dest_gen at end of cycle) and the cell
        // there is still valid.
        if self.heap.is_pinned_cell(from_cell_idx) {
            return Some(w);
        }

        // Real copy. Determine size from the start-bit pattern,
        // not the page kind — mutator TLABs intermix conses and
        // boxed objects on the same Boxed page.
        let (size, is_cons) = if is_cons_start {
            (2, true)
        } else {
            let h = HeapHeader::from_raw(src_header_raw);
            (1 + h.length_cells() as usize, false)
        };

        // Allocate at destination. The allocator may need to acquire
        // a fresh page from the free list; if that fails the heap
        // is full and we have no recourse but to leave the slot
        // alone. Sub-phase 10 will pre-allocate evacuation budget;
        // for sub-phase 7 we panic — running out of room mid-evac is
        // a programming bug in the test sizing, not a recoverable
        // production state.
        let dest_ptr = if is_cons {
            self.heap
                .try_alloc_cons_in(self.dest_gen)
                .expect("page heap exhausted mid-evacuation")
        } else {
            self.heap
                .try_alloc_boxed_in(self.dest_gen, size)
                .expect("page heap exhausted mid-evacuation")
        };
        // Bytewise copy of the cells. Source and destination
        // reservations are disjoint pages, so a plain
        // copy_nonoverlapping is safe even though it's the same
        // arena.
        let src_ptr =
            unsafe { (self.heap.base_ptr() as *mut u64).add(from_cell_idx) };
        unsafe {
            ptr::copy_nonoverlapping(src_ptr, dest_ptr.as_ptr(), size);
        }

        // Compute the destination's global cell index. Used for
        // the BFS queue and (indirectly) for any future caller
        // querying "where is this now?".
        let dest_cell_idx = (dest_ptr.as_ptr() as usize
            - self.heap.base_ptr() as usize)
            / 8;

        // Write the forwarding pointer at the SOURCE. This both
        // (a) lets later visits to the same object short-circuit
        // through the forward check, and (b) tells future passes
        // (e.g., card scanning) that this cell is no longer a
        // live object header.
        unsafe {
            *src_ptr =
                Word::forward(dest_ptr.as_ptr() as *const ()).raw();
        }

        // Push BFS entry. The drain loop will walk this object's
        // payload at the destination.
        self.queue.push(CopiedObject {
            to_cell_idx: dest_cell_idx,
            size,
            is_cons,
        });
        self.objects_copied += 1;
        self.cells_copied += size;
        Some(Word::from_ptr(dest_ptr.as_ptr() as *const u8, tag))
    }

    /// Drain the work queue. Each entry describes one freshly-
    /// copied object at the DESTINATION; we walk its payload
    /// cells and run `maybe_copy` on each as a candidate Word,
    /// updating in place at the destination.
    fn drain(&mut self) {
        // Index-based loop because new entries get pushed during
        // iteration; we don't want to invalidate iterators.
        let mut idx = 0;
        while idx < self.queue.len() {
            let obj = self.queue[idx];
            // For cons: both cells are Words.
            // For boxed: cells 1..size are payload Words. Cell 0
            //   is the header, untouched (only the gc-bits + type
            //   + length-cells field; not a pointer).
            let (payload_start, n_words) = if obj.is_cons {
                (obj.to_cell_idx, 2)
            } else {
                (obj.to_cell_idx + 1, obj.size - 1)
            };
            for i in 0..n_words {
                let cell_idx = payload_start + i;
                let cell_ptr = unsafe {
                    (self.heap.base_ptr() as *mut u64).add(cell_idx)
                };
                unsafe { self.visit_cell(cell_ptr) };
            }
            idx += 1;
        }
    }

    /// Read a raw u64 from a global cell index. Used by
    /// `maybe_copy` to inspect a source header. Bounds-checked in
    /// debug.
    fn read_cell(&self, cell_idx: usize) -> u64 {
        debug_assert!(cell_idx < self.heap.total_cells());
        let p = unsafe {
            (self.heap.base_ptr() as *const u64).add(cell_idx)
        };
        unsafe { *p }
    }
}

impl PageHeap {
    /// Evacuate every reachable object in `from_gen` into pages
    /// belonging to `dest_gen`. Pass `from_gen == dest_gen` for an
    /// in-place mark-evacuate cycle; pass `dest_gen = from_gen.
    /// promoted()` to promote.
    ///
    /// `visit_roots` is the caller's chance to feed in mutator
    /// roots / static roots / dirty-card roots. For each root
    /// slot, call `evac.visit(&mut slot)`; the slot will be
    /// rewritten in place if the target moved.
    ///
    /// Returns an `EvacResult` summarising what happened, suitable
    /// for `(gc-stats)`.
    ///
    /// ## Pre-conditions
    ///
    /// - The caller has stopped the world (only one mutator —
    ///   itself — touches the heap).
    /// - `from_gen` and `dest_gen` are valid generations
    ///   (`G0 / G1 / Tenured`); `Free` is invalid for either.
    ///
    /// ## Post-conditions
    ///
    /// - Every reachable-from-roots object in `from_gen` now lives
    ///   in `dest_gen`.
    /// - Pinned objects remain at their original addresses; their
    ///   page generations have flipped to `dest_gen`.
    /// - Unpinned, fully-evacuated pages are back on the free list.
    /// - `from_gen`'s alloc regions have been reset (their
    ///   `current_page` may have been released).
    /// - The pin set and mark bits for `from_gen` are cleared.
    pub fn evacuate_with_roots<F>(
        &mut self,
        from_gen: Generation,
        dest_gen: Generation,
        mut visit_roots: F,
    ) -> EvacResult
    where
        F: FnMut(&mut PageEvacuator<'_>),
    {
        assert!(
            !matches!(from_gen, Generation::Free),
            "evacuate: from_gen must not be Free"
        );
        assert!(
            !matches!(dest_gen, Generation::Free),
            "evacuate: dest_gen must not be Free"
        );

        // Snapshot from-pages. After evacuation we'll iterate them
        // again to decide release / flip.
        let from_pages: Vec<usize> = self.pages_in_gen(from_gen).collect();

        // Reset every from_gen alloc region. Its current_page may
        // be a from-page that is about to be released or flipped.
        // Future allocations into from_gen (if any — typically a
        // mutator wouldn't ask for from_gen mid-cycle, but be
        // defensive) re-acquire from the free list.
        for kind in [PageKind::Cons, PageKind::Boxed] {
            let r = self.alloc_region_mut(from_gen, kind);
            *r = super::alloc::AllocRegion::empty(from_gen, kind);
        }

        // Run the BFS evacuation.
        let mut evac = PageEvacuator {
            heap: self,
            from_gen,
            dest_gen,
            queue: Vec::new(),
            objects_copied: 0,
            cells_copied: 0,
        };
        visit_roots(&mut evac);
        evac.drain();
        let objects_copied = evac.objects_copied;
        let cells_copied = evac.cells_copied;
        drop(evac);

        // Snapshot each pinned cell's start-bit pattern BEFORE
        // clearing page start bits — mutator TLABs intermix conses
        // and boxed objects on the same `PageKind::Boxed` page, so
        // page.kind is no longer enough to tell which kind of
        // start-bit to re-set. `is_cons_start` is true for pair
        // `11`, false for pair `01`.
        let pinned_with_kind: Vec<(usize, bool)> = self
            .pinned_cells
            .iter()
            .map(|&cell_idx| {
                let is_cons = super::alloc::is_cons_start_at(
                    self.start_bits_slice(),
                    cell_idx,
                );
                (cell_idx, is_cons)
            })
            .collect();

        // Walk from_pages: pinned → flip; un-pinned → release.
        let mut pages_freed = 0;
        let mut pages_flipped = 0;
        for &page_idx in &from_pages {
            if self.desc(page_idx).has_pins() {
                // Pinned objects stay where they are. The page
                // flips generation and resets age — sub-phase 8's
                // age policy starts fresh on the survivors. `kind`,
                // `words_used`, and `scan_start_offset` are
                // preserved by virtue of only writing the fields we
                // need to change. `pin_byte` will be cleared by the
                // end-of-cycle `clear_all_pins` below.
                let d = self.desc_mut(page_idx);
                d.generation = dest_gen;
                d.age = 0;
                // Clear start bits on the page. The next loop re-
                // sets the bits on pinned-object starts so future
                // scanners can still find them; forwarded-cell
                // markers and abandoned garbage on the page become
                // invisible.
                clear_page_start_bits(self.start_bits_slice(), page_idx);
                pages_flipped += 1;
            } else {
                // No pins → page is garbage / fully evacuated.
                // Release back to Free and zero its start bits.
                self.desc_mut(page_idx).release();
                clear_page_start_bits(self.start_bits_slice(), page_idx);
                pages_freed += 1;
            }
        }

        // Re-set start bits for the pinned cells using the snapshot
        // taken before clearing. This preserves cons-vs-boxed
        // distinction even when the page is `PageKind::Boxed` with
        // a mix of object types.
        let bits = self.start_bits_slice();
        for (cell_idx, is_cons) in pinned_with_kind {
            if is_cons {
                set_cons_start_bit_at(bits, cell_idx);
            } else {
                set_start_bit_at(bits, cell_idx);
            }
        }

        // End of cycle: clear pins and mark bits for from_gen.
        // (Pins were tracked against from_gen; flipped pages keep
        // their PageDesc.pin_byte cleared so the next cycle starts
        // fresh.)
        self.clear_all_pins();
        self.clear_mark_bits_in_gen(from_gen);

        EvacResult {
            objects_copied,
            cells_copied,
            pages_freed,
            pages_flipped,
        }
    }

    /// Convenience: evacuate using an array of root Words. Each
    /// root is visited via `PageEvacuator::visit`; the array is
    /// updated in place. Used primarily by tests and by simple
    /// driver code; the production path takes `visit_roots` and
    /// walks stack frames itself.
    pub fn evacuate_from_word_roots(
        &mut self,
        from_gen: Generation,
        dest_gen: Generation,
        roots: &mut [Word],
    ) -> EvacResult {
        self.evacuate_with_roots(from_gen, dest_gen, |evac| {
            for r in roots.iter_mut() {
                evac.visit(r);
            }
        })
    }
}

/// Zero every start-bit pair on the page `page_idx`. The page's
/// 256-cell-worth slice of the global bitmap is one `STARTS_WORDS_
/// PER_PAGE` (= 256 u64) contiguous chunk.
fn clear_page_start_bits(
    bits: &[std::sync::atomic::AtomicU64],
    page_idx: usize,
) {
    use std::sync::atomic::Ordering;
    let first_word = page_idx * STARTS_WORDS_PER_PAGE;
    for w in first_word..first_word + STARTS_WORDS_PER_PAGE {
        bits[w].store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::alloc::{is_cons_start_at, is_start_at};
    use crate::heap::{HeapHeader, HeapType};
    use crate::word::{Tag, Word};

    fn small_heap() -> PageHeap {
        // 8 pages = 512 KB. Plenty for several thousand cons cells
        // and a few pages each in G0 and G1.
        PageHeap::with_reservation(8 * 64 * 1024)
    }

    /// Allocate a chain of `n` cons cells, each pointing back to
    /// the previous via cdr (head is the last alloc).
    fn alloc_cons_chain(h: &mut PageHeap, g: Generation, n: usize) -> Vec<Word> {
        let mut prev = Word::NIL;
        let mut all = Vec::with_capacity(n);
        for i in 0..n {
            let p = h.try_alloc_cons_in(g).expect("cons alloc");
            unsafe {
                *p.as_ptr() = Word::fixnum(i as i64).raw();
                *p.as_ptr().add(1) = prev.raw();
            }
            let w = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
            all.push(w);
            prev = w;
        }
        all
    }

    /// Number of distinct G0 / G1 / Tenured / Free pages in the
    /// heap. Used by tests as a quick "page-state changed?" probe.
    fn gen_counts(h: &PageHeap) -> (usize, usize, usize, usize) {
        (
            h.count_pages_in_gen(Generation::Free),
            h.count_pages_in_gen(Generation::G0),
            h.count_pages_in_gen(Generation::G1),
            h.count_pages_in_gen(Generation::Tenured),
        )
    }

    #[test]
    fn rooted_cons_promotes_to_dest_gen() {
        // Acceptance: one cons, rooted, evacuated G0→G1. The Word
        // is rewritten to point into G1; the original G0 page
        // ends up Free (no pins, all live data moved out).
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *p.as_ptr() = Word::fixnum(42).raw();
            *p.as_ptr().add(1) = Word::NIL.raw();
        }
        let mut root =
            [Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)];
        let before_g0 = h.count_pages_in_gen(Generation::G0);
        assert_eq!(before_g0, 1);

        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut root,
        );
        assert_eq!(result.objects_copied, 1);
        assert_eq!(result.cells_copied, 2);
        assert_eq!(result.pages_freed, 1, "G0 page reclaimed");
        assert_eq!(result.pages_flipped, 0);

        // Root now points into G1.
        let new = root[0];
        assert_eq!(new.tag(), Tag::Cons);
        let new_addr = (new.raw() & PAYLOAD_MASK) as *const u8;
        let new_page = h.page_of(new_addr).expect("new ptr in heap");
        assert_eq!(h.desc(new_page).generation, Generation::G1);
        assert_eq!(h.count_pages_in_gen(Generation::G0), 0);

        // Cell contents preserved.
        let new_ptr = new_addr as *const u64;
        unsafe {
            assert_eq!(*new_ptr, Word::fixnum(42).raw());
            assert_eq!(*new_ptr.add(1), Word::NIL.raw());
        }
    }

    #[test]
    fn unrooted_objects_get_reclaimed() {
        // Allocate 10 conses, root none of them. After evacuation,
        // G0 has zero pages and nothing was copied.
        let mut h = small_heap();
        let _ = alloc_cons_chain(&mut h, Generation::G0, 10);
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut [],
        );
        assert_eq!(result.objects_copied, 0);
        assert_eq!(result.pages_freed, 1, "garbage page reclaimed");
        assert_eq!(h.count_pages_in_gen(Generation::G0), 0);
        assert_eq!(h.count_pages_in_gen(Generation::G1), 0);
    }

    #[test]
    fn chain_head_evacuates_every_link() {
        // 50-cons chain; root only the head. After evacuation,
        // every link should have moved.
        let mut h = small_heap();
        let chain = alloc_cons_chain(&mut h, Generation::G0, 50);
        let head = *chain.last().unwrap();
        let mut roots = [head];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(result.objects_copied, 50);
        assert_eq!(result.cells_copied, 100);
        // Walking the chain via the new head must reach all 50
        // links and the fixnums must match the original order
        // (most-recently-allocated → 49 → 48 → ... → 0).
        let mut cur = roots[0];
        let mut seen = 0;
        let mut expected = 49_i64;
        while !cur.is_nil() {
            assert_eq!(cur.tag(), Tag::Cons);
            let addr = (cur.raw() & PAYLOAD_MASK) as *const u64;
            unsafe {
                let car = Word::from_raw(*addr);
                let cdr = Word::from_raw(*addr.add(1));
                assert_eq!(car.as_fixnum(), Some(expected));
                cur = cdr;
            }
            seen += 1;
            expected -= 1;
        }
        assert_eq!(seen, 50);
    }

    #[test]
    fn cycle_in_object_graph_terminates() {
        // 5-cycle: A→B→C→D→E→A. Root A. After evacuation, all 5
        // copied exactly once and the cycle is preserved at the
        // new locations.
        let mut h = small_heap();
        let mut conses = Vec::new();
        for i in 0..5 {
            let p = h.try_alloc_cons_in(Generation::G0).unwrap();
            unsafe {
                *p.as_ptr() = Word::fixnum(i).raw();
                *p.as_ptr().add(1) = Word::NIL.raw();
            }
            conses.push(p);
        }
        // Stitch the cycle.
        for i in 0..5 {
            let next = (i + 1) % 5;
            let next_word =
                Word::from_ptr(conses[next].as_ptr() as *const u8, Tag::Cons);
            unsafe {
                *conses[i].as_ptr().add(1) = next_word.raw();
            }
        }
        let root = Word::from_ptr(conses[0].as_ptr() as *const u8, Tag::Cons);
        let mut roots = [root];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(result.objects_copied, 5);
        // Walk the new cycle: should return to root[0] in 5 steps.
        let mut cur = roots[0];
        let new_head = cur;
        for _ in 0..5 {
            let addr = (cur.raw() & PAYLOAD_MASK) as *const u64;
            unsafe {
                cur = Word::from_raw(*addr.add(1));
            }
            assert_eq!(cur.tag(), Tag::Cons);
        }
        assert_eq!(cur.raw(), new_head.raw(), "cycle closes after 5 hops");
    }

    #[test]
    fn pinned_object_stays_and_page_flips() {
        // One cons, pinned via a fake stack scan, then evacuated.
        // The cons should NOT move; its page should flip from G0
        // to G1 (because dest_gen = G1).
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *p.as_ptr() = Word::fixnum(7).raw();
            *p.as_ptr().add(1) = Word::NIL.raw();
        }
        let original_addr = p.as_ptr() as usize;
        let original_word =
            Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
        // Build a "stack" containing the pointer — runs the
        // conservative pinner so the cell gets pinned.
        let stack: Box<[u64]> =
            vec![original_word.raw()].into_boxed_slice();
        let lo = stack.as_ptr() as usize;
        let hi = unsafe { stack.as_ptr().add(stack.len()) } as usize;
        let pin_res = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert_eq!(pin_res.n_objects, 1);

        // The pin scan recorded the cell; evacuate.
        let mut root = [original_word];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut root,
        );
        assert_eq!(result.objects_copied, 0, "pinned object not moved");
        assert_eq!(result.pages_freed, 0);
        assert_eq!(result.pages_flipped, 1);

        // Root still points at the original address.
        let now = root[0];
        assert_eq!(
            (now.raw() & PAYLOAD_MASK) as usize,
            original_addr,
            "pinned cons retained its address"
        );
        // Its page is now G1.
        let page_idx = (original_addr - h.base_ptr() as usize)
            / (PAGE_SIZE_CELLS * 8);
        assert_eq!(h.desc(page_idx).generation, Generation::G1);

        // Cell contents unchanged.
        unsafe {
            assert_eq!(*p.as_ptr(), Word::fixnum(7).raw());
            assert_eq!(*p.as_ptr().add(1), Word::NIL.raw());
        }
    }

    #[test]
    fn cdr_pointer_gets_fixed_up_after_evacuation() {
        // Two conses: B has cdr = A. Root B. After evacuation,
        // B's new copy must have cdr pointing at A's new copy
        // (not at A's original address).
        let mut h = small_heap();
        let pa = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *pa.as_ptr() = Word::fixnum(1).raw();
            *pa.as_ptr().add(1) = Word::NIL.raw();
        }
        let a_word = Word::from_ptr(pa.as_ptr() as *const u8, Tag::Cons);
        let pb = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *pb.as_ptr() = Word::fixnum(2).raw();
            *pb.as_ptr().add(1) = a_word.raw();
        }
        let b_word = Word::from_ptr(pb.as_ptr() as *const u8, Tag::Cons);
        let original_a_addr = pa.as_ptr() as usize;

        let mut roots = [b_word];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(result.objects_copied, 2);

        // B's new location:
        let new_b_addr = (roots[0].raw() & PAYLOAD_MASK) as *const u64;
        unsafe {
            // car preserved
            assert_eq!(
                Word::from_raw(*new_b_addr).as_fixnum(),
                Some(2)
            );
            // cdr is a Cons Word pointing to A's NEW location.
            let new_cdr = Word::from_raw(*new_b_addr.add(1));
            assert_eq!(new_cdr.tag(), Tag::Cons);
            let new_a_addr = (new_cdr.raw() & PAYLOAD_MASK) as usize;
            assert_ne!(
                new_a_addr, original_a_addr,
                "A's pointer should have been updated, not stale"
            );
            // A's new location must be in G1 (where we evacuated
            // to), and the cell content must be intact.
            let new_a_page = h
                .page_of(new_a_addr as *const u8)
                .expect("A's new addr in heap");
            assert_eq!(h.desc(new_a_page).generation, Generation::G1);
            assert_eq!(
                Word::from_raw(*(new_a_addr as *const u64)).as_fixnum(),
                Some(1)
            );
            assert_eq!(
                Word::from_raw(*((new_a_addr + 8) as *const u64)).raw(),
                Word::NIL.raw()
            );
        }
    }

    #[test]
    fn already_forwarded_slot_is_re_resolved() {
        // Two distinct root words pointing at the same object.
        // After the first visit, the second visit must follow
        // the forwarding pointer rather than re-copy.
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *p.as_ptr() = Word::fixnum(99).raw();
            *p.as_ptr().add(1) = Word::NIL.raw();
        }
        let w = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
        let mut roots = [w, w, w]; // 3 copies of the same root
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(
            result.objects_copied, 1,
            "duplicate roots copy the object exactly once"
        );
        // All 3 roots end up at the same new address.
        assert_eq!(roots[0].raw(), roots[1].raw());
        assert_eq!(roots[1].raw(), roots[2].raw());
    }

    #[test]
    fn boxed_object_with_word_payload_evacuates_correctly() {
        // 3-cell boxed object: header + 2 Word payload cells.
        // One Word points at a cons. After evacuation, both the
        // boxed object and the cons move, and the boxed's payload
        // pointer is updated.
        let mut h = small_heap();
        let pc = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *pc.as_ptr() = Word::fixnum(50).raw();
            *pc.as_ptr().add(1) = Word::NIL.raw();
        }
        let cons_w = Word::from_ptr(pc.as_ptr() as *const u8, Tag::Cons);
        let pb = h.try_alloc_boxed_in(Generation::G0, 3).unwrap();
        unsafe {
            *pb.as_ptr() = HeapHeader::new(HeapType::Vector, 2).raw();
            *pb.as_ptr().add(1) = cons_w.raw();
            *pb.as_ptr().add(2) = Word::fixnum(0).raw();
        }
        let boxed_w = Word::from_ptr(pb.as_ptr() as *const u8, Tag::Vector);

        let mut roots = [boxed_w];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(result.objects_copied, 2);
        assert_eq!(result.cells_copied, 5); // 3 boxed + 2 cons

        // Boxed's new location, and the Word payload now points at
        // the cons's new location.
        let new_boxed = (roots[0].raw() & PAYLOAD_MASK) as *const u64;
        unsafe {
            let hdr = HeapHeader::from_raw(*new_boxed);
            assert_eq!(hdr.length_cells(), 2);
            let payload_word = Word::from_raw(*new_boxed.add(1));
            assert_eq!(payload_word.tag(), Tag::Cons);
            let cons_new_addr =
                (payload_word.raw() & PAYLOAD_MASK) as *const u64;
            assert_ne!(
                cons_new_addr as usize, pc.as_ptr() as usize,
                "cons should have been moved, not retained"
            );
            assert_eq!(
                Word::from_raw(*cons_new_addr).as_fixnum(),
                Some(50)
            );
        }
    }

    #[test]
    fn mixed_pinned_and_unpinned_on_same_page() {
        // Two conses on the same G0 page. Pin only the first.
        // After evacuation: page flips to G1 (not freed), the
        // pinned cons keeps its address, the unpinned cons gets
        // moved to a different page (also in G1, but a fresh one).
        let mut h = small_heap();
        let p1 = h.try_alloc_cons_in(Generation::G0).unwrap();
        let p2 = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *p1.as_ptr() = Word::fixnum(11).raw();
            *p1.as_ptr().add(1) = Word::NIL.raw();
            *p2.as_ptr() = Word::fixnum(22).raw();
            *p2.as_ptr().add(1) = Word::NIL.raw();
        }
        // Same page?
        let pg1 = (p1.as_ptr() as usize - h.base_ptr() as usize)
            / (PAGE_SIZE_CELLS * 8);
        let pg2 = (p2.as_ptr() as usize - h.base_ptr() as usize)
            / (PAGE_SIZE_CELLS * 8);
        assert_eq!(pg1, pg2, "test premise: same G0 page");

        // Pin p1 only.
        let w1 = Word::from_ptr(p1.as_ptr() as *const u8, Tag::Cons);
        let stack: Box<[u64]> = vec![w1.raw()].into_boxed_slice();
        let lo = stack.as_ptr() as usize;
        let hi = unsafe { stack.as_ptr().add(stack.len()) } as usize;
        h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);

        // Both roots evacuate; one stays, one moves.
        let w2 = Word::from_ptr(p2.as_ptr() as *const u8, Tag::Cons);
        let mut roots = [w1, w2];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(result.objects_copied, 1, "only p2 moved");
        assert_eq!(result.pages_flipped, 1, "the pinned page survives");
        assert_eq!(result.pages_freed, 0);

        // After flip: original page is G1; the unpinned cons's new
        // home is a different page, also G1.
        assert_eq!(h.desc(pg1).generation, Generation::G1);
        let r0_addr = (roots[0].raw() & PAYLOAD_MASK) as usize;
        let r1_addr = (roots[1].raw() & PAYLOAD_MASK) as usize;
        assert_eq!(
            r0_addr, p1.as_ptr() as usize,
            "pinned p1 kept its address"
        );
        assert_ne!(
            r1_addr, p2.as_ptr() as usize,
            "unpinned p2 should have moved"
        );
        let r1_page = h.page_of(r1_addr as *const u8).unwrap();
        assert_eq!(h.desc(r1_page).generation, Generation::G1);
    }

    #[test]
    fn within_gen_evacuation_is_supported() {
        // from_gen == dest_gen — mark-evacuate within G0. Useful
        // for sub-phase 8 when a generation gets collected but
        // nothing gets promoted yet.
        let mut h = small_heap();
        let chain = alloc_cons_chain(&mut h, Generation::G0, 30);
        let head = *chain.last().unwrap();
        let before = gen_counts(&h);

        let mut roots = [head];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G0, // SAME
            &mut roots,
        );
        assert_eq!(result.objects_copied, 30);
        // The from G0 pages should be reclaimed; new G0 pages
        // opened during evacuation hold the survivors. Page count
        // may differ from before depending on fragmentation, but
        // G0 count is non-zero.
        let after = gen_counts(&h);
        let (free_before, g0_before, _, _) = before;
        let (free_after, g0_after, _, _) = after;
        // Same total pages — just shuffled state.
        assert_eq!(free_before + g0_before, free_after + g0_after);
        assert!(g0_after >= 1);

        // Chain still walkable, 30 links.
        let mut cur = roots[0];
        let mut seen = 0;
        while !cur.is_nil() {
            assert_eq!(cur.tag(), Tag::Cons);
            let addr = (cur.raw() & PAYLOAD_MASK) as *const u64;
            unsafe {
                cur = Word::from_raw(*addr.add(1));
            }
            seen += 1;
        }
        assert_eq!(seen, 30);
    }

    #[test]
    fn pins_and_mark_bits_are_cleared_after_cycle() {
        // After evacuate completes, pinned_cells must be empty
        // and the from-gen's mark bits cleared, so the next cycle
        // starts from a clean state.
        let mut h = small_heap();
        let chain = alloc_cons_chain(&mut h, Generation::G0, 10);
        let head = *chain.last().unwrap();
        h.mark_from_roots(Generation::G0, &[head]);
        let stack: Box<[u64]> = vec![head.raw()].into_boxed_slice();
        let lo = stack.as_ptr() as usize;
        let hi = unsafe { stack.as_ptr().add(stack.len()) } as usize;
        h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert!(h.pinned_count() > 0);
        assert!(h.count_marked_in_gen(Generation::G0) > 0);

        let mut roots = [head];
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );

        assert_eq!(h.pinned_count(), 0, "pins cleared post-evacuate");
        assert_eq!(
            h.count_marked_in_gen(Generation::G0),
            0,
            "G0 mark bits cleared post-evacuate"
        );
    }

    #[test]
    fn released_page_can_be_re_acquired() {
        // After releasing a G0 page via evacuation, the next
        // allocation into G0 should be able to acquire it again.
        // This exercises the integration with acquire_free_page's
        // start-bit-clearing path (bug #1 fix from sub-phase 6.5).
        let mut h = small_heap();
        let _ = alloc_cons_chain(&mut h, Generation::G0, 5);
        // Unrooted — gets reclaimed.
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut [],
        );
        assert_eq!(h.count_pages_in_gen(Generation::G0), 0);

        // Now allocate into G0 — must work.
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *p.as_ptr() = Word::fixnum(123).raw();
            *p.as_ptr().add(1) = Word::NIL.raw();
        }
        // And the cons-start bit must be set (no stale state).
        let cell_idx =
            (p.as_ptr() as usize - h.base_ptr() as usize) / 8;
        assert!(is_cons_start_at(h.start_bits_slice(), cell_idx));
    }

    #[test]
    fn false_positive_payload_word_is_rejected() {
        // Regression: a payload Word whose bit pattern tags as
        // Cons but whose target is a non-start cell within
        // from_gen must NOT be followed. Without the start-bit
        // gate in maybe_copy, the evacuator would try to copy
        // the non-start cell, write a forward marker over
        // unrelated data, and corrupt the heap.
        let mut h = small_heap();
        // First, allocate a real cons (will be reachable from
        // root chain). Its first cell is a cons-start, its
        // second cell (cdr) is NOT a start.
        let real = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *real.as_ptr() = Word::fixnum(11).raw();
            *real.as_ptr().add(1) = Word::NIL.raw();
        }
        let cdr_addr =
            unsafe { real.as_ptr().add(1) as usize };
        // Allocate a Vector that holds a SUSPICIOUS Word in its
        // payload: Cons-tagged, pointing at the cdr cell.
        let bogus_word =
            Word::from_raw((cdr_addr as u64) | (Tag::Cons as u64));
        let vec_ptr = h.try_alloc_boxed_in(Generation::G0, 2).unwrap();
        unsafe {
            *vec_ptr.as_ptr() = HeapHeader::new(HeapType::Vector, 1).raw();
            *vec_ptr.as_ptr().add(1) = bogus_word.raw();
        }
        let real_word =
            Word::from_ptr(real.as_ptr() as *const u8, Tag::Cons);
        let vec_word =
            Word::from_ptr(vec_ptr.as_ptr() as *const u8, Tag::Vector);

        let mut roots = [real_word, vec_word];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        // Real-cons + vector = 2 distinct copies. The bogus
        // payload Word must NOT cause a third "ghost" copy.
        assert_eq!(
            result.objects_copied, 2,
            "false-positive interior pointer must not trigger a copy"
        );
        // The vector's payload word is unchanged in shape (still
        // points at the same in-from-gen cdr cell). Since the
        // page got freed after evacuation, the address is stale,
        // but maybe_copy left it alone — that's the contract.
        let new_vec_addr = (roots[1].raw() & PAYLOAD_MASK) as *const u64;
        unsafe {
            let payload = Word::from_raw(*new_vec_addr.add(1));
            assert_eq!(
                payload.raw(), bogus_word.raw(),
                "bogus payload word left untouched"
            );
        }
    }

    #[test]
    fn pointer_tag_must_match_page_kind() {
        // Regression: a Cons-tagged Word pointing at a Boxed page
        // must not be followed (and vice versa). Without the
        // tag-vs-kind gate, evacuation would emit a tag-confused
        // Word into the root slot.
        let mut h = small_heap();
        let b = h.try_alloc_boxed_in(Generation::G0, 2).unwrap();
        unsafe {
            *b.as_ptr() = HeapHeader::new(HeapType::Vector, 1).raw();
            *b.as_ptr().add(1) = Word::NIL.raw();
        }
        // Word tagged as Cons but pointing at a Boxed start.
        let mistagged =
            Word::from_raw((b.as_ptr() as u64) | (Tag::Cons as u64));
        let mut roots = [mistagged];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(
            result.objects_copied, 0,
            "tag/kind mismatch must be rejected"
        );
        // The slot is left alone (and the boxed page gets reclaimed
        // because nothing rooted it correctly — that's fine).
        assert_eq!(roots[0].raw(), mistagged.raw());
    }

    #[test]
    fn pinned_boxed_object_stays_in_place() {
        // Companion to `pinned_object_stays_and_page_flips` for
        // boxed objects. The pin set is keyed by global cell idx
        // regardless of cons-vs-boxed, so the boxed path must
        // also work end-to-end.
        let mut h = small_heap();
        let b = h.try_alloc_boxed_in(Generation::G0, 3).unwrap();
        unsafe {
            *b.as_ptr() = HeapHeader::new(HeapType::Vector, 2).raw();
            *b.as_ptr().add(1) = Word::fixnum(100).raw();
            *b.as_ptr().add(2) = Word::fixnum(200).raw();
        }
        let original_addr = b.as_ptr() as usize;
        let w = Word::from_ptr(b.as_ptr() as *const u8, Tag::Vector);
        let stack: Box<[u64]> = vec![w.raw()].into_boxed_slice();
        let lo = stack.as_ptr() as usize;
        let hi = unsafe { stack.as_ptr().add(stack.len()) } as usize;
        h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);

        let mut roots = [w];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );
        assert_eq!(result.objects_copied, 0);
        assert_eq!(result.pages_flipped, 1);
        assert_eq!(
            (roots[0].raw() & PAYLOAD_MASK) as usize,
            original_addr,
            "pinned boxed kept its address"
        );
        // Start bit still set so the next mark pass can find it.
        let cell_idx = (original_addr - h.base_ptr() as usize) / 8;
        assert!(is_start_at(h.start_bits_slice(), cell_idx));
        // Contents intact.
        unsafe {
            assert_eq!(
                Word::from_raw(*b.as_ptr().add(1)).as_fixnum(),
                Some(100)
            );
            assert_eq!(
                Word::from_raw(*b.as_ptr().add(2)).as_fixnum(),
                Some(200)
            );
        }
    }

    #[test]
    fn empty_from_gen_is_a_noop() {
        // Evacuating a generation with no pages should succeed
        // trivially: no copies, no reclaims, no panics.
        let mut h = small_heap();
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut [],
        );
        assert_eq!(result.objects_copied, 0);
        assert_eq!(result.cells_copied, 0);
        assert_eq!(result.pages_freed, 0);
        assert_eq!(result.pages_flipped, 0);
    }

    #[test]
    fn flipped_page_has_pinned_start_bits_preserved() {
        // After a page gets flipped (pinned, gen changed), the
        // pinned cell's start bit must STILL be set so future
        // walks can find it. The non-pinned cells on the same
        // page (now garbage / forwarding markers) must have had
        // their start bits cleared.
        let mut h = small_heap();
        let p1 = h.try_alloc_cons_in(Generation::G0).unwrap();
        let p2 = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *p1.as_ptr() = Word::fixnum(1).raw();
            *p1.as_ptr().add(1) = Word::NIL.raw();
            *p2.as_ptr() = Word::fixnum(2).raw();
            *p2.as_ptr().add(1) = Word::NIL.raw();
        }
        let w1 = Word::from_ptr(p1.as_ptr() as *const u8, Tag::Cons);
        let w2 = Word::from_ptr(p2.as_ptr() as *const u8, Tag::Cons);
        let stack: Box<[u64]> = vec![w1.raw()].into_boxed_slice();
        let lo = stack.as_ptr() as usize;
        let hi = unsafe { stack.as_ptr().add(stack.len()) } as usize;
        h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        let mut roots = [w1, w2];
        h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            &mut roots,
        );

        // p1 was pinned: its start bit must still be set.
        let p1_cell = (p1.as_ptr() as usize - h.base_ptr() as usize) / 8;
        assert!(
            is_cons_start_at(h.start_bits_slice(), p1_cell),
            "pinned cell keeps its cons-start bit"
        );
        // p2 was evacuated: its old cell should have a forwarding
        // pointer and NO start bit (we cleared the page's start
        // bits and only re-set pinned ones).
        let p2_cell = (p2.as_ptr() as usize - h.base_ptr() as usize) / 8;
        assert!(
            !is_start_at(h.start_bits_slice(), p2_cell),
            "evacuated cell's start bit cleared"
        );
        // p2's CELL still contains a Forward Word for diagnostic
        // purposes, but the start bit is gone so scanners ignore
        // it.
        unsafe {
            let w = Word::from_raw(*p2.as_ptr());
            assert!(w.is_forward());
        }
    }
}
