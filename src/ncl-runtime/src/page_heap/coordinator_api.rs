//! Adapter methods on `PageHeap` matching the inherent API
//! `GcCoordinator` and `MutatorState` expect to call on `gc::Heap`.
//!
//! The semispace `Heap` shaped these method names (`young_*`,
//! `old_*`, `try_alloc_slab`) around its two-semispace + card-anchored
//! geometry. Under build-time feature selection (`gc.rs`), the same
//! call sites need to compile against either heap, so the page heap
//! provides matching names whose semantics map to its own geometry:
//!
//!   - `young_*` reads aggregate over G0 pages.
//!   - `old_*` reads aggregate over G1 + Tenured pages.
//!   - `young_base_ptr` returns the WHOLE reservation base — the
//!     mutator uses it to compute global cell indices, which the
//!     page-heap's start-bit bitmap is also indexed by.
//!   - `young_try_alloc_slab(cells)` reserves a TLAB-sized chunk on
//!     a G0 cons page. The mutator bumps within it and sets start
//!     bits per allocation via the cached `young_starts_handle`.
//!   - `old_cards` returns the reservation-wide card table.
//!   - `old_live_base_ptr` returns the reservation base (the
//!     coordinator's card-mark range check anchors here).
//!   - `old_capacity_bytes_per_semi` returns the whole reservation
//!     size (there's no semispace split in the page-heap geometry —
//!     the barrier just needs a max bound on "old's address range").
//!
//! `collect_minor_with_static` and `last_pin_summary` are also
//! exposed here. The minor-GC method currently panics with a
//! sub-phase 11b pointer; wiring it up is the next landing.

use std::ptr::NonNull;
use std::sync::Arc;

use crate::heap_common::{CardTable, StartBits, CARD_SIZE_CELLS};

use super::evac::PageEvacuator;
use super::mark::{MarkScanner, PageMarker};
use super::page_desc::PageDesc;
use super::scanner::RootScanner;
use super::space::PAGE_SIZE_CELLS;

use super::page_desc::{Generation, PageKind};
use super::space::PageHeap;

impl PageHeap {
    pub(crate) fn gc_free_page_reserve_for_mutator_slab(&self) -> usize {
        if self.page_count() <= 1 {
            return 0;
        }

        self.page_count()
            .div_ceil(4)
            .min(self.page_count() - 1)
    }

    // -- Aggregate usage / capacity --------------------------------------

    /// Total used bytes across all generations. Sum of `words_used *
    /// 8` for every non-Free page.
    pub fn used_bytes(&self) -> usize {
        self.descs()
            .iter()
            .filter(|d| d.generation != Generation::Free)
            .map(|d| d.words_used as usize * 8)
            .sum()
    }

    /// Used bytes in G0 (the "nursery"). Mutators reading this via
    /// `MutatorState::young_used_bytes` see what minor GC will need
    /// to clear.
    pub fn young_used_bytes(&self) -> usize {
        self.descs()
            .iter()
            .filter(|d| d.generation == Generation::G0)
            .map(|d| d.words_used as usize * 8)
            .sum()
    }

    /// Used bytes in G1 + Tenured combined. The mutator's
    /// promote-tracking arithmetic asks "how many bytes survived
    /// into older?"
    pub fn old_used_bytes(&self) -> usize {
        self.descs()
            .iter()
            .filter(|d| {
                matches!(
                    d.generation,
                    Generation::G1 | Generation::Tenured
                )
            })
            .map(|d| d.words_used as usize * 8)
            .sum()
    }

    /// Capacity of one "old semispace." Page-heap has no semispace
    /// split — the barrier needs a max address-range bound and the
    /// whole reservation is the answer.
    pub fn old_capacity_bytes_per_semi(&self) -> usize {
        self.reserved_bytes()
    }

    // -- Nursery TLAB fast-path setup -------------------------------------

    /// Pointer used by mutators to compute global cell indices. The
    /// page-heap's start-bit bitmap is indexed by `(addr - base) /
    /// 8` against the reservation base, so we return the reservation
    /// base directly. (Compare semispace: returns the young semi's
    /// base because young cells live in a contiguous range there.)
    pub fn young_base_ptr(&self) -> *const u64 {
        self.base_ptr() as *const u64
    }

    /// Lock-free handle to the global start-bit bitmap. Mutators
    /// cache this at registration and flip pairs via the same
    /// atomic-OR fast path the semispace heap uses.
    pub fn young_starts_handle(&self) -> StartBits {
        self.start_bits_handle()
    }

    /// Refill a TLAB from G0's cons region. Allocates `cells`
    /// contiguous cells (a slab), returning `None` if no G0 cons
    /// page can be opened or if `cells` exceeds the page size. The
    /// mutator sets per-allocation start bits inside the slab via
    /// the cached `young_starts_handle`.
    ///
    /// Sub-phase 11a only: a slab is one entire fresh cons page's
    /// initial bump. The slab is opened by acquiring a fresh page
    /// (or growing the current G0 cons region) and reserving
    /// `cells` cells via `try_alloc_g0_cons_slab`. Per-cons start
    /// bits aren't set here — the mutator handles them so the cost
    /// stays on the alloc fast path. Returns the slab pointer plus
    /// the granted cell count, which may be smaller than the
    /// request because page-heap slabs are capped to one page.
    pub fn young_try_alloc_slab(
        &mut self,
        cells: usize,
    ) -> Option<(NonNull<u64>, usize)> {
        if cells == 0 {
            return None;
        }
        let granted = cells.min(PAGE_SIZE_CELLS);
        self.try_alloc_g0_cons_slab(cells).map(|slab| (slab, granted))
    }

    // -- Card-marking façade ---------------------------------------------

    /// Reservation-wide card table. The semispace anchors its card
    /// table at the live old-semispace's base and sizes it to one
    /// semispace; the page-heap card table covers the whole
    /// reservation so it works regardless of which generation an
    /// addressed object lives in.
    pub fn old_cards(&self) -> &Arc<CardTable> {
        &self.cards
    }

    /// Base address used as the card-table anchor. For semispace
    /// this is the live old-semispace base; for the page-heap, the
    /// reservation base.
    pub fn old_live_base_ptr(&self) -> *const u8 {
        self.base_ptr() as *const u8
    }

    // -- GC entry point --------------------------------------------------

    /// Minor GC entry called by `MutatorState::trigger_minor_gc`.
    ///
    /// Integration of sub-phase 8's `collect_minor` driver with the
    /// caller-supplied root pipeline:
    ///
    ///   1. Conservative pin pass over each stack range, targeting
    ///      G0. Populates `last_pin_summary` for `(gc-stats)`.
    ///   2. Drive a minor cycle via `cycle::collect_minor`. The
    ///      single visit closure feeds in:
    ///      - **Caller's roots** — mutator stacks / explicit-root
    ///        vectors — visited via the page-heap `RootScanner`
    ///        adapter (which delegates to `PageEvacuator::visit`).
    ///      - **Static-area dirty cards** — every cell in a dirty
    ///        card is offered to `visit_cell`; static→young
    ///        pointers get evacuated and the slot updated in
    ///        place.
    ///      - **Reservation dirty cards** — every dirty card on a
    ///        page in G1/Tenured is scanned for cross-gen refs
    ///        into G0. G0/Free pages are skipped (intra-G0 refs
    ///        come through the BFS drain; Free pages don't host
    ///        live data).
    ///   3. Clear both card tables (they're consumed; the next
    ///      cycle of mutator stores will re-dirty as needed).
    pub fn collect_minor_with_static(
        &mut self,
        static_cards: &CardTable,
        static_base: *mut u64,
        static_cells: usize,
        pin_stack_ranges: &[(usize, usize)],
        visit_roots: &mut dyn FnMut(&mut RootScanner<'_, '_>),
    ) {
        // 1. Conservative pin pass on G0 + G1.
        //
        // Why both: a minor cycle may CASCADE into G1→Tenured every
        // `G0_PROMOTION_THRESHOLD × G1_PROMOTION_THRESHOLD` minors
        // (15 by default). The cascade moves G1 objects, but the
        // mutator's stack-resident pointers to G1 are never
        // rewritten (the stack is conservatively scanned, never
        // updated). If we only pin G0 here, the cascade later
        // invalidates those stack-resident G1 pointers and the
        // mutator crashes on the next dereference (see the
        // `bytes-promoted-total=125 MB / minor-gcs=15` crash in
        // `demos/life.lisp`). Pinning G1 from the same stack ranges
        // is conservative-pin-cheap and keeps G1 targets in place
        // through any cascade in this cycle.
        let pin_g0 =
            self.pin_pointers_in_ranges(Generation::G0, pin_stack_ranges);
        let pin_g1 =
            self.pin_pointers_in_ranges(Generation::G1, pin_stack_ranges);
        self.last_pin_summary = (
            pin_g0.n_objects + pin_g1.n_objects,
            pin_g0.n_cells + pin_g1.n_cells,
        );
        // Extension mark: pinned objects may reach unmarked targets
        // (because the precise mark pass ran before pin). Walk pinned
        // payloads now, marking transitively in BOTH gens — G0 marks
        // already exist from `mark_minor_with_static`; the cascade
        // path inside `collect_minor` will internal-mark G1 again,
        // but extension here ensures pinned-G1 targets that are only
        // reachable via the conservative pin set get marked before
        // the cascade decides what to copy.
        //
        // Cross-gen extension closes the audit hole from
        // `docs/GC_HEAP_WALK_CLOSURE.md`: a pinned-G1 object whose
        // field points at a G0 cell would otherwise leave the G0
        // cell unmarked → unevacuated → page released → dangling
        // pointer in the pinned G1 object's field.
        if self.recycle_live_counts_active_for(Generation::G0) {
            self.extend_mark_from_pinned(Generation::G0);
            self.extend_mark_from_pinned(Generation::G1);
            self.extend_mark_from_cross_gen_pinned(Generation::G0);
            self.prepare_recycle_live_counts_from_marks(Generation::G0);
            self.release_zero_live_unpinned_pages(Generation::G0);
        }

        // Snapshot the per-page dirty-card layout BEFORE evacuation
        // mutates page descriptors. We need: for each card, what
        // generation was the underlying page in at scan time? We
        // record the page indices in G1/Tenured up front so the
        // scanner inside the cycle can skip G0/Free cards.
        let reservation_cards: Arc<CardTable> = Arc::clone(&self.cards);
        let reservation_base: *mut u64 = self.base_ptr() as *mut u64;
        let reservation_cells: usize = self.reserved_bytes() / 8;
        let descs_at_scan_time: Vec<super::page_desc::PageDesc> =
            self.descs().to_vec();

        // 2. Drive the minor cycle. The closure visits caller's
        //    roots, then the two dirty-card scans.
        self.collect_minor(|evac| {
            // 2a. Caller's roots via RootScanner. The scanner
            //     borrows evac for the duration of the visit; when
            //     it drops at end of block, evac is reusable.
            {
                let mut scanner = RootScanner::new(evac);
                visit_roots(&mut scanner);
            }
            // 2b. Static-area dirty cards.
            scan_dirty_cards_as_roots(
                evac,
                static_cards,
                static_base,
                static_cells,
                /*page_filter=*/ None,
            );
            // 2c. Reservation dirty cards in older-than-G0 pages.
            scan_dirty_cards_as_roots(
                evac,
                &reservation_cards,
                reservation_base,
                reservation_cells,
                Some(&descs_at_scan_time),
            );
        });

        // 3. Clear the card tables, BUT only for cards whose cells
        //    no longer contain inter-gen pointers. Cards that still
        //    point at moveable generations stay dirty so the next
        //    cycle's scan finds them.
        //
        // Why this matters: closures created via `ncl_make_closure`
        // mark the static-area `env` card once at construction. The
        // env Vector then moves across minor cycles (G0→G0 evac
        // every cycle, G0→G1 every 3rd, G1→Tenured every 15th).
        // Without persistence the card gets cleared after the first
        // cycle, and every subsequent GC misses the env field,
        // leaving the static-area pointer dangling and the JIT
        // dereferencing freed memory (this is the
        // `demos/life.lisp` crash at minor-gcs=15).
        clear_cards_unless_intergen(&self.cards, reservation_base, reservation_cells);
        clear_cards_unless_intergen(static_cards, static_base, static_cells);
    }

    /// Pre-evac mark pass for the production minor cycle. Streams
    /// the same root shape as evacuation, then seeds per-page live
    /// counts so BFS recycling can release drained from-pages.
    pub fn mark_minor_with_static(
        &mut self,
        static_cards: &CardTable,
        static_base: *mut u64,
        static_cells: usize,
        visit_roots: &mut dyn FnMut(&mut MarkScanner<'_, '_>),
    ) {
        let reservation_cards: Arc<CardTable> = Arc::clone(&self.cards);
        let reservation_base: *mut u64 = self.base_ptr() as *mut u64;
        let reservation_cells: usize = self.reserved_bytes() / 8;
        let descs_at_scan_time: Vec<super::page_desc::PageDesc> =
            self.descs().to_vec();

        {
            let mut marker = PageMarker::new(self, Generation::G0);
            {
                let mut scanner = MarkScanner::new(&mut marker);
                visit_roots(&mut scanner);
            }
            scan_dirty_cards_as_marks(
                &mut marker,
                static_cards,
                static_base,
                static_cells,
                None,
            );
            scan_dirty_cards_as_marks(
                &mut marker,
                &reservation_cards,
                reservation_base,
                reservation_cells,
                Some(&descs_at_scan_time),
            );
            marker.drain();
        }
        self.prepare_recycle_live_counts_from_marks(Generation::G0);
    }

    pub(crate) fn prepare_recycle_live_counts_from_marks(
        &mut self,
        target: Generation,
    ) {
        self.recycle_live_counts.fill(0);
        let target_pages: Vec<usize> = self.pages_in_gen(target).collect();
        let mut live_cells = 0usize;
        let mut live_pages = 0usize;
        for page_idx in target_pages {
            let first_word = page_idx * PAGE_SIZE_CELLS / 64;
            let words_per_page = PAGE_SIZE_CELLS / 64;
            let mut count = 0usize;
            for word_idx in first_word..first_word + words_per_page {
                count += self.mark_bits_slice()[word_idx].count_ones() as usize;
            }
            self.recycle_live_counts[page_idx] = count as u16;
            live_cells += count;
            if count > 0 {
                live_pages += 1;
            }
        }
        self.last_mark_live_cells = live_cells;
        self.last_mark_live_pages = live_pages;
        self.last_zero_live_pages_released = 0;
        self.recycle_live_counts_target = Some(target);
    }

    pub(crate) fn recycle_live_counts_active_for(
        &self,
        target: Generation,
    ) -> bool {
        self.recycle_live_counts_target == Some(target)
    }

    pub(crate) fn clear_recycle_live_counts(&mut self) {
        self.recycle_live_counts.fill(0);
        self.recycle_live_counts_target = None;
    }

    fn release_zero_live_unpinned_pages(&mut self, target: Generation) {
        let releasable: Vec<usize> = self
            .pages_in_gen(target)
            .filter(|&page_idx| {
                !self.desc(page_idx).has_pins()
                    && self.recycle_live_counts[page_idx] == 0
            })
            .collect();

        self.last_zero_live_pages_released = releasable.len();
        for page_idx in releasable {
            self.desc_mut(page_idx).release();
            self.recycle_live_counts[page_idx] = 0;
        }
    }

    pub fn last_mark_live_bytes(&self) -> usize {
        self.last_mark_live_cells * 8
    }

    pub fn last_mark_live_pages(&self) -> usize {
        self.last_mark_live_pages
    }

    pub fn last_zero_live_pages_released(&self) -> usize {
        self.last_zero_live_pages_released
    }

    /// Most recent pin-scan summary. Stored on the heap by the pin
    /// pass (sub-phase 11b will populate it from a real cycle).
    pub fn last_pin_summary(&self) -> (usize, usize) {
        self.last_pin_summary
    }

    // -- Slab allocation primitive used by `young_try_alloc_slab` -------

    /// Reserve `cells` contiguous cells on a G0 page for use as
    /// a mutator TLAB. The slab uses a **`PageKind::Boxed` page**,
    /// not `Cons`, because the mutator writes a mix of cons cells
    /// AND header-bearing objects (vectors, strings, symbols,
    /// etc.) into a single TLAB. The walker dispatches on the
    /// per-cell start-bit pattern (cons `11` vs boxed `01`) to
    /// determine each object's stride, so it doesn't matter that
    /// the page descriptor says "Boxed."
    ///
    /// Why this works: `PageKind::Cons` is an optimization signal
    /// for pages where **every** cell is a 2-cell cons pair — the
    /// walker can skip the start-bit lookup and stride by 2
    /// unconditionally. The page-heap's internal
    /// `try_alloc_cons_in` keeps using Cons pages because it only
    /// allocates conses. Mutator TLABs are mixed, so they use
    /// Boxed pages and the walker reads start bits on each visit.
    ///
    /// Caller responsibilities (matched against the semispace
    /// contract):
    ///   - Set per-allocation start bits via the cached
    ///     `young_starts_handle` as the TLAB is bumped into.
    ///     `set_cons_start_bit_at` for conses (pair `11`),
    ///     `set_start_bit_at` for boxed (pair `01`).
    ///   - Treat the returned chunk as raw memory; the page
    ///     descriptor records `cells` as `words_used`, so any cell
    ///     unwritten inside the chunk is "logical garbage" that
    ///     GC walkers will skip via the missing start-bit.
    ///
    /// `words_used` therefore tracks the reserved slab extent, not
    /// the mutator's later high-water mark inside that slab. That's
    /// scan-safe because page-heap walkers treat start bits as the
    /// object-boundary source of truth; untouched cells in the tail
    /// have no start bits and are skipped.
    fn try_alloc_g0_cons_slab(&mut self, cells: usize) -> Option<NonNull<u64>> {
        // Don't accept oversize slabs — slabs are TLAB-sized
        // (default `tlab_cells` in GcConfig = 65536 cells = 512
        // KB; bigger than a page, so we cap at PAGE_SIZE_CELLS).
        // The mutator's loop in `refill_tlab` will retry with the
        // current page's remaining capacity if we cap.
        use super::space::PAGE_SIZE_CELLS;
        let cells = cells.min(PAGE_SIZE_CELLS);
        // Fast path: the open G0 boxed region has room. Bumps
        // within an already-acquired G0 page — doesn't grow G0,
        // so the young-cap check below doesn't apply.
        if let Some(p) = self.try_bump_g0_mixed(cells) {
            return Some(p);
        }
        if self.count_pages_in_gen(Generation::Free)
            <= self.gc_free_page_reserve_for_mutator_slab()
        {
            return None;
        }
        // Young-cap trigger: page-heap's analogue of "young is full".
        // If G0 has already grown to its configured size, refuse to
        // open another fresh page — `refill_tlab` will see the None
        // and call `trigger_minor_gc`, which empties G0 and lets the
        // retry succeed. Without this gate, G0 keeps swallowing free
        // pages until the whole reservation is gone, and MINOR-GCS
        // stays at zero even on heavy workloads.
        if self.count_pages_in_gen(Generation::G0) >= self.young_page_cap {
            return None;
        }
        // Slow path: open a fresh G0 boxed page.
        let new_page = self.acquire_free_page(Generation::G0, PageKind::Boxed)?;
        let r = self.alloc_region_mut(Generation::G0, PageKind::Boxed);
        r.current_page = new_page;
        r.offset = 0;
        self.try_bump_g0_mixed(cells)
    }

    /// Bump-allocate `cells` from the open G0 boxed region without
    /// setting any start bits (caller's responsibility — see
    /// `try_alloc_g0_cons_slab`). Returns `None` if the region has
    /// no open page or the bump would overflow it.
    fn try_bump_g0_mixed(&mut self, cells: usize) -> Option<NonNull<u64>> {
        use super::space::PAGE_SIZE_CELLS;
        let r = self.alloc_region(Generation::G0, PageKind::Boxed);
        if !r.has_page() || r.offset + cells > PAGE_SIZE_CELLS {
            return None;
        }
        let page_idx = r.current_page;
        let offset = r.offset;
        let page_base = self.page_ptr(page_idx) as *mut u64;
        let ptr = unsafe { page_base.add(offset) };
        {
            let r = self.alloc_region_mut(Generation::G0, PageKind::Boxed);
            r.offset += cells;
        }
        {
            let d = self.desc_mut(page_idx);
            d.words_used = d.words_used.saturating_add(cells as u16);
        }
        // SAFETY: pointer is within a freshly-committed G0 page.
        Some(unsafe { NonNull::new_unchecked(ptr) })
    }
}

/// Clear cards whose cells no longer contain inter-gen pointers.
/// Cards that still contain ANY heap-pointer Word stay dirty so the
/// next minor cycle's scan keeps tracking them — this is how
/// long-lived inter-gen pointers (e.g. a static-area closure's
/// `env` field) survive multiple GC cycles without the mutator
/// having to re-mark the card.
///
/// The pointer-tag check is conservative — it flags any Word whose
/// low 3 bits look like a heap pointer, regardless of where it
/// actually points. False positives keep the card dirty for an
/// extra cycle; false negatives would lose inter-gen refs, so we
/// err in the safe direction.
fn clear_cards_unless_intergen(
    cards: &CardTable,
    base: *mut u64,
    cells: usize,
) {
    use crate::word::{Tag, Word};
    for card_idx in 0..cards.n_cards() {
        if !cards.is_dirty(card_idx) {
            continue;
        }
        let card_start = card_idx * CARD_SIZE_CELLS;
        if card_start >= cells {
            cards.clear(card_idx);
            continue;
        }
        let card_end = (card_start + CARD_SIZE_CELLS).min(cells);
        let mut has_heap_pointer = false;
        for c in card_start..card_end {
            let cell = unsafe { *base.add(c) };
            let tag = Word::from_raw(cell).tag();
            if matches!(
                tag,
                Tag::Cons | Tag::Symbol | Tag::Vector | Tag::Function | Tag::String
            ) {
                has_heap_pointer = true;
                break;
            }
        }
        if !has_heap_pointer {
            cards.clear(card_idx);
        }
    }
}

/// Scan every dirty card in `cards` over the byte range starting at
/// `base` (length `cells` cells), offering each cell to the
/// evacuator as a candidate root.
///
/// `page_filter`: if `Some(descs)`, the scan only fires on cards
/// whose underlying page is in G1 or Tenured (per the snapshotted
/// page-descriptor slice). G0/Free pages are skipped so we don't
/// re-scan from-pages mid-evacuation. If `None`, every dirty
/// card is scanned (used for the static area — it has no page
/// notion).
///
/// Card semantics:
///   - `CARD_SIZE_CELLS = 64` cells per card.
///   - Cells outside `0..cells` are clamped.
///   - The card table covers exactly `cells * 8` bytes from `base`.
fn scan_dirty_cards_as_roots(
    evac: &mut PageEvacuator<'_>,
    cards: &CardTable,
    base: *mut u64,
    cells: usize,
    page_filter: Option<&[PageDesc]>,
) {
    use super::page_desc::Generation;
    for card_idx in 0..cards.n_cards() {
        if !cards.is_dirty(card_idx) {
            continue;
        }
        let card_start = card_idx * CARD_SIZE_CELLS;
        if card_start >= cells {
            break;
        }
        let card_end = (card_start + CARD_SIZE_CELLS).min(cells);

        // If a page filter was supplied, gate by the page's
        // generation. Within a card, all 64 cells live in the
        // same 512-byte slice, which is well within one 64 KB
        // page — so a single page lookup per card is correct.
        if let Some(descs) = page_filter {
            let page_idx = card_start / PAGE_SIZE_CELLS;
            // Defensive bounds.
            if page_idx >= descs.len() {
                continue;
            }
            match descs[page_idx].generation {
                Generation::G1 | Generation::Tenured => {}
                _ => continue, // skip G0/Free
            }
        }

        // Scan each cell as a candidate Word.
        for c in card_start..card_end {
            // SAFETY: c < cells, so `base.add(c)` is in-range of
            // the caller-supplied buffer; `visit_cell`'s contract
            // demands an aligned u64 cell, which `base.add(c)`
            // satisfies since `base` is u64-aligned and `c` is a
            // u64 offset.
            let cell_ptr = unsafe { base.add(c) };
            unsafe { evac.visit_cell(cell_ptr) };
        }
    }
}

fn scan_dirty_cards_as_marks(
    marker: &mut PageMarker<'_>,
    cards: &CardTable,
    base: *mut u64,
    cells: usize,
    page_filter: Option<&[PageDesc]>,
) {
    use super::page_desc::Generation;
    for card_idx in 0..cards.n_cards() {
        if !cards.is_dirty(card_idx) {
            continue;
        }
        let card_start = card_idx * CARD_SIZE_CELLS;
        if card_start >= cells {
            break;
        }
        let card_end = (card_start + CARD_SIZE_CELLS).min(cells);

        if let Some(descs) = page_filter {
            let page_idx = card_start / PAGE_SIZE_CELLS;
            if page_idx >= descs.len() {
                continue;
            }
            match descs[page_idx].generation {
                Generation::G1 | Generation::Tenured => {}
                _ => continue,
            }
        }

        for c in card_start..card_end {
            let cell_ptr = unsafe { base.add(c) };
            unsafe { marker.visit_cell(cell_ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_heap() -> PageHeap {
        PageHeap::with_reservation(8 * 64 * 1024)
    }

    #[test]
    fn ctor_sums_young_and_old_bytes_for_reservation() {
        // 1 MB young + 1 MB old = 2 MB reservation = 32 pages.
        let h = PageHeap::new(1024 * 1024, 1024 * 1024);
        assert_eq!(h.page_count(), 32);
        assert_eq!(h.reserved_bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn ctor_enforces_four_page_minimum() {
        // Tiny config — coordinator could pass small values for
        // unit tests, page-heap rounds up to four pages so
        // within-gen evacuation always has at least one Free
        // page to copy into.
        let h = PageHeap::new(0, 0);
        assert_eq!(h.page_count(), 4);
        assert_eq!(h.reserved_bytes(), 4 * 64 * 1024);
    }

    #[test]
    fn used_bytes_aggregates_by_generation() {
        let mut h = small_heap();
        // No allocations yet → everything zero.
        assert_eq!(h.used_bytes(), 0);
        assert_eq!(h.young_used_bytes(), 0);
        assert_eq!(h.old_used_bytes(), 0);

        // Alloc 10 conses in G0 → 20 cells = 160 bytes.
        for _ in 0..10 {
            h.try_alloc_cons_in(Generation::G0).unwrap();
        }
        assert_eq!(h.young_used_bytes(), 160);
        assert_eq!(h.old_used_bytes(), 0);
        assert_eq!(h.used_bytes(), 160);

        // Alloc 5 conses in G1 → another 10 cells = 80 bytes.
        for _ in 0..5 {
            h.try_alloc_cons_in(Generation::G1).unwrap();
        }
        assert_eq!(h.young_used_bytes(), 160, "G0 untouched");
        assert_eq!(h.old_used_bytes(), 80);
        assert_eq!(h.used_bytes(), 240);
    }

    #[test]
    fn slab_alloc_returns_aligned_pointer_and_advances_words_used() {
        let mut h = small_heap();
        let (slab, granted) = h.young_try_alloc_slab(128).expect("slab alloc");
        assert_eq!(granted, 128);
        assert_eq!(slab.as_ptr() as usize % 8, 0, "8-byte aligned");
        // The G0 page descriptor reflects the slab reservation.
        // Slab pages are `Boxed` (not `Cons`) — they hold a mix
        // of conses and header-bearing objects allocated by the
        // mutator TLAB. The walker dispatches on the per-cell
        // start-bit pattern for these pages.
        let page =
            (slab.as_ptr() as usize - h.base_ptr() as usize) / (64 * 1024);
        assert_eq!(h.desc(page).generation, Generation::G0);
        assert_eq!(h.desc(page).kind, PageKind::Boxed);
        assert_eq!(h.desc(page).words_used, 128);
    }

    #[test]
    fn slab_alloc_fits_multiple_slabs_per_page() {
        let mut h = small_heap();
        // PAGE_SIZE_CELLS = 8192. Two 4096-cell slabs fit in one page.
        let (a, granted_a) = h.young_try_alloc_slab(4096).expect("first slab");
        let (b, granted_b) = h.young_try_alloc_slab(4096).expect("second slab");
        assert_eq!(granted_a, 4096);
        assert_eq!(granted_b, 4096);
        // Same page.
        let pa = (a.as_ptr() as usize - h.base_ptr() as usize) / (64 * 1024);
        let pb = (b.as_ptr() as usize - h.base_ptr() as usize) / (64 * 1024);
        assert_eq!(pa, pb);
        // 4096 cells = 32 KB; b should be 32 KB past a.
        assert_eq!(b.as_ptr() as usize - a.as_ptr() as usize, 32 * 1024);
    }

    #[test]
    fn slab_alloc_caps_at_page_size() {
        let mut h = small_heap();
        // Requesting more than PAGE_SIZE_CELLS hands back a full
        // page's worth, not an error.
        let (slab, granted) = h.young_try_alloc_slab(100_000).expect("slab alloc");
        assert_eq!(granted, super::super::space::PAGE_SIZE_CELLS);
        let page =
            (slab.as_ptr() as usize - h.base_ptr() as usize) / (64 * 1024);
        assert_eq!(
            h.desc(page).words_used,
            super::super::space::PAGE_SIZE_CELLS as u16,
            "request capped at one page"
        );
    }

    #[test]
    fn slab_alloc_zero_cells_returns_none() {
        let mut h = small_heap();
        assert!(h.young_try_alloc_slab(0).is_none());
    }

    #[test]
    fn slab_alloc_uses_current_page_before_reserve_blocks_new_page() {
        let mut h = small_heap();
        // 8-page heap => reserve keeps 2 pages away from the mutator.
        // Fill 5 pages completely, then half-fill page 6.
        for _ in 0..10 {
            h.young_try_alloc_slab(4096)
                .expect("fill pages before reserve boundary");
        }
        h.young_try_alloc_slab(4096)
            .expect("leave one half-full current page at reserve boundary");
        assert_eq!(h.count_pages_in_gen(Generation::Free), 2);

        // Fast path still uses the remainder of the current page.
        assert!(h.young_try_alloc_slab(4096).is_some());
        assert_eq!(h.count_pages_in_gen(Generation::Free), 2);

        // Another slab would need a new page, so the reserve stops it.
        assert!(h.young_try_alloc_slab(1).is_none());
        assert_eq!(h.count_pages_in_gen(Generation::Free), 2);
    }

    #[test]
    fn old_cards_and_base_match_reservation() {
        let h = small_heap();
        // Card table covers the whole reservation (8 pages × 64 KB =
        // 512 KB / 512-byte cards = 1024 cards).
        assert_eq!(h.old_cards().n_cards(), 1024);
        assert_eq!(h.old_live_base_ptr(), h.base_ptr());
        assert_eq!(h.old_capacity_bytes_per_semi(), h.reserved_bytes());
    }
}
