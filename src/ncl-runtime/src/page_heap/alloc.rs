//! Object allocation into the page heap.
//!
//! Sub-phase 4 of the Phase 3 plan in `docs/GC_DESIGN.md`. Replaces
//! the semispace "bump within one big buffer" allocator with
//! per-generation open-allocation regions that bump within
//! individual pages, opening fresh pages from the page table as
//! they fill.
//!
//! ## AllocRegion
//!
//! One per `(Generation, PageKind)` combination. Tracks the
//! currently-open page and the bump offset within it.
//! Allocations check "does it fit in `current_page`?"; if yes,
//! bump the offset; if no, retire the current page and open a
//! fresh one.
//!
//! ## Page retirement
//!
//! When a page can't fit a request, the region "retires" the page
//! (current_page becomes inactive — its `words_used` is final)
//! and acquires a fresh page from the page heap's free list. The
//! retired page stays in its generation cohort; subsequent GC
//! passes will mark/evacuate it like any other.
//!
//! ## Free-page acquisition
//!
//! Sub-phase 4 uses a linear scan over `descs` to find the
//! next Free-generation page. O(n_pages) per page-switch. For
//! 16384 pages this is still microseconds; sub-phase 7 will add a
//! proper free-list (a `VecDeque<usize>` of free page indices) if
//! profiling shows it matters.
//!
//! ## Start bits
//!
//! Boxed-kind pages need a start-bit bitmap so the GC scanner can
//! find object boundaries. Cons-kind pages don't — every 16 bytes
//! is one cons cell, walkers step by 2 cells.
//!
//! For sub-phase 4 the start-bit bitmap is GLOBAL (one big
//! `StartBits` Arc spanning the whole reservation), matching the
//! existing `Semispace` layout. This lets mutators cache a single
//! bitmap handle and use the same atomic-OR fast path they use
//! today. The trade-off vs per-page bitmaps is memory: 32 MB up
//! front for the 1 GB reservation (3% overhead — same ratio as
//! semispace). The agent's plan flagged this as a microbench
//! candidate; if per-page locality matters later, the conversion
//! is contained to this file.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::page_desc::{Generation, PageDesc, PageKind};
use super::space::{PageHeap, PAGE_SIZE_CELLS};

/// Currently-open allocation region for one (generation, kind)
/// pair. The page heap maintains one per pair.
///
/// `current_page == usize::MAX` is the sentinel "no page open."
/// In that state, every allocation goes through the slow path of
/// opening a fresh page.
#[derive(Clone, Copy, Debug)]
pub struct AllocRegion {
    pub generation: Generation,
    pub kind: PageKind,
    pub current_page: usize,
    /// Cell offset within `current_page` where the next allocation
    /// goes. Always `<= PAGE_SIZE_CELLS`.
    pub offset: usize,
}

/// Sentinel for "no page open in this region yet."
const NO_PAGE: usize = usize::MAX;

impl AllocRegion {
    pub const fn empty(generation: Generation, kind: PageKind) -> AllocRegion {
        AllocRegion {
            generation,
            kind,
            current_page: NO_PAGE,
            offset: 0,
        }
    }

    /// Does this region currently have an open page?
    pub fn has_page(&self) -> bool {
        self.current_page != NO_PAGE
    }

    /// Cells remaining in the currently-open page. Zero if no page.
    pub fn remaining_cells(&self) -> usize {
        if self.has_page() {
            PAGE_SIZE_CELLS - self.offset
        } else {
            0
        }
    }
}

// -- Start-bit bitmap (sub-phase 4 uses the semispace shape) --
//
// Same encoding as `Semispace::starts`:
//   - 2 bits per cell, packed into u64 words (32 cells per word)
//   - Pair `01` = header-bearing object start (boxed)
//   - Pair `11` = headerless cons start (both bits set)
//   - Pair `00` = not a start (default)
//
// Mutators can cache an `Arc<[AtomicU64]>` handle and use the same
// `set_start_bit_at` / `set_cons_start_bit_at` helpers from
// `heap.rs`. For sub-phase 4 the helpers are duplicated here as
// inline statics so the page-heap module doesn't depend on
// `heap.rs` internals — sub-phase 11 will unify.

const CELLS_PER_STARTS_WORD: usize = 32;
const STARTS_PAIR_HEADER: u64 = 0b01;
const STARTS_PAIR_CONS: u64 = 0b11;

/// Type alias matching `heap::StartBits` so mutators can hold one
/// handle covering whichever heap backend is active.
pub type PageStartBits = Arc<[AtomicU64]>;

/// Mark cell `idx` as a header-bearing object start. `idx` is
/// global (offset from `PageHeap::base_ptr` in cells).
pub fn set_start_bit_at(bits: &[AtomicU64], idx: usize) {
    let w = idx / CELLS_PER_STARTS_WORD;
    let b = ((idx % CELLS_PER_STARTS_WORD) * 2) as u32;
    bits[w].fetch_or(STARTS_PAIR_HEADER << b, Ordering::Relaxed);
}

/// Mark cell `idx` as a cons-cell start. Both bits in the pair go
/// to `1` so walkers can distinguish cons from header.
pub fn set_cons_start_bit_at(bits: &[AtomicU64], idx: usize) {
    let w = idx / CELLS_PER_STARTS_WORD;
    let b = ((idx % CELLS_PER_STARTS_WORD) * 2) as u32;
    bits[w].fetch_or(STARTS_PAIR_CONS << b, Ordering::Relaxed);
}

/// Test whether cell `idx` is any kind of object start.
pub fn is_start_at(bits: &[AtomicU64], idx: usize) -> bool {
    let w = idx / CELLS_PER_STARTS_WORD;
    let b = ((idx % CELLS_PER_STARTS_WORD) * 2) as u32;
    (bits[w].load(Ordering::Relaxed) >> b) & 1 != 0
}

/// Test whether cell `idx` is a cons start specifically.
pub fn is_cons_start_at(bits: &[AtomicU64], idx: usize) -> bool {
    let w = idx / CELLS_PER_STARTS_WORD;
    let b = ((idx % CELLS_PER_STARTS_WORD) * 2) as u32;
    let pair = (bits[w].load(Ordering::Relaxed) >> b) & 0b11;
    pair == STARTS_PAIR_CONS
}

impl PageHeap {
    /// Find a free page and convert it to (`generation`, `kind`).
    /// Commits the page if not already committed. Returns
    /// `None` if no free pages are available (heap is full).
    ///
    /// Sub-phase 4: O(n_pages) linear scan. Sub-phase 7 may add a
    /// proper free-list.
    pub(crate) fn acquire_free_page(
        &mut self,
        generation: Generation,
        kind: PageKind,
    ) -> Option<usize> {
        let n = self.page_count();
        // Linear scan for the first Free-generation page. Start
        // from index 0 every time — locality matters less for the
        // slow path than for the alloc inner loop.
        let mut found: Option<usize> = None;
        for i in 0..n {
            if self.desc(i).generation == Generation::Free {
                found = Some(i);
                break;
            }
        }
        let idx = found?;
        // Commit the page if needed.
        self.commit_page(idx).ok()?;
        // Assign generation / kind via desc_mut.
        *self.desc_mut(idx) = PageDesc::fresh(generation, kind);
        // Bug #1 from the code review (docs/GC_DESIGN.md sub-phase
        // 6.5): a previously-freed page may still carry stale
        // start bits in the global bitmap from its prior tenant.
        // After evacuation lands in sub-phase 7, recycled pages
        // become routine; without this zero pass a conservative
        // pointer landing on a stale `01`/`11` bit pattern would
        // pass the start-bit gate and a) be incorrectly pinned, or
        // b) be followed by the marker. Zero the page's 256-word
        // slice of the start-bit table here.
        const WORDS_PER_PAGE: usize = PAGE_SIZE_CELLS / CELLS_PER_STARTS_WORD;
        let first_word = idx * WORDS_PER_PAGE;
        let bits = self.start_bits_slice();
        for w in first_word..first_word + WORDS_PER_PAGE {
            bits[w].store(0, Ordering::Relaxed);
        }
        Some(idx)
    }

    /// Allocate a cons cell (2 cells = 16 bytes) in the given
    /// generation. Returns a pointer to the first cell, or `None`
    /// if the heap is full.
    ///
    /// Cons cells have no header and no start bit (cons-page
    /// walkers step by 2 cells unconditionally), but allocations
    /// still bump the per-page `words_used` so the page table
    /// reflects the bump-pointer high water.
    pub fn try_alloc_cons_in(
        &mut self,
        generation: Generation,
    ) -> Option<NonNull<u64>> {
        self.try_alloc_in_region(generation, PageKind::Cons, 2, /*is_cons*/ true)
    }

    /// Allocate `n_cells` for a boxed object (header + payload) in
    /// the given generation. Sets the header start bit at the
    /// returned cell index. Returns a pointer to the first cell
    /// or `None` if the heap is full.
    ///
    /// `n_cells` must include the HeapHeader cell (so 1 for
    /// header-only, 5 for a Function, etc.). Allocations larger
    /// than a single page (8192 cells) are rejected — Large
    /// objects use a different path that lands in sub-phase 7.
    pub fn try_alloc_boxed_in(
        &mut self,
        generation: Generation,
        n_cells: usize,
    ) -> Option<NonNull<u64>> {
        if n_cells == 0 || n_cells > PAGE_SIZE_CELLS {
            return None;
        }
        self.try_alloc_in_region(generation, PageKind::Boxed, n_cells, /*is_cons*/ false)
    }

    /// Inner allocation. Common path for cons and boxed; the
    /// `is_cons` flag controls start-bit semantics. The
    /// `(generation, kind)` pair indexes into the alloc-region
    /// table; if the current page can't fit, a fresh one is
    /// acquired from the free list.
    fn try_alloc_in_region(
        &mut self,
        generation: Generation,
        kind: PageKind,
        n_cells: usize,
        is_cons: bool,
    ) -> Option<NonNull<u64>> {
        // Try the fast path — fits in the current region.
        if let Some(ptr) = self.try_bump_in_current(generation, kind, n_cells, is_cons) {
            return Some(ptr);
        }
        // Slow path: acquire a fresh page and retry. The retry
        // can still fail if a single allocation exceeds the page
        // size (Boxed > 8192 cells), which is currently rejected
        // upstream — but defensive None is correct for any other
        // OOM.
        let new_page = self.acquire_free_page(generation, kind)?;
        // Reset the region to point at the new page. Any pages
        // previously held in `current_page` of this region stay
        // assigned to their generation; the next GC sees their
        // `words_used` as final.
        let region_ref = self.alloc_region_mut(generation, kind);
        region_ref.current_page = new_page;
        region_ref.offset = 0;
        self.try_bump_in_current(generation, kind, n_cells, is_cons)
    }

    /// Fast-path bump within the region's current page. Returns
    /// `None` if the page has no current open page yet (sentinel
    /// `NO_PAGE`) or if `n_cells` doesn't fit in the remaining
    /// space.
    fn try_bump_in_current(
        &mut self,
        generation: Generation,
        kind: PageKind,
        n_cells: usize,
        is_cons: bool,
    ) -> Option<NonNull<u64>> {
        // Read region state, decide if alloc fits.
        let (page_idx, alloc_offset) = {
            let r = self.alloc_region(generation, kind);
            if !r.has_page() || r.offset + n_cells > PAGE_SIZE_CELLS {
                return None;
            }
            (r.current_page, r.offset)
        };
        // Compute pointer + global cell index for start-bit mark.
        let page_base = self.page_ptr(page_idx) as *mut u64;
        let ptr = unsafe { page_base.add(alloc_offset) };
        let global_cell_idx = self.global_cell_index(page_idx, alloc_offset);

        // Update bookkeeping: advance region offset, advance
        // page's words_used, set start bit on Boxed.
        {
            let r = self.alloc_region_mut(generation, kind);
            r.offset += n_cells;
        }
        {
            let d = self.desc_mut(page_idx);
            d.words_used = d.words_used.saturating_add(n_cells as u16);
        }
        if is_cons {
            set_cons_start_bit_at(self.start_bits_slice(), global_cell_idx);
        } else {
            set_start_bit_at(self.start_bits_slice(), global_cell_idx);
        }
        // SAFETY: pointer is within a freshly-committed page; the
        // caller initialises the cells via `*p.add(i) = ...`.
        Some(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Global cell index (offset from `base_ptr` in cells) for
    /// the cell at `(page_idx, offset_within_page)`.
    pub fn global_cell_index(&self, page_idx: usize, cell_offset: usize) -> usize {
        page_idx * PAGE_SIZE_CELLS + cell_offset
    }

    /// Total cells covered by the reservation (every cell has 2
    /// bits in `start_bits`).
    pub fn total_cells(&self) -> usize {
        self.page_count() * PAGE_SIZE_CELLS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::page_desc::Generation;

    fn small_heap() -> PageHeap {
        // 8 pages = 512 KB. Enough to exercise multi-page allocation
        // (one page = 8192 cells = 4096 cons cells).
        PageHeap::new(8 * 64 * 1024)
    }

    #[test]
    fn alloc_region_starts_empty() {
        let h = small_heap();
        let r = h.alloc_region(Generation::G0, PageKind::Cons);
        assert!(!r.has_page());
        assert_eq!(r.offset, 0);
        assert_eq!(r.generation, Generation::G0);
        assert_eq!(r.kind, PageKind::Cons);
    }

    #[test]
    fn first_alloc_acquires_a_page() {
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).expect("first cons");
        assert!(!p.as_ptr().is_null());
        // Some G0 page got assigned.
        assert_eq!(h.count_pages_in_gen(Generation::G0), 1);
        // The region now has a current page.
        let r = h.alloc_region(Generation::G0, PageKind::Cons);
        assert!(r.has_page());
        assert_eq!(r.offset, 2);
    }

    #[test]
    fn alloc_cons_returns_aligned_pointer() {
        let mut h = small_heap();
        for _ in 0..100 {
            let p = h.try_alloc_cons_in(Generation::G0).unwrap();
            let addr = p.as_ptr() as usize;
            assert_eq!(addr % 8, 0, "cons pointer must be 8-byte aligned");
        }
    }

    #[test]
    fn cons_allocs_set_cons_start_bit() {
        let mut h = small_heap();
        let p1 = h.try_alloc_cons_in(Generation::G0).unwrap();
        let p2 = h.try_alloc_cons_in(Generation::G0).unwrap();
        // Both must be cons-start.
        let base = h.base_ptr() as usize;
        let idx1 = (p1.as_ptr() as usize - base) / 8;
        let idx2 = (p2.as_ptr() as usize - base) / 8;
        assert!(is_cons_start_at(h.start_bits_slice(), idx1));
        assert!(is_cons_start_at(h.start_bits_slice(), idx2));
        // And they must be a cons-cell apart (2 cells).
        assert_eq!(idx2, idx1 + 2);
    }

    #[test]
    fn allocs_within_one_page_are_contiguous() {
        let mut h = small_heap();
        let first = h.try_alloc_cons_in(Generation::G0).unwrap().as_ptr();
        let mut last = first;
        // PAGE_SIZE_CELLS = 8192, /2 = 4096 cons cells per page.
        // Run 100 to stay safely within one page.
        for _ in 1..100 {
            let p = h.try_alloc_cons_in(Generation::G0).unwrap().as_ptr();
            // Next cons must be exactly 2 cells = 16 bytes past
            // the previous one.
            assert_eq!(p as usize, last as usize + 16);
            last = p;
        }
    }

    #[test]
    fn alloc_overflows_first_page_into_second() {
        let mut h = small_heap();
        // PAGE_SIZE_CELLS = 8192; cons takes 2; fill exactly one
        // page worth (4096 conses) and then one more.
        for _ in 0..4096 {
            h.try_alloc_cons_in(Generation::G0).unwrap();
        }
        assert_eq!(h.count_pages_in_gen(Generation::G0), 1);
        let r = h.alloc_region(Generation::G0, PageKind::Cons);
        assert_eq!(r.offset, PAGE_SIZE_CELLS, "first page is full");

        // The next alloc must open a second page.
        let _ = h.try_alloc_cons_in(Generation::G0).unwrap();
        assert_eq!(h.count_pages_in_gen(Generation::G0), 2);
        let r = h.alloc_region(Generation::G0, PageKind::Cons);
        assert_eq!(r.offset, 2, "fresh page, one cons in");
    }

    #[test]
    fn one_hundred_k_cons_cells_across_pages() {
        // 100k cons × 16 bytes = 1.6 MB. At 32 KB of conses per
        // page (4096 conses), needs 25 pages. Use a 32-page heap
        // = 2 MB so there's headroom.
        let mut h = PageHeap::new(32 * 64 * 1024);
        // Stash a few pointers to verify content round-trips
        // after the cells are written.
        let mut samples: Vec<*mut u64> = Vec::with_capacity(10);
        for i in 0..100_000_u32 {
            let p = h.try_alloc_cons_in(Generation::G0).expect("alloc");
            let cell = p.as_ptr();
            unsafe {
                *cell = i as u64;
                *cell.add(1) = (i as u64).wrapping_mul(31);
            }
            if i % 10_000 == 0 {
                samples.push(cell);
            }
        }
        // Should have spread over ~25 pages.
        let pages_used = h.count_pages_in_gen(Generation::G0);
        assert!(
            (24..=26).contains(&pages_used),
            "expected 24-26 G0 pages, got {pages_used}"
        );
        // Sample-point contents survived.
        for (idx, &p) in samples.iter().enumerate() {
            let i = (idx as u32) * 10_000;
            unsafe {
                assert_eq!(*p, i as u64, "sample {idx} car corrupted");
                assert_eq!(
                    *p.add(1),
                    (i as u64).wrapping_mul(31),
                    "sample {idx} cdr corrupted"
                );
            }
        }
    }

    #[test]
    fn heap_exhaustion_returns_none() {
        let mut h = small_heap();
        // Fill all 8 pages with cons cells. 8 × 4096 = 32768 allocs.
        for _ in 0..32_768 {
            assert!(h.try_alloc_cons_in(Generation::G0).is_some());
        }
        // 33rd alloc onto a fresh page fails — no free pages.
        assert!(h.try_alloc_cons_in(Generation::G0).is_none());
    }

    #[test]
    fn boxed_alloc_sets_header_start_bit() {
        let mut h = small_heap();
        // 5-cell object (matches Function record size).
        let p = h.try_alloc_boxed_in(Generation::G0, 5).unwrap();
        let base = h.base_ptr() as usize;
        let idx = (p.as_ptr() as usize - base) / 8;
        assert!(is_start_at(h.start_bits_slice(), idx));
        assert!(
            !is_cons_start_at(h.start_bits_slice(), idx),
            "boxed start is `01`, not `11`"
        );
    }

    #[test]
    fn cons_and_boxed_use_different_pages() {
        let mut h = small_heap();
        h.try_alloc_cons_in(Generation::G0).unwrap();
        h.try_alloc_boxed_in(Generation::G0, 4).unwrap();
        // Two pages assigned: one Cons, one Boxed.
        let mut cons_pages = 0;
        let mut boxed_pages = 0;
        for (i, d) in h.descs().iter().enumerate() {
            if d.generation == Generation::G0 {
                match d.kind {
                    PageKind::Cons => cons_pages += 1,
                    PageKind::Boxed => boxed_pages += 1,
                    other => panic!("unexpected G0 kind {other:?} on page {i}"),
                }
            }
        }
        assert_eq!(cons_pages, 1);
        assert_eq!(boxed_pages, 1);
    }

    #[test]
    fn boxed_too_big_returns_none() {
        let mut h = small_heap();
        // PAGE_SIZE_CELLS = 8192. One more than that overflows.
        assert!(h.try_alloc_boxed_in(Generation::G0, 8193).is_none());
        // Zero-cells is also rejected.
        assert!(h.try_alloc_boxed_in(Generation::G0, 0).is_none());
    }

    #[test]
    fn words_used_tracks_allocation() {
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        let base = h.base_ptr() as usize;
        let cell_idx = (p.as_ptr() as usize - base) / 8;
        let page = cell_idx / PAGE_SIZE_CELLS;
        assert_eq!(h.desc(page).words_used, 2);

        for _ in 0..9 {
            h.try_alloc_cons_in(Generation::G0).unwrap();
        }
        assert_eq!(h.desc(page).words_used, 20, "10 conses = 20 cells");
    }

    #[test]
    fn recycle_page_clears_start_bits() {
        // Regression test for bug #1 from the code review: after a
        // page is freed and re-acquired, the global start_bits
        // bitmap must NOT carry the prior tenant's `11`/`01`
        // patterns. Before the fix, evacuation would have shipped
        // pages back to the Free pool with stale bits set; the
        // conservative pinner would then accept bogus pointers
        // that happened to land on them.
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        let base = h.base_ptr() as usize;
        let cell_idx = (p.as_ptr() as usize - base) / 8;
        let page_idx = cell_idx / PAGE_SIZE_CELLS;
        assert!(
            is_start_at(h.start_bits_slice(), cell_idx),
            "fresh cons must have its start bit set"
        );
        // Manually release the page (simulates sub-phase 7 freeing
        // an evacuated page back to the Free pool).
        h.desc_mut(page_idx).release();
        // Reset the corresponding alloc region so the next acquire
        // doesn't try to re-use the now-Free page through the
        // cached current_page pointer.
        {
            let r = h.alloc_region_mut(Generation::G0, PageKind::Cons);
            *r = AllocRegion::empty(Generation::G0, PageKind::Cons);
        }
        // Re-acquire (should be the same page since it's the first
        // free one in our tiny 8-page heap).
        let new_idx = h
            .acquire_free_page(Generation::G0, PageKind::Cons)
            .expect("re-acquire");
        assert_eq!(new_idx, page_idx, "should recycle the just-freed page");
        // EVERY cell of the page must now have a clear start bit —
        // the stale `11` from the previous tenant is gone.
        let first_cell = page_idx * PAGE_SIZE_CELLS;
        for c in first_cell..first_cell + PAGE_SIZE_CELLS {
            assert!(
                !is_start_at(h.start_bits_slice(), c),
                "cell {c} still has a stale start bit after recycle"
            );
        }
    }
}
