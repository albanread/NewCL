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

use crate::traits::HeapLayout;
use std::ptr;
use std::sync::OnceLock;

use crate::heap_common::HeapHeader;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GcStallReason {
    MidEvacOOM,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GcStallError {
    pub reason: GcStallReason,
    pub from_gen: Generation,
    pub dest_gen: Generation,
    pub attempted_cells: usize,
    pub attempted_kind: PageKind,
    pub free_pages: usize,
    pub g0_pages: usize,
    pub g1_pages: usize,
    pub tenured_pages: usize,
    pub pinned_pages: usize,
    pub pin_set_size: usize,
    pub objects_copied_before_failure: usize,
    pub cells_copied_before_failure: usize,
    pub reserve_pages: usize,
    pub mark_live_bytes: usize,
    pub mark_live_pages: usize,
    pub zero_live_pages_released: usize,
    pub pages_recycled_mid_evac: usize,
}

impl GcStallError {
    fn mid_evac_oom<L: HeapLayout>(
        heap: &PageHeap<L>,
        from_gen: Generation,
        dest_gen: Generation,
        attempted_cells: usize,
        attempted_kind: PageKind,
        objects_copied_before_failure: usize,
        cells_copied_before_failure: usize,
        pages_recycled_mid_evac: usize,
    ) -> Self {
        Self {
            reason: GcStallReason::MidEvacOOM,
            from_gen,
            dest_gen,
            attempted_cells,
            attempted_kind,
            free_pages: heap.count_pages_in_gen(Generation::Free),
            g0_pages: heap.count_pages_in_gen(Generation::G0),
            g1_pages: heap.count_pages_in_gen(Generation::G1),
            tenured_pages: heap.count_pages_in_gen(Generation::Tenured),
            pinned_pages: heap.descs().iter().filter(|d| d.has_pins()).count(),
            pin_set_size: heap.pinned_count(),
            objects_copied_before_failure,
            cells_copied_before_failure,
            reserve_pages: heap.gc_free_page_reserve_for_mutator_slab(),
            mark_live_bytes: heap.last_mark_live_bytes(),
            mark_live_pages: heap.last_mark_live_pages(),
            zero_live_pages_released: heap.last_zero_live_pages_released(),
            pages_recycled_mid_evac,
        }
    }

    pub fn render_with_runtime_context(
        &self,
        trigger: &str,
        static_used_bytes: usize,
        static_committed_bytes: usize,
    ) -> String {
        format!(
            "gc-stall: reason={:?} trigger={trigger} from={:?} dest={:?} attempted-kind={:?} attempted-cells={} pages(free/g0/g1/tenured)={}/{}/{}/{} pinned-pages={} pin-set={} reserve-pages={} copied(objects/cells)={}/{} mark(live-bytes/live-pages/zero-live-pages-released)={}/{}/{} recycled-mid-evac={} static(used/committed)={}/{}",
            self.reason,
            self.from_gen,
            self.dest_gen,
            self.attempted_kind,
            self.attempted_cells,
            self.free_pages,
            self.g0_pages,
            self.g1_pages,
            self.tenured_pages,
            self.pinned_pages,
            self.pin_set_size,
            self.reserve_pages,
            self.objects_copied_before_failure,
            self.cells_copied_before_failure,
            self.mark_live_bytes,
            self.mark_live_pages,
            self.zero_live_pages_released,
            self.pages_recycled_mid_evac,
            static_used_bytes,
            static_committed_bytes,
        )
    }
}

/// Recoverable error from a `try_*` GC entry point. Sub-phase 10
/// follow-up: the existing `collect_minor` / `collect_major` panic
/// via `panic_any(GcStallError)` when allocation fails mid-
/// evacuation. The `try_*` variants below catch that panic and
/// surface it here so clients can decide what to do (drop the heap,
/// grow it, log + abort, etc.) without process termination.
///
/// **Heap state after a `GcError`:** indeterminate (the heap is now
/// "poisoned"). Some objects may have been copied with forwarding
/// markers; the corresponding destination pages may exist; cards
/// may be partially marked. The heap is no longer safe for mutator
/// allocations or further GC cycles — `try_collect_*` on a poisoned
/// heap short-circuits to [`GcError::HeapPoisoned`] without
/// attempting another collection (a second mid-state collect would
/// compound the corruption rather than recover). The safe response
/// is to drop the heap and either abort or rebuild from durable
/// state. `Drop` itself is always safe to run.
#[derive(Clone, Debug)]
pub enum GcError {
    /// Mid-evacuation out of memory. Contains the underlying
    /// `GcStallError` diagnostic.
    MidEvacOom(GcStallError),
    /// Heap was already poisoned by an earlier failed `try_collect_*`
    /// — no fresh diagnostic is generated because no collection ran.
    /// Drop the heap.
    HeapPoisoned,
}

impl GcError {
    pub fn render(&self) -> String {
        match self {
            GcError::MidEvacOom(stall) => stall.render_with_runtime_context(
                "try_collect",
                0,
                0,
            ),
            GcError::HeapPoisoned => {
                "gc-stall: reason=HeapPoisoned trigger=try_collect (heap \
                 was poisoned by an earlier failed try_collect; drop the \
                 heap and rebuild)"
                    .to_string()
            }
        }
    }
}

impl std::fmt::Display for GcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render())
    }
}

impl std::error::Error for GcError {}

pub fn install_quiet_gc_stall_panic_hook() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info.payload().downcast_ref::<GcStallError>().is_some() {
                return;
            }
            prev(info);
        }));
    });
}

/// Mode flag controlling how [`PageEvacuator::visit`] interprets a
/// slot during the chunked two-phase mark-evacuate-rewrite cycle.
/// See [`PageHeap::evacuate_with_roots`] for the driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvacMode {
    /// Internal mark pass (test path only): visit sets the mark
    /// bit at the target's start cell and queues for recursive
    /// payload traversal. Mirrors `page_heap::mark::PageMarker`.
    Mark,
    /// Phase 2 rewrite (every chunk): visit reads the source cell
    /// at the target address; if it holds a `Word::forward`,
    /// rewrite the slot to point at the new location.
    Rewrite,
}

/// Scanner handed to the caller's root-walking closure. The mode
/// flag determines what `visit` does, but the call shape stays
/// `evac.visit(slot)` either way — so neither the mutator-side
/// closures nor the in-heap card scan need to know which phase
/// they're feeding.
pub struct PageEvacuator<'a, L: HeapLayout> {
    heap: &'a mut PageHeap<L>,
    from_gen: Generation,
    dest_gen: Generation,
    mode: EvacMode,
    /// Cells queued for recursive payload mark traversal in `Mark`
    /// mode. Empty in `Rewrite` mode.
    mark_queue: Vec<usize>,
}

impl<'a, L: HeapLayout> PageEvacuator<'a, L> {
    /// Generation being evacuated from. Used by the dirty-card scan
    /// to filter pages strictly older than this — the card barrier
    /// only finds cross-gen pointers, not intra-gen.
    pub(super) fn from_gen(&self) -> Generation {
        self.from_gen
    }

    /// Visit a root slot. Behavior depends on the mode: `Mark`
    /// marks the reachable target's start cell and queues for
    /// recursive payload mark; `Rewrite` consults the source cell
    /// at the target address and rewrites the slot if a forwarding
    /// marker is present.
    pub fn visit(&mut self, slot: &mut Word) {
        match self.mode {
            EvacMode::Mark => self.mark_visit_slot(slot),
            EvacMode::Rewrite => {
                if let Some(new) = self.maybe_rewrite(*slot) {
                    *slot = new;
                }
            }
        }
    }

    /// Same as [`Self::visit`], but for a raw cell address. Used
    /// by the dirty-card scanner in
    /// `coordinator_api::collect_minor_with_static` to scan
    /// external regions (the static area, older-generation pages)
    /// for cross-gen pointers into `from_gen`.
    ///
    /// SAFETY: caller asserts `cell_ptr` is a valid `*mut u64`
    /// inside the page heap's reservation OR an externally-supplied
    /// region and points at an 8-byte-aligned cell. The cell content
    /// is read as a `Word`. In `Mark` mode the slot is not written;
    /// in `Rewrite` mode the cell is updated in place if it holds a
    /// pointer whose source has a forwarding marker.
    pub unsafe fn visit_cell(&mut self, cell_ptr: *mut u64) {
        let raw = unsafe { *cell_ptr };
        let w = Word::from_raw(raw);
        match self.mode {
            EvacMode::Mark => {
                let mut tmp = w;
                self.mark_visit_slot(&mut tmp);
                // Mark never writes through to the slot.
            }
            EvacMode::Rewrite => {
                if let Some(new) = self.maybe_rewrite(w) {
                    unsafe { *cell_ptr = new.raw() };
                }
            }
        }
    }

    /// `Mark`-mode body. Mirrors `mark::PageHeap::try_mark_root`:
    /// same gates (tag, page lookup, generation, kind, start-bit,
    /// tag-vs-start consistency), same effect (sets a mark bit at
    /// the target's start and queues for payload scan).
    fn mark_visit_slot(&mut self, slot: &mut Word) {
        use crate::traits::{PointerKind, WordKind};
        let w = *slot;
        let (target_addr, ptr_kind) = match L::classify(w.raw()) {
            WordKind::PointerCons(a) => (a, PointerKind::Cons),
            WordKind::PointerHeader(a) => (a, PointerKind::Header),
            _ => return,
        };
        let page_idx = match self.heap.page_of(target_addr) {
            Some(p) => p,
            None => return,
        };
        if self.heap.desc(page_idx).generation != self.from_gen {
            return;
        }
        let kind = self.heap.desc(page_idx).kind;

        // Large objects are never evacuated — they stay at their original
        // address. Pin all pages in the run so phase3_reclaim flips their
        // generation in-place. Record the head cell in pinned_cells so
        // phase2_rewrite can walk the large object's payload and fix up
        // any pointers to evacuated small objects. Also mark the head cell
        // and queue it for payload BFS — without this, small cells reached
        // only via a Large object's heap-pointer payload would be missed
        // by the mark pass and reclaimed.
        if kind == PageKind::Large {
            // Find the head page (a slot should always point to a head cell,
            // but scan backwards defensively for a cont page).
            let head_page_idx = if self.heap.desc(page_idx).is_large_head() {
                page_idx
            } else {
                let mut h = page_idx;
                while h > 0 && self.heap.desc(h).is_large_cont() {
                    h -= 1;
                }
                h
            };
            let head_cell = head_page_idx * PAGE_SIZE_CELLS;
            // Mark first; if we've already seen this Large head via another
            // root, the pin + queue work below was done on the earlier visit.
            if self.heap.mark_cell(head_cell) {
                return;
            }
            let n_span = self.heap.desc(head_page_idx).n_span as usize;
            // Pin every page in the run so phase3_reclaim flips them all.
            for i in 0..n_span {
                let pidx = head_page_idx + i;
                self.heap.desc_mut(pidx).set_pin(0);
            }
            // Record only the head cell in pinned_cells.
            // rewrite_pinned_object will walk the header to find the full
            // payload extent (which spans all pages in the run).
            self.heap.pinned_cells.insert(head_cell);
            // Queue for payload BFS so heap-pointer fields inside the Large
            // object reach `mark_scan_object` and their children get marked.
            self.mark_queue.push(head_cell);
            // Leave the slot unchanged — the large object didn't move.
            return;
        }

        if !matches!(kind, PageKind::Cons | PageKind::Boxed) {
            return;
        }
        let cell_idx =
            (target_addr as usize - self.heap.base_ptr() as usize) / 8;
        if !is_start_at(self.heap.start_bits_slice(), cell_idx) {
            return;
        }
        let is_cons_start = super::alloc::is_cons_start_at(
            self.heap.start_bits_slice(),
            cell_idx,
        );
        match (ptr_kind, is_cons_start) {
            (PointerKind::Cons, true) | (PointerKind::Header, false) => {}
            _ => return,
        }
        if self.heap.mark_cell(cell_idx) {
            return;
        }
        self.mark_queue.push(cell_idx);
    }

    /// Drain the mark queue, walking each marked object's payload
    /// and recursively marking heap-pointer children.
    fn mark_drain(&mut self) {
        while let Some(cell_idx) = self.mark_queue.pop() {
            self.mark_scan_object(cell_idx);
        }
    }

    /// Walk the payload cells of a marked object and call
    /// `mark_visit_slot` on each. Same dispatch as
    /// `mark::PageHeap::scan_marked_object`.
    fn mark_scan_object(&mut self, cell_idx: usize) {
        let page_idx = cell_idx / PAGE_SIZE_CELLS;
        let kind = self.heap.desc(page_idx).kind;
        let is_cons = match kind {
            PageKind::Cons => true,
            PageKind::Boxed => super::alloc::is_cons_start_at(
                self.heap.start_bits_slice(),
                cell_idx,
            ),
            // A Large object's head cell carries a boxed-style HeapHeader;
            // its payload may span multiple contiguous pages in the
            // reservation but indexes the same global cell space. Decoding
            // via `L::header_layout` gives the pointer-cell range across
            // the whole run.
            PageKind::Large => false,
            PageKind::Free => return,
        };
        let (payload_start, payload_end_inclusive) = if is_cons {
            (cell_idx, cell_idx + 1)
        } else {
            let header_ptr =
                unsafe { (self.heap.base_ptr() as *const u64).add(cell_idx) };
            let layout = unsafe { L::header_layout(header_ptr) };
            if layout.pointer_cell_count() == 0 {
                return;
            }
            (
                cell_idx + layout.pointer_cells_start,
                cell_idx + layout.pointer_cells_end - 1,
            )
        };
        for c in payload_start..=payload_end_inclusive {
            let w = Word::from_raw(self.read_cell(c));
            let mut tmp = w;
            self.mark_visit_slot(&mut tmp);
        }
    }

    /// `Rewrite`-mode body. Returns the post-copy Word for a slot
    /// whose target has a forwarding marker; otherwise `None`
    /// (slot left untouched).
    ///
    /// Does NOT gate on `page.gen == from_gen`: a page that was a
    /// from_gen source can have been flipped to `dest_gen` already
    /// (for pinned pages, end of an earlier chunk's Phase 3), with
    /// forward markers still sitting in its non-pinned cells. Words
    /// elsewhere pointing at those source cells must still follow
    /// the forward. `is_real_forward_target` validates the encoded
    /// target lives in the reservation, which is the safety net.
    fn maybe_rewrite(&self, w: Word) -> Option<Word> {
        use crate::traits::WordKind;
        // Only heap pointers can be rewritten. Use classify to filter
        // immediates and forwarding markers consistently with the
        // layout trait.
        let target_addr = match L::classify(w.raw()) {
            WordKind::PointerCons(a) | WordKind::PointerHeader(a) => a,
            _ => return None,
        };
        let page_idx = self.heap.page_of(target_addr)?;
        if self.heap.desc(page_idx).generation == Generation::Free {
            return None;
        }
        let cell_idx =
            (target_addr as usize - self.heap.base_ptr() as usize) / 8;
        let src_raw = self.read_cell(cell_idx);
        is_real_forward_target_at::<L>(self.heap, cell_idx, src_raw).map(|new_addr| {
            // Preserve the original pointer tag (Cons / Symbol /
            // Vector / etc.) by rewriting only the address part.
            Word::from_raw(L::rewrite_pointer_addr(w.raw(), new_addr as *const u8))
        })
    }

    /// Read a raw u64 from a global cell index. Bounds-checked in
    /// debug.
    fn read_cell(&self, cell_idx: usize) -> u64 {
        debug_assert!(cell_idx < self.heap.total_cells());
        let p =
            unsafe { (self.heap.base_ptr() as *const u64).add(cell_idx) };
        unsafe { *p }
    }

    /// Reservation base as a `*mut u64`, so a caller (the dirty-card
    /// scanner) can tell whether the region it's scanning is the heap
    /// reservation (object-aware scan available) or an external area
    /// like the static segment (cell-by-cell only).
    pub fn reservation_base(&self) -> *mut u64 {
        self.heap.base_ptr() as *mut u64
    }

    /// **Object-aware** dirty-card scan over reservation cells
    /// `[start, end)`: walk live objects via start bits + layout and
    /// offer only each object's *pointer* cells to `visit_cell`.
    ///
    /// This replaces a naive cell-by-cell scan, which treated every cell
    /// in a dirty card as a candidate pointer. That was unsound for
    /// objects with **opaque byte payloads** (`<byte-string>`): arbitrary
    /// bytes can alias an in-reservation pointer to a real object start,
    /// so the mark phase would resurrect dead objects and the rewrite
    /// phase would *overwrite the opaque payload* with a relocated
    /// address (GAP-010). Tagged-Word payloads (cons / vector) can't
    /// alias — a fixnum's low bit is 0 — which is why this only ever bit
    /// byte-strings, and why no test caught it until one was card-scanned.
    ///
    /// SAFETY: `start`/`end` are global cell indices within the
    /// reservation; the card table guarantees `end <= total_cells`.
    pub unsafe fn visit_card_pointer_cells(&mut self, start: usize, end: usize) {
        let base = self.heap.base_ptr() as *mut u64;
        // Position at the object covering `start` (may begin earlier), or
        // the first object start within the card if `start` is in a tail.
        let mut s = match object_start_at_or_before(self.heap, start) {
            Some(s) => s,
            None => {
                let mut t = start;
                while t < end
                    && !is_start_at(self.heap.start_bits_slice(), t)
                {
                    t += 1;
                }
                t
            }
        };
        while s < end {
            if !is_start_at(self.heap.start_bits_slice(), s) {
                // Unused tail between the last object and the card end:
                // bump-allocated pages have no inter-object gaps, so this
                // means there are no further objects in this card span.
                break;
            }
            let (size, p_start, p_end) = object_pointer_cells(self.heap, s);
            for off in p_start..p_end {
                // SAFETY: a pointer cell of a live object; in-reservation,
                // u64-aligned.
                unsafe { self.visit_cell(base.add(s + off)) };
            }
            if size == 0 {
                break;
            }
            s += size;
        }
    }
}

/// `(total_cells, pointer_cells_start, pointer_cells_end)` for the object
/// starting at `cell_idx`, as cell offsets from the start. Cons = 2 cells,
/// both pointers; boxed/large = `header_layout`. An opaque payload (e.g. a
/// `<byte-string>`) reports an empty pointer range, so a card scan visits
/// none of its bytes. Shared by the evacuator's and the marker's
/// object-aware card scans (free fn so both `&PageHeap` callers reach it).
pub(super) fn object_pointer_cells<L: HeapLayout>(
    heap: &PageHeap<L>,
    cell_idx: usize,
) -> (usize, usize, usize) {
    let page = cell_idx / PAGE_SIZE_CELLS;
    let kind = heap.desc(page).kind;
    let is_cons = match kind {
        PageKind::Cons => true,
        PageKind::Boxed => {
            super::alloc::is_cons_start_at(heap.start_bits_slice(), cell_idx)
        }
        PageKind::Large => false,
        PageKind::Free => return (1, 0, 0),
    };
    if is_cons {
        (2, 0, 2)
    } else {
        let header_ptr = unsafe { (heap.base_ptr() as *const u64).add(cell_idx) };
        let layout = unsafe { L::header_layout(header_ptr) };
        (layout.total_cells, layout.pointer_cells_start, layout.pointer_cells_end)
    }
}

/// Start cell of the object whose extent covers `cell` (it may begin
/// earlier in the page, or — for a Large run — on an earlier page), or
/// `None` if `cell` precedes the first object on its page / sits in an
/// unused tail with no start at or below it. Shared by both card scans.
pub(super) fn object_start_at_or_before<L: HeapLayout>(
    heap: &PageHeap<L>,
    cell: usize,
) -> Option<usize> {
    let page = cell / PAGE_SIZE_CELLS;
    let kind = heap.desc(page).kind;
    if kind == PageKind::Large {
        // Resolve to the run head (possibly on an earlier page).
        let mut h = page;
        if !heap.desc(h).is_large_head() {
            while h > 0 && heap.desc(h).is_large_cont() {
                h -= 1;
            }
        }
        return Some(h * PAGE_SIZE_CELLS);
    }
    if matches!(kind, PageKind::Free) {
        return None;
    }
    let page_first = page * PAGE_SIZE_CELLS;
    let mut s = cell;
    loop {
        if is_start_at(heap.start_bits_slice(), s) {
            return Some(s);
        }
        if s == page_first {
            return None;
        }
        s -= 1;
    }
}

impl<L: HeapLayout> PageHeap<L> {
    /// Evacuate every reachable object in `from_gen` into pages
    /// belonging to `dest_gen`. Pass `from_gen == dest_gen` for an
    /// in-place mark-evacuate cycle; pass `dest_gen =
    /// from_gen.promoted()` to promote.
    ///
    /// ## Algorithm — block-incremental two-phase mark-evacuate-rewrite
    ///
    /// 1. (Optional) **Internal mark pass**. If the caller hasn't
    ///    run `mark_minor_with_static` first, drive a mark pass via
    ///    `visit_roots` in `Mark` mode. Production path skips this
    ///    (the coordinator runs mark first); test path uses it.
    /// 2. **Pre-chunk release**. Snapshot `from_pages` and release
    ///    every zero-mark unpinned page straight to Free, growing
    ///    the dest budget for the first chunk.
    /// 3. **Chunked loop**, iterating until every `from_page` is
    ///    processed. Each chunk's size is bounded by current Free
    ///    so Phase 1 can't run out of destination pages:
    ///    - **Phase 1 (Copy)**: iterate marked starts on the
    ///      chunk's source pages; copy each to `dest_gen` and
    ///      write `Word::forward` at the source cell. Pinned cells
    ///      are skipped (their pages will flip in Phase 3).
    ///    - **Phase 2 (Rewrite)**: invoke `visit_roots` with the
    ///      evacuator in `Rewrite` mode (rewrites mutator-root
    ///      slots + dirty-card cells via the closure), then walk
    ///      every live page in `from_gen` / `dest_gen` and rewrite
    ///      payload Words whose targets carry a forwarding marker.
    ///    - **Phase 3 (Reclaim)**: walk the chunk's source pages.
    ///      Pages with pins flip to `dest_gen` in place, preserving
    ///      pinned objects; pages without pins release to Free,
    ///      growing the budget for the next chunk.
    /// 4. **Cleanup**: clear pin set, mark bits, recycle-live-counts.
    ///
    /// ## Pre-conditions
    ///
    /// - The caller has stopped the world.
    /// - `from_gen` and `dest_gen` are valid generations
    ///   (`G0 / G1 / Tenured`); `Free` is invalid for either.
    ///
    /// ## Post-conditions
    ///
    /// - Every reachable-from-roots object in `from_gen` now lives
    ///   in `dest_gen` (with its in-heap references rewritten),
    ///   except pinned objects which kept their original addresses.
    /// - Pinned-page `from_gen` pages have flipped to `dest_gen`;
    ///   their start bits are cleared except for the pinned starts.
    /// - Unpinned, fully-evacuated pages are back on the free list.
    /// - `from_gen`'s alloc regions have been reset.
    /// - The pin set, mark bits, and recycle-live-counts are cleared.
    pub fn evacuate_with_roots<F>(
        &mut self,
        from_gen: Generation,
        dest_gen: Generation,
        mut visit_roots: F,
    ) -> EvacResult
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        assert!(
            !matches!(from_gen, Generation::Free),
            "evacuate: from_gen must not be Free"
        );
        assert!(
            !matches!(dest_gen, Generation::Free),
            "evacuate: dest_gen must not be Free"
        );

        // Step 1: ensure marks are populated. The production path
        // runs `mark_minor_with_static` and seeds
        // `recycle_live_counts` first; tests typically don't, so
        // we drive an internal mark via the caller's closure.
        let need_internal_mark =
            !self.recycle_live_counts_active_for(from_gen);
        if need_internal_mark {
            self.clear_mark_bits_in_gen(from_gen);
            let mut marker = PageEvacuator {
                heap: self,
                from_gen,
                dest_gen,
                mode: EvacMode::Mark,
                mark_queue: Vec::new(),
            };
            visit_roots(&mut marker);
            marker.mark_drain();
            drop(marker);
            self.prepare_recycle_live_counts_from_marks(from_gen);
        }

        // Reconcile the pin set (fold durable FFI pins) and extend the
        // mark from EVERY pinned object — conservative pins from
        // `pin_pointers_in_ranges` as well as explicit ones — so a pinned
        // object's transitive children survive instead of being reclaimed
        // out from under its pointers. Must run after the mark /
        // recycle-count seed is established and before the pinned-cells
        // snapshot below. The logical-cycle driver (cycle.rs) calls
        // `clear_all_pins` after all its sub-evacs complete, so this
        // re-applies the durable explicit pins every evacuation.
        self.apply_pins_and_extend_mark(from_gen);

        // Snapshot the pinned cells with their is_cons bit BEFORE
        // any start-bit clearing happens. Each chunk's Phase 3 uses
        // this to restore start bits on flipped pages.
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

        // Snapshot from_pages. Phase 3 of each chunk releases the
        // chunk's zero-mark unpinned pages and counts them in
        // `pages_freed`; we don't pre-release here so the EvacResult
        // tally captures every reclaim.
        let from_pages: Vec<usize> =
            self.pages_in_gen(from_gen).collect();

        // Reset from_gen alloc regions. Any prior `current_page`
        // may be a page that got pre-released or will be released
        // by a chunk; future allocs into from_gen re-acquire from
        // the free list.
        for kind in [PageKind::Cons, PageKind::Boxed] {
            let r = self.alloc_region_mut(from_gen, kind);
            *r = super::alloc::AllocRegion::empty(from_gen, kind);
        }

        let mut total_objects_copied = 0usize;
        let mut total_cells_copied = 0usize;
        let mut total_pages_freed = 0usize;
        let mut total_pages_flipped = 0usize;

        // Distinct from_gen pages that carry a pin. Pinned pages flip in
        // place in Phase 3 (they keep their pinned objects) instead of
        // returning to Free, so they NEVER replenish the chunk budget.
        // Excluding them from the budget keeps a chunk from over-committing
        // dest demand against pages that won't free — a false mid-evac OOM
        // on a heap with enough Free overall but badly distributed. This is
        // a no-op for precise-only clients (the pin set is empty), and only
        // ever shrinks a chunk, so it can never make evacuation *less* safe.
        let pinned_pages: Vec<usize> = {
            let mut v: Vec<usize> =
                self.pinned_cells.iter().map(|&c| c / PAGE_SIZE_CELLS).collect();
            v.sort_unstable();
            v.dedup();
            v
        };

        // Step 3: chunked loop.
        let mut idx = 0;
        while idx < from_pages.len() {
            let avail_free = self.count_pages_in_gen(Generation::Free);
            // Pages still pinned in from_gen (already-flipped ones have left
            // from_gen, so they drop out naturally as the loop progresses).
            let pinned_pending = pinned_pages
                .iter()
                .filter(|&&p| self.desc(p).generation == from_gen)
                .count();
            // Pick chunk_size at 7/8 of the *replenishable* free budget. The
            // 1/8 margin absorbs two sources of dest-demand slop:
            //   - per-page density variance (older source pages
            //     have more dead cells than newer ones; the
            //     "1 source → 1 dest" worst case is conservative on
            //     average but can be exceeded on dense tails),
            //   - dest allocator fragmentation when a boxed object
            //     can't fit in the current dest page's tail.
            // Floor at 1 to guarantee progress; cap at remaining.
            let effective_free = avail_free.saturating_sub(pinned_pending);
            let chunk_size = ((effective_free * 7) / 8)
                .max(1)
                .min(from_pages.len() - idx);
            let chunk_pages: Vec<usize> =
                from_pages[idx..idx + chunk_size].to_vec();

            let (chunk_objects, chunk_cells) = self.phase1_copy_chunk(
                from_gen,
                dest_gen,
                &chunk_pages,
                total_objects_copied,
                total_cells_copied,
                total_pages_freed,
            );
            total_objects_copied += chunk_objects;
            total_cells_copied += chunk_cells;

            self.phase2_rewrite(from_gen, dest_gen, &mut visit_roots);

            let (released, flipped) = self.phase3_reclaim(
                dest_gen,
                &chunk_pages,
                &pinned_with_kind,
            );
            total_pages_freed += released;
            total_pages_flipped += flipped;

            idx += chunk_size;
        }

        // DEBUG: every pinned object must survive this cycle in place,
        // with its start bit intact. If a pinned cell lost its start bit
        // it was zeroed/released by Phase 3 — the smoking gun for the
        // pinned-object-destroyed corruption.
        #[cfg(debug_assertions)]
        {
            let pc: Vec<usize> = self.pinned_cells.iter().copied().collect();
            for cell_idx in pc {
                debug_assert!(
                    is_start_at(self.start_bits_slice(), cell_idx),
                    "pinned cell {cell_idx} (page {}) lost its start bit during evac \
                     from_gen={from_gen:?} dest_gen={dest_gen:?}",
                    cell_idx / PAGE_SIZE_CELLS
                );
            }
        }

        // Step 4: end-of-cycle cleanup.
        //
        // NB: `clear_all_pins` is NOT called here. A logical GC cycle (one
        // `collect_minor` / `collect_major` / `collect_full` invocation)
        // may run multiple `evacuate_with_roots` passes back-to-back —
        // notably a `collect_minor` that cascades G0→G1 then G1→Tenured.
        // The conservative pin scan runs ONCE before the logical cycle
        // starts and seeds pins for ALL generations the cycle may touch
        // (e.g. drive_collect pins both G0 and G1 before calling
        // `collect_minor`). Clearing pins between the G0→G1 and
        // G1→Tenured passes would wipe the G1 pins, leaving live G1
        // objects unprotected during the cascade — they'd be reclaimed
        // because no precise root reached them either. (This was the
        // cons-elision / random-data-loss bug: NCL's stack-resident G1
        // pointers got their pages released by the cascade.)
        //
        // The logical-cycle entry points (`cycle.rs` `collect_minor`,
        // `collect_major`, `collect_full`) are responsible for calling
        // `clear_all_pins` AFTER all their sub-evacs complete.
        self.clear_mark_bits_in_gen(from_gen);
        self.clear_recycle_live_counts();

        EvacResult {
            objects_copied: total_objects_copied,
            cells_copied: total_cells_copied,
            pages_freed: total_pages_freed,
            pages_flipped: total_pages_flipped,
        }
    }

    /// Phase 1: iterate marked starts on each of `chunk`'s source
    /// pages and copy them to `dest_gen`, writing in-heap forwarding
    /// markers at the source. Pinned cells are skipped (they stay
    /// in place; their pages will flip in Phase 3).
    ///
    /// Returns `(objects_copied, cells_copied)` for this chunk.
    /// The carry-in tallies feed `GcStallError` on dest exhaustion.
    fn phase1_copy_chunk(
        &mut self,
        from_gen: Generation,
        dest_gen: Generation,
        chunk: &[usize],
        total_objects_so_far: usize,
        total_cells_so_far: usize,
        total_pages_freed_so_far: usize,
    ) -> (usize, usize) {
        let mut objs = 0usize;
        let mut cells = 0usize;
        for &page_idx in chunk {
            // A page may have been pre-released for zero-mark or
            // flipped during an earlier chunk; filter by current
            // generation each iteration.
            if self.desc(page_idx).generation != from_gen {
                continue;
            }
            // Large pages are never evacuated (their objects stay in place
            // and are handled by phase3_reclaim). Skip them here.
            if self.desc(page_idx).kind == PageKind::Large {
                continue;
            }
            let first_cell = page_idx * PAGE_SIZE_CELLS;
            let last_cell =
                first_cell + self.desc(page_idx).words_used as usize;
            let mut cell_idx = first_cell;
            while cell_idx < last_cell {
                if !self.is_marked(cell_idx) {
                    cell_idx += 1;
                    continue;
                }
                if !is_start_at(self.start_bits_slice(), cell_idx) {
                    cell_idx += 1;
                    continue;
                }
                let src = read_heap_cell(self, cell_idx);
                if is_real_forward_target_at::<L>(self, cell_idx, src).is_some() {
                    // Defensive — shouldn't fire under mark-driven
                    // iteration, but harmless. `is_real_forward_target`
                    // validates the encoded target sits inside the
                    // reservation, ruling out a header bit-pattern
                    // that aliases the forwarding tag.
                    cell_idx += 1;
                    continue;
                }
                let is_cons = super::alloc::is_cons_start_at(
                    self.start_bits_slice(),
                    cell_idx,
                );
                let size = if is_cons {
                    2
                } else {
                    let header_ptr = unsafe {
                        (self.base_ptr() as *const u64).add(cell_idx)
                    };
                    unsafe { L::header_layout(header_ptr) }.total_cells
                };
                if self.is_pinned_cell(cell_idx) {
                    cell_idx += size;
                    continue;
                }

                let dest_ptr = if is_cons {
                    match self.try_alloc_cons_in(dest_gen) {
                        Some(p) => p,
                        None => std::panic::panic_any(
                            GcStallError::mid_evac_oom(
                                self,
                                from_gen,
                                dest_gen,
                                size,
                                PageKind::Cons,
                                total_objects_so_far + objs,
                                total_cells_so_far + cells,
                                total_pages_freed_so_far,
                            ),
                        ),
                    }
                } else {
                    match self.try_alloc_boxed_in(dest_gen, size) {
                        Some(p) => p,
                        None => std::panic::panic_any(
                            GcStallError::mid_evac_oom(
                                self,
                                from_gen,
                                dest_gen,
                                size,
                                PageKind::Boxed,
                                total_objects_so_far + objs,
                                total_cells_so_far + cells,
                                total_pages_freed_so_far,
                            ),
                        ),
                    }
                };

                let src_ptr = unsafe {
                    (self.base_ptr() as *mut u64).add(cell_idx)
                };
                unsafe {
                    ptr::copy_nonoverlapping(
                        src_ptr,
                        dest_ptr.as_ptr(),
                        size,
                    );
                }
                unsafe {
                    *src_ptr = L::make_forward(dest_ptr.as_ptr() as *const u8);
                }

                // Sub-phase 9: mark every dest card unconditionally
                // (any dest_gen including G0). The mutator's card
                // marks lived on the SOURCE page, which is about to
                // be reclaimed. The dest page must inherit the
                // "may-contain-heap-pointers" status so subsequent
                // GC cycles can find any cross-gen pointers the
                // object carries. For dest_gen=G0 this only matters
                // in MAJOR cycles (where G1→Tenured's card scan
                // walks G0 pages to find G0→G1 cross-gen pointers);
                // minor cycles skip G0 cards via the page filter
                // either way.
                let dest_byte_offset =
                    (dest_ptr.as_ptr() as usize)
                        - (self.base_ptr() as usize);
                let dest_byte_end = dest_byte_offset + size * 8;
                let mut byte = dest_byte_offset;
                while byte < dest_byte_end {
                    self.shared.cards.mark_offset(byte);
                    let next_card_start =
                        (byte / crate::heap_common::CARD_SIZE_BYTES + 1)
                            * crate::heap_common::CARD_SIZE_BYTES;
                    byte = next_card_start;
                }

                objs += 1;
                cells += size;
                cell_idx += size;
            }
        }
        (objs, cells)
    }

    /// Phase 2: rewrite Words that point at forwarding markers.
    ///
    /// Three sources of stale pointers:
    ///   (a) Caller-supplied roots — mutator stacks, static area,
    ///       reservation dirty cards on G1/Tenured. Walked via the
    ///       `visit_roots` closure.
    ///   (b) Newly-copied objects in `dest_gen` — Phase 1 copied
    ///       payload bytes verbatim, so any intra-from-gen Word
    ///       still references the source location; that source now
    ///       has a forward marker that needs to be followed.
    ///   (c) Pinned objects — they stayed at their original address
    ///       and aren't on any dirty-card list, but their payload
    ///       Words may point at evacuated from-gen targets.
    ///
    /// Earlier this code walked EVERY live page in EVERY generation
    /// "over-cautiously" while debugging Life's stale-pointer crash
    /// class. Once the env-arg-rooting fix landed, the over-walk
    /// became pure overhead: scanning 4096 Tenured pages on every
    /// minor for cells that haven't moved since they were promoted.
    /// The dirty-card scan in (a) already covers older-gen → from-gen
    /// references; the per-page sweep added nothing the cards didn't.
    ///
    /// Reduced sweep: walk `dest_gen` pages only, plus the precise
    /// pinned-cells set. Skips the Tenured fleet entirely (typically
    /// 90%+ of the heap on a long-running session) and skips
    /// from-gen pages (forward markers there are the *target* of
    /// rewrites, not the source).
    fn phase2_rewrite<F>(
        &mut self,
        from_gen: Generation,
        dest_gen: Generation,
        visit_roots: &mut F,
    ) where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        // 2a: walk caller-provided roots in Rewrite mode. The
        // production closure also walks static-area and reservation
        // dirty cards via `scan_dirty_cards_as_roots` (which goes
        // through `evac.visit_cell`); both paths see the same mode.
        {
            let mut rewriter = PageEvacuator {
                heap: self,
                from_gen,
                dest_gen,
                mode: EvacMode::Rewrite,
                mark_queue: Vec::new(),
            };
            visit_roots(&mut rewriter);
        }

        // 2b: walk dest-gen pages — these contain the just-copied
        // objects whose payload pointers still reference from-gen
        // (where the source object now has a forward marker).
        //
        // For within-gen evac (G0 → G0) this also catches from-gen
        // pages since they share the generation, which is harmless
        // — `rewrite_page` skips forward-marker headers and only
        // rewrites real Word fields.
        let dest_pages: Vec<usize> = self
            .descs()
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                if d.generation == dest_gen {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        for page_idx in dest_pages {
            self.rewrite_page(from_gen, page_idx);
        }

        // 2c: walk pinned objects' payloads. Pinned objects stay
        // at their original addresses in from-gen until Phase 3
        // flips their page, so they're not in `dest_pages` above.
        // Their fields can still point at from-gen objects that
        // got evacuated, and those references need rewriting before
        // Phase 3 clears the source forward markers' start bits.
        let pinned_cells: Vec<usize> =
            self.pinned_cells.iter().copied().collect();
        for cell_idx in pinned_cells {
            self.rewrite_pinned_object(from_gen, cell_idx);
        }
    }

    /// Rewrite the payload of a single pinned object. Same shape as
    /// `rewrite_page`'s inner loop but with the start cell known
    /// (not discovered by scanning start bits) so we don't depend
    /// on the page's start-bit table being intact.
    fn rewrite_pinned_object(&mut self, from_gen: Generation, cell_idx: usize) {
        let page_idx = cell_idx / PAGE_SIZE_CELLS;
        if self.desc(page_idx).generation == Generation::Free {
            return;
        }
        let header_raw = read_heap_cell(self, cell_idx);
        if is_real_forward_target_at::<L>(self, cell_idx, header_raw).is_some() {
            return;
        }
        let is_cons =
            super::alloc::is_cons_start_at(self.start_bits_slice(), cell_idx);
        let (payload_start, payload_end_inclusive) = if is_cons {
            (cell_idx, cell_idx + 1)
        } else {
            let header_ptr =
                unsafe { (self.base_ptr() as *const u64).add(cell_idx) };
            let layout = unsafe { L::header_layout(header_ptr) };
            if layout.pointer_cell_count() == 0 {
                return;
            }
            (
                cell_idx + layout.pointer_cells_start,
                cell_idx + layout.pointer_cells_end - 1,
            )
        };
        for c in payload_start..=payload_end_inclusive {
            let cell_ptr = unsafe { (self.base_ptr() as *mut u64).add(c) };
            let raw = unsafe { *cell_ptr };
            let w = Word::from_raw(raw);
            if let Some(new) = self.maybe_rewrite_word(from_gen, w) {
                unsafe { *cell_ptr = new.raw() };
            }
        }
    }

    /// Walk one page's objects (via start bits), rewriting payload
    /// Words that point at forwarding markers in `from_gen`.
    fn rewrite_page(&mut self, from_gen: Generation, page_idx: usize) {
        let desc = self.desc(page_idx);
        if desc.generation == Generation::Free {
            return;
        }
        let first_cell = page_idx * PAGE_SIZE_CELLS;
        let last_cell = first_cell + desc.words_used as usize;
        let mut cell_idx = first_cell;
        while cell_idx < last_cell {
            if !is_start_at(self.start_bits_slice(), cell_idx) {
                cell_idx += 1;
                continue;
            }
            let header_raw = read_heap_cell(self, cell_idx);
            // A forwarding marker sits at the start cell of a copied
            // source object. Its start bit is still set (from the
            // original alloc) but there's no payload to walk — the
            // cells past the marker were overwritten by neighbouring
            // copies or are stale. Skip. The reservation check
            // distinguishes a real forward marker from a `Float`
            // HeapHeader (whose TYPE=7 also has low 3 = 0b111).
            if is_real_forward_target_at::<L>(self, cell_idx, header_raw).is_some() {
                cell_idx += 1;
                continue;
            }
            let is_cons = super::alloc::is_cons_start_at(
                self.start_bits_slice(),
                cell_idx,
            );
            let (size, word_range_inclusive) = if is_cons {
                (2usize, Some((cell_idx, cell_idx + 1)))
            } else {
                let header_ptr =
                    unsafe { (self.base_ptr() as *const u64).add(cell_idx) };
                let layout = unsafe { L::header_layout(header_ptr) };
                let range = if layout.pointer_cell_count() == 0 {
                    None
                } else {
                    Some((
                        cell_idx + layout.pointer_cells_start,
                        cell_idx + layout.pointer_cells_end - 1,
                    ))
                };
                (layout.total_cells, range)
            };
            if let Some((payload_start, payload_end_inclusive)) = word_range_inclusive {
                for c in payload_start..=payload_end_inclusive {
                    let cell_ptr =
                        unsafe { (self.base_ptr() as *mut u64).add(c) };
                    let raw = unsafe { *cell_ptr };
                    let w = Word::from_raw(raw);
                    if let Some(new) = self.maybe_rewrite_word(from_gen, w) {
                        unsafe { *cell_ptr = new.raw() };
                    }
                }
            }
            cell_idx += size;
        }
    }

    /// Free-function-style `maybe_rewrite` for use inside
    /// `rewrite_page` where we hold a `&mut self` and don't want to
    /// construct a `PageEvacuator`.
    ///
    /// Unlike the `PageEvacuator::maybe_rewrite` method, this does
    /// NOT gate on `page.gen == from_gen`. After a promotion
    /// (`G0 → G1`), a page that was a from_gen source can end up
    /// flipped to `dest_gen` (if pinned), with forward markers still
    /// in its non-pinned cells. A Word elsewhere pointing at one of
    /// those source cells must still follow the forward — the
    /// generation gate would mis-fire and leave the Word dangling.
    /// `is_real_forward_target` already validates the encoded target
    /// lives in the heap reservation, which is enough.
    fn maybe_rewrite_word(
        &self,
        _from_gen: Generation,
        w: Word,
    ) -> Option<Word> {
        use crate::traits::WordKind;
        let target_addr = match L::classify(w.raw()) {
            WordKind::PointerCons(a) | WordKind::PointerHeader(a) => a,
            _ => return None,
        };
        let page_idx = self.page_of(target_addr)?;
        if self.desc(page_idx).generation == Generation::Free {
            return None;
        }
        let cell_idx =
            (target_addr as usize - self.base_ptr() as usize) / 8;
        let src_raw = read_heap_cell(self, cell_idx);
        is_real_forward_target_at::<L>(self, cell_idx, src_raw).map(|new_addr| {
            Word::from_raw(L::rewrite_pointer_addr(w.raw(), new_addr as *const u8))
        })
    }

    /// Phase 3: reclaim a chunk's source pages. Pages with pins
    /// flip to `dest_gen` in place (preserving pinned objects);
    /// pages without pins release to Free. Forwarding markers on
    /// released pages drop with the page; markers on flipped pages
    /// persist as unreachable cells (their start bits cleared so
    /// future scans don't see them).
    fn phase3_reclaim(
        &mut self,
        dest_gen: Generation,
        chunk: &[usize],
        pinned_with_kind: &[(usize, bool)],
    ) -> (usize, usize) {
        let mut released = 0usize;
        let mut flipped = 0usize;
        for &page_idx in chunk {
            let desc = self.desc(page_idx);
            if desc.generation == Generation::Free {
                // Pre-released for zero-mark — skip here.
                continue;
            }

            // Large continuation pages are handled when their head page is
            // processed. Skip them here so we don't double-process.
            if desc.is_large_cont() {
                continue;
            }

            // Large head pages: handle the whole run together.
            if desc.is_large_head() {
                let n_span = desc.n_span as usize;
                if desc.has_pins() {
                    // Live large object: flip all pages in the run to dest_gen.
                    for i in 0..n_span {
                        let pidx = page_idx + i;
                        {
                            let d = self.desc_mut(pidx);
                            d.generation = dest_gen;
                            d.age = 0;
                            d.pin_byte = 0;
                        }
                        // Continuation pages have no object starts — clear
                        // their start bits. The head page keeps its single
                        // object-start bit (set by try_alloc_large).
                        if i > 0 {
                            clear_page_start_bits(self.start_bits_slice(), pidx);
                        }
                    }
                    flipped += n_span;
                } else {
                    // Dead large object: release all pages in the run.
                    for i in 0..n_span {
                        let pidx = page_idx + i;
                        clear_page_start_bits(self.start_bits_slice(), pidx);
                        zero_whole_page(self, pidx);
                        self.release_page_to_free(pidx, true);
                        if self.is_committed(pidx) {
                            let _ = self.decommit_page(pidx);
                        }
                    }
                    released += n_span;
                }
                continue;
            }

            if desc.has_pins() {
                // FLIP. Collect the pinned objects' byte ranges
                // FIRST (we need to read each boxed header to know
                // its length, before we zero anything around it).
                let mut pinned_ranges: Vec<(usize, usize)> = Vec::new();
                for &(cell_idx, is_cons) in pinned_with_kind {
                    if cell_idx / PAGE_SIZE_CELLS != page_idx {
                        continue;
                    }
                    let size = if is_cons {
                        2
                    } else {
                        let header_ptr = unsafe {
                            (self.base_ptr() as *const u64).add(cell_idx)
                        };
                        unsafe { L::header_layout(header_ptr) }.total_cells
                    };
                    pinned_ranges.push((cell_idx, size));
                }

                let d = self.desc_mut(page_idx);
                d.generation = dest_gen;
                d.age = 0;
                clear_page_start_bits(self.start_bits_slice(), page_idx);
                let bits = self.start_bits_slice();
                for &(cell_idx, is_cons) in pinned_with_kind {
                    if cell_idx / PAGE_SIZE_CELLS != page_idx {
                        continue;
                    }
                    if is_cons {
                        set_cons_start_bit_at(bits, cell_idx);
                    } else {
                        set_start_bit_at(bits, cell_idx);
                    }
                }

                // ZERO every cell on the page that isn't inside a
                // pinned object's byte range. Non-pinned cells held
                // either forward markers (from Phase 1) or original
                // payload bytes of objects that have since been
                // copied to dest. Leaving them readable means a
                // stale Word elsewhere — one Phase 2 didn't reach
                // — can be dereferenced and yield a
                // valid-looking Lisp value (forward marker, cons
                // header, etc.). Zeroing them turns any such
                // dereference into "Fixnum 0", which the JIT
                // can't mistake for a pointer.
                zero_page_outside_ranges(self, page_idx, &pinned_ranges);

                // The flip PROMOTES these pinned objects IN PLACE to
                // `dest_gen`. Phase 1 cards every *copied* object's dest
                // page (above) so a later minor's cross-gen card scan
                // finds the pointers it carries into a younger generation;
                // a flip must do the same, or the remembered set is
                // incomplete for in-place-promoted objects. Concretely: a
                // pinned cons promoted G0→G1→Tenured by successive flips
                // keeps a `cdr` into a still-younger node; without carding
                // it here, a future minor never scans this object, never
                // treats that pointer as a root, and reclaims the younger
                // target — leaving the `cdr` dangling (the cons-elision /
                // interior-node-splice bug). Mirror Phase 1's per-object
                // byte-range card marking.
                for &(pinned_cell, size) in &pinned_ranges {
                    let byte_offset = pinned_cell * 8;
                    let byte_end = byte_offset + size * 8;
                    let mut byte = byte_offset;
                    while byte < byte_end {
                        self.shared.cards.mark_offset(byte);
                        let next_card_start =
                            (byte / crate::heap_common::CARD_SIZE_BYTES + 1)
                                * crate::heap_common::CARD_SIZE_BYTES;
                        byte = next_card_start;
                    }
                }

                flipped += 1;
            } else {
                // DEBUG: a page that holds a pinned cell must take the
                // FLIP path, never RELEASE — releasing zeroes the pinned
                // object. If this fires, `has_pins()` disagrees with the
                // pinned-cell set (the smoking gun for the elision).
                debug_assert!(
                    !pinned_with_kind
                        .iter()
                        .any(|&(c, _)| c / PAGE_SIZE_CELLS == page_idx),
                    "phase3 RELEASING page {page_idx} that holds a pinned cell \
                     (has_pins={})",
                    self.desc(page_idx).has_pins()
                );
                // RELEASE. The page goes to Free; its bytes are
                // useless to anyone post-cycle. Zero them now so a
                // stale Word that points into this page between
                // release and the next `acquire_free_page` reads
                // Fixnum 0 instead of a forward marker (or live-
                // looking bytes from the just-copied object).
                // (`zeroed = true`: the zero two lines down happens
                // inside this same exclusive borrow, before any
                // mutator can re-acquire the page.)
                self.release_page_to_free(page_idx, true);
                clear_page_start_bits(self.start_bits_slice(), page_idx);
                zero_whole_page(self, page_idx);
                released += 1;
            }
        }
        (released, flipped)
    }

    /// Convenience: evacuate using an array of root Words. Each
    /// root is visited via [`PageEvacuator::visit`]; the array is
    /// updated in place. Used primarily by tests; the production
    /// path passes its own `visit_roots` closure.
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
/// Module-level helper: read a raw u64 from a global cell index.
/// Free-function form to avoid clashing with `mark::PageHeap::read_cell`
/// (which is private to that module).
fn read_heap_cell<L: HeapLayout>(heap: &PageHeap<L>, cell_idx: usize) -> u64 {
    debug_assert!(cell_idx < heap.total_cells());
    let p = unsafe { (heap.base_ptr() as *const u64).add(cell_idx) };
    unsafe { *p }
}

/// Zero every cell of page `page_idx`. Used by Phase 3 when a page
/// is released to Free — between release and the next
/// `acquire_free_page`, the cells must not look like Lisp values to
/// any stale Word that points at them.
fn zero_whole_page<L: HeapLayout>(heap: &PageHeap<L>, page_idx: usize) {
    let first_cell = page_idx * PAGE_SIZE_CELLS;
    let base = heap.base_ptr() as *mut u64;
    unsafe {
        core::ptr::write_bytes(base.add(first_cell), 0, PAGE_SIZE_CELLS);
    }
}

/// Zero every cell on page `page_idx` that does NOT lie inside one
/// of the given `(start_cell, size)` ranges. Used by Phase 3 when a
/// page is flipped (gen changed in place because of pins): pinned
/// objects' bytes must be preserved; everything else — including
/// the forward markers Phase 1 wrote at non-pinned object starts —
/// must be erased so the page can't accidentally answer a stale
/// dereference with a Lisp-looking Word.
///
/// The ranges are arbitrary positions on the page; this walks
/// cell-by-cell and skips past any range it lands inside.
fn zero_page_outside_ranges<L: HeapLayout>(
    heap: &PageHeap<L>,
    page_idx: usize,
    pinned_ranges: &[(usize, usize)],
) {
    let first_cell = page_idx * PAGE_SIZE_CELLS;
    let last_cell = first_cell + PAGE_SIZE_CELLS;
    let base = heap.base_ptr() as *mut u64;
    let mut c = first_cell;
    while c < last_cell {
        // If `c` falls inside a pinned-object range, jump past it.
        let mut inside = None;
        for &(start, size) in pinned_ranges {
            if c >= start && c < start + size {
                inside = Some(start + size);
                break;
            }
        }
        match inside {
            Some(skip_to) => {
                c = skip_to;
            }
            None => {
                unsafe { *base.add(c) = 0 };
                c += 1;
            }
        }
    }
}

/// Check whether the cell at `cell_idx` is a real `Word::forward`
/// marker written by Phase 1 in the CURRENT cycle.
///
/// Three gates:
///   1. Cell content must have low 3 bits = `Tag::Forward` (0b111).
///   2. The encoded target must lie inside the heap reservation.
///   3. The cell's start bit must still be set.
///
/// Why each matters:
///   - Gate 1 alone matches `Word::from_raw(...).is_forward()`, but
///     a `HeapHeader` for `HeapType::Float` (TYPE=7=0b111) looks
///     identical and would otherwise be followed as a forward.
///   - Gate 2 rejects Float headers — their decoded "target" is the
///     `length / gc_bits` field, typically under a few hundred,
///     never a heap address.
///   - Gate 3 distinguishes a CURRENT-cycle forward marker from a
///     STALE one. Phase 1 writes the marker at an object's start
///     cell (which has its start bit set). Phase 3 clears start
///     bits for non-pinned cells on flipped pages and on released
///     pages. So a stale marker from a prior cycle, lingering on a
///     flipped page that survived to a later cycle, has had its
///     start bit cleared — gate 3 rejects it. Without this check,
///     `maybe_rewrite_word` would follow stale markers and rewrite
///     references to invalid addresses (the source of the
///     `<Cons:0x...>` + `<forward:0x...>` mutator crashes in
///     `demos/life.lisp`).
fn is_real_forward_target_at<L: HeapLayout>(
    heap: &PageHeap<L>,
    cell_idx: usize,
    raw: u64,
) -> Option<*const ()> {
    use crate::traits::WordKind;
    let target = match L::classify(raw) {
        WordKind::Forwarded(a) => a,
        _ => return None,
    };
    if !is_start_at(heap.start_bits_slice(), cell_idx) {
        return None;
    }
    if heap.page_of(target).is_none() {
        return None;
    }
    Some(target as *const ())
}

/// Variant that doesn't take a cell index (skips gate 3). Used in
/// defensive paths where the caller doesn't have a cell index yet.
#[allow(dead_code)]
fn is_real_forward_target<L: HeapLayout>(heap: &PageHeap<L>, raw: u64) -> Option<*const ()> {
    use crate::traits::WordKind;
    let target = match L::classify(raw) {
        WordKind::Forwarded(a) => a,
        _ => return None,
    };
    if heap.page_of(target).is_none() {
        return None;
    }
    Some(target as *const ())
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
    use crate::heap_common::{HeapHeader, HeapType};
    use crate::word::{Tag, Word};

    fn small_heap() -> PageHeap<crate::lisp_layout::LispLayout> {
        // 8 pages = 512 KB. Plenty for several thousand cons cells
        // and a few pages each in G0 and G1.
        PageHeap::<crate::lisp_layout::LispLayout>::with_reservation(8 * 64 * 1024)
    }

    /// Allocate a chain of `n` cons cells, each pointing back to
    /// the previous via cdr (head is the last alloc).
    fn alloc_cons_chain(h: &mut PageHeap<crate::lisp_layout::LispLayout>, g: Generation, n: usize) -> Vec<Word> {
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
    fn gen_counts(h: &PageHeap<crate::lisp_layout::LispLayout>) -> (usize, usize, usize, usize) {
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

    #[cfg(feature = "conservative-pin")]

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

    #[cfg(feature = "conservative-pin")]

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

    #[cfg(feature = "conservative-pin")]

    #[test]

    fn pins_and_mark_bits_are_cleared_after_cycle() {
        // After a complete GC cycle (mark + pin + evacuate +
        // end-of-cycle cleanup), pinned_cells must be empty and the
        // from-gen's mark bits cleared so the next cycle starts
        // clean.
        //
        // `evacuate_with_roots` itself clears mark bits. Pin cleanup
        // is a *logical-cycle* responsibility (cycle.rs's
        // `collect_minor`/`collect_major`/`collect_full`), because a
        // single cycle may run multiple sub-evacs (e.g. cascade
        // G0→G1 then G1→Tenured) that all share the conservative
        // pin set from one pre-cycle scan. Mimic the cycle-level
        // cleanup here with an explicit `clear_all_pins` so this
        // test still verifies the end-of-cycle invariant.
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
        // Cycle-level cleanup — mirrors what `cycle.rs::collect_*`
        // does at the end of its sub-evac sequence.
        h.clear_all_pins();

        assert_eq!(h.pinned_count(), 0, "pins cleared post-cycle");
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
    fn boxed_evacuation_can_use_pages_reserved_from_mutator_slabs() {
        // Regression: slab growth must stop before it consumes
        // every Free page, otherwise within-gen evacuation has
        // nowhere to land a boxed survivor.
        let mut h = small_heap();
        let boxed = h.try_alloc_boxed_in(Generation::G0, 2).unwrap();
        unsafe {
            *boxed.as_ptr() = HeapHeader::new(HeapType::Vector, 1).raw();
            *boxed.as_ptr().add(1) = Word::fixnum(7).raw();
        }
        let original_addr = boxed.as_ptr() as usize;

        while h.young_try_alloc_slab(super::super::space::PAGE_SIZE_CELLS).is_some() {}

        let free_before_gc = h.count_pages_in_gen(Generation::Free);
        assert!(
            free_before_gc > 0,
            "mutator slab growth must stop before consuming every free page"
        );

        let mut roots = [Word::from_ptr(boxed.as_ptr() as *const u8, Tag::Vector)];
        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G0,
            &mut roots,
        );
        assert_eq!(result.objects_copied, 1, "boxed survivor should evacuate");

        let new_addr = (roots[0].raw() & PAYLOAD_MASK) as usize;
        assert_ne!(new_addr, original_addr, "boxed root should move during evacuation");
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

    #[cfg(feature = "conservative-pin")]

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

    #[cfg(feature = "conservative-pin")]

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
        // p2 was evacuated: its old cell should have NO start bit
        // (Phase 3 cleared the page's start bits and only re-set
        // pinned ones) AND the cell content should be zero (Phase 3
        // also zeroes non-pinned cells on flipped pages so a stale
        // Word elsewhere reading this cell yields Fixnum 0 instead
        // of a forward marker that the JIT might dereference). See
        // `zero_page_outside_ranges` and `docs/GC_CHUNKED_INVARIANTS.md`
        // invariant I-6.
        let p2_cell = (p2.as_ptr() as usize - h.base_ptr() as usize) / 8;
        assert!(
            !is_start_at(h.start_bits_slice(), p2_cell),
            "evacuated cell's start bit cleared"
        );
        unsafe {
            assert_eq!(
                *p2.as_ptr(),
                0,
                "evacuated cell zeroed on flip (no stale forward marker)"
            );
        }
    }

    #[test]
    fn mid_evac_oom_reports_structured_gc_stall() {
        let mut h = PageHeap::<crate::lisp_layout::LispLayout>::with_reservation(64 * 1024);
        let boxed = h.try_alloc_boxed_in(Generation::G0, 2).unwrap();
        unsafe {
            *boxed.as_ptr() = HeapHeader::new(crate::heap_common::HeapType::Vector, 1).raw();
            *boxed.as_ptr().add(1) = Word::fixnum(9).raw();
        }
        let mut roots = [Word::from_ptr(boxed.as_ptr() as *const u8, Tag::Vector)];

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            h.evacuate_from_word_roots(Generation::G0, Generation::G0, &mut roots)
        }))
        .expect_err("within-gen evac on a one-page heap must stall");

        let stall = panic
            .downcast_ref::<GcStallError>()
            .expect("panic payload should be GcStallError");
        assert_eq!(stall.reason, GcStallReason::MidEvacOOM);
        assert_eq!(stall.from_gen, Generation::G0);
        assert_eq!(stall.dest_gen, Generation::G0);
        assert_eq!(stall.attempted_kind, PageKind::Boxed);
        assert_eq!(stall.attempted_cells, 2);
        assert_eq!(stall.free_pages, 0);
        assert_eq!(stall.g0_pages, 1);
        assert_eq!(stall.objects_copied_before_failure, 0);
        assert_eq!(stall.cells_copied_before_failure, 0);
    }

    #[test]
    fn g0_to_g0_evacuation_bypasses_mutator_young_cap() {
        // 2 young pages + 6 old pages => reservation of 8 pages.
        // Mutator allocation should stop opening fresh G0 pages at 2,
        // but GC-internal G0->G0 evacuation still needs to copy into
        // fresh G0 pages during a minor cycle.
        let mut h = PageHeap::<crate::lisp_layout::LispLayout>::new(
            2 * 64 * 1024,
            6 * 64 * 1024,
        );
        let mut roots = Vec::new();

        for marker in 0..2 {
            let ptr = h
                .try_alloc_boxed_in(Generation::G0, PAGE_SIZE_CELLS)
                .expect("one page-sized object per young page");
            unsafe {
                *ptr.as_ptr() =
                    HeapHeader::new(HeapType::Vector, (PAGE_SIZE_CELLS - 1) as u32).raw();
                for i in 1..PAGE_SIZE_CELLS {
                    *ptr.as_ptr().add(i) = Word::fixnum(marker).raw();
                }
            }
            roots.push(Word::from_ptr(ptr.as_ptr() as *const u8, Tag::Vector));
        }

        assert_eq!(h.count_pages_in_gen(Generation::G0), 2);
        assert_eq!(h.count_pages_in_gen(Generation::Free), 6);

        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G0,
            roots.as_mut_slice(),
        );

        assert_eq!(result.objects_copied, 2);
        assert_eq!(h.count_pages_in_gen(Generation::G0), 2);
        assert_eq!(h.count_pages_in_gen(Generation::Free), 6);

        for root in &roots {
            let addr = (root.raw() & crate::word::PAYLOAD_MASK) as *const u8;
            let page = h.page_of(addr).expect("root in heap");
            assert_eq!(h.desc(page).generation, Generation::G0);
        }
    }

    #[test]
    fn marked_within_gen_evacuation_recycles_from_pages() {
        let mut h = small_heap();
        let mut roots = Vec::new();

        for marker in 0..7 {
            let ptr = h
                .try_alloc_boxed_in(Generation::G0, PAGE_SIZE_CELLS)
                .expect("one page-sized object per page");
            unsafe {
                *ptr.as_ptr() =
                    HeapHeader::new(HeapType::Vector, (PAGE_SIZE_CELLS - 1) as u32).raw();
                for i in 1..PAGE_SIZE_CELLS {
                    *ptr.as_ptr().add(i) = Word::fixnum(marker).raw();
                }
            }
            roots.push(Word::from_ptr(ptr.as_ptr() as *const u8, Tag::Vector));
        }

        h.mark_from_roots(Generation::G0, &roots);
        h.prepare_recycle_live_counts_from_marks(Generation::G0);

        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G0,
            roots.as_mut_slice(),
        );

        assert_eq!(result.objects_copied, 7);
        assert_eq!(h.count_pages_in_gen(Generation::G0), 7);
        assert_eq!(h.count_pages_in_gen(Generation::Free), 1);
    }

    #[cfg(feature = "conservative-pin")]

    #[test]

    fn promotion_recycle_skips_reused_pages_and_flips_pinned_pages() {
        // With the chunked two-phase evacuator a chunk's source
        // pages are released only after that chunk's Phase 3, so
        // mid-cycle source-page reuse (Cheney's old recycler) no
        // longer fires. The test now verifies the cycle's exposed
        // invariants:
        //
        // - both non-pinned objects copied,
        // - pinned page flipped (not released),
        // - G0 fully drained,
        // - G1 ends with one page per surviving object (two new
        //   dest pages + one flipped),
        // - root Words rewritten to live on G1 pages.
        let mut h = small_heap();

        let first = h
            .try_alloc_boxed_in(Generation::G0, PAGE_SIZE_CELLS)
            .expect("first page-sized object");
        let pinned = h
            .try_alloc_boxed_in(Generation::G0, PAGE_SIZE_CELLS)
            .expect("pinned page-sized object");
        let third = h
            .try_alloc_boxed_in(Generation::G0, PAGE_SIZE_CELLS)
            .expect("third page-sized object");

        for (ptr, marker) in [(first, 11), (pinned, 22), (third, 33)] {
            unsafe {
                *ptr.as_ptr() =
                    HeapHeader::new(HeapType::Vector, (PAGE_SIZE_CELLS - 1) as u32).raw();
                for i in 1..PAGE_SIZE_CELLS {
                    *ptr.as_ptr().add(i) = Word::fixnum(marker).raw();
                }
            }
        }

        let pinned_word = Word::from_ptr(pinned.as_ptr() as *const u8, Tag::Vector);
        let pinned_stack: Box<[u64]> = vec![pinned_word.raw()].into_boxed_slice();
        let lo = pinned_stack.as_ptr() as usize;
        let hi = unsafe { pinned_stack.as_ptr().add(pinned_stack.len()) } as usize;
        h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);

        let mut roots = vec![
            Word::from_ptr(first.as_ptr() as *const u8, Tag::Vector),
            Word::from_ptr(third.as_ptr() as *const u8, Tag::Vector),
        ];

        h.mark_from_roots(Generation::G0, &roots);
        h.prepare_recycle_live_counts_from_marks(Generation::G0);

        let result = h.evacuate_from_word_roots(
            Generation::G0,
            Generation::G1,
            roots.as_mut_slice(),
        );

        assert_eq!(result.objects_copied, 2);
        assert_eq!(result.pages_flipped, 1);
        assert_eq!(h.count_pages_in_gen(Generation::G0), 0);
        assert_eq!(h.count_pages_in_gen(Generation::G1), 3);

        // Both root Words must now point at G1 pages — either at
        // a freshly-allocated dest or at the flipped pinned page.
        for root in &roots {
            let addr =
                (root.raw() & crate::word::PAYLOAD_MASK) as *const u8;
            let page = h.page_of(addr).expect("root in heap");
            assert_eq!(
                h.desc(page).generation,
                Generation::G1,
                "root rewritten to a G1 page"
            );
        }
    }
}

