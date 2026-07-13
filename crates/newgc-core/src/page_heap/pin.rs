//! Conservative pin scan for the page heap.
//!
//! Sub-phase 6 of the Phase 3 plan in `docs/GC_DESIGN.md`. Walks
//! one or more stack-range slices word-by-word; for each value
//! that decodes as a heap-pointer Word into the target
//! generation, sets the matching `PageDesc.pin_byte` slot and
//! records the global cell index in `PageHeap::pinned_cells`.
//!
//! ## Two-level pin index
//!
//! - **Fast path**: `PageDesc.pin_byte` holds 8 bits per page, one
//!   per 8 KB sub-region. Bit `i` set = "at least one object in
//!   sub-region `i` is pinned." A single byte-load + bit-test
//!   answers "could anything on this page be pinned?" — the bulk
//!   of conservative-pin queries answer No on this check alone.
//! - **Precise path**: `PageHeap::pinned_cells` is a `HashSet`
//!   of global cell indices. Consulted only when the page byte's
//!   relevant bit is set. Lets evacuation distinguish "this
//!   specific object is pinned" from "this page has some other
//!   pinned object near it."
//!
//! Why two levels: the conservative pinner is called every minor
//! GC, often with thousands of candidate Words; most aren't real
//! pointers, but every candidate that LOOKS like a heap pointer
//! has to be checked. Without the page-byte fast path,
//! every false positive does a hashset lookup. With it, only
//! page-byte hits hit the set.
//!
//! ## Self-stack-pointer exclusion
//!
//! The pinner accepts an explicit "skip-range" — values that
//! point back into the stack range being scanned. On rare OS
//! layouts the stack and heap can land in overlapping VMAs; a
//! saved RBP or return address inside the same stack as the
//! scan would otherwise be misinterpreted as a heap pointer.
//! Same trick the semispace pinner uses (`heap.rs:432`, sub-phase
//! 1 of the design doc).
//!
//! ## Start-bit gate
//!
//! Even after passing the tag check and the page-generation
//! check, a candidate is rejected if its target cell isn't a real
//! object start (per `start_bits`). This rejects pointers into
//! object payloads — important for cons-payload cells that
//! coincidentally match a heap-pointer bit pattern.

use crate::traits::HeapLayout;
use crate::word::{Tag, Word};

use super::alloc::is_start_at;
use super::page_desc::Generation;
use super::space::{PageHeap, PAGE_SIZE_CELLS};

/// Number of sub-regions per page (slots in `pin_byte`). 8 slots
/// over 64 KB = one slot per 8 KB.
pub const PIN_SLOTS_PER_PAGE: usize = 8;

/// Cells per pin slot. 8192 cells / 8 slots = 1024 cells per slot.
pub const CELLS_PER_PIN_SLOT: usize = PAGE_SIZE_CELLS / PIN_SLOTS_PER_PAGE;

/// Result of a pin scan — surfaced via `(gc-stats)` for parity
/// with the semispace heap's per-cycle pin summary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PinScanResult {
    /// Distinct objects pinned this scan.
    pub n_objects: usize,
    /// Total cells the pinned objects occupy. Unknown at pin time
    /// for sub-phase 6 (header lookup happens during marking /
    /// evacuation); reported as 0 for now and filled in when the
    /// evacuation pass runs in sub-phase 7. Kept on the result
    /// struct so the API stabilises now.
    pub n_cells: usize,
}

/// Handle returned by [`PageHeap::pin`]; release it via
/// [`PageHeap::unpin`]. The inner `Option` is `None` when the pinned
/// `Word` was an immediate / non-heap value (nothing to pin), so
/// `unpin` is then a no-op. `Copy` so an FFI layer can stash it as a
/// plain value — the refcount in `PageHeap::explicit_pins` is the real
/// bookkeeping. MM-0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "release the pin with unpin(); dropping the handle leaks the pin"]
pub struct PinHandle(Option<usize>);

/// Per-gate counters for one pin pass. Exposed only via the
/// `NCL_GC_VERBOSE` diagnostic dump in `pin_pointers_in_ranges`.
#[derive(Default)]
struct PinStats {
    candidates: u64,
    tag_pass: u64,
    in_range_pass: u64,
    in_heap_pass: u64,
    in_gen_pass: u64,
    start_bit_pass: u64,
    new_pins: usize,
}

impl<L: HeapLayout> PageHeap<L> {
    /// Conservative-pin scan. For each candidate `Word` found in
    /// the byte ranges, set the matching pin bit if the target is
    /// a real object start on a page in `target` generation.
    ///
    /// `ranges` is a slice of `(lo, hi)` half-open byte ranges,
    /// one per mutator stack window. Sub-phase 6 uses the same
    /// per-thread "skip targets pointing into the scanned range
    /// itself" exclusion as the semispace pinner (see
    /// `Semispace::pin_pointers_in_range`).
    ///
    /// Returns a `PinScanResult` carrying the number of distinct
    /// objects pinned across all ranges. The result is intended
    /// for `(gc-stats)` and for the trigger policy in
    /// sub-phase 10.
    ///
    /// **Gated by Cargo feature `conservative-pin` (default on).**
    /// Clients that supply precise roots only (e.g. an
    /// LLVM-statepoint-emitting JIT) can compile this away with
    /// `--no-default-features`, eliminating both the API surface
    /// and the linked code.
    #[cfg(feature = "conservative-pin")]
    pub fn pin_pointers_in_ranges(
        &mut self,
        target: Generation,
        ranges: &[(usize, usize)],
    ) -> PinScanResult {
        // Diagnostic: NCL_NO_CONSERVATIVE_PIN=1 makes this a no-op.
        // If Life crashes earlier and in a different place, precise
        // roots are missing a genuinely-live value (the conservative
        // scan was masking the gap). If Life crashes the same way,
        // the bug is reachable through precise roots and conservative
        // pin isn't relevant.
        if std::env::var_os("NCL_NO_CONSERVATIVE_PIN").is_some() {
            return PinScanResult::default();
        }
        let verbose = std::env::var_os("NCL_GC_VERBOSE").is_some();
        let mut stats = PinStats::default();
        for &(range_lo, range_hi) in ranges {
            self.pin_range_one(target, range_lo, range_hi, &mut stats);
        }
        if verbose {
            eprintln!(
                "[pin target={target:?}] candidates={} tag_pass={} \
                 in_range_pass={} in_heap_pass={} in_gen_pass={} \
                 start_bit_pass={} new_pins={}",
                stats.candidates,
                stats.tag_pass,
                stats.in_range_pass,
                stats.in_heap_pass,
                stats.in_gen_pass,
                stats.start_bit_pass,
                stats.new_pins,
            );
        }
        PinScanResult {
            n_objects: stats.new_pins,
            n_cells: 0, // populated by sub-phase 7 once size is known
        }
    }

    /// Pin candidates from one stack range. Returns the number of
    /// distinct new pins recorded (so the caller can sum across
    /// ranges).
    ///
    /// We re-resolve the heap base / cells slice once and walk
    /// the range as `*const u64`. Each candidate Word goes through:
    ///   1. Tag check — accept only Cons/Symbol/Vector/Function/String
    ///   2. Self-stack-exclusion — skip targets back into this range
    ///   3. Page lookup — skip if outside the reservation
    ///   4. Generation check — skip if not in `target`
    ///   5. Start-bit check — skip if not a real object start
    ///   6. Pin: set page byte slot, record cell index
    fn pin_range_one(
        &mut self,
        target: Generation,
        range_lo: usize,
        range_hi: usize,
        stats: &mut PinStats,
    ) {
        if range_lo >= range_hi {
            return;
        }
        let scan_start = (range_lo + 7) & !7;
        let scan_end = range_hi & !7;
        let mut p = scan_start as *const u64;
        let end = scan_end as *const u64;
        while p < end {
            stats.candidates += 1;
            let raw = unsafe { *p };
            // 1. Tag must look like a heap pointer (cons-shaped or
            //    header-bearing). Immediates and forwarding markers
            //    skip immediately.
            let target_addr = match L::classify(raw) {
                crate::traits::WordKind::PointerCons(a)
                | crate::traits::WordKind::PointerHeader(a) => a as usize,
                _ => {
                    p = unsafe { p.add(1) };
                    continue;
                }
            };
            stats.tag_pass += 1;
            // 2. Self-stack-exclusion — pointer back into our own
            //    scan range. Same trick as Semispace::pin_*.
            if target_addr >= range_lo && target_addr < range_hi {
                p = unsafe { p.add(1) };
                continue;
            }
            stats.in_range_pass += 1;
            // 3. Page lookup.
            let target_ptr = target_addr as *const u8;
            let page_idx = match self.page_of(target_ptr) {
                Some(i) => i,
                None => {
                    p = unsafe { p.add(1) };
                    continue;
                }
            };
            stats.in_heap_pass += 1;
            // 4. Generation gate.
            if self.desc(page_idx).generation != target {
                p = unsafe { p.add(1) };
                continue;
            }
            stats.in_gen_pass += 1;
            // 5. Cell-start gate. Compute global cell index and
            //    require the start bit be set.
            let cell_idx = (target_addr - self.base_ptr() as usize) / 8;
            if !is_start_at(self.start_bits_slice(), cell_idx) {
                p = unsafe { p.add(1) };
                continue;
            }
            stats.start_bit_pass += 1;
            // 6. Record the pin. HashSet::insert returns true if
            //    this is a new entry; track for the result count.
            if self.pinned_cells.insert(cell_idx) {
                stats.new_pins += 1;
            }
            // Set page-byte slot. The slot is the sub-region the
            // pinned cell falls into.
            let cell_offset_in_page = cell_idx % PAGE_SIZE_CELLS;
            let slot = (cell_offset_in_page / CELLS_PER_PIN_SLOT) as u8;
            self.desc_mut(page_idx).set_pin(slot);

            p = unsafe { p.add(1) };
        }
    }

    /// Test whether the object at `cell_idx` is pinned by the most
    /// recent scan. The two-level lookup: page byte fast-rejects;
    /// hashtable confirms.
    pub fn is_pinned_cell(&self, cell_idx: usize) -> bool {
        let page_idx = cell_idx / PAGE_SIZE_CELLS;
        if page_idx >= self.page_count() {
            return false;
        }
        let cell_offset_in_page = cell_idx % PAGE_SIZE_CELLS;
        let slot = (cell_offset_in_page / CELLS_PER_PIN_SLOT) as u8;
        if !self.desc(page_idx).is_pinned(slot) {
            return false;
        }
        // Fast path lied; consult the precise set.
        self.pinned_cells.contains(&cell_idx)
    }

    /// Clear every pin bit AND empty the pinned-cells set. Called
    /// at the start of each GC cycle so stale pins from earlier
    /// cycles don't carry forward.
    pub fn clear_all_pins(&mut self) {
        // Reset every PageDesc.pin_byte. Walking all descs is
        // O(n_pages), tiny (16384 max).
        for d in self.descs.iter_mut() {
            d.clear_pins();
        }
        self.pinned_cells.clear();
    }

    /// Diagnostic: number of pinned objects since the last
    /// `clear_all_pins`.
    pub fn pinned_count(&self) -> usize {
        self.pinned_cells.len()
    }

    // -- MM-0: explicit (FFI) pinning ------------------------------------
    //
    // Distinct from the conservative *stack* pin above: these are
    // explicit, client-driven, persistent pins for objects whose address
    // has escaped into foreign code. They are NOT gated by the
    // `conservative-pin` feature — precise-primary builds (which compile
    // the conservative scan out) still need them, because that scan is
    // what *incidentally* pins FFI objects today. See §5.4 of the
    // multi-mutator design.

    /// Resolve a `Word` to a pinnable start-cell index, or `None` if it
    /// is an immediate, points outside the reservation, lands on a Free
    /// page, or isn't a real object start.
    fn pin_target_cell(&self, w: Word) -> Option<usize> {
        let target_addr = match L::classify(w.raw()) {
            crate::traits::WordKind::PointerCons(a)
            | crate::traits::WordKind::PointerHeader(a) => a,
            _ => return None,
        };
        let page_idx = self.page_of(target_addr)?;
        if self.desc(page_idx).generation == Generation::Free {
            return None;
        }
        let cell_idx = (target_addr as usize - self.base_ptr() as usize) / 8;
        if !is_start_at(self.start_bits_slice(), cell_idx) {
            return None;
        }
        Some(cell_idx)
    }

    /// Pin `w`'s target so the collector never moves it until the
    /// matching [`unpin`](Self::unpin), across any number of GC cycles.
    /// Refcounted: N pins need N unpins. Pinning an immediate / non-heap
    /// `Word` is a harmless no-op (returns a handle whose `unpin` does
    /// nothing). Cheap — one hash-map bump; the pin is folded into the
    /// active pin set at the start of the next (and every subsequent)
    /// evacuation by `apply_explicit_pins`.
    pub fn pin(&mut self, w: Word) -> PinHandle {
        match self.pin_target_cell(w) {
            Some(cell_idx) => {
                *self.explicit_pins.entry(cell_idx).or_insert(0) += 1;
                PinHandle(Some(cell_idx))
            }
            None => PinHandle(None),
        }
    }

    /// Release one pin recorded by [`pin`](Self::pin). After the last
    /// unpin of an object it becomes movable/collectable by the next
    /// cycle. Unpinning a no-op handle, or unpinning more times than
    /// pinned, is harmless.
    pub fn unpin(&mut self, handle: PinHandle) {
        let cell_idx = match handle.0 {
            Some(c) => c,
            None => return,
        };
        if let Some(count) = self.explicit_pins.get_mut(&cell_idx) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.explicit_pins.remove(&cell_idx);
            }
        }
    }

    /// Number of distinct objects with at least one live explicit pin.
    /// Diagnostic / test hook.
    pub fn pinned_explicit_count(&self) -> usize {
        self.explicit_pins.len()
    }

    /// Reconcile this cycle's pin set and extend the mark from it, at the
    /// start of an evacuation of `from_gen`.
    ///
    /// Two kinds of pin reach this point:
    ///   - **conservative** pins, recorded by `pin_pointers_in_ranges`
    ///     (the Mutator/`drive_collect` and coordinator paths run it
    ///     before evacuation); and
    ///   - **explicit** (FFI) pins held durably in `explicit_pins`.
    ///
    /// Explicit pins are folded into `pinned_cells` (insert + set the
    /// page's pin byte so Phase 1 skips them and Phase 3 flips their page
    /// in place). Then — for the FULL pin set — extend the mark from every
    /// pinned object so its transitive children survive even when the
    /// pinned object isn't reachable from the caller's precise roots, and
    /// re-seed the per-page live counts so pin-reachable cells aren't
    /// released.
    ///
    /// The extension mark MUST run whenever any cell is pinned, NOT only
    /// when there are explicit pins. The conservative pin scan runs after
    /// the precise mark, so a conservatively-pinned object's children are
    /// unmarked until this extension; skipping it (e.g. early-returning on
    /// an empty `explicit_pins` set) silently reclaims those children,
    /// dropping whole sub-chains and dangling the pinned object's pointers
    /// into freed memory. The coordinator path extends the mark itself;
    /// the bare `collect_minor`/`drive_collect` path relies entirely on
    /// this call.
    ///
    /// Runs every evacuation because `clear_all_pins` wipes `pinned_cells`
    /// at cycle end; `explicit_pins` is the durable source for FFI pins.
    pub(super) fn apply_pins_and_extend_mark(&mut self, from_gen: Generation) {
        // Fold durable explicit (FFI) pins into this cycle's pin set.
        if !self.explicit_pins.is_empty() {
            let cells: Vec<usize> =
                self.explicit_pins.keys().copied().collect();
            for &cell_idx in &cells {
                let page_idx = cell_idx / PAGE_SIZE_CELLS;
                // The object's page may have been relabeled to another gen
                // by an earlier pin-flip; pin regardless of gen so whichever
                // pass touches its gen flips it in place. Skip only the
                // impossible-while-pinned Free case, defensively.
                if self.desc(page_idx).generation == Generation::Free {
                    continue;
                }
                self.pinned_cells.insert(cell_idx);
                let slot =
                    ((cell_idx % PAGE_SIZE_CELLS) / CELLS_PER_PIN_SLOT) as u8;
                self.desc_mut(page_idx).set_pin(slot);
            }
        }
        // A pinned object is effectively a root: mark its children
        // (same-gen and cross-gen into `from_gen`) so they survive, then
        // re-seed live counts. Runs for the whole pin set (conservative +
        // explicit) — see the method doc. Only meaningful while a
        // mark/recycle pass is active for `from_gen`, which the evac
        // driver seeds just before calling us.
        if self.recycle_live_counts_active_for(from_gen)
            && !self.pinned_cells.is_empty()
        {
            self.extend_mark_from_pinned(from_gen);
            self.extend_mark_from_cross_gen_pinned(from_gen);
            self.prepare_recycle_live_counts_from_marks(from_gen);
        }
    }
}

#[cfg(all(test, feature = "conservative-pin"))]
mod tests {
    use super::*;
    use crate::word::{Tag, Word};
    use super::super::space::PageHeap;

    fn small_heap() -> PageHeap<crate::lisp_layout::LispLayout> {
        // 4 pages = 256 KB. Plenty for several thousand cons cells.
        PageHeap::<crate::lisp_layout::LispLayout>::with_reservation(4 * 64 * 1024)
    }

    /// Build a stack-like array of u64 values containing the given
    /// raw words, in the order provided. Returns the address range
    /// of the underlying buffer plus a Pin'd Box keeping it alive.
    fn build_fake_stack(words: &[u64]) -> (usize, usize, Box<[u64]>) {
        let b: Box<[u64]> = words.to_vec().into_boxed_slice();
        let lo = b.as_ptr() as usize;
        let hi = unsafe { b.as_ptr().add(b.len()) } as usize;
        (lo, hi, b)
    }

    #[test]
    fn empty_range_pins_nothing() {
        let mut h = small_heap();
        let _ = h.try_alloc_cons_in(Generation::G0);
        let result = h.pin_pointers_in_ranges(Generation::G0, &[(0, 0)]);
        assert_eq!(result.n_objects, 0);
        assert_eq!(h.pinned_count(), 0);
    }

    #[test]
    fn cons_pointer_on_fake_stack_pins_the_object() {
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        let w = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
        let (lo, hi, _b) = build_fake_stack(&[w.raw()]);
        let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert_eq!(result.n_objects, 1);
        let cell_idx = (p.as_ptr() as usize - h.base_ptr() as usize) / 8;
        assert!(h.is_pinned_cell(cell_idx), "cell should be pinned");
        // PageDesc.pin_byte must record the slot too.
        let page_idx = cell_idx / PAGE_SIZE_CELLS;
        let slot = ((cell_idx % PAGE_SIZE_CELLS) / CELLS_PER_PIN_SLOT) as u8;
        assert!(h.desc(page_idx).is_pinned(slot));
    }

    #[test]
    fn fixnum_on_fake_stack_pins_nothing() {
        let mut h = small_heap();
        let _ = h.try_alloc_cons_in(Generation::G0);
        let (lo, hi, _b) = build_fake_stack(&[
            Word::fixnum(0).raw(),
            Word::fixnum(42).raw(),
            Word::fixnum(-1).raw(),
            Word::NIL.raw(),
            Word::T.raw(),
        ]);
        let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert_eq!(result.n_objects, 0);
    }

    #[test]
    fn out_of_range_pointer_pins_nothing() {
        let mut h = small_heap();
        let _ = h.try_alloc_cons_in(Generation::G0);
        // A Cons-tagged Word whose target is outside the
        // reservation. Should be rejected by the page lookup.
        let bogus = Word::from_raw(0xdead_beef_0000 | (Tag::Cons as u64));
        let (lo, hi, _b) = build_fake_stack(&[bogus.raw()]);
        let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert_eq!(result.n_objects, 0);
    }

    #[test]
    fn pointer_into_cons_cdr_is_rejected_by_start_bit_check() {
        let mut h = small_heap();
        // Allocate one cons. Its first cell is a cons-start.
        // Construct a Word pointing at the SECOND cell (cdr
        // position) — not a start, so the pinner must reject.
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        let cdr_addr = unsafe { p.as_ptr().add(1) as *const u8 } as usize;
        let bogus = Word::from_raw((cdr_addr as u64) | (Tag::Cons as u64));
        let (lo, hi, _b) = build_fake_stack(&[bogus.raw()]);
        let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert_eq!(result.n_objects, 0, "cdr-of-cons isn't a start");
    }

    #[test]
    fn cross_generational_pointer_does_not_pin_outside_target() {
        let mut h = small_heap();
        let g0_cons = h.try_alloc_cons_in(Generation::G0).unwrap();
        let g1_cons = h.try_alloc_cons_in(Generation::G1).unwrap();
        let g0_w = Word::from_ptr(g0_cons.as_ptr() as *const u8, Tag::Cons);
        let g1_w = Word::from_ptr(g1_cons.as_ptr() as *const u8, Tag::Cons);
        let (lo, hi, _b) = build_fake_stack(&[g0_w.raw(), g1_w.raw()]);
        // Scan G0 only: G1 word should be skipped by the gen gate.
        let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert_eq!(result.n_objects, 1);
        let g0_cell = (g0_cons.as_ptr() as usize - h.base_ptr() as usize) / 8;
        let g1_cell = (g1_cons.as_ptr() as usize - h.base_ptr() as usize) / 8;
        assert!(h.is_pinned_cell(g0_cell));
        assert!(!h.is_pinned_cell(g1_cell));
    }

    #[test]
    fn duplicate_pointers_pin_once() {
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        let w = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
        let (lo, hi, _b) = build_fake_stack(&[w.raw(), w.raw(), w.raw(), w.raw()]);
        let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert_eq!(result.n_objects, 1, "n_objects counts DISTINCT pins");
        assert_eq!(h.pinned_count(), 1);
    }

    #[test]
    fn self_stack_exclusion_skips_intra_range_pointers() {
        // Fabricate a "stack" range, then plant a Word inside it
        // pointing back into the same range. The pinner must skip.
        let mut h = small_heap();
        // First need a real cons so the heap has something to
        // potentially pin (just to make the test less trivial).
        let _ = h.try_alloc_cons_in(Generation::G0);
        let buf: Box<[u64]> = vec![0u64; 4].into_boxed_slice();
        let lo = buf.as_ptr() as usize;
        let hi = unsafe { buf.as_ptr().add(buf.len()) } as usize;
        // Plant a Cons-tagged "pointer" into the SECOND slot,
        // pointing at the THIRD slot. Pointer is in [lo, hi) AND
        // its target is in [lo, hi), so it's a self-stack ref.
        let inner_target = unsafe { buf.as_ptr().add(2) as usize };
        let self_ref = (inner_target as u64) | (Tag::Cons as u64);
        unsafe {
            (buf.as_ptr().add(1) as *mut u64).write(self_ref);
        }
        let result = h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        assert_eq!(result.n_objects, 0, "self-stack pointer must be skipped");
    }

    #[test]
    fn clear_all_pins_resets_state() {
        let mut h = small_heap();
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        let w = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
        let (lo, hi, _b) = build_fake_stack(&[w.raw()]);
        h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        let cell_idx = (p.as_ptr() as usize - h.base_ptr() as usize) / 8;
        assert!(h.is_pinned_cell(cell_idx));
        h.clear_all_pins();
        assert!(!h.is_pinned_cell(cell_idx));
        assert_eq!(h.pinned_count(), 0);
        // PageDesc.pin_byte cleared too.
        let page_idx = cell_idx / PAGE_SIZE_CELLS;
        assert!(!h.desc(page_idx).has_pins());
    }

    #[test]
    fn pin_byte_groups_into_correct_slot() {
        // Allocate enough conses that we span two pin-slots
        // within the first page. Slot size = 1024 cells = 512
        // conses. So conses 0..512 land in slot 0, 512..1024 in
        // slot 1, etc. Allocate 600 conses and verify both
        // slots 0 and 1 get pinned when we feed pointers to a
        // cons in each half.
        let mut h = small_heap();
        let mut cons_ptrs = Vec::with_capacity(600);
        for _ in 0..600 {
            cons_ptrs.push(h.try_alloc_cons_in(Generation::G0).unwrap());
        }
        let w0 = Word::from_ptr(cons_ptrs[10].as_ptr() as *const u8, Tag::Cons);
        let w1 = Word::from_ptr(cons_ptrs[550].as_ptr() as *const u8, Tag::Cons);
        let (lo, hi, _b) = build_fake_stack(&[w0.raw(), w1.raw()]);
        h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
        // Both should land on page 0.
        let page = 0usize;
        let cell_a = (cons_ptrs[10].as_ptr() as usize - h.base_ptr() as usize) / 8;
        let cell_b = (cons_ptrs[550].as_ptr() as usize - h.base_ptr() as usize) / 8;
        let slot_a = ((cell_a % PAGE_SIZE_CELLS) / CELLS_PER_PIN_SLOT) as u8;
        let slot_b = ((cell_b % PAGE_SIZE_CELLS) / CELLS_PER_PIN_SLOT) as u8;
        // cons #10 is at cell ~20 → slot 0; cons #550 is at cell
        // ~1100 → slot 1. Both slots should now be set on page 0.
        assert_eq!(slot_a, 0);
        assert_eq!(slot_b, 1);
        assert!(h.desc(page).is_pinned(slot_a));
        assert!(h.desc(page).is_pinned(slot_b));
        // Sub-region without a pinned cons (slot 7, page tail) is
        // NOT pinned.
        assert!(!h.desc(page).is_pinned(7));
    }
}
