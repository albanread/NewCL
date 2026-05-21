//! Per-page metadata for the page heap.
//!
//! Sub-phase 3 of the Phase 3 plan in `docs/GC_DESIGN.md`. The
//! types here are pure data; no GC behaviour yet (that lands in
//! sub-phases 5-7).
//!
//! ## Why a separate metadata table
//!
//! The page-heap reservation (`space.rs`) holds Lisp objects in
//! 64 KB pages. To collect garbage in generations, mark live
//! objects, evacuate survivors, and reclaim empty pages, we need
//! per-page bookkeeping: which generation each page belongs to,
//! what shape of objects it holds, how full it is, where to start
//! scanning, and which sub-regions have pinned objects.
//!
//! Storing this metadata *inside* the page is awkward — it'd
//! shrink the usable region and complicate scan logic. Instead we
//! keep a parallel `Vec<PageDesc>` indexed by page number. 12
//! bytes per page × 16384 pages (= 1 GB reservation) = 192 KB of
//! descriptor storage, paid up front. Tiny compared to the data
//! it describes.
//!
//! ## Atomic / non-atomic
//!
//! Sub-phase 3 keeps fields plain (non-atomic). All mutation
//! happens during stop-the-world GC. Sub-phase 9 will introduce
//! the write barrier and at that point the relevant fields (most
//! likely `gen` and `pin_byte`) become atomic for lock-free
//! mutator-side reads. Until then, simpler is better.

/// Which generation a page belongs to. The lifecycle is
/// `Free → G0 → G1 → Tenured`, with each transition triggered by
/// surviving `num_gcs_before_promotion` collections at the
/// current level.
///
/// Encoded as `u8` so it packs cleanly into `PageDesc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Generation {
    /// Page is on the free list. Not assigned to any cohort, not
    /// scanned by the GC, may or may not be currently committed.
    Free = 0,
    /// Nursery. All freshly-allocated objects land here.
    /// Minor GC collects G0; survivors stay in G0 until they've
    /// lived through `num_gcs_before_promotion` cycles, then they
    /// promote to G1.
    G0 = 1,
    /// Intermediate generation. Objects that survived G0 long
    /// enough to be considered "probably long-lived" but haven't
    /// proven it yet. Collected less frequently than G0.
    G1 = 2,
    /// Tenured. Old objects unlikely to die soon. Collected only
    /// during full GC.
    Tenured = 3,
}

impl Generation {
    /// Decode from raw byte. Returns `None` for invalid values
    /// (which indicates corruption of the page table).
    pub fn from_u8(b: u8) -> Option<Generation> {
        match b {
            0 => Some(Generation::Free),
            1 => Some(Generation::G0),
            2 => Some(Generation::G1),
            3 => Some(Generation::Tenured),
            _ => None,
        }
    }

    /// Next generation the page promotes to. `Free` and `Tenured`
    /// are fixed points — Free → Free (still free), Tenured →
    /// Tenured (already at the top).
    pub fn promoted(self) -> Generation {
        match self {
            Generation::Free => Generation::Free,
            Generation::G0 => Generation::G1,
            Generation::G1 => Generation::Tenured,
            Generation::Tenured => Generation::Tenured,
        }
    }

    /// Human-readable name for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Generation::Free => "free",
            Generation::G0 => "g0",
            Generation::G1 => "g1",
            Generation::Tenured => "tenured",
        }
    }
}

/// What kind of objects live on this page. The kind determines
/// how the GC walks the page's contents (cons cells have no
/// headers; boxed objects have a per-object header; large objects
/// occupy one or more whole pages).
///
/// Sub-phase 4 will use this to decide which allocation region
/// the page belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PageKind {
    /// Page is free / unassigned. No objects.
    Free = 0,
    /// Headerless cons cells. Each pair of 8-byte cells is one
    /// cons cell. Every cell is therefore an object start; no
    /// per-cell start-bit bitmap needed. Walking is a fixed
    /// stride of 2 cells.
    Cons = 1,
    /// Boxed objects of mixed size (Symbol, Vector, Function,
    /// String, …). Each object has a `HeapHeader` at its first
    /// cell carrying the type and length. Walking requires a
    /// start-bit bitmap so the scanner can find object
    /// boundaries.
    Boxed = 2,
    /// One large object spanning the whole page (or multiple
    /// consecutive pages for very large objects). Allocated rare
    /// large vectors / strings. Freed page-at-a-time during GC.
    Large = 3,
}

impl PageKind {
    pub fn from_u8(b: u8) -> Option<PageKind> {
        match b {
            0 => Some(PageKind::Free),
            1 => Some(PageKind::Cons),
            2 => Some(PageKind::Boxed),
            3 => Some(PageKind::Large),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PageKind::Free => "free",
            PageKind::Cons => "cons",
            PageKind::Boxed => "boxed",
            PageKind::Large => "large",
        }
    }
}

/// Per-page metadata. Twelve bytes per page; one entry in the
/// parallel `Vec<PageDesc>` owned by `PageHeap`.
///
/// Layout (with `#[repr(C)]`, fields ordered for natural alignment):
///
/// ```text
///   offset 0   scan_start_offset  u32   (4 bytes)
///   offset 4   words_used         u16   (2 bytes)
///   offset 6   gen                u8    (1 byte)
///   offset 7   kind               u8    (1 byte)
///   offset 8   pin_byte           u8    (1 byte)
///   offset 9   age                u8    (1 byte)
///   offset 10  n_span             u16   (2 bytes)
///   total                              = 12 bytes
/// ```
///
/// `words_used` is 16 bits — page is 8192 cells = 13 bits; 16-bit
/// max of 65535 cells is plenty. `scan_start_offset` is 32 bits
/// for future-compat with multi-page large objects (could
/// reference a position outside one page).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PageDesc {
    /// Cell offset where the linear scan should begin on the next
    /// mark pass. Updated by evacuation when objects compact down.
    /// Zero for a freshly-allocated page (scan from the start).
    pub scan_start_offset: u32,
    /// Cells consumed by allocation. Bump-pointer high-water mark.
    /// Zero for a Free page. Allocators advance this; evacuation
    /// rewrites it after compaction.
    pub words_used: u16,
    /// Generation byte. `Free` for pages on the free list.
    /// (Named `generation` rather than `gen` because Rust 2024
    /// reserved `gen` as a keyword.)
    pub generation: Generation,
    /// Page kind. `Free` for free pages.
    pub kind: PageKind,
    /// Sub-page pin bitmap. Eight slots per page (one per 8 KB
    /// sub-region). Bit `i` set = at least one object in the
    /// `i`-th sub-region is pinned by a conservative root.
    /// Checked first on the pinning fast path before consulting
    /// the per-page pinned-objects hashtable.
    pub pin_byte: u8,
    /// Number of minor-GC cycles this page has survived in its
    /// current generation. Used by the promotion policy in
    /// sub-phase 8: when `age >= num_gcs_before_promotion[gen]`,
    /// the page flips to `gen.promoted()` after the next cycle.
    /// Reset to 0 on every page-state change.
    pub age: u8,
    /// For `PageKind::Large` pages only — number of consecutive
    /// pages in this large-object run. Set to `n_pages` on the
    /// HEAD page of the run (the first page, where the object
    /// header lives). Set to `0` on all CONTINUATION pages. For
    /// all non-Large pages this field is `1` (Cons/Boxed) or `0`
    /// (Free), but must never be read as a span length unless the
    /// kind is Large.
    ///
    /// Repurposes the former `_pad: u16` field; the `PageDesc` size
    /// (12 bytes) and alignment (4 bytes) are unchanged.
    pub n_span: u16,
}

impl PageDesc {
    /// Canonical descriptor for a free page. Used to initialise
    /// the page table at construction time.
    pub const FREE: PageDesc = PageDesc {
        scan_start_offset: 0,
        words_used: 0,
        generation: Generation::Free,
        kind: PageKind::Free,
        pin_byte: 0,
        age: 0,
        n_span: 0,
    };

    /// Construct a fresh page descriptor for a page just assigned
    /// to a generation/kind cohort. `words_used` starts at 0;
    /// `scan_start_offset` starts at 0; `pin_byte` and `age` are
    /// cleared.
    ///
    /// `n_span` is initialised to `1` for normal object pages
    /// (Cons, Boxed) and `0` for Free and Large pages. For Large
    /// pages the caller must set `n_span` on the head page after
    /// calling `fresh`.
    pub fn fresh(generation: Generation, kind: PageKind) -> PageDesc {
        PageDesc {
            scan_start_offset: 0,
            words_used: 0,
            generation,
            kind,
            pin_byte: 0,
            age: 0,
            n_span: match kind {
                PageKind::Cons | PageKind::Boxed => 1,
                _ => 0, // Free and Large start at 0
            },
        }
    }

    /// Returns `true` if this is the head page of a large-object run
    /// (`kind == Large` and `n_span >= 1`).
    pub fn is_large_head(&self) -> bool {
        self.kind == PageKind::Large && self.n_span >= 1
    }

    /// Returns `true` if this is a continuation page of a large-object
    /// run (`kind == Large` and `n_span == 0`).
    pub fn is_large_cont(&self) -> bool {
        self.kind == PageKind::Large && self.n_span == 0
    }

    /// Reset to FREE while preserving the page-index identity.
    /// Called when evacuation reclaims an empty page.
    pub fn release(&mut self) {
        *self = PageDesc::FREE;
    }

    /// Test whether any sub-region of this page is pinned. Used
    /// by evacuation to decide whether the page can be freed
    /// outright or must be kept in place.
    pub fn has_pins(&self) -> bool {
        self.pin_byte != 0
    }

    /// Mark a sub-region as pinned. `slot` is `0..8` — the page's
    /// 64 KB range divided into 8 KB sub-regions. Idempotent.
    pub fn set_pin(&mut self, slot: u8) {
        debug_assert!(slot < 8, "pin slot {slot} out of range");
        self.pin_byte |= 1 << slot;
    }

    /// Clear all pin bits. Called at the start of a GC cycle.
    pub fn clear_pins(&mut self) {
        self.pin_byte = 0;
    }

    /// Test whether sub-region `slot` is pinned.
    pub fn is_pinned(&self, slot: u8) -> bool {
        debug_assert!(slot < 8, "pin slot {slot} out of range");
        self.pin_byte & (1 << slot) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_desc_is_twelve_bytes() {
        assert_eq!(std::mem::size_of::<PageDesc>(), 12);
    }

    #[test]
    fn page_desc_align_is_four() {
        // u32 field forces 4-byte alignment; we want this so a
        // Vec<PageDesc> has no extra padding between elements.
        assert_eq!(std::mem::align_of::<PageDesc>(), 4);
    }

    #[test]
    fn free_descriptor_is_zero_initialised() {
        let d = PageDesc::FREE;
        assert_eq!(d.generation, Generation::Free);
        assert_eq!(d.kind, PageKind::Free);
        assert_eq!(d.words_used, 0);
        assert_eq!(d.scan_start_offset, 0);
        assert_eq!(d.pin_byte, 0);
        assert_eq!(d.age, 0);
    }

    #[test]
    fn fresh_descriptor_carries_generation_and_kind() {
        let d = PageDesc::fresh(Generation::G0, PageKind::Cons);
        assert_eq!(d.generation, Generation::G0);
        assert_eq!(d.kind, PageKind::Cons);
        assert_eq!(d.words_used, 0);
        assert_eq!(d.scan_start_offset, 0);
        assert!(!d.has_pins());
    }

    #[test]
    fn generation_promotion_ladder() {
        assert_eq!(Generation::Free.promoted(), Generation::Free);
        assert_eq!(Generation::G0.promoted(), Generation::G1);
        assert_eq!(Generation::G1.promoted(), Generation::Tenured);
        assert_eq!(
            Generation::Tenured.promoted(),
            Generation::Tenured,
            "tenured is a fixed point — no super-old generation"
        );
    }

    #[test]
    fn generation_byte_roundtrip() {
        for g in [
            Generation::Free,
            Generation::G0,
            Generation::G1,
            Generation::Tenured,
        ] {
            assert_eq!(Generation::from_u8(g as u8), Some(g));
        }
        assert_eq!(Generation::from_u8(99), None);
    }

    #[test]
    fn page_kind_byte_roundtrip() {
        for k in [PageKind::Free, PageKind::Cons, PageKind::Boxed, PageKind::Large] {
            assert_eq!(PageKind::from_u8(k as u8), Some(k));
        }
        assert_eq!(PageKind::from_u8(99), None);
    }

    #[test]
    fn pin_bitmap_set_and_clear() {
        let mut d = PageDesc::fresh(Generation::G0, PageKind::Boxed);
        assert!(!d.has_pins());

        d.set_pin(0);
        assert!(d.is_pinned(0));
        assert!(!d.is_pinned(1));
        assert!(d.has_pins());

        d.set_pin(3);
        d.set_pin(7);
        assert!(d.is_pinned(0));
        assert!(d.is_pinned(3));
        assert!(d.is_pinned(7));
        assert!(!d.is_pinned(4));
        assert_eq!(d.pin_byte, 0b1000_1001);

        // Setting a pin twice doesn't toggle.
        d.set_pin(0);
        assert!(d.is_pinned(0));

        d.clear_pins();
        assert!(!d.has_pins());
        assert_eq!(d.pin_byte, 0);
    }

    #[test]
    fn release_sets_back_to_free() {
        let mut d = PageDesc::fresh(Generation::G1, PageKind::Boxed);
        d.words_used = 4000;
        d.set_pin(2);
        d.age = 5;
        d.scan_start_offset = 17;
        d.release();
        assert_eq!(d, PageDesc::FREE);
    }

    #[test]
    fn n_span_is_zero_for_free_page() {
        assert_eq!(PageDesc::FREE.n_span, 0);
    }

    #[test]
    fn n_span_is_one_for_cons_and_boxed() {
        let d_cons = PageDesc::fresh(Generation::G0, PageKind::Cons);
        assert_eq!(d_cons.n_span, 1, "Cons page n_span must be 1");
        let d_boxed = PageDesc::fresh(Generation::G1, PageKind::Boxed);
        assert_eq!(d_boxed.n_span, 1, "Boxed page n_span must be 1");
    }

    #[test]
    fn large_head_and_cont_predicates() {
        // Fresh Large desc has n_span == 0 → cont (caller sets n_span on head).
        let mut head = PageDesc::fresh(Generation::G0, PageKind::Large);
        assert!(!head.is_large_head(), "before n_span set, not head");
        assert!(head.is_large_cont(), "n_span == 0 is a cont");

        head.n_span = 3;
        assert!(head.is_large_head(), "n_span = 3 is the head");
        assert!(!head.is_large_cont());

        let cont = PageDesc::fresh(Generation::G0, PageKind::Large);
        // n_span stays 0
        assert!(!cont.is_large_head());
        assert!(cont.is_large_cont());
    }
}
