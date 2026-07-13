//! Language-binding traits for newgc-core.
//!
//! The GC engine stores raw `u64` cells. Everything language-specific
//! — tag layout, fixnum encoding, header decoding, the choice between
//! cons-shaped and header-bearing objects — is hidden behind the
//! `HeapLayout` trait. A client implements `HeapLayout` once for their
//! language, parameterises `PageHeap<L>` with it, and the GC inlines
//! every dispatch against `L`'s concrete methods.
//!
//! `GC_LESSONS.md` Pattern 1 ("runtime `dyn` is a permanent tax") is
//! load-bearing here: this trait is monomorphised, never object-safe,
//! never `dyn`. The page-heap's hot paths inline the
//! tag-check and pointer-extraction at the call site.

use std::fmt::Debug;

/// Classification of a 64-bit cell value.
///
/// The GC reads a cell, asks the layout to classify it, and dispatches:
///   - `Immediate`: leave untouched. Includes fixnums, characters,
///     booleans, NIL — anything that isn't a tagged heap pointer or
///     forwarding marker.
///   - `PointerCons(addr)`: tagged pointer to a 2-cell, header-less
///     object (a Lisp cons pair, a Dylan `<pair>`, etc.). The target
///     page must be `PageKind::Cons`.
///   - `PointerHeader(addr)`: tagged pointer to an object whose first
///     cell is a header (`HeapLayout::header_layout` decodes it). The
///     target page must be `PageKind::Boxed`.
///   - `Forwarded(addr)`: marker written by the evacuator into the
///     first cell of a moved object. Tells subsequent visits "this
///     object lives at `addr` now."
///
/// The two `Pointer*` variants are how the GC enforces the
/// tag-vs-page-kind consistency gate (the fifth of the five gates in
/// `evac::maybe_copy`). A language that doesn't distinguish cons-shaped
/// from header-bearing objects implements `classify` to always return
/// `PointerHeader` — the cons-start bit in the start-bit bitmap then
/// goes unused, costing one bit per cell but no correctness.
#[derive(Copy, Clone, Debug)]
pub enum WordKind {
    Immediate,
    PointerCons(*const u8),
    PointerHeader(*const u8),
    Forwarded(*const u8),
}

/// Whether a tagged pointer points at a cons-shaped or header-bearing
/// object. Used by `make_pointer` and by start-bit dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PointerKind {
    Cons,
    Header,
}

/// The pointer-bearing payload range of a header-bearing object.
///
/// Returned by `HeapLayout::header_layout`. The GC uses this to know
///   - how many cells to skip past when walking the heap linearly
///     (`total_cells`),
///   - which payload cells are pointer-typed and should be scanned
///     (`pointer_cells_start..pointer_cells_end`).
///
/// Cells outside `[pointer_cells_start, pointer_cells_end)` are opaque
/// — raw f64 bits, bignum limbs, string bytes, FFI scratch. The GC
/// will not classify them as `Word`s and will not follow them.
///
/// All cell counts are relative to the header cell (offset 0). The
/// payload starts at offset 1. `total_cells` includes the header
/// cell itself.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ObjectLayout {
    pub total_cells: usize,
    /// Inclusive lower bound of pointer cells. Cell offset (not cell
    /// index in the heap).
    pub pointer_cells_start: usize,
    /// Exclusive upper bound of pointer cells. `pointer_cells_start ==
    /// pointer_cells_end` means no pointer cells.
    pub pointer_cells_end: usize,
}

impl ObjectLayout {
    /// No pointer-bearing payload (string, raw bytes, opaque payload).
    pub fn opaque(total_cells: usize) -> ObjectLayout {
        ObjectLayout {
            total_cells,
            pointer_cells_start: 0,
            pointer_cells_end: 0,
        }
    }

    /// All payload cells hold pointer-typed Words (vector-shaped).
    pub fn all_pointers(total_cells: usize) -> ObjectLayout {
        debug_assert!(total_cells >= 1);
        ObjectLayout {
            total_cells,
            pointer_cells_start: 1,
            pointer_cells_end: total_cells,
        }
    }

    pub fn pointer_cell_count(self) -> usize {
        self.pointer_cells_end.saturating_sub(self.pointer_cells_start)
    }
}

/// Per-language layout adapter.
///
/// Implementors are zero-sized markers (`pub struct LispLayout;`); the
/// trait is implemented on the type itself, not on instances. All
/// methods are `fn`, not `&self fn` — the GC never holds a layout
/// instance, only its type.
///
/// **Inlining contract**: implementations MUST be small (most are 5–20
/// instructions). The GC depends on `classify` inlining into the
/// mark/evac scanners; a 200-line `classify` would obliterate the hot
/// path. If a tag scheme needs many cases, consider splitting it.
pub trait HeapLayout: Copy + Clone + Debug + Default + 'static {
    /// What the allocator writes into freshly-allocated cells. The
    /// language's nil/null/false value, so reads of fresh cells return
    /// something the language naturally treats as "absent" rather
    /// than a stale pointer.
    const FILL_WORD: u64;

    /// Decode the raw 64-bit content of a heap cell.
    ///
    /// Called on **every cell read** during mark/evac. Must be branchy
    /// but small — the tag dispatch is the safety boundary between
    /// "leave this alone" (immediate) and "follow this" (pointer).
    fn classify(raw: u64) -> WordKind;

    /// Encode a forwarding marker pointing at `new_addr`. Written by
    /// the evacuator into the first cell of a moved object. Subsequent
    /// `classify` calls on that cell must return
    /// `WordKind::Forwarded(new_addr)`.
    fn make_forward(new_addr: *const u8) -> u64;

    /// Encode a heap pointer of the given kind. Convenience for tests
    /// and debug paths. The production rewrite path uses
    /// `rewrite_pointer_addr` instead, which preserves language-
    /// specific tag bits the GC doesn't see.
    fn make_pointer(addr: *const u8, kind: PointerKind) -> u64;

    /// Rewrite a pointer Word's address while preserving any
    /// language-specific tag bits the GC isn't aware of.
    ///
    /// Called by the evacuator on every pointer-slot rewrite: given
    /// the original Word's raw bits and the target object's new
    /// address, produce the rewritten raw bits. The original tag
    /// (cons/symbol/vector/function/string in Lisp; cons/header in
    /// a simpler language) is preserved.
    ///
    /// The default `make_pointer` would lose fine-grained tags by
    /// collapsing to PointerCons/PointerHeader. `rewrite_pointer_addr`
    /// is the production path; `make_pointer` is the test path.
    fn rewrite_pointer_addr(old_raw: u64, new_addr: *const u8) -> u64;

    /// Decode the header at `header_cell` and return the object's
    /// layout. Called by the scanner before walking payload cells.
    ///
    /// SAFETY: `header_cell` is guaranteed to point at a valid header
    /// cell (the GC has already verified it via the start-bit bitmap).
    unsafe fn header_layout(header_cell: *const u64) -> ObjectLayout;
}
