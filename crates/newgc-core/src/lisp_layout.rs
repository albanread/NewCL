//! `HeapLayout` impl for NCL's 3-bit-tagged `Word` + `HeapHeader`.
//!
//! This is the default-shipped language binding. It exists for two
//! reasons:
//!   1. NCL itself can adopt this crate by importing
//!      `newgc_core::PageHeap<LispLayout>` and the existing 3-bit-tag
//!      tooling continues to work.
//!   2. The crate's test suite (both unit tests in `page_heap/*.rs`
//!      and the synthetic integration tests in `tests/synthetic.rs`)
//!      exercises the GC engine through this layout — proving the
//!      trait surface is sufficient to express a realistic
//!      tagged-pointer Lisp.
//!
//! A second binding (e.g. `DylanLayout` with 1-bit tags + Wrapper
//! header) lives downstream of this crate. The synthetic test client
//! in Phase 2's follow-up will be a third reference binding.

use crate::heap_common::{HeapHeader, HeapType};
use crate::traits::{HeapLayout, ObjectLayout, PointerKind, WordKind};
use crate::word::{PAYLOAD_MASK, Tag, Word};

/// NCL's 3-bit-tag Lisp layout.
///
/// Tag map (bits 0..3 of every cell):
///   - `000` Fixnum    → `Immediate`
///   - `001` Cons      → `PointerCons`
///   - `010` Symbol    → `PointerHeader`
///   - `011` Vector    → `PointerHeader`
///   - `100` Function  → `PointerHeader`
///   - `101` String    → `PointerHeader`
///   - `110` Immediate → `Immediate` (NIL, T, char, unbound)
///   - `111` Forward   → `Forwarded`
#[derive(Copy, Clone, Debug, Default)]
pub struct LispLayout;

impl HeapLayout for LispLayout {
    /// Newly-allocated cells get NIL — the empty list / falsy value.
    /// `Word::NIL` is `Tag::Immediate | (SUBTAG_NIL << TAG_BITS)`, a
    /// non-zero immediate. Distinct from `Word::fixnum(0)`.
    const FILL_WORD: u64 = Word::NIL.raw();

    #[inline(always)]
    fn classify(raw: u64) -> WordKind {
        let w = Word::from_raw(raw);
        match w.tag() {
            Tag::Fixnum | Tag::Immediate => WordKind::Immediate,
            Tag::Cons => {
                WordKind::PointerCons((raw & PAYLOAD_MASK) as *const u8)
            }
            Tag::Symbol | Tag::Vector | Tag::Function | Tag::String => {
                WordKind::PointerHeader((raw & PAYLOAD_MASK) as *const u8)
            }
            Tag::Forward => {
                WordKind::Forwarded((raw & PAYLOAD_MASK) as *const u8)
            }
        }
    }

    #[inline(always)]
    fn make_forward(new_addr: *const u8) -> u64 {
        Word::forward(new_addr as *const ()).raw()
    }

    #[inline(always)]
    fn rewrite_pointer_addr(old_raw: u64, new_addr: *const u8) -> u64 {
        // Preserve the original 3-bit tag (bits 0..3) and the
        // immediate sub-tag bits (3..8) if any. For heap pointers,
        // bits 3..64 are the payload; for immediates the whole word
        // is meaningful. The evacuator only calls this when classify
        // returned `PointerCons` or `PointerHeader`, so bits 0..3 are
        // a heap-pointer tag and the rest is the old address. Mask
        // those tag bits and OR in the new address.
        let tag_bits = old_raw & crate::word::TAG_MASK;
        (new_addr as u64) | tag_bits
    }

    #[inline(always)]
    fn make_pointer(addr: *const u8, kind: PointerKind) -> u64 {
        let tag = match kind {
            // The GC only distinguishes Cons vs Header; for Header it
            // doesn't know the language's finer-grained type. We use
            // `Tag::Vector` as the canonical "header-bearing object"
            // tag — same width and start-bit semantics as Symbol /
            // Function / String, and the language's downstream code
            // re-reads the original tag from the HeapHeader's type
            // field anyway.
            //
            // The original tag IS preserved across evacuation by the
            // evacuator copying the source object's first cell (header
            // or content) unchanged. The pointer-tag in *the
            // referring slot* is set from the source pointer's
            // original tag — `make_pointer` is called with the
            // same `kind` the source pointer had, but distinguishing
            // Symbol-vs-Vector-vs-Function within Header isn't a GC
            // concern.
            //
            // Practical implication: a `Symbol`-tagged Word evacuated
            // through this path keeps `Symbol` in its tag because the
            // evacuator preserves the original `tag` when rewriting,
            // not by calling `make_pointer`. `make_pointer` is the
            // *fallback* for synthesised pointers in tests / debug
            // paths — not the production rewrite path.
            PointerKind::Cons => Tag::Cons,
            PointerKind::Header => Tag::Vector,
        };
        Word::from_ptr(addr, tag).raw()
    }

    #[inline(always)]
    unsafe fn header_layout(header_cell: *const u64) -> ObjectLayout {
        let h = HeapHeader::from_raw(unsafe { *header_cell });
        let length = h.length_cells();
        // Header occupies one cell; payload is `length` cells; total
        // is `1 + length`.
        let total_cells = 1 + length as usize;
        match h.ty().word_field_range(length) {
            Some((first, last)) => ObjectLayout {
                total_cells,
                pointer_cells_start: first,
                pointer_cells_end: last + 1,
            },
            None => ObjectLayout::opaque(total_cells),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_fixnum_is_immediate() {
        let w = Word::fixnum(42);
        assert!(matches!(LispLayout::classify(w.raw()), WordKind::Immediate));
    }

    #[test]
    fn classify_nil_is_immediate() {
        assert!(matches!(LispLayout::classify(Word::NIL.raw()), WordKind::Immediate));
    }

    #[test]
    fn classify_t_is_immediate() {
        assert!(matches!(LispLayout::classify(Word::T.raw()), WordKind::Immediate));
    }

    #[test]
    fn classify_cons_pointer_is_pointer_cons() {
        let dummy: u64 = 0;
        let p = &dummy as *const u64 as *const u8;
        let w = Word::from_ptr(p, Tag::Cons);
        match LispLayout::classify(w.raw()) {
            WordKind::PointerCons(addr) => assert_eq!(addr, p),
            other => panic!("expected PointerCons, got {other:?}"),
        }
    }

    #[test]
    fn classify_vector_pointer_is_pointer_header() {
        let dummy: u64 = 0;
        let p = &dummy as *const u64 as *const u8;
        let w = Word::from_ptr(p, Tag::Vector);
        match LispLayout::classify(w.raw()) {
            WordKind::PointerHeader(addr) => assert_eq!(addr, p),
            other => panic!("expected PointerHeader, got {other:?}"),
        }
    }

    #[test]
    fn classify_function_and_symbol_and_string_are_pointer_header() {
        let dummy: u64 = 0;
        let p = &dummy as *const u64 as *const u8;
        for tag in [Tag::Function, Tag::Symbol, Tag::String] {
            let w = Word::from_ptr(p, tag);
            assert!(
                matches!(LispLayout::classify(w.raw()), WordKind::PointerHeader(_)),
                "expected PointerHeader for {tag:?}"
            );
        }
    }

    #[test]
    fn classify_forward_is_forwarded() {
        // Forward tag pointer must be 8-byte aligned. Construct a
        // dummy on the stack and convert to ()-pointer.
        let dummy: u64 = 0;
        let p = (&dummy as *const u64 as *const u8).wrapping_offset(8);
        let raw = LispLayout::make_forward(p);
        match LispLayout::classify(raw) {
            WordKind::Forwarded(addr) => assert_eq!(addr, p),
            other => panic!("expected Forwarded, got {other:?}"),
        }
    }

    #[test]
    fn fill_word_is_nil() {
        assert_eq!(LispLayout::FILL_WORD, Word::NIL.raw());
        assert!(matches!(
            LispLayout::classify(LispLayout::FILL_WORD),
            WordKind::Immediate
        ));
    }

    #[test]
    fn make_pointer_roundtrips_for_both_kinds() {
        let dummy: u64 = 0;
        let p = &dummy as *const u64 as *const u8;
        let cons_raw = LispLayout::make_pointer(p, PointerKind::Cons);
        let header_raw = LispLayout::make_pointer(p, PointerKind::Header);
        match LispLayout::classify(cons_raw) {
            WordKind::PointerCons(a) => assert_eq!(a, p),
            other => panic!("expected PointerCons, got {other:?}"),
        }
        match LispLayout::classify(header_raw) {
            WordKind::PointerHeader(a) => assert_eq!(a, p),
            other => panic!("expected PointerHeader, got {other:?}"),
        }
    }

    #[test]
    fn header_layout_vector_returns_full_payload_range() {
        let h = HeapHeader::new(HeapType::Vector, 5);
        let raw = h.raw();
        let cell_ptr = &raw as *const u64;
        let layout = unsafe { LispLayout::header_layout(cell_ptr) };
        assert_eq!(layout.total_cells, 6);
        assert_eq!(layout.pointer_cells_start, 1);
        assert_eq!(layout.pointer_cells_end, 6);
    }

    #[test]
    fn header_layout_string_returns_opaque() {
        let h = HeapHeader::new(HeapType::String, 10);
        let raw = h.raw();
        let cell_ptr = &raw as *const u64;
        let layout = unsafe { LispLayout::header_layout(cell_ptr) };
        assert_eq!(layout.total_cells, 11);
        assert_eq!(layout.pointer_cell_count(), 0);
    }

    #[test]
    fn header_layout_function_skips_code_ptr_and_arity() {
        // Function payload: cells 1, 2 are non-Word (code_ptr, arity);
        // cells 3, 4 are Word-typed (env, name). word_field_range
        // returns (3, 4) inclusive.
        let h = HeapHeader::new(HeapType::Function, 6);
        let raw = h.raw();
        let cell_ptr = &raw as *const u64;
        let layout = unsafe { LispLayout::header_layout(cell_ptr) };
        assert_eq!(layout.total_cells, 7);
        assert_eq!(layout.pointer_cells_start, 3);
        assert_eq!(layout.pointer_cells_end, 5);
    }
}
