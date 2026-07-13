//! `TinyLayout` — a minimal `HeapLayout` impl independent of NCL.
//!
//! Purpose: prove the `HeapLayout` trait surface is sufficient to
//! express a SECOND, structurally different language binding —
//! validating that the trait is genuinely polymorphic, not just
//! "the LispLayout type signature rephrased."
//!
//! Differences from `LispLayout`:
//!   - **2-bit tag** (vs Lisp's 3-bit). Tag bits 0..2 of every cell:
//!     - `00` = immediate (62-bit signed fixnum payload)
//!     - `01` = cons-shaped pointer
//!     - `10` = header-bearing pointer
//!     - `11` = forwarding marker
//!   - **Headers carry length only**, no type tag. The full payload
//!     is always pointer-typed (no opaque cells like Lisp's String
//!     or FfiBlock). Header value = payload-cell count.
//!   - **All non-zero immediates are fixnums.** No NIL, T, characters,
//!     subtypes. The language has fixnums and heap objects, period.
//!
//! This is a faithful but minimal alternative — equivalent in
//! expressiveness to e.g. an early-stage research VM.

use crate::traits::{HeapLayout, ObjectLayout, PointerKind, WordKind};

/// Number of tag bits in a TinyLayout word.
pub const TINY_TAG_BITS: u32 = 2;
/// Mask for the tag bits.
pub const TINY_TAG_MASK: u64 = 0b11;
/// Mask for the payload (everything but the tag bits).
pub const TINY_PAYLOAD_MASK: u64 = !TINY_TAG_MASK;

/// Tag for an immediate value (low 2 bits = 00).
pub const TINY_TAG_IMMEDIATE: u64 = 0b00;
/// Tag for a pointer to a cons-shaped (2-cell, header-less) object.
pub const TINY_TAG_CONS: u64 = 0b01;
/// Tag for a pointer to a header-bearing object.
pub const TINY_TAG_HEADER: u64 = 0b10;
/// Tag for a forwarding marker.
pub const TINY_TAG_FORWARD: u64 = 0b11;

/// Construct a fixnum from a signed integer.
/// Panics in debug if `n` exceeds 62-bit signed range.
pub const fn tiny_fixnum(n: i64) -> u64 {
    debug_assert!(n >= -(1 << 61) && n < (1 << 61));
    (n as u64) << TINY_TAG_BITS
}

/// Construct a cons-shaped heap pointer. The address must be
/// 4-byte aligned (low 2 bits = 0). 8-byte alignment of GC heap
/// pointers satisfies this trivially.
pub fn tiny_cons_ptr(addr: *const u8) -> u64 {
    debug_assert!((addr as u64) & TINY_TAG_MASK == 0);
    (addr as u64) | TINY_TAG_CONS
}

/// Construct a header-bearing heap pointer.
pub fn tiny_header_ptr(addr: *const u8) -> u64 {
    debug_assert!((addr as u64) & TINY_TAG_MASK == 0);
    (addr as u64) | TINY_TAG_HEADER
}

/// Construct a forwarding marker pointing at `new_addr`.
pub fn tiny_forward(new_addr: *const u8) -> u64 {
    debug_assert!((new_addr as u64) & TINY_TAG_MASK == 0);
    (new_addr as u64) | TINY_TAG_FORWARD
}

/// Construct a header cell for a boxed object holding `payload_cells`
/// pointer-typed payload slots. The header itself occupies one cell;
/// the payload follows.
pub const fn tiny_header(payload_cells: u32) -> u64 {
    // Header values are raw u32 counts; they're not interpreted as
    // tagged Words. `classify` is only called on cells the GC
    // believes are pointer slots (via the start-bit dispatch), so
    // header bytes are never misread as immediates.
    payload_cells as u64
}

/// The minimal layout binding. Zero-sized marker type.
#[derive(Copy, Clone, Debug, Default)]
pub struct TinyLayout;

impl HeapLayout for TinyLayout {
    /// Fill cells with immediate zero. A `0u64` decodes as
    /// `WordKind::Immediate` under TinyLayout's tag scheme — safer
    /// than leaving uninitialised memory the scanner could misread
    /// as a stale pointer.
    const FILL_WORD: u64 = 0;

    #[inline(always)]
    fn classify(raw: u64) -> WordKind {
        let addr = (raw & TINY_PAYLOAD_MASK) as *const u8;
        match raw & TINY_TAG_MASK {
            TINY_TAG_IMMEDIATE => WordKind::Immediate,
            TINY_TAG_CONS => WordKind::PointerCons(addr),
            TINY_TAG_HEADER => WordKind::PointerHeader(addr),
            TINY_TAG_FORWARD => WordKind::Forwarded(addr),
            _ => unreachable!("tag is 2 bits"),
        }
    }

    #[inline(always)]
    fn make_forward(new_addr: *const u8) -> u64 {
        tiny_forward(new_addr)
    }

    #[inline(always)]
    fn make_pointer(addr: *const u8, kind: PointerKind) -> u64 {
        match kind {
            PointerKind::Cons => tiny_cons_ptr(addr),
            PointerKind::Header => tiny_header_ptr(addr),
        }
    }

    #[inline(always)]
    fn rewrite_pointer_addr(old_raw: u64, new_addr: *const u8) -> u64 {
        let tag = old_raw & TINY_TAG_MASK;
        (new_addr as u64) | tag
    }

    #[inline(always)]
    unsafe fn header_layout(header_cell: *const u64) -> ObjectLayout {
        let payload_cells = unsafe { *header_cell } as usize;
        // Every payload cell is pointer-typed under TinyLayout.
        ObjectLayout::all_pointers(1 + payload_cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixnum_round_trip() {
        for n in [0_i64, 1, -1, 42, -42, 1 << 30, -(1 << 30)] {
            let raw = tiny_fixnum(n);
            assert!(matches!(TinyLayout::classify(raw), WordKind::Immediate));
            // Decode: shift right 2, sign-extended.
            assert_eq!((raw as i64) >> TINY_TAG_BITS, n);
        }
    }

    #[test]
    fn classify_zero_is_immediate() {
        // Fill-word case: a 0u64 is the "no value" immediate.
        assert!(matches!(TinyLayout::classify(0), WordKind::Immediate));
    }

    #[test]
    fn classify_cons_pointer() {
        let dummy: u64 = 0;
        let p = &dummy as *const u64 as *const u8;
        let raw = tiny_cons_ptr(p);
        match TinyLayout::classify(raw) {
            WordKind::PointerCons(addr) => assert_eq!(addr, p),
            other => panic!("expected PointerCons, got {other:?}"),
        }
    }

    #[test]
    fn classify_header_pointer() {
        let dummy: u64 = 0;
        let p = &dummy as *const u64 as *const u8;
        let raw = tiny_header_ptr(p);
        match TinyLayout::classify(raw) {
            WordKind::PointerHeader(addr) => assert_eq!(addr, p),
            other => panic!("expected PointerHeader, got {other:?}"),
        }
    }

    #[test]
    fn classify_forwarding_marker() {
        let dummy: u64 = 0;
        let p = (&dummy as *const u64 as *const u8).wrapping_offset(8);
        let raw = TinyLayout::make_forward(p);
        match TinyLayout::classify(raw) {
            WordKind::Forwarded(addr) => assert_eq!(addr, p),
            other => panic!("expected Forwarded, got {other:?}"),
        }
    }

    #[test]
    fn make_pointer_roundtrip_both_kinds() {
        let dummy: u64 = 0;
        let p = &dummy as *const u64 as *const u8;
        for kind in [PointerKind::Cons, PointerKind::Header] {
            let raw = TinyLayout::make_pointer(p, kind);
            let classified = TinyLayout::classify(raw);
            match (kind, classified) {
                (PointerKind::Cons, WordKind::PointerCons(a)) => {
                    assert_eq!(a, p)
                }
                (PointerKind::Header, WordKind::PointerHeader(a)) => {
                    assert_eq!(a, p)
                }
                (k, c) => {
                    panic!("kind {k:?} classified as {c:?}")
                }
            }
        }
    }

    #[test]
    fn rewrite_preserves_tag() {
        let old_addr: u64 = 0x1000;
        let new_addr: u64 = 0x2000;
        for &tag in &[
            TINY_TAG_CONS,
            TINY_TAG_HEADER,
        ] {
            let old_raw = old_addr | tag;
            let new_raw =
                TinyLayout::rewrite_pointer_addr(old_raw, new_addr as *const u8);
            assert_eq!(new_raw & TINY_TAG_MASK, tag);
            assert_eq!(new_raw & TINY_PAYLOAD_MASK, new_addr);
        }
    }

    #[test]
    fn fill_word_classifies_as_immediate() {
        assert!(matches!(
            TinyLayout::classify(TinyLayout::FILL_WORD),
            WordKind::Immediate
        ));
    }

    #[test]
    fn header_layout_returns_correct_total() {
        let header = tiny_header(5);
        let cell_ptr = &header as *const u64;
        let layout = unsafe { TinyLayout::header_layout(cell_ptr) };
        assert_eq!(layout.total_cells, 6);
        assert_eq!(layout.pointer_cells_start, 1);
        assert_eq!(layout.pointer_cells_end, 6);
    }

    #[test]
    fn header_layout_zero_payload() {
        let header = tiny_header(0);
        let cell_ptr = &header as *const u64;
        let layout = unsafe { TinyLayout::header_layout(cell_ptr) };
        assert_eq!(layout.total_cells, 1);
        assert_eq!(layout.pointer_cell_count(), 0);
    }
}
