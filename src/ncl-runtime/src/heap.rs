//! Heap, header, and bump allocator.
//!
//! See `docs/GC.md`. This module is GC step 2: a fixed-capacity
//! semispace with a bump allocator and the canonical 8-byte header
//! word that sits in front of every non-cons heap object. No
//! collection yet — exhaustion panics. Step 3 adds the second
//! semispace and forwarding-pointer copy; step 5 adds the card
//! table.
//!
//! Cons cells are headerless (16 bytes, just car and cdr) per the
//! design. Everything else carries one `HeapHeader` cell in front
//! of its payload.

use std::ptr::NonNull;

use crate::word::{Tag, Word};

// -- HeapHeader --------------------------------------------------------------
//
// One 8-byte cell sitting in front of every non-cons heap object.
//
//   0..5     type           5 bits   (HeapType — 32 codes available)
//   5..29   length cells   24 bits   (payload cells, not including header)
//   29..37  gc bits         8 bits   (mark, age, pinned, has-finalizer, ...)
//   37..64  reserved       27 bits   (future: extended length, class ptr, ...)

const TYPE_SHIFT: u32 = 0;
const TYPE_BITS: u32 = 5;
const TYPE_MASK: u64 = (1 << TYPE_BITS) - 1;

const LEN_SHIFT: u32 = TYPE_SHIFT + TYPE_BITS;
const LEN_BITS: u32 = 24;
const LEN_MASK: u64 = (1 << LEN_BITS) - 1;

const GC_SHIFT: u32 = LEN_SHIFT + LEN_BITS;
const GC_BITS: u32 = 8;
const GC_MASK: u64 = (1 << GC_BITS) - 1;

/// Maximum length (in cells) of a single heap object's payload.
/// 16 MiB-1 cells = 128 MiB. Bigger objects need a different scheme.
pub const MAX_OBJECT_CELLS: u32 = (1 << LEN_BITS) - 1;

/// Type codes carried in a header. The numeric values are stable —
/// never reorder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum HeapType {
    /// Symbols.
    Symbol = 0,
    /// Simple vectors.
    Vector = 1,
    /// Functions / closures.
    Function = 2,
    /// Strings.
    String = 3,
    /// Foreign-declaration block (the `#!...!#` reader output).
    FfiBlock = 4,
    /// Catch-all for typed objects we haven't grown a dedicated code
    /// for yet. Discriminant of the actual type lives in payload[0].
    Other = 5,
}

impl HeapType {
    pub fn from_bits(bits: u8) -> Option<HeapType> {
        match bits {
            0 => Some(HeapType::Symbol),
            1 => Some(HeapType::Vector),
            2 => Some(HeapType::Function),
            3 => Some(HeapType::String),
            4 => Some(HeapType::FfiBlock),
            5 => Some(HeapType::Other),
            _ => None,
        }
    }
}

/// 8-byte header. All bit fiddling is encapsulated; callers ask for
/// type, length, and gc bits with named methods.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct HeapHeader(u64);

impl HeapHeader {
    pub fn new(ty: HeapType, length_cells: u32) -> HeapHeader {
        debug_assert!(
            length_cells <= MAX_OBJECT_CELLS,
            "object length {length_cells} exceeds {MAX_OBJECT_CELLS}",
        );
        let bits = ((ty as u64) << TYPE_SHIFT)
            | (((length_cells as u64) & LEN_MASK) << LEN_SHIFT);
        HeapHeader(bits)
    }

    pub fn raw(self) -> u64 { self.0 }
    pub fn from_raw(bits: u64) -> HeapHeader { HeapHeader(bits) }

    pub fn ty(self) -> HeapType {
        let code = ((self.0 >> TYPE_SHIFT) & TYPE_MASK) as u8;
        HeapType::from_bits(code).expect("invalid header type code")
    }

    pub fn length_cells(self) -> u32 {
        ((self.0 >> LEN_SHIFT) & LEN_MASK) as u32
    }

    pub fn gc_bits(self) -> u8 {
        ((self.0 >> GC_SHIFT) & GC_MASK) as u8
    }

    pub fn set_gc_bit(&mut self, bit: GcBit) {
        self.0 |= (bit as u64) << GC_SHIFT;
    }

    pub fn clear_gc_bit(&mut self, bit: GcBit) {
        self.0 &= !((bit as u64) << GC_SHIFT);
    }

    pub fn has_gc_bit(self, bit: GcBit) -> bool {
        (self.0 >> GC_SHIFT) & (bit as u64) != 0
    }
}

impl std::fmt::Debug for HeapHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeapHeader")
            .field("ty", &self.ty())
            .field("length_cells", &self.length_cells())
            .field("gc_bits", &format_args!("{:#010b}", self.gc_bits()))
            .finish()
    }
}

/// GC bit slots. Each is a single bit position within the 8-bit
/// gc-bits region of a HeapHeader. Add new bits by giving them their
/// own constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum GcBit {
    /// "I have been visited / copied this cycle." Cleared by the
    /// scanner after each pass.
    Mark = 0b0000_0001,
    /// "I have been promoted to old generation." Set when the
    /// minor GC copies an object out of young.
    Tenured = 0b0000_0010,
    /// "I am pinned — do not move me." Used for objects that
    /// conservative roots reach.
    Pinned = 0b0000_0100,
}

// -- Heap (single semispace, bump allocator) --------------------------------

/// One semispace, fixed capacity, bump-allocated.
///
/// Storage is `Vec<u64>` cells (8 bytes each) so every allocation is
/// 8-byte aligned by construction. The bump pointer grows upward
/// from cell 0.
pub struct Heap {
    cells: Box<[u64]>,
    /// Cell index of the next free slot.
    top: usize,
}

impl Heap {
    /// Construct a heap with capacity for `size_bytes / 8` cells.
    /// Sizes are rounded down to a multiple of 8.
    pub fn new(size_bytes: usize) -> Heap {
        let n_cells = size_bytes / 8;
        let cells = vec![0u64; n_cells].into_boxed_slice();
        Heap { cells, top: 0 }
    }

    pub fn capacity_cells(&self) -> usize { self.cells.len() }
    pub fn capacity_bytes(&self) -> usize { self.cells.len() * 8 }
    pub fn used_cells(&self) -> usize { self.top }
    pub fn used_bytes(&self) -> usize { self.top * 8 }
    pub fn free_cells(&self) -> usize { self.cells.len() - self.top }
    pub fn free_bytes(&self) -> usize { self.free_cells() * 8 }

    /// Reset the bump pointer. The backing storage is NOT zeroed —
    /// callers who care must zero the cells they're about to use.
    /// Used by the GC after copying live objects to the other space.
    pub fn reset(&mut self) {
        self.top = 0;
    }

    /// Pointer-into-this-heap test. Used by the GC to classify
    /// pointers as young / old / static.
    pub fn contains_ptr(&self, ptr: *const u8) -> bool {
        let base = self.cells.as_ptr() as usize;
        let end = base + self.cells.len() * 8;
        let p = ptr as usize;
        p >= base && p < end
    }

    /// Allocate `cells` raw 8-byte cells. The returned pointer is
    /// 8-byte aligned and points at the first cell. Panics on
    /// exhaustion (step 2 — collection lands later).
    pub fn alloc_cells(&mut self, cells: usize) -> NonNull<u64> {
        if self.top + cells > self.cells.len() {
            panic!(
                "heap exhausted: requested {cells} cells, have {} free of {} total",
                self.cells.len() - self.top,
                self.cells.len(),
            );
        }
        // SAFETY: bounds checked above; pointer arithmetic stays
        // within the boxed slice.
        let p = unsafe { self.cells.as_mut_ptr().add(self.top) };
        self.top += cells;
        // SAFETY: as_mut_ptr() never returns null for a non-empty Box.
        // For an empty heap (size_bytes < 8) we'd have panicked above.
        unsafe { NonNull::new_unchecked(p) }
    }

    /// Allocate a cons cell. Returns a tagged Cons `Word` ready to
    /// store anywhere. Cons cells are headerless — just two raw
    /// `Word` slots — which is the design's space-saving choice for
    /// the dominant heap object.
    pub fn alloc_cons(&mut self, car: Word, cdr: Word) -> Word {
        let p = self.alloc_cells(2);
        // SAFETY: alloc_cells returned 2 cells of valid storage.
        unsafe {
            *p.as_ptr() = car.raw();
            *p.as_ptr().add(1) = cdr.raw();
        }
        Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
    }

    /// Allocate a header + `length_cells` payload cells. Caller must
    /// fill in the payload before exposing the pointer; the header is
    /// written here with the requested type and length, and zero gc
    /// bits.
    pub fn alloc_with_header(
        &mut self,
        ty: HeapType,
        length_cells: u32,
    ) -> NonNull<HeapHeader> {
        let total = 1 + length_cells as usize;
        let p = self.alloc_cells(total);
        // SAFETY: alloc_cells returned 1 + length_cells cells. We
        // write the header into cell 0; the caller is responsible
        // for the payload cells.
        unsafe {
            *p.as_ptr() = HeapHeader::new(ty, length_cells).raw();
        }
        // SAFETY: alignment of u64 cell suffices for HeapHeader
        // (transparent over u64).
        unsafe { NonNull::new_unchecked(p.as_ptr() as *mut HeapHeader) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = HeapHeader::new(HeapType::Symbol, 8);
        assert_eq!(h.ty(), HeapType::Symbol);
        assert_eq!(h.length_cells(), 8);
        assert_eq!(h.gc_bits(), 0);
    }

    #[test]
    fn header_all_types() {
        for ty in [
            HeapType::Symbol,
            HeapType::Vector,
            HeapType::Function,
            HeapType::String,
            HeapType::FfiBlock,
            HeapType::Other,
        ] {
            let h = HeapHeader::new(ty, 1);
            assert_eq!(h.ty(), ty);
            assert_eq!(h.length_cells(), 1);
        }
    }

    #[test]
    fn header_max_length() {
        let h = HeapHeader::new(HeapType::Vector, MAX_OBJECT_CELLS);
        assert_eq!(h.length_cells(), MAX_OBJECT_CELLS);
    }

    #[test]
    fn gc_bits_independent() {
        let mut h = HeapHeader::new(HeapType::Symbol, 4);
        assert!(!h.has_gc_bit(GcBit::Mark));
        assert!(!h.has_gc_bit(GcBit::Tenured));

        h.set_gc_bit(GcBit::Mark);
        assert!(h.has_gc_bit(GcBit::Mark));
        assert!(!h.has_gc_bit(GcBit::Tenured));
        // Type and length unaffected.
        assert_eq!(h.ty(), HeapType::Symbol);
        assert_eq!(h.length_cells(), 4);

        h.set_gc_bit(GcBit::Tenured);
        assert!(h.has_gc_bit(GcBit::Mark));
        assert!(h.has_gc_bit(GcBit::Tenured));

        h.clear_gc_bit(GcBit::Mark);
        assert!(!h.has_gc_bit(GcBit::Mark));
        assert!(h.has_gc_bit(GcBit::Tenured));
    }

    #[test]
    fn header_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<HeapHeader>(), 8);
        assert_eq!(std::mem::align_of::<HeapHeader>(), 8);
    }

    #[test]
    fn empty_heap_starts_at_zero() {
        let h = Heap::new(1024);
        assert_eq!(h.used_bytes(), 0);
        assert_eq!(h.free_bytes(), 1024);
        assert_eq!(h.capacity_bytes(), 1024);
    }

    #[test]
    fn cons_round_trip() {
        let mut h = Heap::new(1024);
        let c = h.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        assert!(c.is_cons());
        let p = c.as_ptr::<u64>(Tag::Cons).expect("cons ptr");
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*p.add(1)).as_fixnum(), Some(2));
        }
        assert_eq!(h.used_bytes(), 16);
    }

    #[test]
    fn allocations_are_disjoint() {
        let mut h = Heap::new(1024);
        let a = h.alloc_cons(Word::NIL, Word::NIL);
        let b = h.alloc_cons(Word::T, Word::T);
        let pa = a.as_ptr::<u8>(Tag::Cons).unwrap() as usize;
        let pb = b.as_ptr::<u8>(Tag::Cons).unwrap() as usize;
        assert_eq!(pb - pa, 16, "second cons should sit immediately after first");
    }

    #[test]
    fn header_alloc_writes_header_and_reserves_payload() {
        let mut h = Heap::new(1024);
        let p = h.alloc_with_header(HeapType::Symbol, 7);
        let header = unsafe { *p.as_ptr() };
        assert_eq!(header.ty(), HeapType::Symbol);
        assert_eq!(header.length_cells(), 7);
        // 1 header cell + 7 payload cells = 8 cells = 64 bytes.
        assert_eq!(h.used_bytes(), 64);
    }

    #[test]
    fn pointer_classification() {
        let mut h = Heap::new(1024);
        let c = h.alloc_cons(Word::NIL, Word::NIL);
        let p = c.as_ptr::<u8>(Tag::Cons).unwrap();
        assert!(h.contains_ptr(p));
        // A pointer to the heap struct itself isn't in the heap.
        assert!(!h.contains_ptr(&h as *const Heap as *const u8));
    }

    #[test]
    fn reset_lets_us_re_allocate() {
        let mut h = Heap::new(64); // 8 cells
        let _ = h.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        let _ = h.alloc_cons(Word::fixnum(3), Word::fixnum(4));
        // 4 cells used, 4 left. A 5-cell alloc would fail.
        assert_eq!(h.used_cells(), 4);
        h.reset();
        assert_eq!(h.used_cells(), 0);
        // Now 8 cells free again.
        let _ = h.alloc_with_header(HeapType::Vector, 7);
        assert_eq!(h.used_cells(), 8);
    }

    #[test]
    #[should_panic(expected = "heap exhausted")]
    fn exhaustion_panics() {
        let mut h = Heap::new(16); // exactly 2 cells
        let _ = h.alloc_cons(Word::NIL, Word::NIL); // fills it
        let _ = h.alloc_cons(Word::NIL, Word::NIL); // boom
    }

    #[test]
    fn pointers_are_eight_byte_aligned() {
        let mut h = Heap::new(1024);
        for _ in 0..10 {
            let c = h.alloc_cons(Word::NIL, Word::NIL);
            let addr = c.as_ptr::<u8>(Tag::Cons).unwrap() as usize;
            assert_eq!(addr & 7, 0, "address {addr:#x} not 8-byte aligned");
        }
    }
}
