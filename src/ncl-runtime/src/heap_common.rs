//! Heap types shared between GC backends.
//!
//! Sub-phase 6.5 of `docs/GC_DESIGN.md`. Extracted from `heap.rs`
//! so the page-based heap can reach for them without going through
//! semispace-specific machinery, and so sub-phase 12 can delete
//! `heap.rs`'s body without losing the shared scaffolding.
//!
//! What lives here:
//!   - `CardTable` — soft-card write-barrier bookkeeping.
//!   - `HeapHeader` — the 8-byte word every header-bearing heap
//!     object carries as cell 0.
//!   - `HeapType` — the type tag inside the header.
//!   - `GcBit` — per-object GC flag bits packed into the header.
//!   - `MAX_OBJECT_CELLS` — declared object-length cap.
//!   - `CARD_SIZE_BYTES`, `CARD_SIZE_CELLS` — card-table geometry.
//!   - `StartBits` — alias for the lock-free start-bit bitmap Arc.
//!
//! `heap.rs` re-exports everything in this module via `pub use
//! crate::heap_common::*;` so existing imports like
//! `use crate::heap::HeapHeader` keep compiling.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

// -- Card table --------------------------------------------------------------
//
// Software card-marking for the write barrier. One byte per
// CARD_SIZE_BYTES of heap storage. Mutators write to it lock-free
// via atomic byte stores; the GC reads it during minor cycles to
// scan only regions known to (possibly) hold young pointers.
//
// False positives are fine — copy_into filters non-pointers and
// pointers that aren't into young. False negatives (an unmarked
// card containing a young pointer) are NOT fine; the discipline is
// every old-heap store must mark.

pub const CARD_SIZE_BYTES: usize = 512;
pub const CARD_SIZE_CELLS: usize = CARD_SIZE_BYTES / 8;

pub struct CardTable {
    bytes: Box<[AtomicU8]>,
}

impl CardTable {
    pub fn new(coverage_bytes: usize) -> CardTable {
        let n_cards = coverage_bytes.div_ceil(CARD_SIZE_BYTES);
        let v: Vec<AtomicU8> = (0..n_cards).map(|_| AtomicU8::new(0)).collect();
        CardTable { bytes: v.into_boxed_slice() }
    }

    pub fn n_cards(&self) -> usize { self.bytes.len() }

    /// Mark the card containing the given byte offset (relative to
    /// the start of the covered region). Lock-free, single byte store.
    pub fn mark_offset(&self, byte_offset: usize) {
        let card = byte_offset / CARD_SIZE_BYTES;
        if let Some(b) = self.bytes.get(card) {
            b.store(1, Ordering::Relaxed);
        }
    }

    pub fn is_dirty(&self, card: usize) -> bool {
        self.bytes.get(card).is_some_and(|b| b.load(Ordering::Relaxed) != 0)
    }

    pub fn clear(&self, card: usize) {
        if let Some(b) = self.bytes.get(card) {
            b.store(0, Ordering::Relaxed);
        }
    }

    pub fn clear_all(&self) {
        for b in self.bytes.iter() {
            b.store(0, Ordering::Relaxed);
        }
    }

    /// Count dirty cards. Useful for tests and for diagnostics.
    pub fn dirty_count(&self) -> usize {
        self.bytes.iter().filter(|b| b.load(Ordering::Relaxed) != 0).count()
    }
}

// -- HeapHeader --------------------------------------------------------------

// `pub(crate)` so `heap.rs`'s linear walker (which decodes type and
// gc bits manually for diagnostic dumps) can still see them after
// the move. The previously-private status was an organisational
// accident, not a design choice — these constants belong wherever
// HeapHeader lives.
pub(crate) const TYPE_SHIFT: u32 = 0;
pub(crate) const TYPE_BITS: u32 = 5;
pub(crate) const TYPE_MASK: u64 = (1 << TYPE_BITS) - 1;

pub(crate) const LEN_SHIFT: u32 = TYPE_SHIFT + TYPE_BITS;
pub(crate) const LEN_BITS: u32 = 24;
pub(crate) const LEN_MASK: u64 = (1 << LEN_BITS) - 1;

pub(crate) const GC_SHIFT: u32 = LEN_SHIFT + LEN_BITS;
pub(crate) const GC_BITS: u32 = 8;
pub(crate) const GC_MASK: u64 = (1 << GC_BITS) - 1;

pub const MAX_OBJECT_CELLS: u32 = (1 << LEN_BITS) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum HeapType {
    Symbol = 0,
    Vector = 1,
    Function = 2,
    String = 3,
    FfiBlock = 4,
    Other = 5,
    /// Arbitrary-precision integer. Layout under `bignum.rs`:
    ///   cell 1: %BIGNUM marker symbol
    ///   cell 2: sign (fixnum +1 or -1)
    ///   cell 3: n_limbs (fixnum)
    ///   cell 4: reserved (cached fixnum-equivalent / hash)
    ///   cell 5..5+n_limbs: raw u64 limbs, little-endian
    /// GC scans cells 0..=4 (header + boxed values), skips the
    /// limb data — same shape as FfiBlock but with the bignum
    /// marker for printer / typep recognition.
    Bignum = 6,
    /// IEEE 754 double-precision float. Layout under `float.rs`:
    ///   cell 1: %FLOAT marker symbol
    ///   cell 2: raw f64 bits (transmute, not a Word)
    /// 2-cell payload (3 with header). GC scans the marker as a
    /// Word; the f64 bits are opaque (probabilistic-correctness
    /// is fine, same as bignum limbs).
    Float = 7,
    /// Exact rational. Layout under `ratio.rs`:
    ///   cell 1: %RATIO marker symbol
    ///   cell 2: numerator (Word — fixnum or bignum)
    ///   cell 3: denominator (Word — fixnum or bignum, always > 1
    ///           because we simplify and demote on construction)
    /// 3-cell payload (4 with header). Both num and den ARE Words,
    /// so the GC scan path treats them as live pointers naturally.
    Ratio = 8,
    /// Complex number. Layout under `complex.rs`:
    ///   cell 1: %COMPLEX marker symbol
    ///   cell 2: real part (Word — any real-number subtype)
    ///   cell 3: imaginary part (Word — any real-number subtype,
    ///           guaranteed non-zero after canonicalisation —
    ///           imag-zero would demote to the real part)
    /// 3-cell payload, identical shape to Ratio.
    Complex = 9,
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
            6 => Some(HeapType::Bignum),
            7 => Some(HeapType::Float),
            8 => Some(HeapType::Ratio),
            9 => Some(HeapType::Complex),
            _ => None,
        }
    }

    /// Range of cell offsets within an object of this type that
    /// hold Word-valued fields the GC must scan. Offsets are
    /// relative to the object's first cell (the header is at
    /// offset 0; payload starts at offset 1).
    ///
    /// Returns `None` if there are no Word-bearing cells (string
    /// data, raw float bits, raw bignum limbs, FFI scratch).
    ///
    /// Otherwise returns `Some((first, last))` where both bounds
    /// are inclusive cell offsets. The walker iterates
    /// `cell_idx + first ..= cell_idx + last`.
    ///
    /// The pre-typed walker treated every payload cell as a
    /// candidate Word, which:
    ///   - wasted dispatch on raw u64 fields (code_ptr in Function,
    ///     f64 bits in Float, limbs in Bignum, char data in String,
    ///     raw bytes in FfiBlock);
    ///   - risked false-positive matches when a raw u64 happened to
    ///     have heap-pointer-shaped tag bits — the downstream
    ///     reservation + start-bit gates always rejected those, but
    ///     the work to get there is the slow part.
    ///
    /// For `length_cells` = the value in the header (excludes the
    /// header itself), this returns:
    ///
    /// | type          | Word-cell range | non-Word cells              |
    /// |---------------|-----------------|------------------------------|
    /// | Vector        | 1..=length      | —                            |
    /// | Symbol        | 1..=length (7)  | —                            |
    /// | Ratio         | 1..=length (3)  | —                            |
    /// | Complex       | 1..=length (3)  | —                            |
    /// | Other         | 1..=length      | (treated conservatively)     |
    /// | Function      | 3..=4           | code_ptr (1), arity (2)      |
    /// | Bignum        | 1..=4           | limbs (5..length)            |
    /// | Float         | 1..=1           | f64 bits (2)                 |
    /// | String        | None            | UTF-8 / char data            |
    /// | FfiBlock      | None            | foreign declaration bytes    |
    pub fn word_field_range(self, length_cells: u32) -> Option<(usize, usize)> {
        let len = length_cells as usize;
        match self {
            HeapType::Vector
            | HeapType::Symbol
            | HeapType::Ratio
            | HeapType::Complex
            | HeapType::Other => {
                if len == 0 { None } else { Some((1, len)) }
            }
            HeapType::Function => Some((3, 4)),
            HeapType::Bignum => Some((1, 4)),
            HeapType::Float => Some((1, 1)),
            HeapType::String | HeapType::FfiBlock => None,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct HeapHeader(u64);

impl HeapHeader {
    pub fn new(ty: HeapType, length_cells: u32) -> HeapHeader {
        debug_assert!(length_cells <= MAX_OBJECT_CELLS);
        let bits = ((ty as u64) << TYPE_SHIFT)
            | (((length_cells as u64) & LEN_MASK) << LEN_SHIFT);
        HeapHeader(bits)
    }
    pub fn raw(self) -> u64 { self.0 }
    pub fn from_raw(bits: u64) -> HeapHeader { HeapHeader(bits) }
    pub fn ty(self) -> HeapType {
        HeapType::from_bits(((self.0 >> TYPE_SHIFT) & TYPE_MASK) as u8)
            .expect("invalid header type")
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum GcBit {
    Mark = 0b0000_0001,
    Tenured = 0b0000_0010,
    Pinned = 0b0000_0100,
}

// -- Start bits --------------------------------------------------------------
//
// Lock-free Arc<[AtomicU64]> alias. Both semispace and page-heap
// allocators flip start-bit pairs on this bitmap via the same
// fetch_or atomic pattern. The bitmap encoding (`01` for boxed
// start, `11` for cons start, packed 32 cells per u64) is shared
// — though semispace's bitmap covers `young` only, and the page
// heap's covers the whole reservation.

pub type StartBits = Arc<[AtomicU64]>;
