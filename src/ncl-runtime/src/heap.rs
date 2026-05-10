//! Heap, header, semispace allocator, and Cheney-style copying GC.
//!
//! See `docs/GC.md`. This module is GC steps 2 and 3:
//!   - the canonical 8-byte header word for non-cons heap objects,
//!   - a fixed-capacity `Semispace` with a bump allocator,
//!   - a `Heap` that owns from/to semispaces and runs Cheney's
//!     copy on `collect()`.
//!
//! Cons cells are headerless (two raw `Word` slots) per the design.
//! Everything else carries one `HeapHeader` cell in front of its
//! payload.
//!
//! Step 3 limitation: every heap object is treated as a payload of
//! `Word`s. Strings (UTF-8 bytes) and other types with non-Word
//! payloads will need per-type scan functions; that's a small refactor
//! when we add them.
//!
//! Cheney's two-pointer algorithm in this implementation: instead of
//! discriminating cons vs header'd by inspecting to-space bytes (which
//! would be ambiguous — a cons's car-word can look like a header), we
//! maintain a parallel queue recording `(to-space offset, size, is_cons)`
//! for every copied object. The scan phase walks that queue.

use std::ptr::NonNull;

use crate::word::{Tag, Word};

// -- HeapHeader --------------------------------------------------------------
//
//   0..5    type           5 bits   (HeapType — 32 codes available)
//   5..29   length cells  24 bits   (payload, not including header)
//   29..37  gc bits        8 bits   (mark, tenured, pinned, ...)
//   37..64  reserved      27 bits

const TYPE_SHIFT: u32 = 0;
const TYPE_BITS: u32 = 5;
const TYPE_MASK: u64 = (1 << TYPE_BITS) - 1;

const LEN_SHIFT: u32 = TYPE_SHIFT + TYPE_BITS;
const LEN_BITS: u32 = 24;
const LEN_MASK: u64 = (1 << LEN_BITS) - 1;

const GC_SHIFT: u32 = LEN_SHIFT + LEN_BITS;
const GC_BITS: u32 = 8;
const GC_MASK: u64 = (1 << GC_BITS) - 1;

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

// -- Semispace ---------------------------------------------------------------

pub struct Semispace {
    cells: Box<[u64]>,
    top: usize,
}

impl Semispace {
    pub fn new(size_bytes: usize) -> Semispace {
        let n_cells = size_bytes / 8;
        let cells = vec![0u64; n_cells].into_boxed_slice();
        Semispace { cells, top: 0 }
    }

    pub fn capacity_cells(&self) -> usize { self.cells.len() }
    pub fn capacity_bytes(&self) -> usize { self.cells.len() * 8 }
    pub fn used_cells(&self) -> usize { self.top }
    pub fn used_bytes(&self) -> usize { self.top * 8 }
    pub fn free_cells(&self) -> usize { self.cells.len() - self.top }
    pub fn free_bytes(&self) -> usize { self.free_cells() * 8 }

    pub fn reset(&mut self) { self.top = 0; }

    pub fn contains_ptr(&self, ptr: *const u8) -> bool {
        let base = self.cells.as_ptr() as usize;
        let end = base + self.cells.len() * 8;
        let p = ptr as usize;
        p >= base && p < end
    }

    /// Cell index of `ptr` within this semispace, or `None` if outside.
    fn cell_index_of(&self, ptr: *const u8) -> Option<usize> {
        let base = self.cells.as_ptr() as usize;
        let end = base + self.cells.len() * 8;
        let p = ptr as usize;
        if p >= base && p < end { Some((p - base) / 8) } else { None }
    }

    fn cell_ptr(&self, idx: usize) -> *const u64 {
        debug_assert!(idx < self.cells.len());
        unsafe { self.cells.as_ptr().add(idx) }
    }

    fn cell_ptr_mut(&mut self, idx: usize) -> *mut u64 {
        debug_assert!(idx < self.cells.len());
        unsafe { self.cells.as_mut_ptr().add(idx) }
    }

    pub fn alloc_cells(&mut self, cells: usize) -> NonNull<u64> {
        if self.top + cells > self.cells.len() {
            panic!(
                "semispace exhausted: requested {cells} cells, have {} free of {} total",
                self.cells.len() - self.top,
                self.cells.len(),
            );
        }
        let p = unsafe { self.cells.as_mut_ptr().add(self.top) };
        self.top += cells;
        unsafe { NonNull::new_unchecked(p) }
    }

    pub fn alloc_cons(&mut self, car: Word, cdr: Word) -> Word {
        let p = self.alloc_cells(2);
        unsafe {
            *p.as_ptr() = car.raw();
            *p.as_ptr().add(1) = cdr.raw();
        }
        Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
    }

    pub fn alloc_with_header(
        &mut self,
        ty: HeapType,
        length_cells: u32,
    ) -> NonNull<HeapHeader> {
        let total = 1 + length_cells as usize;
        let p = self.alloc_cells(total);
        unsafe { *p.as_ptr() = HeapHeader::new(ty, length_cells).raw(); }
        unsafe { NonNull::new_unchecked(p.as_ptr() as *mut HeapHeader) }
    }
}

// -- Heap (from + to + Cheney copy) -----------------------------------------

pub struct Heap {
    from: Semispace,
    to: Semispace,
}

impl Heap {
    pub fn new(size_bytes: usize) -> Heap {
        Heap {
            from: Semispace::new(size_bytes),
            to: Semispace::new(size_bytes),
        }
    }

    pub fn alloc_cons(&mut self, car: Word, cdr: Word) -> Word {
        self.from.alloc_cons(car, cdr)
    }

    pub fn alloc_with_header(
        &mut self,
        ty: HeapType,
        length_cells: u32,
    ) -> NonNull<HeapHeader> {
        self.from.alloc_with_header(ty, length_cells)
    }

    pub fn used_bytes(&self) -> usize { self.from.used_bytes() }
    pub fn capacity_bytes(&self) -> usize { self.from.capacity_bytes() }
    pub fn free_bytes(&self) -> usize { self.from.free_bytes() }
    pub fn contains_ptr(&self, ptr: *const u8) -> bool { self.from.contains_ptr(ptr) }

    /// Run a copying collection. The closure is called with a
    /// `RootScanner` and must visit every root `Word`. Roots are
    /// updated in place to point at their post-copy locations.
    ///
    /// After the closure returns, the to-space scan completes, then
    /// from/to are swapped and the (now empty) to-space is reset.
    pub fn collect(&mut self, mut visit_roots: impl FnMut(&mut RootScanner<'_, '_>)) {
        let mut state = CollectState {
            from: &mut self.from,
            to: &mut self.to,
            queue: Vec::new(),
        };
        // Visit roots, then drop the scanner so we can use `state`
        // mutably again for the to-space scan phase.
        {
            let mut scanner = RootScanner { state: &mut state };
            visit_roots(&mut scanner);
        }
        state.scan_to_completion();

        std::mem::swap(&mut self.from, &mut self.to);
        self.to.reset();
    }
}

struct CopiedObject {
    /// Cell index in to-space of the start of the copied object.
    /// For cons: cell 0 is car, cell 1 is cdr.
    /// For header'd: cell 0 is header, cells 1.. are payload.
    to_offset: usize,
    /// Total size in cells.
    size: usize,
    /// True iff this is a headerless cons (size == 2, no header word).
    is_cons: bool,
}

struct CollectState<'a> {
    from: &'a mut Semispace,
    to: &'a mut Semispace,
    queue: Vec<CopiedObject>,
}

impl<'a> CollectState<'a> {
    /// If `w` is a heap pointer into `from`, copy the referenced
    /// object into `to` (or follow an existing forwarding pointer)
    /// and return the relocated `Word`. Otherwise return `None`.
    fn maybe_copy(&mut self, w: Word) -> Option<Word> {
        let tag = w.tag();
        match tag {
            Tag::Fixnum | Tag::Immediate | Tag::Forward => None,
            Tag::Cons | Tag::Symbol | Tag::Vector | Tag::Function | Tag::String => {
                let from_ptr = w.as_mut_ptr::<u64>(tag).expect("heap ptr");
                let from_idx = match self.from.cell_index_of(from_ptr as *const u8) {
                    Some(i) => i,
                    None => return None, // pointer outside this heap
                };

                // Check for an existing forwarding pointer.
                let header_word = unsafe { *self.from.cell_ptr(from_idx) };
                if Word::from_raw(header_word).is_forward() {
                    let new_ptr = Word::from_raw(header_word).forward_target().unwrap();
                    return Some(Word::from_ptr(new_ptr as *const u8, tag));
                }

                // Compute size and copy.
                let is_cons = tag == Tag::Cons;
                let size = if is_cons {
                    2
                } else {
                    1 + HeapHeader::from_raw(header_word).length_cells() as usize
                };

                let dest = self.to.alloc_cells(size);
                let to_offset = self.to.cell_index_of(dest.as_ptr() as *const u8)
                    .expect("dest in to-space");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        from_ptr as *const u64,
                        dest.as_ptr(),
                        size,
                    );
                }

                // Install forwarding pointer at the from-space header.
                unsafe {
                    *self.from.cell_ptr_mut(from_idx) =
                        Word::forward(dest.as_ptr() as *const ()).raw();
                }

                self.queue.push(CopiedObject { to_offset, size, is_cons });

                Some(Word::from_ptr(dest.as_ptr() as *const u8, tag))
            }
        }
    }

    /// Walk the queue of copied objects, scanning each one's Word
    /// slots and copying any reachable referents. New objects copied
    /// during this phase are pushed onto the queue and processed in
    /// turn.
    fn scan_to_completion(&mut self) {
        let mut idx = 0;
        while idx < self.queue.len() {
            let obj = CopiedObject {
                to_offset: self.queue[idx].to_offset,
                size: self.queue[idx].size,
                is_cons: self.queue[idx].is_cons,
            };
            self.scan_object(&obj);
            idx += 1;
        }
    }

    fn scan_object(&mut self, obj: &CopiedObject) {
        let (payload_offset, n_words) = if obj.is_cons {
            (obj.to_offset, 2)
        } else {
            // Skip the header cell.
            (obj.to_offset + 1, obj.size - 1)
        };
        for i in 0..n_words {
            let cell_idx = payload_offset + i;
            let current = unsafe { Word::from_raw(*self.to.cell_ptr(cell_idx)) };
            if let Some(new) = self.maybe_copy(current) {
                unsafe { *self.to.cell_ptr_mut(cell_idx) = new.raw(); }
            }
        }
    }
}

pub struct RootScanner<'s, 'a: 's> {
    state: &'s mut CollectState<'a>,
}

impl<'s, 'a: 's> RootScanner<'s, 'a> {
    /// Visit one root. If `slot` holds a heap pointer that gets
    /// copied (or has already been forwarded), `slot` is updated
    /// in place to point at the new location.
    pub fn visit(&mut self, slot: &mut Word) {
        let w = *slot;
        if let Some(new) = self.state.maybe_copy(w) {
            *slot = new;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- HeapHeader tests ----------------------------------------------------

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
            HeapType::Symbol, HeapType::Vector, HeapType::Function,
            HeapType::String, HeapType::FfiBlock, HeapType::Other,
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
        h.set_gc_bit(GcBit::Mark);
        h.set_gc_bit(GcBit::Tenured);
        assert!(h.has_gc_bit(GcBit::Mark));
        assert!(h.has_gc_bit(GcBit::Tenured));
        assert_eq!(h.ty(), HeapType::Symbol);
        assert_eq!(h.length_cells(), 4);
        h.clear_gc_bit(GcBit::Mark);
        assert!(!h.has_gc_bit(GcBit::Mark));
        assert!(h.has_gc_bit(GcBit::Tenured));
    }

    #[test]
    fn header_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<HeapHeader>(), 8);
        assert_eq!(std::mem::align_of::<HeapHeader>(), 8);
    }

    // -- Semispace tests -----------------------------------------------------

    #[test]
    fn semispace_starts_empty() {
        let s = Semispace::new(1024);
        assert_eq!(s.used_bytes(), 0);
        assert_eq!(s.free_bytes(), 1024);
    }

    #[test]
    fn semispace_cons_round_trip() {
        let mut s = Semispace::new(1024);
        let c = s.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        assert!(c.is_cons());
        let p = c.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*p.add(1)).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn semispace_pointers_eight_byte_aligned() {
        let mut s = Semispace::new(1024);
        for _ in 0..10 {
            let c = s.alloc_cons(Word::NIL, Word::NIL);
            let addr = c.as_ptr::<u8>(Tag::Cons).unwrap() as usize;
            assert_eq!(addr & 7, 0);
        }
    }

    #[test]
    fn semispace_reset_lets_us_re_allocate() {
        let mut s = Semispace::new(64);
        s.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        s.alloc_cons(Word::fixnum(3), Word::fixnum(4));
        assert_eq!(s.used_cells(), 4);
        s.reset();
        assert_eq!(s.used_cells(), 0);
        s.alloc_with_header(HeapType::Vector, 7);
        assert_eq!(s.used_cells(), 8);
    }

    #[test]
    #[should_panic(expected = "semispace exhausted")]
    fn semispace_exhaustion_panics() {
        let mut s = Semispace::new(16);
        s.alloc_cons(Word::NIL, Word::NIL);
        s.alloc_cons(Word::NIL, Word::NIL);
    }

    // -- Heap copying-collector tests ---------------------------------------

    #[test]
    fn empty_collection_is_safe() {
        let mut heap = Heap::new(1024);
        heap.collect(|_| {});
        assert_eq!(heap.used_bytes(), 0);
    }

    #[test]
    fn collection_drops_unreferenced_cons() {
        let mut heap = Heap::new(1024);
        for _ in 0..3 {
            heap.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        }
        assert_eq!(heap.used_bytes(), 48);
        heap.collect(|_| {});
        assert_eq!(heap.used_bytes(), 0);
    }

    #[test]
    fn collection_keeps_rooted_cons() {
        let mut heap = Heap::new(1024);
        heap.alloc_cons(Word::fixnum(99), Word::fixnum(99));
        let mut root = heap.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        heap.alloc_cons(Word::fixnum(99), Word::fixnum(99));

        heap.collect(|s| s.visit(&mut root));

        assert_eq!(heap.used_bytes(), 16);
        let p = root.as_ptr::<u64>(Tag::Cons).expect("still a cons");
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*p.add(1)).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn collection_follows_cons_chain() {
        let mut heap = Heap::new(1024);
        let tail = heap.alloc_cons(Word::fixnum(3), Word::NIL);
        let mid = heap.alloc_cons(Word::fixnum(2), tail);
        let mut head = heap.alloc_cons(Word::fixnum(1), mid);

        heap.collect(|s| s.visit(&mut head));

        assert_eq!(heap.used_bytes(), 48); // 3 conses
        unsafe {
            let p1 = head.as_ptr::<u64>(Tag::Cons).unwrap();
            assert_eq!(Word::from_raw(*p1).as_fixnum(), Some(1));
            let p2_w = Word::from_raw(*p1.add(1));
            let p2 = p2_w.as_ptr::<u64>(Tag::Cons).unwrap();
            assert_eq!(Word::from_raw(*p2).as_fixnum(), Some(2));
            let p3_w = Word::from_raw(*p2.add(1));
            let p3 = p3_w.as_ptr::<u64>(Tag::Cons).unwrap();
            assert_eq!(Word::from_raw(*p3).as_fixnum(), Some(3));
            assert!(Word::from_raw(*p3.add(1)).is_nil());
        }
    }

    #[test]
    fn forwarding_dedupes_shared_subgraph() {
        let mut heap = Heap::new(1024);
        let shared = heap.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        let mut a = heap.alloc_cons(shared, Word::NIL);
        let mut b = heap.alloc_cons(shared, Word::NIL);

        heap.collect(|s| {
            s.visit(&mut a);
            s.visit(&mut b);
        });

        // a, b, and the shared cons — three conses live.
        assert_eq!(heap.used_bytes(), 48);
        unsafe {
            let pa = a.as_ptr::<u64>(Tag::Cons).unwrap();
            let pb = b.as_ptr::<u64>(Tag::Cons).unwrap();
            // Both a's car and b's car point at the SAME copy of the shared cons.
            assert_eq!(Word::from_raw(*pa), Word::from_raw(*pb));
        }
    }

    #[test]
    fn header_object_with_word_payload_copies() {
        let mut heap = Heap::new(1024);
        let vec_ptr = heap.alloc_with_header(HeapType::Vector, 3);
        unsafe {
            let payload = vec_ptr.as_ptr().add(1) as *mut u64;
            *payload = Word::fixnum(10).raw();
            *payload.add(1) = Word::fixnum(20).raw();
            *payload.add(2) = Word::fixnum(30).raw();
        }
        let mut root = Word::from_ptr(vec_ptr.as_ptr() as *const u8, Tag::Vector);

        heap.collect(|s| s.visit(&mut root));

        assert_eq!(heap.used_bytes(), 32); // header + 3 cells
        let new_ptr = root.as_mut_ptr::<HeapHeader>(Tag::Vector).unwrap();
        unsafe {
            assert_eq!((*new_ptr).ty(), HeapType::Vector);
            assert_eq!((*new_ptr).length_cells(), 3);
            let payload = (new_ptr as *const u64).add(1);
            assert_eq!(Word::from_raw(*payload).as_fixnum(), Some(10));
            assert_eq!(Word::from_raw(*payload.add(1)).as_fixnum(), Some(20));
            assert_eq!(Word::from_raw(*payload.add(2)).as_fixnum(), Some(30));
        }
    }

    #[test]
    fn header_with_pointer_payload_traces_through() {
        let mut heap = Heap::new(1024);
        // Build (vector: cons1 cons2)
        let c1 = heap.alloc_cons(Word::fixnum(1), Word::NIL);
        let c2 = heap.alloc_cons(Word::fixnum(2), Word::NIL);
        let v = heap.alloc_with_header(HeapType::Vector, 2);
        unsafe {
            let payload = v.as_ptr().add(1) as *mut u64;
            *payload = c1.raw();
            *payload.add(1) = c2.raw();
        }
        let mut root = Word::from_ptr(v.as_ptr() as *const u8, Tag::Vector);

        heap.collect(|s| s.visit(&mut root));

        // 1 vector (1 header + 2 payload = 3 cells) + 2 conses
        // (2 cells each = 4 cells) = 7 cells = 56 bytes.
        assert_eq!(heap.used_bytes(), 56);

        let v_new = root.as_mut_ptr::<HeapHeader>(Tag::Vector).unwrap();
        unsafe {
            let payload = (v_new as *const u64).add(1);
            let c1_w = Word::from_raw(*payload);
            let c2_w = Word::from_raw(*payload.add(1));
            assert!(c1_w.is_cons());
            assert!(c2_w.is_cons());
            let p1 = c1_w.as_ptr::<u64>(Tag::Cons).unwrap();
            let p2 = c2_w.as_ptr::<u64>(Tag::Cons).unwrap();
            assert_eq!(Word::from_raw(*p1).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*p2).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn surviving_multiple_collections() {
        let mut heap = Heap::new(1024);
        let mut root = heap.alloc_cons(Word::fixnum(42), Word::NIL);

        for cycle in 0..5 {
            // Allocate some garbage between collections.
            for _ in 0..3 {
                heap.alloc_cons(Word::fixnum(99), Word::fixnum(99));
            }
            heap.collect(|s| s.visit(&mut root));
            assert_eq!(heap.used_bytes(), 16, "cycle {cycle}");
            let p = root.as_ptr::<u64>(Tag::Cons).expect("alive");
            unsafe { assert_eq!(Word::from_raw(*p).as_fixnum(), Some(42)); }
        }
    }
}
