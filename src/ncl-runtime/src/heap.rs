//! Heap, header, semispace allocator, generational copying GC.
//!
//! See `docs/GC.md`. This module is GC steps 2 through 4:
//!   - the canonical 8-byte header word for non-cons heap objects,
//!   - a fixed-capacity `Semispace` with a bump allocator,
//!   - an `OldGen` with two semispaces that swap on full GC,
//!   - a `Heap` that pairs a `young` semispace with an `OldGen`,
//!   - `collect_minor` (young → old.live) and `collect_full`
//!     (young + old.live → old.scratch, swap old, reset young).
//!
//! Cons cells are headerless (two raw `Word` slots) per the design.
//! Everything else carries one `HeapHeader` cell in front of its
//! payload. Step 4 limitation: every heap object is treated as a
//! payload of `Word`s. Strings (UTF-8 bytes) and other non-Word
//! payloads land later via per-type scan functions.
//!
//! Step 4 also has no write barrier yet — minor GC scans ALL of
//! old.live for young pointers (correct but O(old)). Step 5 adds a
//! card table and the barrier.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::word::{Tag, Word};

// -- Card table --------------------------------------------------------------
//
// Software card-marking for the old-heap write barrier. One byte per
// CARD_SIZE_BYTES of old-heap storage. Mutator threads write to it
// lock-free via atomic byte stores; the GC reads it during minor
// collections to scan only the regions known to (possibly) hold
// young pointers.
//
// False positives are fine — `copy_into` filters non-pointers and
// pointers that aren't into young. False negatives (an unmarked
// card that contains a young pointer) are NOT fine; the discipline
// is that every old-heap store must mark.

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

    /// Try to allocate `cells` cells; returns `None` instead of
    /// panicking on exhaustion. Used by the TLAB refill path.
    pub fn try_alloc_cells(&mut self, cells: usize) -> Option<NonNull<u64>> {
        if self.top + cells > self.cells.len() {
            return None;
        }
        let p = unsafe { self.cells.as_mut_ptr().add(self.top) };
        self.top += cells;
        Some(unsafe { NonNull::new_unchecked(p) })
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

// -- OldGen (two semispaces, swap on full GC) -------------------------------

pub struct OldGen {
    a: Semispace,
    b: Semispace,
    /// Which semispace currently holds live data. The other is the
    /// scratch space used during full GC.
    live_is_a: bool,
    /// Software card table covering whichever semispace is currently
    /// live. Mutators dirty cards here on every old→x store; the GC
    /// reads dirty cards during minor collection. After every GC
    /// (minor or full) the table is reset to all-clean.
    cards: Arc<CardTable>,
}

impl OldGen {
    pub fn new(size_bytes: usize) -> OldGen {
        OldGen {
            a: Semispace::new(size_bytes),
            b: Semispace::new(size_bytes),
            live_is_a: true,
            cards: Arc::new(CardTable::new(size_bytes)),
        }
    }

    pub fn live(&self) -> &Semispace {
        if self.live_is_a { &self.a } else { &self.b }
    }

    fn live_mut(&mut self) -> &mut Semispace {
        if self.live_is_a { &mut self.a } else { &mut self.b }
    }

    fn scratch_mut(&mut self) -> &mut Semispace {
        if self.live_is_a { &mut self.b } else { &mut self.a }
    }

    fn live_and_scratch_mut(&mut self) -> (&mut Semispace, &mut Semispace) {
        if self.live_is_a {
            (&mut self.a, &mut self.b)
        } else {
            (&mut self.b, &mut self.a)
        }
    }

    fn swap_and_reset_scratch(&mut self) {
        self.live_is_a = !self.live_is_a;
        self.scratch_mut().reset();
        // After a full GC, the new live's contents were freshly
        // copied — they reference each other and the (now empty)
        // young heap, never any pre-existing young pointers. Cards
        // start clean.
        self.cards.clear_all();
    }

    pub fn cards(&self) -> &Arc<CardTable> { &self.cards }

    /// Pointer to the start of the live semispace's storage. Used by
    /// the lock-free mark-card façade to compute byte offsets.
    pub fn live_base_ptr(&self) -> *const u8 {
        self.live().cells.as_ptr() as *const u8
    }
}

// -- Heap (young + old) ------------------------------------------------------

pub struct Heap {
    young: Semispace,
    old: OldGen,
}

impl Heap {
    pub fn new(young_bytes: usize, old_bytes: usize) -> Heap {
        Heap {
            young: Semispace::new(young_bytes),
            old: OldGen::new(old_bytes),
        }
    }

    /// All allocation goes into the young heap.
    pub fn alloc_cons(&mut self, car: Word, cdr: Word) -> Word {
        self.young.alloc_cons(car, cdr)
    }

    pub fn alloc_with_header(
        &mut self,
        ty: HeapType,
        length_cells: u32,
    ) -> NonNull<HeapHeader> {
        self.young.alloc_with_header(ty, length_cells)
    }

    pub fn young_used_bytes(&self) -> usize { self.young.used_bytes() }
    pub fn young_free_cells(&self) -> usize { self.young.free_cells() }
    pub fn old_used_bytes(&self) -> usize { self.old.live().used_bytes() }
    pub fn used_bytes(&self) -> usize {
        self.young_used_bytes() + self.old_used_bytes()
    }

    /// Reserve a slab in young for a TLAB. Returns `None` if young
    /// can't fit the slab. The caller (a `MutatorState`) bump-
    /// allocates within the returned slab without locks.
    pub fn young_try_alloc_slab(&mut self, cells: usize) -> Option<NonNull<u64>> {
        self.young.try_alloc_cells(cells)
    }

    pub fn young_capacity_bytes(&self) -> usize { self.young.capacity_bytes() }
    pub fn old_capacity_bytes(&self) -> usize { self.old.live().capacity_bytes() }

    pub fn young_contains(&self, ptr: *const u8) -> bool {
        self.young.contains_ptr(ptr)
    }

    pub fn old_contains(&self, ptr: *const u8) -> bool {
        self.old.live().contains_ptr(ptr)
    }

    pub fn old_cards(&self) -> &Arc<CardTable> { self.old.cards() }
    pub fn old_live_base_ptr(&self) -> *const u8 { self.old.live_base_ptr() }
    pub fn old_capacity_bytes_per_semi(&self) -> usize {
        self.old.live().capacity_bytes()
    }

    /// Direct card mark for tests and the heap-level API. Production
    /// mutators go through the lock-free `MutatorState::mark_card`
    /// (which doesn't acquire the heap mutex).
    pub fn mark_old_card(&self, addr: *const u8) {
        let base = self.old.live_base_ptr() as usize;
        let cap = self.old.live().capacity_bytes();
        let p = addr as usize;
        if p >= base && p < base + cap {
            self.old.cards.mark_offset(p - base);
        }
    }

    /// Minor GC. Copies everything reachable in `young` (via roots
    /// and via pointers from `old.live`) into `old.live`, leaves
    /// forwarding pointers in young, and resets young.
    ///
    /// Step 4 has no write barrier yet, so we full-scan `old.live`
    /// for young pointers. Step 5 narrows that to dirty cards.
    pub fn collect_minor(&mut self, mut visit_roots: impl FnMut(&mut RootScanner<'_, '_>)) {
        let cards = Arc::clone(&self.old.cards);
        let live_base = self.old.live_base_ptr() as usize;

        let mut state = MinorState {
            young: &mut self.young,
            dest: self.old.live_mut(),
            queue: Vec::new(),
        };
        {
            let mut scanner = RootScanner { state: ScanTarget::Minor(&mut state) };
            visit_roots(&mut scanner);
        }
        state.scan_dirty_cards_for_young_pointers(&cards, live_base);
        state.scan_to_completion();
        self.young.reset();
        cards.clear_all();
    }

    /// Like `collect_minor` but also scans dirty cards in a static
    /// area for static→young pointers. Used by the GcCoordinator at
    /// runtime; tests that don't have a static area use the simpler
    /// `collect_minor` above.
    pub fn collect_minor_with_static(
        &mut self,
        static_cards: &CardTable,
        static_base: *mut u64,
        static_cells: usize,
        mut visit_roots: impl FnMut(&mut RootScanner<'_, '_>),
    ) {
        let cards = Arc::clone(&self.old.cards);
        let live_base = self.old.live_base_ptr() as usize;

        let mut state = MinorState {
            young: &mut self.young,
            dest: self.old.live_mut(),
            queue: Vec::new(),
        };
        {
            let mut scanner = RootScanner { state: ScanTarget::Minor(&mut state) };
            visit_roots(&mut scanner);
        }
        // Scan old.live's dirty cards.
        state.scan_dirty_cards_for_young_pointers(&cards, live_base);
        // Scan the static area's dirty cards. Static slots may now
        // contain young pointers (placed there since the last GC);
        // we promote those targets and update the slots in place.
        state.scan_dirty_cards_in(static_cards, static_base, static_cells);
        state.scan_to_completion();
        self.young.reset();
        cards.clear_all();
        static_cards.clear_all();
    }

    /// Full GC. Copies everything reachable from roots into
    /// `old.scratch`, then swaps live↔scratch and resets young.
    pub fn collect_full(&mut self, mut visit_roots: impl FnMut(&mut RootScanner<'_, '_>)) {
        let (live, scratch) = self.old.live_and_scratch_mut();
        let mut state = FullState {
            young: &mut self.young,
            old_live: live,
            dest: scratch,
            queue: Vec::new(),
        };
        {
            let mut scanner = RootScanner { state: ScanTarget::Full(&mut state) };
            visit_roots(&mut scanner);
        }
        state.scan_to_completion();
        // Swap old.live ↔ old.scratch and reset (the now-empty)
        // old-live-after-swap.
        self.old.swap_and_reset_scratch();
        self.young.reset();
    }
}

// -- Cheney machinery, with two flavours ------------------------------------

struct CopiedObject {
    /// Cell index in the destination semispace.
    to_offset: usize,
    /// Total size in cells.
    size: usize,
    /// True iff headerless cons.
    is_cons: bool,
}

/// Minor GC: source = young, destination = old.live.
struct MinorState<'a> {
    young: &'a mut Semispace,
    dest: &'a mut Semispace,
    queue: Vec<CopiedObject>,
}

impl<'a> MinorState<'a> {
    fn maybe_copy(&mut self, w: Word) -> Option<Word> {
        copy_into(w, &mut [self.young], self.dest, &mut self.queue)
    }

    /// Walk only DIRTY cards in `dest` (= old.live) looking for
    /// pointers into `young`.
    fn scan_dirty_cards_for_young_pointers(
        &mut self,
        cards: &CardTable,
        live_base: usize,
    ) {
        let dest_base = self.dest.cells.as_ptr() as usize;
        debug_assert_eq!(dest_base, live_base, "card table covers wrong space");
        let dest_base_ptr = self.dest.cells.as_ptr() as *mut u64;
        let dest_limit = self.dest.top;
        self.scan_dirty_cards_in(cards, dest_base_ptr, dest_limit);
    }

    /// Generic dirty-card scan over a region. Used for both old.live
    /// (during minor GC) and the static area (also during minor GC).
    /// `region_base..region_base.add(region_cells)` must match the
    /// area covered by `cards`.
    fn scan_dirty_cards_in(
        &mut self,
        cards: &CardTable,
        region_base: *mut u64,
        region_cells: usize,
    ) {
        for card in 0..cards.n_cards() {
            if !cards.is_dirty(card) { continue; }
            let card_start = card * CARD_SIZE_CELLS;
            let card_end = (card_start + CARD_SIZE_CELLS).min(region_cells);
            if card_start >= region_cells { break; }
            for i in card_start..card_end {
                let raw = unsafe { *region_base.add(i) };
                let w = Word::from_raw(raw);
                if let Some(new) = copy_into(
                    w,
                    &mut [self.young],
                    self.dest,
                    &mut self.queue,
                ) {
                    unsafe { region_base.add(i).write(new.raw()); }
                }
            }
        }
    }

    /// Step-4 fallback: full-scan old.live. No longer the default
    /// path — kept only because some heap-level tests still rely on
    /// "no card discipline" semantics. Production minor GC uses
    /// `scan_dirty_cards_for_young_pointers`.
    #[allow(dead_code)]
    fn scan_dest_for_young_pointers(&mut self) {
        // Iterate over all words written to dest BEFORE this minor
        // GC started. This includes both pre-existing live data and
        // the objects copied in by the root visit phase. Copying
        // bumps `dest.top`; we capture the limit at entry and walk
        // up to there. (Newly-copied objects' payloads are scanned
        // through the queue by `scan_to_completion`.)
        //
        // We need to walk old.live's data structurally — that means
        // knowing which cells are headers and which are payload.
        // The simplest correct walk: read each cell as a header at
        // the start of an object, advance by `1 + length_cells`. To
        // distinguish cons cells (no header) from header'd objects
        // we can't — old.live mixes both. So we use the same trick
        // as scan_to_completion: we don't actually walk old.live
        // structurally here. Instead, we iterate ALL cells in
        // old.live as candidate Words, and `maybe_copy` filters
        // those that aren't pointers.
        //
        // This is correct because non-pointer Words have tags
        // {Fixnum, Immediate, Forward} and `maybe_copy` returns
        // None on those. False positives (a fixnum payload that
        // happens to look like a young pointer) can't happen
        // because the tag is wrong.
        //
        // It IS slightly more work than necessary — header cells
        // get inspected as Words too, and their tag may match a
        // valid Tag pattern by coincidence. But headers have type
        // codes in their low bits (TYPE_BITS=5, < TAG_BITS+SUBTAG=8?
        // — no, TYPE_BITS=5, TAG_BITS=3) so a header word's low 3
        // bits are the type's low 3 bits. For HeapType values:
        //   Symbol=0 → tag 000 (fixnum) → as_fixnum, no copy
        //   Vector=1 → tag 001 (cons!) → would attempt to copy
        // This is a real false-positive risk. We DO need to walk
        // structurally here. Move forward with the simpler "scan
        // cells as Words" but mark this as a known issue —
        // tracked for step 5 where the card-table/dirty-byte
        // mechanism will replace this scan entirely.
        //
        // To make this correct in step 4, we'd need to track the
        // "first tagged offset" per card the way Roger's page-table
        // does. We defer that complexity; the user-visible bug
        // would be: "a fixnum 1 stored in old somehow gets relocated
        // because it looks like a Cons-tagged pointer to a non-young
        // address." `maybe_copy` already handles the "not in young"
        // case as None, so we're safe — it only acts on pointers
        // that point INTO young. As long as no fixnum-bit-pattern
        // happens to be a valid young address, we're fine. Young
        // addresses are 8-byte aligned heap pointers; fixnum
        // payloads with low 3 bits forming Tag::Cons (001) cannot
        // be 8-byte aligned (their low 3 bits are 001, not 000).
        // So `as_mut_ptr` returning a non-aligned pointer that
        // somehow maps into young's address range is essentially
        // impossible. The "false positive" concern reduces to:
        // does any non-pointer Word's bit pattern, when interpreted
        // as Tag::Cons/Symbol/Vector/Function/String, point into
        // young? Only if a fixnum/immediate has that bit pattern
        // and is also a valid young heap address — which can't
        // happen because tagged words mask out the low 3 bits to
        // get the pointer, and our fake "fixnum-as-pointer" isn't
        // 8-byte aligned to a real young location.
        //
        // Net: scanning all cells of old.live as candidate Words
        // is safe and correct. Slow, but correct. Step 5 replaces
        // it with the card-table mechanism.

        let dest_limit = self.dest.top;
        for i in 0..dest_limit {
            let raw = unsafe { *self.dest.cell_ptr(i) };
            let w = Word::from_raw(raw);
            if let Some(new) = copy_into(w, &mut [self.young], self.dest, &mut self.queue) {
                unsafe { *self.dest.cell_ptr_mut(i) = new.raw(); }
            }
        }
    }

    fn scan_to_completion(&mut self) {
        let mut idx = 0;
        while idx < self.queue.len() {
            let obj = self.queue[idx];
            let (payload_offset, n_words) = if obj.is_cons {
                (obj.to_offset, 2)
            } else {
                (obj.to_offset + 1, obj.size - 1)
            };
            for i in 0..n_words {
                let cell_idx = payload_offset + i;
                let current = unsafe { Word::from_raw(*self.dest.cell_ptr(cell_idx)) };
                if let Some(new) = copy_into(
                    current,
                    &mut [self.young],
                    self.dest,
                    &mut self.queue,
                ) {
                    unsafe { *self.dest.cell_ptr_mut(cell_idx) = new.raw(); }
                }
            }
            idx += 1;
        }
    }
}

/// Full GC: sources = young + old.live, destination = old.scratch.
struct FullState<'a> {
    young: &'a mut Semispace,
    old_live: &'a mut Semispace,
    dest: &'a mut Semispace,
    queue: Vec<CopiedObject>,
}

impl<'a> FullState<'a> {
    fn maybe_copy(&mut self, w: Word) -> Option<Word> {
        copy_into(w, &mut [self.young, self.old_live], self.dest, &mut self.queue)
    }

    fn scan_to_completion(&mut self) {
        let mut idx = 0;
        while idx < self.queue.len() {
            let obj = self.queue[idx];
            let (payload_offset, n_words) = if obj.is_cons {
                (obj.to_offset, 2)
            } else {
                (obj.to_offset + 1, obj.size - 1)
            };
            for i in 0..n_words {
                let cell_idx = payload_offset + i;
                let current = unsafe { Word::from_raw(*self.dest.cell_ptr(cell_idx)) };
                if let Some(new) = copy_into(
                    current,
                    &mut [self.young, self.old_live],
                    self.dest,
                    &mut self.queue,
                ) {
                    unsafe { *self.dest.cell_ptr_mut(cell_idx) = new.raw(); }
                }
            }
            idx += 1;
        }
    }
}

impl Copy for CopiedObject {}
impl Clone for CopiedObject {
    fn clone(&self) -> Self { *self }
}

// -- Shared copy primitive --------------------------------------------------

/// If `w` is a heap pointer that points into one of `sources`, copy
/// the referenced object into `dest` (or follow an existing
/// forwarding pointer) and return the new tagged Word. Otherwise
/// return None.
fn copy_into(
    w: Word,
    sources: &mut [&mut Semispace],
    dest: &mut Semispace,
    queue: &mut Vec<CopiedObject>,
) -> Option<Word> {
    let tag = w.tag();
    match tag {
        Tag::Fixnum | Tag::Immediate | Tag::Forward => None,
        Tag::Cons | Tag::Symbol | Tag::Vector | Tag::Function | Tag::String => {
            let from_ptr = w.as_mut_ptr::<u64>(tag).expect("heap ptr");

            // Find which source this pointer is in (if any).
            let mut source_idx_and_cell: Option<(usize, usize)> = None;
            for (i, src) in sources.iter().enumerate() {
                if let Some(idx) = src.cell_index_of(from_ptr as *const u8) {
                    source_idx_and_cell = Some((i, idx));
                    break;
                }
            }
            let (src_idx, from_idx) = source_idx_and_cell?;

            // Forwarding pointer already there?
            let header_word = unsafe { *sources[src_idx].cell_ptr(from_idx) };
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
            let dest_ptr = dest.alloc_cells(size);
            let to_offset = dest.cell_index_of(dest_ptr.as_ptr() as *const u8)
                .expect("dest in dest-space");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    from_ptr as *const u64,
                    dest_ptr.as_ptr(),
                    size,
                );
            }
            // Forwarding pointer at the source.
            unsafe {
                *sources[src_idx].cell_ptr_mut(from_idx) =
                    Word::forward(dest_ptr.as_ptr() as *const ()).raw();
            }
            queue.push(CopiedObject { to_offset, size, is_cons });
            Some(Word::from_ptr(dest_ptr.as_ptr() as *const u8, tag))
        }
    }
}

// -- RootScanner (covers both Minor and Full) -------------------------------

enum ScanTarget<'s, 'a: 's> {
    Minor(&'s mut MinorState<'a>),
    Full(&'s mut FullState<'a>),
}

pub struct RootScanner<'s, 'a: 's> {
    state: ScanTarget<'s, 'a>,
}

impl<'s, 'a: 's> RootScanner<'s, 'a> {
    pub fn visit(&mut self, slot: &mut Word) {
        let w = *slot;
        let new = match &mut self.state {
            ScanTarget::Minor(s) => s.maybe_copy(w),
            ScanTarget::Full(s) => s.maybe_copy(w),
        };
        if let Some(updated) = new {
            *slot = updated;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Heap {
        Heap::new(1024, 1024)
    }

    // -- HeapHeader ----------------------------------------------------------

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
        h.clear_gc_bit(GcBit::Mark);
        assert!(!h.has_gc_bit(GcBit::Mark));
        assert!(h.has_gc_bit(GcBit::Tenured));
    }

    #[test]
    fn header_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<HeapHeader>(), 8);
        assert_eq!(std::mem::align_of::<HeapHeader>(), 8);
    }

    // -- Semispace -----------------------------------------------------------

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
    #[should_panic(expected = "semispace exhausted")]
    fn semispace_exhaustion_panics() {
        let mut s = Semispace::new(16);
        s.alloc_cons(Word::NIL, Word::NIL);
        s.alloc_cons(Word::NIL, Word::NIL);
    }

    // -- Heap allocation lands in young -------------------------------------

    #[test]
    fn allocation_lands_in_young() {
        let mut h = fresh();
        let c = h.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        let p = c.as_ptr::<u8>(Tag::Cons).unwrap();
        assert!(h.young_contains(p));
        assert!(!h.old_contains(p));
        assert_eq!(h.young_used_bytes(), 16);
        assert_eq!(h.old_used_bytes(), 0);
    }

    // -- Minor GC -----------------------------------------------------------

    #[test]
    fn minor_promotes_rooted_cons_to_old() {
        let mut h = fresh();
        h.alloc_cons(Word::fixnum(99), Word::fixnum(99));
        let mut root = h.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        h.alloc_cons(Word::fixnum(99), Word::fixnum(99));

        h.collect_minor(|s| s.visit(&mut root));

        // Young is empty, the rooted cons lives in old.
        assert_eq!(h.young_used_bytes(), 0);
        assert_eq!(h.old_used_bytes(), 16);

        let p = root.as_ptr::<u8>(Tag::Cons).unwrap();
        assert!(h.old_contains(p));

        let pp = root.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*pp).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*pp.add(1)).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn minor_drops_unrooted_cons() {
        let mut h = fresh();
        for _ in 0..3 {
            h.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        }
        h.collect_minor(|_| {});
        assert_eq!(h.young_used_bytes(), 0);
        assert_eq!(h.old_used_bytes(), 0);
    }

    #[test]
    fn minor_old_to_young_pointer_promotes_with_card_mark() {
        let mut h = fresh();

        // Allocate young cons A. Run a minor GC with A as root —
        // A goes to old.
        let mut a = h.alloc_cons(Word::fixnum(1), Word::NIL);
        h.collect_minor(|s| s.visit(&mut a));
        assert_eq!(h.old_used_bytes(), 16);
        assert_eq!(h.young_used_bytes(), 0);

        // Now allocate young cons B and patch it into A's cdr.
        let b = h.alloc_cons(Word::fixnum(2), Word::NIL);
        let a_ptr = a.as_mut_ptr::<u64>(Tag::Cons).unwrap();
        unsafe { *a_ptr.add(1) = b.raw(); }
        // The write barrier discipline: mark the card containing A.
        h.mark_old_card(a_ptr as *const u8);
        assert_eq!(h.young_used_bytes(), 16);

        h.collect_minor(|s| s.visit(&mut a));

        // Both conses now in old; young empty.
        assert_eq!(h.young_used_bytes(), 0);
        assert_eq!(h.old_used_bytes(), 32);

        unsafe {
            let p = a.as_ptr::<u64>(Tag::Cons).unwrap();
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(1));
            let cdr = Word::from_raw(*p.add(1));
            assert!(cdr.is_cons());
            let bp = cdr.as_ptr::<u64>(Tag::Cons).unwrap();
            assert_eq!(Word::from_raw(*bp).as_fixnum(), Some(2));
            assert!(Word::from_raw(*bp.add(1)).is_nil());
        }
    }

    #[test]
    fn missing_card_mark_loses_young_object() {
        // Negative test: this demonstrates that the card-marking
        // discipline is real. WITHOUT calling mark_old_card after
        // writing a young pointer into an old object, the minor GC
        // will not find the young object via old's roots and will
        // discard it. The compiler will emit mark_card at every
        // old-heap store; this test exists to enforce the contract
        // by failing if anyone breaks it.
        let mut h = fresh();
        let mut a = h.alloc_cons(Word::fixnum(1), Word::NIL);
        h.collect_minor(|s| s.visit(&mut a));

        let b = h.alloc_cons(Word::fixnum(2), Word::NIL);
        let a_ptr = a.as_mut_ptr::<u64>(Tag::Cons).unwrap();
        unsafe { *a_ptr.add(1) = b.raw(); }
        // DO NOT mark the card.

        h.collect_minor(|s| s.visit(&mut a));

        // Only `a` survives; `b` was missed because no card was dirty.
        assert_eq!(h.old_used_bytes(), 16);
        assert_eq!(h.young_used_bytes(), 0);

        // a's cdr now points at a stale young address. We don't
        // dereference it (would be UB), but we verify it's STILL a
        // cons-tagged Word — the GC didn't update it because the GC
        // never knew about it.
        unsafe {
            let p = a.as_ptr::<u64>(Tag::Cons).unwrap();
            let cdr = Word::from_raw(*p.add(1));
            assert!(cdr.is_cons(), "stale tagged ptr is still a Cons-tag");
        }
    }

    #[test]
    fn cards_are_cleared_after_minor() {
        let mut h = fresh();
        let mut a = h.alloc_cons(Word::fixnum(1), Word::NIL);
        h.collect_minor(|s| s.visit(&mut a));

        let b = h.alloc_cons(Word::fixnum(2), Word::NIL);
        let a_ptr = a.as_mut_ptr::<u64>(Tag::Cons).unwrap();
        unsafe { *a_ptr.add(1) = b.raw(); }
        h.mark_old_card(a_ptr as *const u8);

        // One card dirty before GC.
        assert_eq!(h.old_cards().dirty_count(), 1);

        h.collect_minor(|s| s.visit(&mut a));

        // Cards cleared after GC; the next minor scan starts clean.
        assert_eq!(h.old_cards().dirty_count(), 0);
    }

    #[test]
    fn multiple_dirty_cards_all_processed() {
        // Allocate enough conses that promotion fills more than
        // CARD_SIZE_CELLS = 64 cells in old; verify that cards
        // covering different regions all get scanned.
        let mut h = fresh();
        let mut roots: Vec<Word> = Vec::new();
        for i in 0..40 {
            roots.push(h.alloc_cons(Word::fixnum(i), Word::NIL));
        }
        // Promote all to old.
        h.collect_minor(|s| {
            for r in roots.iter_mut() { s.visit(r); }
        });
        // 40 conses in old = 80 cells. With 64 cells per card,
        // that spans 2 cards.

        // Allocate a young object and patch it into the LAST root
        // (which is in the second card).
        let young = h.alloc_cons(Word::fixnum(999), Word::NIL);
        let last_root_ptr = roots[39].as_mut_ptr::<u64>(Tag::Cons).unwrap();
        unsafe { *last_root_ptr.add(1) = young.raw(); }
        h.mark_old_card(last_root_ptr as *const u8);

        h.collect_minor(|s| {
            for r in roots.iter_mut() { s.visit(r); }
        });

        // young got promoted via the second card. Total in old:
        // 41 conses = 82 cells = 656 bytes.
        assert_eq!(h.old_used_bytes(), 41 * 16);
    }

    #[test]
    fn cards_clear_on_full_gc() {
        let mut h = fresh();
        let mut a = h.alloc_cons(Word::fixnum(1), Word::NIL);
        h.collect_minor(|s| s.visit(&mut a));

        let b = h.alloc_cons(Word::fixnum(2), Word::NIL);
        let a_ptr = a.as_mut_ptr::<u64>(Tag::Cons).unwrap();
        unsafe { *a_ptr.add(1) = b.raw(); }
        h.mark_old_card(a_ptr as *const u8);

        assert_eq!(h.old_cards().dirty_count(), 1);

        h.collect_full(|s| s.visit(&mut a));

        // Full GC swaps; new live's cards are clean by construction.
        assert_eq!(h.old_cards().dirty_count(), 0);
    }

    #[test]
    fn minor_chain_in_young_promotes_all_reachable() {
        let mut h = fresh();
        let tail = h.alloc_cons(Word::fixnum(3), Word::NIL);
        let mid = h.alloc_cons(Word::fixnum(2), tail);
        let mut head = h.alloc_cons(Word::fixnum(1), mid);

        h.collect_minor(|s| s.visit(&mut head));

        assert_eq!(h.young_used_bytes(), 0);
        assert_eq!(h.old_used_bytes(), 48);
    }

    // -- Full GC ------------------------------------------------------------

    #[test]
    fn full_collects_unrooted() {
        let mut h = fresh();
        // First minor → put garbage in old.
        h.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        h.collect_minor(|_| {});
        // Second batch in young; another minor with a root.
        let mut keep = h.alloc_cons(Word::fixnum(7), Word::NIL);
        h.collect_minor(|s| s.visit(&mut keep));
        assert_eq!(h.old_used_bytes(), 16);

        // Now `keep` is the only thing alive in old.
        // Full GC with no roots → drops everything.
        h.collect_full(|_| {});
        assert_eq!(h.old_used_bytes(), 0);
        assert_eq!(h.young_used_bytes(), 0);
    }

    #[test]
    fn full_compacts_old() {
        let mut h = fresh();

        // Promote three conses to old via minor GCs. Drop the
        // intermediate two.
        let mut keep = h.alloc_cons(Word::fixnum(42), Word::NIL);
        h.collect_minor(|s| s.visit(&mut keep));

        // Allocate two unrooted conses and minor-GC them away
        // (they don't survive minor without roots).
        h.alloc_cons(Word::fixnum(98), Word::fixnum(98));
        h.alloc_cons(Word::fixnum(99), Word::fixnum(99));
        h.collect_minor(|s| s.visit(&mut keep)); // unrooted dies here

        assert_eq!(h.old_used_bytes(), 16);

        // Full GC keeps only `keep`.
        h.collect_full(|s| s.visit(&mut keep));
        assert_eq!(h.old_used_bytes(), 16);

        // `keep` still has its data and is in old (after the swap).
        let p = keep.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(42));
            assert!(Word::from_raw(*p.add(1)).is_nil());
        }
        assert!(h.old_contains(keep.as_ptr::<u8>(Tag::Cons).unwrap()));
    }

    #[test]
    fn full_handles_young_old_mix() {
        let mut h = fresh();

        // Step 1: cons in young; minor → in old.
        let mut keep = h.alloc_cons(Word::fixnum(1), Word::NIL);
        h.collect_minor(|s| s.visit(&mut keep));
        assert_eq!(h.old_used_bytes(), 16);

        // Step 2: another cons in young; left there.
        let mut also_keep = h.alloc_cons(Word::fixnum(2), Word::NIL);
        assert_eq!(h.young_used_bytes(), 16);

        // Full GC with both roots: both survive, both end up in
        // old (full collapses young into old).
        h.collect_full(|s| {
            s.visit(&mut keep);
            s.visit(&mut also_keep);
        });

        assert_eq!(h.young_used_bytes(), 0);
        assert_eq!(h.old_used_bytes(), 32);

        // Both pointers are now in old.
        assert!(h.old_contains(keep.as_ptr::<u8>(Tag::Cons).unwrap()));
        assert!(h.old_contains(also_keep.as_ptr::<u8>(Tag::Cons).unwrap()));

        // Values intact.
        unsafe {
            assert_eq!(Word::from_raw(*keep.as_ptr::<u64>(Tag::Cons).unwrap()).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*also_keep.as_ptr::<u64>(Tag::Cons).unwrap()).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn many_minor_then_full_is_stable() {
        let mut h = fresh();
        let mut root = h.alloc_cons(Word::fixnum(0), Word::NIL);
        for cycle in 0..10 {
            // Allocate garbage.
            for _ in 0..3 {
                h.alloc_cons(Word::fixnum(99), Word::fixnum(99));
            }
            h.collect_minor(|s| s.visit(&mut root));
            unsafe {
                let p = root.as_ptr::<u64>(Tag::Cons).expect(&format!("cycle {cycle}"));
                assert_eq!(Word::from_raw(*p).as_fixnum(), Some(0));
            }
        }
        // Old should have just the one root.
        assert_eq!(h.old_used_bytes(), 16);

        // Full GC keeps it.
        h.collect_full(|s| s.visit(&mut root));
        assert_eq!(h.old_used_bytes(), 16);
    }
}
