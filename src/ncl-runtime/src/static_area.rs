//! Static area: pinned, never-moved memory for things that never die.
//!
//! See `docs/GC.md`. The static area is for:
//!   - JIT-emitted machine code (when the JIT lands)
//!   - The loaded image's interned constants
//!   - The package and symbol registries
//!
//! Allocation is bump-pointer with atomic CAS so multiple Lisp
//! threads can allocate concurrently without locks. Storage is never
//! freed individually — the static area lives for the process
//! lifetime. The loader's retirement-with-quiescent-epoch mechanism
//! handles "old code that's no longer reachable" separately from the
//! GC; see MANIFESTO.md, "The loader".
//!
//! Static→young pointers are handled by a card table (same kind as
//! old's). Mutators dirty cards on every store of a heap pointer
//! into a static cell. Minor GC scans dirty cards for young
//! pointers; full GC scans the whole static area for any heap
//! pointer.
//!
//! Static cells are read and written by multiple threads; all access
//! goes through `*mut u64` after a successful CAS bump. The
//! park/unpark synchronisation barrier in the GC coordinator
//! provides the cross-thread happens-before relationship; within a
//! thread, the alloc-then-write pattern guarantees exclusive access
//! to a freshly allocated range until the next safe point.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::heap::{CardTable, HeapHeader, HeapType};

/// Pinned, never-moved memory region. One per process.
pub struct StaticArea {
    /// Backing storage. `Box<[u64]>` with raw pointer access; the
    /// allocation discipline (CAS bump → exclusive ownership of the
    /// returned range until safe point) keeps this sound.
    cells: Box<[u64]>,
    /// Bump pointer (cell index). Atomically advanced on every alloc.
    top: AtomicUsize,
    /// Card table for static→young pointers. Cleared after minor GC.
    cards: Arc<CardTable>,
}

impl StaticArea {
    pub fn new(size_bytes: usize) -> StaticArea {
        let n_cells = size_bytes / 8;
        let cells = vec![0u64; n_cells].into_boxed_slice();
        let cards = Arc::new(CardTable::new(size_bytes));
        StaticArea {
            cells,
            top: AtomicUsize::new(0),
            cards,
        }
    }

    pub fn capacity_cells(&self) -> usize { self.cells.len() }
    pub fn capacity_bytes(&self) -> usize { self.cells.len() * 8 }
    pub fn used_cells(&self) -> usize { self.top.load(Ordering::Acquire) }
    pub fn used_bytes(&self) -> usize { self.used_cells() * 8 }
    pub fn free_cells(&self) -> usize { self.capacity_cells() - self.used_cells() }
    pub fn free_bytes(&self) -> usize { self.free_cells() * 8 }

    pub fn cards(&self) -> &Arc<CardTable> { &self.cards }

    pub fn base_ptr(&self) -> *const u8 { self.cells.as_ptr() as *const u8 }

    pub fn contains_ptr(&self, ptr: *const u8) -> bool {
        let base = self.cells.as_ptr() as usize;
        let end = base + self.cells.len() * 8;
        let p = ptr as usize;
        p >= base && p < end
    }

    /// Cell index of `ptr` within this area, or `None` if outside.
    pub fn cell_index_of(&self, ptr: *const u8) -> Option<usize> {
        let base = self.cells.as_ptr() as usize;
        let end = base + self.cells.len() * 8;
        let p = ptr as usize;
        if p >= base && p < end { Some((p - base) / 8) } else { None }
    }

    /// Pointer to the cell at index `idx`. Caller must ensure
    /// `idx < capacity_cells()`.
    pub(crate) fn cell_ptr(&self, idx: usize) -> *mut u64 {
        debug_assert!(idx < self.cells.len());
        unsafe { (self.cells.as_ptr() as *mut u64).add(idx) }
    }

    /// Allocate `cells` cells via lock-free CAS. Returns a pointer
    /// to the first cell, or `None` if exhausted. Multiple threads
    /// may call concurrently; each gets a disjoint range.
    pub fn try_alloc_cells(&self, cells: usize) -> Option<NonNull<u64>> {
        loop {
            let cur = self.top.load(Ordering::Acquire);
            let new = cur.checked_add(cells)?;
            if new > self.cells.len() {
                return None;
            }
            if self.top
                .compare_exchange(cur, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let p = unsafe { (self.cells.as_ptr() as *mut u64).add(cur) };
                return Some(unsafe { NonNull::new_unchecked(p) });
            }
            // Lost the race — retry.
        }
    }

    /// Allocate a header'd object in static. Caller fills the payload
    /// cells.
    pub fn try_alloc_with_header(
        &self,
        ty: HeapType,
        length_cells: u32,
    ) -> Option<NonNull<HeapHeader>> {
        let total = 1 + length_cells as usize;
        let p = self.try_alloc_cells(total)?;
        unsafe { *p.as_ptr() = HeapHeader::new(ty, length_cells).raw(); }
        Some(unsafe { NonNull::new_unchecked(p.as_ptr() as *mut HeapHeader) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn static_starts_empty() {
        let s = StaticArea::new(1024);
        assert_eq!(s.used_bytes(), 0);
        assert_eq!(s.free_bytes(), 1024);
        assert_eq!(s.capacity_bytes(), 1024);
    }

    #[test]
    fn static_basic_alloc() {
        let s = StaticArea::new(1024);
        let p = s.try_alloc_cells(4).expect("alloc 4 cells");
        unsafe {
            *p.as_ptr() = 11;
            *p.as_ptr().add(1) = 22;
            *p.as_ptr().add(2) = 33;
            *p.as_ptr().add(3) = 44;
        }
        assert_eq!(s.used_cells(), 4);

        // Read back via cell_ptr.
        let idx = s.cell_index_of(p.as_ptr() as *const u8).expect("in range");
        unsafe {
            assert_eq!(*s.cell_ptr(idx), 11);
            assert_eq!(*s.cell_ptr(idx + 1), 22);
        }
    }

    #[test]
    fn static_exhaustion_returns_none() {
        let s = StaticArea::new(64); // 8 cells
        let _ = s.try_alloc_cells(8);
        assert_eq!(s.used_cells(), 8);
        assert!(s.try_alloc_cells(1).is_none());
        assert!(s.try_alloc_cells(100).is_none());
    }

    #[test]
    fn static_pointer_classification() {
        let s = StaticArea::new(1024);
        let p = s.try_alloc_cells(2).unwrap();
        assert!(s.contains_ptr(p.as_ptr() as *const u8));
        let stack: u64 = 0;
        assert!(!s.contains_ptr(&stack as *const u64 as *const u8));
    }

    #[test]
    fn static_alloc_with_header() {
        let s = StaticArea::new(1024);
        let p = s.try_alloc_with_header(HeapType::Symbol, 7).unwrap();
        unsafe {
            assert_eq!((*p.as_ptr()).ty(), HeapType::Symbol);
            assert_eq!((*p.as_ptr()).length_cells(), 7);
        }
        assert_eq!(s.used_cells(), 8); // 1 header + 7 payload
    }

    #[test]
    fn static_concurrent_alloc_disjoint() {
        // Multiple threads CAS-bump simultaneously; we verify each
        // gets a disjoint range with no overlap.
        let s = Arc::new(StaticArea::new(64 * 1024)); // 8K cells
        let n_threads = 4;
        let allocs_per_thread = 100;

        let handles: Vec<_> = (0..n_threads).map(|_| {
            let s = Arc::clone(&s);
            thread::spawn(move || {
                let mut my_ptrs = Vec::new();
                for _ in 0..allocs_per_thread {
                    let p = s.try_alloc_cells(2).expect("alloc");
                    my_ptrs.push(p.as_ptr() as usize);
                }
                my_ptrs
            })
        }).collect();

        let mut all: Vec<usize> = Vec::new();
        for h in handles {
            all.extend(h.join().expect("thread"));
        }
        // All pointers distinct and 8-byte aligned.
        all.sort();
        for w in all.windows(2) {
            assert!(w[0] < w[1], "overlap: {} >= {}", w[0], w[1]);
            assert!(w[1] - w[0] >= 16, "alloc 2 cells = 16 bytes apart minimum");
            assert_eq!(w[0] & 7, 0);
        }
        assert_eq!(s.used_cells(), n_threads * allocs_per_thread * 2);
    }

    #[test]
    fn static_cards_are_initially_clean() {
        let s = StaticArea::new(1024);
        assert_eq!(s.cards.dirty_count(), 0);
    }
}
