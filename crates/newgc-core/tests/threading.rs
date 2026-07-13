//! Threading exploration — what does multi-thread support look
//! like in newgc-core today?
//!
//! Findings summary (see `THREADING.md` for the long form):
//!   - `PageHeap<L>` is `Send + Sync` (compile-time verified below).
//!   - **Independent heaps in parallel: works.** N threads each
//!     with their own `PageHeap` allocate freely. The GC is per-
//!     heap STW; no cross-heap coordination exists.
//!   - **Shared heap via `Mutex<PageHeap>`: works but serializes
//!     all allocation.** Throughput collapses to that of one
//!     thread plus mutex overhead. There is no concurrent-allocation
//!     fast path today.
//!   - **Lock-free read-only access** works for accessors like
//!     `count_pages_in_gen`, `committed_pages`, `is_committed`
//!     (which use atomics for state) — but the only useful mutator
//!     operation is allocation, which requires `&mut`.
//!   - **Concurrent mutators on one heap is not supported.** The
//!     missing pieces, in order: per-thread TLABs, safepoint/
//!     poll-word API, cooperative parking, per-thread root
//!     enumeration. None of these exist today.

use std::sync::{Arc, Mutex};
use std::thread;

use newgc_core::page_heap::space::PageHeap;
use newgc_core::{Generation, LispLayout};

type Heap = PageHeap<LispLayout>;

// =========================================================================
// Compile-time Send / Sync
// =========================================================================

#[test]
fn pageheap_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Heap>();
}

#[test]
fn pageheap_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<Heap>();
}

// =========================================================================
// Independent heaps in parallel
// =========================================================================

#[test]
fn n_independent_heaps_allocate_in_parallel() {
    let n_threads = 4;
    let allocs_per = 1000;

    let handles: Vec<_> = (0..n_threads)
        .map(|tid| {
            thread::spawn(move || {
                let mut heap = Heap::with_reservation(8 * 64 * 1024);
                let mut count = 0;
                for _ in 0..allocs_per {
                    if heap.try_alloc_cons_in(Generation::G0).is_some() {
                        count += 1;
                    }
                }
                heap.evacuate_from_word_roots(
                    Generation::G0,
                    Generation::G1,
                    &mut [],
                );
                (tid, count, heap.count_pages_in_gen(Generation::G0))
            })
        })
        .collect();

    for h in handles {
        let (tid, allocs, g0_after_gc) = h.join().expect("thread panicked");
        assert_eq!(allocs, allocs_per, "thread {tid} alloc shortfall");
        assert_eq!(g0_after_gc, 0,
            "thread {tid}: G0 should be empty after GC with no roots");
    }
}

#[test]
fn independent_heaps_are_genuinely_independent() {
    // Two threads, each fills its heap to OOM. Independence means
    // one running out doesn't affect the other.
    let t1 = thread::spawn(|| {
        let mut heap = Heap::with_reservation(2 * 64 * 1024);
        let mut count = 0;
        while heap.try_alloc_cons_in(Generation::G0).is_some() {
            count += 1;
            if count > 100_000 { break; }
        }
        count
    });
    let t2 = thread::spawn(|| {
        let mut heap = Heap::with_reservation(8 * 64 * 1024);
        let mut count = 0;
        while heap.try_alloc_cons_in(Generation::G0).is_some() {
            count += 1;
            if count > 100_000 { break; }
        }
        count
    });
    let c1 = t1.join().unwrap();
    let c2 = t2.join().unwrap();
    // The 8-page heap should fit ~4× as many conses as the 2-page one
    // (each page holds 4096 conses; minus alloc-region overhead).
    assert!(c1 > 0 && c2 > 0);
    assert!(c2 > c1, "8-page heap fit {c2}, 2-page heap fit {c1}");
}

// =========================================================================
// Shared heap via Mutex — serialized allocation
// =========================================================================

#[test]
fn shared_heap_via_mutex_serializes_allocation() {
    // Wrap one heap in a Mutex; spawn N threads; each tries to
    // allocate. The Mutex serializes — but it works.
    let heap = Arc::new(Mutex::new(
        Heap::with_reservation(32 * 64 * 1024),
    ));
    let n_threads = 4;
    let allocs_per_thread = 500;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let heap = Arc::clone(&heap);
            thread::spawn(move || {
                let mut local_count = 0;
                for _ in 0..allocs_per_thread {
                    let mut h = heap.lock().unwrap();
                    if h.try_alloc_cons_in(Generation::G0).is_some() {
                        local_count += 1;
                    }
                }
                local_count
            })
        })
        .collect();

    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total, n_threads * allocs_per_thread);

    // Heap is consistent after the parallel work.
    let h = heap.lock().unwrap();
    let g0 = h.count_pages_in_gen(Generation::G0);
    assert!(g0 > 0);
}

#[test]
fn shared_heap_can_gc_after_concurrent_alloc() {
    // After many threads allocate, one collects, and the heap
    // empties out (no roots). Verifies the GC works when called
    // from a context where multiple threads previously held the
    // mutex.
    let heap = Arc::new(Mutex::new(
        Heap::with_reservation(16 * 64 * 1024),
    ));
    let n_threads = 4;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let heap = Arc::clone(&heap);
            thread::spawn(move || {
                for _ in 0..500 {
                    let mut h = heap.lock().unwrap();
                    h.try_alloc_cons_in(Generation::G0);
                }
            })
        })
        .collect();
    for h in handles { h.join().unwrap(); }

    let mut h = heap.lock().unwrap();
    let before = h.count_pages_in_gen(Generation::G0);
    h.evacuate_from_word_roots(Generation::G0, Generation::G1, &mut []);
    let after = h.count_pages_in_gen(Generation::G0);
    assert!(before > 0);
    assert_eq!(after, 0, "no roots, G0 should be fully reclaimed");
}

// =========================================================================
// Lock-free reads on shared &Heap
// =========================================================================

#[test]
fn read_only_accessors_work_concurrently() {
    // Build a heap, then have many threads read its stats
    // concurrently. The stats methods take &self, so this is the
    // pattern a future "GC stats endpoint" would use.
    let mut heap = Heap::with_reservation(16 * 64 * 1024);
    for _ in 0..1000 {
        heap.try_alloc_cons_in(Generation::G0);
    }
    let heap = Arc::new(heap);
    let n_threads = 8;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let heap = Arc::clone(&heap);
            thread::spawn(move || {
                let mut acc = 0usize;
                for _ in 0..1000 {
                    acc = acc.wrapping_add(heap.committed_pages());
                    acc = acc.wrapping_add(heap.count_pages_in_gen(Generation::G0));
                    acc = acc.wrapping_add(heap.count_pages_in_gen(Generation::G1));
                    acc = acc.wrapping_add(heap.count_pages_in_gen(Generation::Tenured));
                    acc = acc.wrapping_add(heap.page_count());
                    acc = acc.wrapping_add(heap.committed_bytes());
                }
                acc
            })
        })
        .collect();

    let mut totals = Vec::new();
    for h in handles {
        totals.push(h.join().unwrap());
    }
    // All threads saw the SAME state (no concurrent mutator), so
    // their accumulated reads should all be equal.
    let first = totals[0];
    for (i, t) in totals.iter().enumerate() {
        assert_eq!(*t, first, "thread {i} saw different totals");
    }
}

// =========================================================================
// What we deliberately can't do
// =========================================================================

// The compiler enforces that try_alloc_cons_in (&mut self) cannot be
// called concurrently on the same heap. You can't write a test that
// races — the borrow checker rejects it. That's the type-system
// signature of "single-mutator-only today."
//
// To get concurrent allocation, the API would need:
//   - Per-thread TLABs holding pre-reserved page slots.
//   - A safepoint protocol so the collector can park other threads.
//   - A poll-word the JIT emits at back edges.
// These are NCL design-doc Phase 4 work; not present in newgc-core.
