//! Page heap: raw memory infrastructure.
//!
//! Sub-phase 2 of the Phase 3 plan in `docs/GC_DESIGN.md`. This
//! file ONLY handles the kernel-level memory dance — reserve a
//! large virtual range, commit individual pages on demand,
//! decommit pages when they're no longer needed. No object
//! semantics, no GC, no page descriptors — those live in sub-
//! phases 3+.
//!
//! ## Design
//!
//! A `PageHeap` owns a fixed-size virtual reservation (default
//! 2 GB) divided into 64 KB pages. Each page is in one of two
//! states:
//!
//!   - **Reserved-but-uncommitted** — address range valid but no
//!     physical backing. Reading or writing the page faults.
//!   - **Committed** — backed by pages in RAM / page-file.
//!     Read/write succeeds.
//!
//! The reservation lives for the process lifetime; only individual
//! pages move between states. `Drop` releases the entire
//! reservation back to the OS via `VirtualFree(MEM_RELEASE)`.
//!
//! ## Page size choice
//!
//! 64 KB matches Windows' `VirtualAlloc` allocation granularity —
//! the smallest unit `VirtualAlloc` will hand out as a separate
//! allocation. Using anything smaller would mean multiple
//! "logical" pages share a single VirtualAlloc-granule and we
//! couldn't independently decommit them. On Linux the page size
//! is 4 KB but mmap with MAP_FIXED handles arbitrary alignments,
//! so 64 KB still works fine cross-platform — just a bit chunkier
//! than necessary.
//!
//! Each page is 8192 cells (64 KB / 8 bytes-per-cell) → ~16 bits
//! of object addresses can be encoded inside-a-page if needed.
//!
//! ## Concurrent commit
//!
//! Multiple Lisp threads will hit this when fresh pages are
//! needed during allocation. `commit_page` takes a per-heap mutex,
//! checks the commit-bit, calls `VirtualAlloc(MEM_COMMIT)` if not
//! already committed, sets the bit, drops the lock. Idempotent.
//! Read path (`is_committed`) is a relaxed atomic load — no lock.
//!
//! ## Non-Windows
//!
//! Falls back to a single `Box<[u8]>` allocation with all "pages"
//! permanently "committed" (since Rust's allocator commits at the
//! OS layer anyway). Decommit is a no-op. Proper
//! `mmap(MAP_NORESERVE)` + `madvise(MADV_DONTNEED)` support is
//! future work — Windows is NCL's primary platform.

use std::marker::PhantomData;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::heap_common::CardTable;
use crate::traits::HeapLayout;

use super::alloc::{AllocRegion, PageStartBits};
use super::page_desc::{Generation, PageDesc, PageKind};
use super::shared::SharedHeap;

/// Size of a single page in bytes. 64 KB matches Windows'
/// VirtualAlloc allocation granularity (the smallest size that
/// VirtualAlloc will return as a separately-decommittable region).
pub const PAGE_SIZE_BYTES: usize = 64 * 1024;

/// Size of a page in cells (64-bit words).
pub const PAGE_SIZE_CELLS: usize = PAGE_SIZE_BYTES / 8;

/// Default reservation size: 2 GB → 32768 pages. Sized for a
/// long-running session with plenty of headroom. Costs ~32 KB of
/// commit-bitmap storage and one entry in the OS VAD tree; no
/// physical RAM until pages are committed.
pub const DEFAULT_RESERVATION_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// The page-heap reservation.
///
/// `Send + Sync` because all interior mutability goes through atomic
/// loads/stores on `committed_bits` and through `commit_lock` for
/// the VirtualAlloc calls themselves.
pub struct PageHeap<L: HeapLayout> {
    /// Phantom binding to the language layout. Zero-sized; carries
    /// the type parameter through to all methods.
    pub(super) _phantom: PhantomData<fn() -> L>,
    /// Backing storage. Either an OS reservation (Windows) or a
    /// Box-backed fully-committed fallback.
    storage: Backing,
    /// Number of pages in the reservation.
    n_pages: usize,
    /// Per-page commit-state bitmap. One bit per page, packed into
    /// `AtomicU64` words for cache efficiency. Bit `i % 64` of word
    /// `i / 64` is set when page `i` is committed.
    ///
    /// Atomics are used so `is_committed` reads can be lock-free.
    /// Writes go through `commit_lock` to serialize the
    /// `VirtualAlloc(MEM_COMMIT)` call itself (the OS handles
    /// concurrent commits gracefully but we'd waste system calls).
    committed_bits: Vec<AtomicU64>,
    /// Count of currently-committed pages. Atomic for lock-free
    /// reads. Reported by `committed_pages()` for diagnostics and
    /// the page-heap's `(gc-stats)` extension.
    committed_count: AtomicUsize,
    /// Serializes commit / decommit calls so two threads can't race
    /// on the same page. Held briefly across one `VirtualAlloc`
    /// or `VirtualFree(MEM_DECOMMIT)` call.
    commit_lock: Mutex<()>,
    /// Per-page metadata table. Parallel array to the page
    /// reservation: `descs[i]` describes page `i`. 12 bytes per
    /// entry × `n_pages` entries.
    ///
    /// Sub-phase 3 of `docs/GC_DESIGN.md`. Accessed only during
    /// stop-the-world GC for now (no atomics, no lock). Sub-phase 9
    /// will add atomic-field variants for the fields the write
    /// barrier needs to read from mutator threads (most likely
    /// `gen` and `pin_byte`); the rest stay plain.
    ///
    /// `pub(super)` so sibling modules (`mark`, `pin`, `alloc`)
    /// can mutate descriptors directly during their own passes
    /// without going through accessor methods.
    pub(super) descs: Vec<PageDesc>,
    /// Stack of page indices believed Free — the O(1) source for
    /// `acquire_free_page` (which previously linear-scanned all
    /// descriptors per TLAB refill). Invariant: every page whose
    /// descriptor transitions to `Generation::Free` is pushed
    /// (`release_page_to_free`); seeded with all pages at
    /// construction. Entries can go stale — the contiguous
    /// large-object allocator claims Free pages without popping
    /// them — so consumers pop-and-validate against the
    /// descriptor, discarding entries that are no longer Free.
    /// A page can briefly appear twice (released, claimed by a
    /// large run, released again); the validate step makes the
    /// duplicate harmless.
    pub(super) free_page_list: Vec<usize>,
    /// `1 + (max page index ever transitioned out of Free)`, i.e. an
    /// upper bound on the touched prefix of the reservation. Pages at
    /// index `>= pages_high_water` have never been acquired, so they
    /// are still Free with start-bits and cards untouched since
    /// construction (all-zero). Monotonic — only grows — so it stays
    /// a valid bound even after pages are released. Lets the
    /// per-cycle O(n_pages) sweeps (card rebuild, descriptor scans)
    /// iterate `0..pages_high_water` instead of the full 2 GB
    /// reservation (~32K pages), which is the ~9 ms fixed floor of
    /// every minor on a near-empty heap.
    pub(super) pages_high_water: usize,
    /// Open allocation regions, one per (generation, kind). Indexed
    /// by `(generation_idx, kind_idx)` — see `region_index` for
    /// the encoding. Sub-phase 4 supports `Cons` and `Boxed`
    /// kinds across `G0`, `G1`, `Tenured` generations = 6
    /// regions; `Free` and `Large` get no region.
    alloc_regions: [[AllocRegion; 2]; 3],
    /// Lock-free shared heap state (MM-2): start-bit bitmap, card table,
    /// poison flag, and the alloc counter. Held behind an `Arc` so a
    /// mutator can touch these without the heap lock (MM-3) and so the
    /// collector's `&mut PageHeap` never aliases a mutator's reference.
    /// `start_bits` / `cards` / `poisoned` / `bytes_alloc_since_gc` are
    /// reached through `self.shared` + the existing accessors.
    pub(super) shared: Arc<SharedHeap>,
    /// Mark bitmap covering the whole reservation. One bit per
    /// cell, packed 64 cells per `u64` word. Bit `c % 64` of
    /// word `c / 64` is set when cell `c` is the start of a
    /// reachable object on the most recent mark pass.
    ///
    /// 32 MB for the 2 GB default reservation. Plain `Box<[u64]>`,
    /// not atomic, because mark is STW — exclusive `&mut self` on
    /// `PageHeap` keeps races impossible. Sub-phase 5 of the
    /// design doc; consumed by sub-phase 7 evacuation.
    mark_bits: Box<[u64]>,
    /// Hashtable of object starts (global cell indices) pinned by
    /// the most recent conservative pin scan. Populated by
    /// `pin_pointers_in_ranges`; queried by evacuation to decide
    /// "may we move this?"; cleared at end of GC cycle.
    ///
    /// Sub-phase 6 of the design doc. Two-level lookup: PageDesc's
    /// `pin_byte` is the fast path (one byte-load + bit test); the
    /// hashtable is only consulted on a pin-byte hit. False-
    /// positive rate on the page-level bitmap is acceptable because
    /// the hashtable refines.
    ///
    /// Plain `HashSet<usize>` for now — simple, well-understood.
    /// Sub-phase 7 may swap for a sorted Vec or hopscotch map if
    /// profiling demands.
    ///
    /// `pub(super)` so sibling modules (`pin`, future `evacuate`)
    /// can mutate without going through accessors.
    pub(super) pinned_cells: std::collections::HashSet<usize>,
    /// Explicit (FFI) pins — MM-0. Maps a pinned object's global start
    /// cell index to a refcount. Unlike `pinned_cells` (rebuilt every
    /// cycle by the conservative scan and wiped by `clear_all_pins`),
    /// this set is **persistent**: it survives collections and is
    /// re-applied into `pinned_cells` + page pin-bytes at the start of
    /// every evacuation (see `apply_explicit_pins`). An object stays at
    /// a fixed address from `pin()` until the matching `unpin()`, across
    /// any number of cycles — the guarantee FFI needs (object address
    /// escaped into Win32 / held by a callback for the process lifetime).
    pub(super) explicit_pins: std::collections::HashMap<usize, u32>,
    /// Per-page count of marked live object starts for the current
    /// recycling-enabled evacuation cycle. Zeroed when inactive.
    pub(super) recycle_live_counts: Vec<u16>,
    /// Generation whose per-page live counts are currently valid.
    /// `None` disables mid-evacuation page recycling.
    pub(super) recycle_live_counts_target: Option<Generation>,
    /// Most recent mark pass's total live start count in cells.
    pub(super) last_mark_live_cells: usize,
    /// Most recent mark pass's count of pages with at least one
    /// marked live start.
    pub(super) last_mark_live_pages: usize,
    /// Number of zero-live, unpinned pages reclaimed before the
    /// most recent evacuation started.
    pub(super) last_zero_live_pages_released: usize,
    /// Minor cycles since the last G0 → G1 promotion. Incremented
    /// by `collect_minor`; reset to 0 on the cycle that promotes.
    /// Sub-phase 8 of `docs/GC_DESIGN.md`.
    pub(super) minors_since_g0_promote: u32,
    /// G0-promotion events since the last G1 → Tenured promotion.
    /// Ticks only on cycles that already promoted G0; reset on the
    /// cycle that cascades into G1 promotion.
    pub(super) g0_promotes_since_g1_promote: u32,
    /// (MM-2: the soft card table moved into `shared`
    /// (`SharedHeap::cards`); reached via `self.cards()` / `self.shared.cards`.)
    /// Most recent pin-scan result (n_objects, n_cells), surfaced
    /// to `(gc-stats)` via `last_pin_summary`. Updated by every
    /// `pin_pointers_in_ranges`; sub-phase 11b populates the
    /// `n_cells` field too (currently always 0 — `PinScanResult`
    /// hasn't computed object sizes yet).
    pub(super) last_pin_summary: (usize, usize),
    /// Soft cap on the number of G0 pages before the allocator
    /// refuses to open a fresh G0 page and forces a minor cycle.
    /// Set from `young_bytes` in `PageHeap::new`; defaults to
    /// `n_pages` (effectively unlimited) in `with_reservation`.
    ///
    /// This is the page-heap analogue of the semispace "young is
    /// full" trigger. Without it, the page-heap freely promotes
    /// G0 pages out of the shared reservation and `MINOR-GCS`
    /// can stay zero indefinitely.
    pub(super) young_page_cap: usize,

    // -- Sub-phase 10: trigger policy ------------------------------------
    //
    // The allocator bumps `bytes_alloc_since_gc` after every cell
    // it hands out. `should_collect()` returns true once that
    // counter exceeds `auto_gc_trigger_bytes`. After each collection
    // the trigger is recomputed as
    //   `current_alloc + max(budget_min, 0.5 * tenured_used)`
    // so older heaps with more tenured data get longer cycles
    // between collections (the absolute allocation budget grows
    // with the live set, matching SBCL's GENCGC policy).
    /// (MM-2: `bytes_alloc_since_gc` moved into `shared` as `AtomicUsize`;
    /// reached via `self.shared.bytes_alloc_since_gc`.)
    /// Threshold for `should_collect`. Updated by `collect_auto`.
    pub(super) auto_gc_trigger_bytes: usize,
    /// Minimum allocation budget between collections. Default 8 MB.
    pub(super) gc_budget_min_bytes: usize,
    /// Tenured-fill threshold for `should_collect_major`. Basis
    /// points (`10000 = 100%`). Default 7500 = 75% of reservation.
    pub(super) tenured_full_threshold_bps: u32,
    // (MM-2: the `poisoned` flag moved into `shared` as `AtomicBool`;
    // reached via `self.is_poisoned()` and `self.shared.poisoned`.)
}

enum Backing {
    /// Box-backed fallback for platforms without an OS-level
    /// reservation primitive, or for tests that want a small fully-
    /// committed reservation. All "pages" are always "committed";
    /// `commit_page` and `decommit_page` are no-ops on this path.
    Boxed(Box<[u8]>),
    /// Windows `VirtualAlloc(MEM_RESERVE)` reservation. Pages are
    /// individually committed/decommitted via `MEM_COMMIT` /
    /// `MEM_DECOMMIT`.
    #[cfg(windows)]
    Virtual {
        base: *mut u8,
        reserved_bytes: usize,
    },
    /// Unix `mmap(PROT_NONE | MAP_PRIVATE | MAP_ANONYMOUS |
    /// MAP_NORESERVE)` reservation. Pages are "committed" by
    /// `mprotect(PROT_READ | PROT_WRITE)` and "decommitted" by
    /// `madvise(MADV_DONTNEED)` + `mprotect(PROT_NONE)`.
    /// `MADV_DONTNEED` returns physical pages to the OS without
    /// releasing the address range; the next read/write re-faults
    /// and re-allocates zero pages.
    #[cfg(unix)]
    Mmap {
        base: *mut u8,
        reserved_bytes: usize,
    },
}

// SAFETY: Backing::Virtual holds a raw pointer to a VirtualAlloc'd
// region. The region is process-lifetime stable and access is
// mediated by the commit-bit bitmap + commit_lock. Box<[u8]> is
// naturally Send+Sync.
unsafe impl Send for Backing {}
unsafe impl Sync for Backing {}

impl Backing {
    fn base(&self) -> *mut u8 {
        match self {
            Backing::Boxed(b) => b.as_ptr() as *mut u8,
            #[cfg(windows)]
            Backing::Virtual { base, .. } => *base,
            #[cfg(unix)]
            Backing::Mmap { base, .. } => *base,
        }
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        match self {
            Backing::Boxed(_) => {} // Box drop handles it
            #[cfg(windows)]
            Backing::Virtual { base, .. } => {
                // VirtualFree(addr, 0, MEM_RELEASE) drops the entire
                // reservation (committed + uncommitted). Size MUST
                // be 0 for MEM_RELEASE.
                use windows::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
                unsafe {
                    let _ = VirtualFree(*base as *mut _, 0, MEM_RELEASE);
                }
            }
            #[cfg(unix)]
            Backing::Mmap { base, reserved_bytes } => {
                // munmap releases the entire reservation.
                unsafe {
                    libc::munmap(*base as *mut _, *reserved_bytes);
                }
            }
        }
    }
}

impl<L: HeapLayout> PageHeap<L> {
    /// Coordinator-facing constructor. Mirrors `Heap::new(young_bytes,
    /// old_bytes)` so `GcCoordinator::new` can use the same signature
    /// for either backend under build-time feature selection.
    ///
    /// For the page heap, both byte counts feed into a single
    /// reservation: total = `young_bytes + old_bytes`, rounded up to
    /// a whole number of 64 KB pages, with a **4-page minimum**
    /// (256 KB). The 4-page floor matters because:
    /// - Within-gen evacuation (`G0 → G0` on a non-threshold minor)
    ///   needs at least one Free page to copy survivors into, AND
    ///   the original page still in G0 at the time the BFS runs.
    /// - Cascading promotion wants a Free page for the destination
    ///   cohort plus working slack for the BFS.
    /// - The sub-phase 7 mid-evacuation OOM panic is avoided on
    ///   typical test configs that pass 32 KB / 32 KB sizes.
    pub fn new(young_bytes: usize, old_bytes: usize) -> Self {
        const MIN_BYTES: usize = 4 * PAGE_SIZE_BYTES;
        let bytes = (young_bytes + old_bytes).max(MIN_BYTES);
        let mut heap = Self::with_reservation(bytes);
        // Make `young_bytes` a real soft cap: the allocator stops
        // opening fresh G0 pages once G0 reaches this many pages,
        // forcing a minor cycle. Floor at 2 so a within-gen
        // evacuation can copy survivors into at least one page
        // while the other still holds the from-data.
        let cap_pages = (young_bytes / PAGE_SIZE_BYTES).max(2);
        heap.young_page_cap = cap_pages.min(heap.n_pages);
        heap
    }

    /// Internal / test constructor: reserve `reserved_bytes` of
    /// address space rounded up to a whole number of pages
    /// (64 KB each). On Windows uses `VirtualAlloc(MEM_RESERVE,
    /// PAGE_NOACCESS)`; pages must be individually committed via
    /// `commit_page` before use. On non-Windows allocates a
    /// `Box<[u8]>` of the same size with all pages permanently
    /// "committed" (the kernel decommit semantics aren't faithfully
    /// reproduced — proper mmap-based support is future work).
    pub fn with_reservation(reserved_bytes: usize) -> Self {
        let n_pages = reserved_bytes.div_ceil(PAGE_SIZE_BYTES);
        let total_bytes = n_pages * PAGE_SIZE_BYTES;
        let n_bitmap_words = n_pages.div_ceil(64);
        let committed_bits = (0..n_bitmap_words).map(|_| AtomicU64::new(0)).collect();
        // Per-page metadata table — every page starts as Free.
        // ~12 bytes × n_pages of allocation (384 KB for the 2 GB
        // default reservation; tiny compared to what it describes).
        let descs = vec![PageDesc::FREE; n_pages];
        // Open allocation regions, all empty (no current page).
        // Indexed via `region_index(generation, kind)`. Sub-phase 4
        // supports 6 regions: {G0, G1, Tenured} × {Cons, Boxed}.
        let alloc_regions: [[AllocRegion; 2]; 3] = [
            [
                AllocRegion::empty(Generation::G0, PageKind::Cons),
                AllocRegion::empty(Generation::G0, PageKind::Boxed),
            ],
            [
                AllocRegion::empty(Generation::G1, PageKind::Cons),
                AllocRegion::empty(Generation::G1, PageKind::Boxed),
            ],
            [
                AllocRegion::empty(Generation::Tenured, PageKind::Cons),
                AllocRegion::empty(Generation::Tenured, PageKind::Boxed),
            ],
        ];
        // Global start-bit bitmap. 2 bits per cell, 32 cells per
        // u64 word, n_pages × PAGE_SIZE_CELLS cells total. For the
        // 2 GB default reservation: 32768 × 8192 / 32 = 8M words
        // = 64 MB.
        let total_cells = n_pages * PAGE_SIZE_CELLS;
        let n_start_words = total_cells.div_ceil(32);
        let start_vec: Vec<AtomicU64> =
            (0..n_start_words).map(|_| AtomicU64::new(0)).collect();
        let start_bits: PageStartBits = Arc::from(start_vec.into_boxed_slice());
        // Mark bitmap: 1 bit per cell, 64 cells per u64.
        let n_mark_words = total_cells.div_ceil(64);
        let mark_bits: Box<[u64]> = vec![0u64; n_mark_words].into_boxed_slice();
        // Pinned-cells set starts empty — no scan run yet.
        let pinned_cells = std::collections::HashSet::new();
        // Explicit (FFI) pins — persistent, starts empty.
        let explicit_pins = std::collections::HashMap::new();
        let recycle_live_counts = vec![0u16; n_pages];
        // Card table covering the whole reservation. Same 512-byte
        // card granularity as the semispace heap so the IR-level
        // barrier shape is identical.
        let cards = Arc::new(CardTable::new(total_bytes));
        // MM-2: bundle the lock-free shared fields. Moved into one of
        // the cfg-gated literals below (only one compiles per platform).
        let shared = Arc::new(SharedHeap::new(start_bits, cards));

        #[cfg(windows)]
        {
            use windows::Win32::System::Memory::{
                VirtualAlloc, MEM_RESERVE, PAGE_NOACCESS,
            };
            let base = unsafe {
                VirtualAlloc(None, total_bytes, MEM_RESERVE, PAGE_NOACCESS)
            };
            if base.is_null() {
                panic!(
                    "PageHeap::new: VirtualAlloc(MEM_RESERVE, {total_bytes}) failed"
                );
            }
            PageHeap {
                _phantom: PhantomData,
                storage: Backing::Virtual {
                    base: base as *mut u8,
                    reserved_bytes: total_bytes,
                },
                n_pages,
                committed_bits,
                committed_count: AtomicUsize::new(0),
                commit_lock: Mutex::new(()),
                descs,
                free_page_list: (0..n_pages).rev().collect(),
                pages_high_water: 0,
                alloc_regions,
                shared,
                mark_bits,
                pinned_cells,
                explicit_pins,
                recycle_live_counts,
                recycle_live_counts_target: None,
                last_mark_live_cells: 0,
                last_mark_live_pages: 0,
                last_zero_live_pages_released: 0,
                minors_since_g0_promote: 0,
                g0_promotes_since_g1_promote: 0,
                last_pin_summary: (0, 0),
                young_page_cap: n_pages,
                auto_gc_trigger_bytes: 8 * 1024 * 1024,
                gc_budget_min_bytes: 8 * 1024 * 1024,
                tenured_full_threshold_bps: 7500,
            }
        }
        #[cfg(unix)]
        {
            // mmap(PROT_NONE | MAP_PRIVATE | MAP_ANONYMOUS |
            // MAP_NORESERVE). PROT_NONE means reading or writing
            // faults — pages must be commit-promoted via mprotect
            // before use. MAP_NORESERVE tells the kernel not to
            // pre-reserve swap for the address range (matches
            // VirtualAlloc(MEM_RESERVE) semantics).
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    total_bytes,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if base == libc::MAP_FAILED {
                panic!(
                    "PageHeap::new: mmap(PROT_NONE, {total_bytes}) failed: errno={}",
                    std::io::Error::last_os_error()
                );
            }
            return PageHeap {
                _phantom: PhantomData,
                storage: Backing::Mmap {
                    base: base as *mut u8,
                    reserved_bytes: total_bytes,
                },
                n_pages,
                committed_bits,
                committed_count: AtomicUsize::new(0),
                commit_lock: Mutex::new(()),
                descs,
                free_page_list: (0..n_pages).rev().collect(),
                pages_high_water: 0,
                alloc_regions,
                shared,
                mark_bits,
                pinned_cells,
                explicit_pins,
                recycle_live_counts,
                recycle_live_counts_target: None,
                last_mark_live_cells: 0,
                last_mark_live_pages: 0,
                last_zero_live_pages_released: 0,
                minors_since_g0_promote: 0,
                g0_promotes_since_g1_promote: 0,
                last_pin_summary: (0, 0),
                young_page_cap: n_pages,
                auto_gc_trigger_bytes: 8 * 1024 * 1024,
                gc_budget_min_bytes: 8 * 1024 * 1024,
                tenured_full_threshold_bps: 7500,
            };
        }
        #[cfg(not(any(windows, unix)))]
        {
            // Box-backed fallback for platforms without an OS
            // reservation primitive. All "pages" are permanently
            // "committed"; decommit is a no-op (the kernel may not
            // release pages even if we want it to).
            let boxed = vec![0u8; total_bytes].into_boxed_slice();
            // Pre-fill the commit bitmap so `is_committed` returns
            // true uniformly — matches the production behaviour of
            // "yes this address is backed."
            for w in &committed_bits {
                w.store(u64::MAX, Ordering::Relaxed);
            }
            PageHeap {
                _phantom: PhantomData,
                storage: Backing::Boxed(boxed),
                n_pages,
                committed_bits,
                committed_count: AtomicUsize::new(n_pages),
                commit_lock: Mutex::new(()),
                descs,
                free_page_list: (0..n_pages).rev().collect(),
                pages_high_water: 0,
                alloc_regions,
                shared,
                mark_bits,
                pinned_cells,
                explicit_pins,
                recycle_live_counts,
                recycle_live_counts_target: None,
                last_mark_live_cells: 0,
                last_mark_live_pages: 0,
                last_zero_live_pages_released: 0,
                minors_since_g0_promote: 0,
                g0_promotes_since_g1_promote: 0,
                last_pin_summary: (0, 0),
                young_page_cap: n_pages,
                auto_gc_trigger_bytes: 8 * 1024 * 1024,
                gc_budget_min_bytes: 8 * 1024 * 1024,
                tenured_full_threshold_bps: 7500,
            }
        }
    }

    /// Reservation base address. Constant for the lifetime of the
    /// heap.
    pub fn base_ptr(&self) -> *mut u8 {
        self.storage.base()
    }

    /// Number of pages in the reservation.
    pub fn page_count(&self) -> usize {
        self.n_pages
    }

    /// Total reserved size in bytes (= page_count * 64 KB).
    pub fn reserved_bytes(&self) -> usize {
        self.n_pages * PAGE_SIZE_BYTES
    }

    /// Number of currently-committed pages. Lock-free atomic
    /// read — useful for diagnostics and `(gc-stats)` extensions.
    pub fn committed_pages(&self) -> usize {
        self.committed_count.load(Ordering::Acquire)
    }

    /// Currently-committed bytes (= `committed_pages() * 64 KB`).
    pub fn committed_bytes(&self) -> usize {
        self.committed_pages() * PAGE_SIZE_BYTES
    }

    /// Mark the card containing `slot_addr` as dirty. The soft card
    /// barrier — mutators call this after every store of a
    /// heap-pointer Word into an object that may live in old gen.
    /// The next minor GC will scan dirty cards on G1/Tenured pages
    /// for cross-gen pointers into G0, picking up the new pointer
    /// without the mutator having to register it as an explicit root.
    ///
    /// Cheap: one byte store via `AtomicU8::store(Relaxed)`. Safe to
    /// call unconditionally — false positives just keep a card dirty
    /// for one extra cycle. Addresses outside the reservation are a
    /// no-op (the underlying `mark_offset` clamps).
    ///
    // -- Sub-phase 10: trigger policy ------------------------------------

    /// Returns true when the mutator has allocated enough bytes
    /// since the last collection to warrant a GC cycle. Cheap to
    /// call frequently — a single load and compare.
    ///
    /// Pair with `collect_auto` to let the GC pick minor vs major
    /// based on Tenured pressure:
    ///
    /// ```ignore
    /// if heap.should_collect() {
    ///     heap.collect_auto(|evac| visit_my_roots(evac));
    /// }
    /// ```
    ///
    /// Sub-phase 10 of `docs/GC_DESIGN.md`.
    #[inline]
    pub fn should_collect(&self) -> bool {
        self.shared.bytes_alloc_since_gc.load(Ordering::Relaxed) >= self.auto_gc_trigger_bytes
    }

    /// Returns true when Tenured occupancy has exceeded the
    /// configured fraction of the total reservation. Used by
    /// `collect_auto` to upgrade a minor to a major.
    pub fn should_collect_major(&self) -> bool {
        let tenured_bytes = self.tenured_used_bytes();
        let cap_bytes = self.reserved_bytes();
        // bps = basis points; 7500 = 75%. Compare scaled values to
        // avoid floating-point.
        tenured_bytes.saturating_mul(10000)
            >= cap_bytes.saturating_mul(self.tenured_full_threshold_bps as usize)
    }

    /// Run a collection automatically — chooses minor vs major
    /// based on `should_collect_major`. After the cycle, recomputes
    /// the next trigger threshold as
    /// `current_alloc + max(gc_budget_min_bytes, 0.5 * tenured_used)`.
    /// SBCL's GENCGC trigger heuristic.
    pub fn collect_auto<F>(
        &mut self,
        visit_roots: F,
    ) -> super::cycle::CollectResult
    where
        F: FnMut(&mut super::evac::PageEvacuator<'_, L>),
    {
        let pick_major = self.should_collect_major();
        let result = if pick_major {
            self.collect_major(visit_roots)
        } else {
            self.collect_minor(visit_roots)
        };
        self.recompute_auto_trigger();
        result
    }

    /// Recompute `auto_gc_trigger_bytes` after a collection.
    /// Budget = max(min, 0.5 * tenured_used). Called by
    /// `collect_auto` and exposed so clients invoking the explicit
    /// `collect_minor` / `collect_major` paths can keep the
    /// auto-trigger in sync.
    pub fn recompute_auto_trigger(&mut self) {
        let tenured_used = self.tenured_used_bytes();
        let budget = (tenured_used / 2).max(self.gc_budget_min_bytes);
        // Reset alloc counter; next trigger fires after `budget`
        // more bytes.
        self.shared.bytes_alloc_since_gc.store(0, Ordering::Relaxed);
        self.auto_gc_trigger_bytes = budget;
    }

    /// Total bytes used by Tenured pages. Computed from page
    /// descriptors.
    pub fn tenured_used_bytes(&self) -> usize {
        self.descs
            .iter()
            .filter(|d| d.generation == Generation::Tenured)
            .map(|d| d.words_used as usize * 8)
            .sum()
    }

    /// Bytes allocated since the last collection. Diagnostic.
    pub fn bytes_alloc_since_gc(&self) -> usize {
        self.shared.bytes_alloc_since_gc.load(Ordering::Relaxed)
    }

    /// Current auto-GC trigger threshold (bytes). Diagnostic.
    pub fn auto_gc_trigger_bytes(&self) -> usize {
        self.auto_gc_trigger_bytes
    }

    /// Set the minimum allocation budget between collections.
    /// Default is 8 MB. Lower for tighter heaps / more aggressive
    /// GC; higher for throughput-sensitive workloads. Recomputes
    /// the trigger threshold immediately so the new setting takes
    /// effect from the next `should_collect()` call.
    pub fn set_gc_budget_min_bytes(&mut self, bytes: usize) {
        self.gc_budget_min_bytes = bytes.max(1);
        self.recompute_auto_trigger();
    }

    /// Set the Tenured-fill threshold for `should_collect_major`,
    /// in basis points (10000 = 100%). Default 7500 = 75%.
    pub fn set_tenured_full_threshold_bps(&mut self, bps: u32) {
        self.tenured_full_threshold_bps = bps.min(10000);
    }

    /// Snapshot of the heap's state. Cheap to compute (one walk of
    /// the `descs` array). Stable struct: every public diagnostic
    /// getter on `PageHeap` is also a field here, so clients can
    /// take a single `stats()` and render a complete status without
    /// chasing a dozen methods.
    pub fn stats(&self) -> GcStats {
        let mut g0_pages = 0usize;
        let mut g1_pages = 0usize;
        let mut tenured_pages = 0usize;
        let mut free_pages = 0usize;
        let mut g0_used_bytes = 0usize;
        let mut g1_used_bytes = 0usize;
        let mut tenured_used_bytes = 0usize;
        for d in self.descs.iter() {
            let bytes = d.words_used as usize * 8;
            match d.generation {
                Generation::G0 => {
                    g0_pages += 1;
                    g0_used_bytes += bytes;
                }
                Generation::G1 => {
                    g1_pages += 1;
                    g1_used_bytes += bytes;
                }
                Generation::Tenured => {
                    tenured_pages += 1;
                    tenured_used_bytes += bytes;
                }
                Generation::Free => {
                    free_pages += 1;
                }
            }
        }
        GcStats {
            reserved_bytes: self.reserved_bytes(),
            committed_bytes: self.committed_bytes(),
            page_count: self.n_pages,
            committed_pages: self.committed_pages(),
            g0_pages,
            g1_pages,
            tenured_pages,
            free_pages,
            g0_used_bytes,
            g1_used_bytes,
            tenured_used_bytes,
            total_used_bytes: g0_used_bytes + g1_used_bytes + tenured_used_bytes,
            bytes_alloc_since_gc: self.shared.bytes_alloc_since_gc.load(Ordering::Relaxed),
            auto_gc_trigger_bytes: self.auto_gc_trigger_bytes,
            gc_budget_min_bytes: self.gc_budget_min_bytes,
            tenured_full_threshold_bps: self.tenured_full_threshold_bps,
            last_mark_live_bytes: self.last_mark_live_cells * 8,
            last_mark_live_pages: self.last_mark_live_pages,
            last_zero_live_pages_released: self.last_zero_live_pages_released,
            last_pin_summary_objects: self.last_pin_summary.0,
            last_pin_summary_cells: self.last_pin_summary.1,
            minors_since_g0_promote: self.minors_since_g0_promote,
            g0_promotes_since_g1_promote: self.g0_promotes_since_g1_promote,
        }
    }

    // -- Sub-phase 10 follow-up: try_* variants returning Result --------

    /// True if a previous `try_collect_*` returned `Err` and the
    /// heap is now in an indeterminate state (forwarding markers,
    /// partial pin set, stale cards). Further GC calls short-circuit
    /// to `Err`; allocation calls refuse. `Drop` is still safe.
    ///
    /// The intended client response is to drop the heap and either
    /// abort the program or rebuild from durable state.
    pub fn is_poisoned(&self) -> bool {
        self.shared.poisoned.load(Ordering::Acquire)
    }

    /// `collect_minor` with mid-evacuation-OOM caught and returned
    /// as a `Result::Err(GcError::MidEvacOom)`. Use this in clients
    /// that need to handle out-of-memory without process termination.
    ///
    /// **Important:** on `Err`, the heap is poisoned — see
    /// [`PageHeap::is_poisoned`]. Subsequent `try_collect_*` calls
    /// on the same heap will short-circuit to the same `Err`
    /// payload without attempting another cycle. The safe response
    /// is to drop the heap.
    pub fn try_collect_minor<F>(
        &mut self,
        visit_roots: F,
    ) -> Result<super::cycle::CollectResult, super::evac::GcError>
    where
        F: FnMut(&mut super::evac::PageEvacuator<'_, L>),
    {
        if let Some(err) = self.poisoned_err() {
            return Err(err);
        }
        let result = run_catching_oom(|| self.collect_minor(visit_roots));
        if result.is_err() {
            self.shared.poisoned.store(true, Ordering::Release);
        }
        result
    }

    /// `collect_major` with mid-evacuation-OOM caught as `Result::Err`.
    /// Poisons the heap on `Err` — see [`PageHeap::try_collect_minor`].
    pub fn try_collect_major<F>(
        &mut self,
        visit_roots: F,
    ) -> Result<super::cycle::CollectResult, super::evac::GcError>
    where
        F: FnMut(&mut super::evac::PageEvacuator<'_, L>),
    {
        if let Some(err) = self.poisoned_err() {
            return Err(err);
        }
        let result = run_catching_oom(|| self.collect_major(visit_roots));
        if result.is_err() {
            self.shared.poisoned.store(true, Ordering::Release);
        }
        result
    }

    /// `collect_auto` with mid-evacuation-OOM caught as `Result::Err`.
    /// Poisons the heap on `Err` — see [`PageHeap::try_collect_minor`].
    pub fn try_collect_auto<F>(
        &mut self,
        visit_roots: F,
    ) -> Result<super::cycle::CollectResult, super::evac::GcError>
    where
        F: FnMut(&mut super::evac::PageEvacuator<'_, L>),
    {
        if let Some(err) = self.poisoned_err() {
            return Err(err);
        }
        let result = run_catching_oom(|| self.collect_auto(visit_roots));
        if result.is_err() {
            self.shared.poisoned.store(true, Ordering::Release);
        }
        result
    }

    /// If poisoned, return `GcError::HeapPoisoned`. Used by
    /// `try_collect_*` to short-circuit before running another cycle.
    fn poisoned_err(&self) -> Option<super::evac::GcError> {
        if self.shared.poisoned.load(Ordering::Acquire) {
            Some(super::evac::GcError::HeapPoisoned)
        } else {
            None
        }
    }

    /// Sub-phase 9 of `docs/GC_DESIGN.md`.
    #[inline]
    pub fn mark_card_at(&self, slot_addr: *const u8) {
        let base = self.storage.base() as usize;
        let p = slot_addr as usize;
        if p < base {
            return;
        }
        let byte_offset = p - base;
        if byte_offset < self.reserved_bytes() {
            self.shared.cards.mark_offset(byte_offset);
        }
    }

    /// Sub-phase 9: rebuild the card table for pages in
    /// `Generation::G1` and `Generation::Tenured` by examining
    /// actual cell contents.
    ///
    /// Called at end-of-collection by `collect_minor` and
    /// `collect_major`. Replaces the "incremental" approach of
    /// marking cards during the mutator's `mark_card_at` call plus
    /// some carry-over heuristic — that approach loses card-marks
    /// when objects move between pages during evacuation.
    ///
    /// Cost: iterates the *touched prefix* of the reservation
    /// (`0..pages_high_water`), not the full 2 GB reservation. Pages
    /// at index `>= pages_high_water` were never acquired, so their
    /// cards are still all-zero from construction and need no
    /// clearing. Before this bound, sweeping all ~32K mostly-Free
    /// reservation pages (128 card-clears each) was the ~9 ms fixed
    /// floor of every minor on a near-empty heap. The per-cell
    /// pointer scan over the *live* pages within the prefix remains —
    /// that is the live-proportional part of the pause.
    pub(super) fn rebuild_cards_for_old_gens(&self) {
        let base = self.storage.base() as *const u64;
        let cards_per_page = PAGE_SIZE_BYTES / crate::heap_common::CARD_SIZE_BYTES;
        for page_idx in 0..self.pages_high_water {
            let page_gen = self.descs[page_idx].generation;
            if matches!(page_gen, Generation::Free) {
                // Free pages: clear all cards. The cells are zeroed
                // and shouldn't track any pointer state.
                let page_first_card = page_idx * cards_per_page;
                for c in 0..cards_per_page {
                    self.shared.cards.clear(page_first_card + c);
                }
                continue;
            }
            // For any live page (G0, G1, Tenured), scan cells and
            // mark cards that contain heap pointers. G0 cards
            // persist because major cycles' G1→Tenured pass scans
            // G0 cards to find G0→G1 cross-gen pointers; clearing
            // them unconditionally would lose those pointers.
            // Minor cycles skip G0 cards via the page filter.
            //
            // Only scan the cards covering the page's *used* extent
            // (`words_used`). A page is bump-allocated from cell 0, so no
            // live object — hence no cross-gen pointer — exists at or
            // beyond `words_used`; cards past it cover only
            // never-written / dead cells and are simply cleared without
            // touching memory. evac (`evac.rs` first_cell+words_used) and
            // mark (`mark.rs` first_cell+words_used) already bound their
            // scans the same way; this makes the card rebuild match,
            // turning a full-page (8192-cell) classify on a 1%-live
            // Tenured page into a words_used-proportional one.
            let page_base_cell = page_idx * PAGE_SIZE_CELLS;
            let words_used = self.descs[page_idx].words_used as usize;
            let used_cards =
                words_used.div_ceil(crate::heap_common::CARD_SIZE_CELLS);
            for card_offset_in_page in 0..cards_per_page {
                let card_idx =
                    page_idx * cards_per_page + card_offset_in_page;
                if card_offset_in_page >= used_cards {
                    // Entirely beyond the live extent — no pointers.
                    self.shared.cards.clear(card_idx);
                    continue;
                }
                let card_first_cell = page_base_cell
                    + card_offset_in_page
                        * crate::heap_common::CARD_SIZE_CELLS;
                let mut has_heap_pointer = false;
                for c in card_first_cell
                    ..card_first_cell + crate::heap_common::CARD_SIZE_CELLS
                {
                    let cell = unsafe { *base.add(c) };
                    if matches!(
                        L::classify(cell),
                        crate::traits::WordKind::PointerCons(_)
                            | crate::traits::WordKind::PointerHeader(_)
                    ) {
                        has_heap_pointer = true;
                        break;
                    }
                }
                if has_heap_pointer {
                    self.shared.cards
                        .mark_offset(card_first_cell * 8);
                } else {
                    self.shared.cards.clear(card_idx);
                }
            }
        }
    }

    /// Pointer to the first byte of page `idx`. Panics if `idx >=
    /// page_count()`.
    pub fn page_ptr(&self, idx: usize) -> *mut u8 {
        assert!(idx < self.n_pages, "PageHeap::page_ptr: {idx} >= {}", self.n_pages);
        unsafe { self.storage.base().add(idx * PAGE_SIZE_BYTES) }
    }

    /// Page index containing `ptr`, or `None` if `ptr` is outside
    /// the reservation. Used by the conservative pinner and the
    /// write barrier to look up which page an address belongs to.
    pub fn page_of(&self, ptr: *const u8) -> Option<usize> {
        let base = self.storage.base() as usize;
        let end = base + self.reserved_bytes();
        let p = ptr as usize;
        if p >= base && p < end {
            Some((p - base) / PAGE_SIZE_BYTES)
        } else {
            None
        }
    }

    /// Is page `idx` currently committed? Lock-free atomic read.
    pub fn is_committed(&self, idx: usize) -> bool {
        if idx >= self.n_pages {
            return false;
        }
        let word = self.committed_bits[idx / 64].load(Ordering::Acquire);
        (word >> (idx % 64)) & 1 != 0
    }

    /// Commit page `idx` so its backing memory becomes accessible.
    /// Idempotent — if the page is already committed, this is a
    /// fast lock-free check followed by an early return.
    ///
    /// On Windows: `VirtualAlloc(MEM_COMMIT, PAGE_READWRITE)` on
    /// the page's range. On non-Windows (Box-backed) this is a
    /// no-op because the Rust allocator already committed the
    /// whole region.
    ///
    /// Returns `Ok(())` on success or after observing an existing
    /// commit; `Err(...)` if the OS commit call fails (page-file
    /// full, etc.).
    /// Bug #3 from the code review (docs/GC_DESIGN.md sub-phase
    /// 6.5): both `commit_page` and `decommit_page` previously
    /// took `&self`. Two threads racing — one committing, the
    /// other decommitting — could observe each other's mid-state
    /// and return Ok with the page in the wrong terminal state,
    /// causing an AV on the next write. The fix is to require
    /// `&mut self` so the borrow checker enforces exclusivity.
    /// The internal `commit_lock` Mutex is now redundant; we
    /// leave the field on the struct (avoiding churn) but no
    /// longer lock it inside these methods. Sub-phase 7 routinely
    /// decommits empty pages, so this protection becomes load-
    /// bearing then.
    /// Returns `Ok(true)` when this call performed the OS commit —
    /// in which case the page's memory is guaranteed all-zero
    /// (Windows `MEM_COMMIT` zero-fills, as does first-touch after
    /// the unix `madvise(MADV_DONTNEED)`/initial `PROT_NONE` mmap).
    /// `Ok(false)` means the page was already committed and may
    /// hold a prior tenant's bytes. The Box-backed fallback
    /// pre-fills the commit bitmap, so it always reports `false`.
    pub fn commit_page(&mut self, idx: usize) -> Result<bool, CommitError> {
        if idx >= self.n_pages {
            return Err(CommitError::OutOfRange(idx));
        }
        // Idempotent — already committed.
        if self.is_committed(idx) {
            return Ok(false);
        }
        #[cfg(windows)]
        {
            use windows::Win32::System::Memory::{
                VirtualAlloc, MEM_COMMIT, PAGE_READWRITE,
            };
            let addr = self.page_ptr(idx);
            let result = unsafe {
                VirtualAlloc(
                    Some(addr as *const _),
                    PAGE_SIZE_BYTES,
                    MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            if result.is_null() {
                return Err(CommitError::OsRefused);
            }
        }
        #[cfg(unix)]
        {
            // mprotect to enable read+write. The page was
            // mmap'd PROT_NONE so writing would have faulted.
            let addr = self.page_ptr(idx);
            let r = unsafe {
                libc::mprotect(
                    addr as *mut _,
                    PAGE_SIZE_BYTES,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
            if r != 0 {
                return Err(CommitError::OsRefused);
            }
        }
        // Set the commit bit + bump the count.
        let word_idx = idx / 64;
        let bit = 1u64 << (idx % 64);
        self.committed_bits[word_idx].fetch_or(bit, Ordering::AcqRel);
        self.committed_count.fetch_add(1, Ordering::AcqRel);
        // Zero-fill is an OS guarantee — only the windows/unix
        // paths above actually ran a syscall. On the Box-backed
        // fallback a decommit/recommit round-trip moves no memory,
        // so the prior bytes survive and we must report false.
        Ok(cfg!(any(windows, unix)))
    }

    /// Decommit page `idx`, returning its backing memory to the
    /// OS. The address range stays reserved, so the page can be
    /// re-committed later at the same address. Reads after
    /// decommit fault.
    ///
    /// On Windows: `VirtualFree(MEM_DECOMMIT)`. On non-Windows the
    /// Box-backed implementation can't actually decommit (Rust
    /// doesn't expose this), so this clears the bit but the memory
    /// stays resident. Diagnostics-only on that path.
    ///
    /// Idempotent on already-uncommitted pages. Takes `&mut self`
    /// (per bug #3 fix) so the borrow checker rules out the
    /// commit/decommit race.
    pub fn decommit_page(&mut self, idx: usize) -> Result<(), CommitError> {
        if idx >= self.n_pages {
            return Err(CommitError::OutOfRange(idx));
        }
        if !self.is_committed(idx) {
            return Ok(());
        }
        #[cfg(windows)]
        {
            use windows::Win32::System::Memory::{VirtualFree, MEM_DECOMMIT};
            let addr = self.page_ptr(idx);
            let ok = unsafe {
                VirtualFree(addr as *mut _, PAGE_SIZE_BYTES, MEM_DECOMMIT)
            };
            if ok.is_err() {
                return Err(CommitError::OsRefused);
            }
        }
        #[cfg(unix)]
        {
            // MADV_DONTNEED returns physical pages to the OS.
            // Then mprotect back to PROT_NONE so reads/writes
            // fault (matching the post-commit-bit-cleared state).
            // The next commit_page does mprotect(PROT_READ|WRITE)
            // and the kernel zero-fills.
            let addr = self.page_ptr(idx);
            let r1 = unsafe {
                libc::madvise(addr as *mut _, PAGE_SIZE_BYTES, libc::MADV_DONTNEED)
            };
            let r2 = unsafe {
                libc::mprotect(addr as *mut _, PAGE_SIZE_BYTES, libc::PROT_NONE)
            };
            if r1 != 0 || r2 != 0 {
                return Err(CommitError::OsRefused);
            }
        }
        let word_idx = idx / 64;
        let bit = 1u64 << (idx % 64);
        self.committed_bits[word_idx].fetch_and(!bit, Ordering::AcqRel);
        self.committed_count.fetch_sub(1, Ordering::AcqRel);

        // A decommitted page is unmapped — reading it faults. Clear the
        // reservation cards covering it so a later card scan *in the same
        // cycle* never dereferences it. This closes a stale-snapshot
        // hazard: `collect_minor`'s G1→Tenured cascade (and `collect_full`'s
        // passes) scan dirty cards against a page-descriptor snapshot taken
        // at cycle start, in which a G0 page that an earlier pass has since
        // released + decommitted still appears live (non-Free, ≠ from_gen),
        // so the filter would pass it and `visit_cell` would read freed,
        // unmapped memory (EXCEPTION_ACCESS_VIOLATION). Any cross-gen
        // pointers the page held are gone (its live objects were evacuated,
        // or it was dead), and old-gen cards are rebuilt from the live heap
        // at cycle end, so clearing here loses nothing.
        let cards_per_page = PAGE_SIZE_CELLS / crate::heap_common::CARD_SIZE_CELLS;
        let first_card = idx * cards_per_page;
        let n_cards = self.shared.cards.n_cards();
        for c in first_card..(first_card + cards_per_page).min(n_cards) {
            self.shared.cards.clear(c);
        }
        Ok(())
    }

    // -- Page descriptor accessors ------------------------------------
    //
    // Sub-phase 3 of `docs/GC_DESIGN.md`. Plain mutable access for
    // now (no atomics, no internal locking) — call sites are GC
    // passes running under stop-the-world. Sub-phase 9 may
    // refactor to atomic field access for write-barrier reads.

    /// Read a copy of the descriptor for page `idx`. Cheap — 12
    /// bytes copied. Panics if `idx >= page_count()`.
    pub fn desc(&self, idx: usize) -> PageDesc {
        assert!(
            idx < self.n_pages,
            "PageHeap::desc: {idx} >= {}",
            self.n_pages
        );
        self.descs[idx]
    }

    /// Mutable reference to the descriptor for page `idx`. Used by
    /// GC passes to update generation, kind, words_used, pin
    /// bitmap. Requires `&mut self` — production access goes
    /// through `MutexGuard<Box<dyn HeapBackend>>`.
    pub fn desc_mut(&mut self, idx: usize) -> &mut PageDesc {
        assert!(
            idx < self.n_pages,
            "PageHeap::desc_mut: {idx} >= {}",
            self.n_pages
        );
        &mut self.descs[idx]
    }

    /// Read-only slice of all descriptors. Useful for scanning
    /// passes that need to look at many pages without paying for
    /// per-page bounds checks.
    pub fn descs(&self) -> &[PageDesc] {
        &self.descs
    }

    /// Iterate page indices whose descriptor has the given
    /// generation. Order is ascending by page index — matches
    /// physical-page order in the reservation, which gives evac
    /// passes good cache locality.
    pub fn pages_in_gen<'a>(
        &'a self,
        target: Generation,
    ) -> impl Iterator<Item = usize> + 'a {
        self.descs
            .iter()
            .enumerate()
            .filter_map(move |(i, d)| if d.generation == target { Some(i) } else { None })
    }

    /// Count pages with the given generation. O(n_pages) — used by
    /// diagnostics and the trigger policy in sub-phase 10.
    pub fn count_pages_in_gen(&self, target: Generation) -> usize {
        self.descs.iter().filter(|d| d.generation == target).count()
    }

    // -- Allocation regions + start bits (sub-phase 4) ----------------

    /// Indexing helper. Returns `(generation_idx, kind_idx)` for
    /// the `alloc_regions` 2D array. Free / Large kinds and
    /// the Free generation have no region — passing them panics.
    fn region_index(generation: Generation, kind: PageKind) -> (usize, usize) {
        let gen_idx = match generation {
            Generation::G0 => 0,
            Generation::G1 => 1,
            Generation::Tenured => 2,
            Generation::Free => panic!("PageHeap: Free has no alloc region"),
        };
        let kind_idx = match kind {
            PageKind::Cons => 0,
            PageKind::Boxed => 1,
            other => panic!("PageHeap: no alloc region for kind {other:?}"),
        };
        (gen_idx, kind_idx)
    }

    /// Read-only view of the alloc region for a given
    /// (generation, kind). Used by allocators to check the
    /// fast-path fit before bumping.
    pub fn alloc_region(&self, generation: Generation, kind: PageKind) -> &AllocRegion {
        let (g, k) = Self::region_index(generation, kind);
        &self.alloc_regions[g][k]
    }

    /// Mutable view of the alloc region for a given
    /// (generation, kind). Allocators advance the bump offset and
    /// the current-page index through this.
    pub fn alloc_region_mut(
        &mut self,
        generation: Generation,
        kind: PageKind,
    ) -> &mut AllocRegion {
        let (g, k) = Self::region_index(generation, kind);
        &mut self.alloc_regions[g][k]
    }

    /// Cheap clone of the start-bit bitmap handle for mutators
    /// that need to set start bits from their alloc fast path
    /// without taking the heap lock. The mutator caches one of
    /// these at registration.
    pub fn start_bits_handle(&self) -> PageStartBits {
        Arc::clone(&self.shared.start_bits)
    }

    /// Clone of the `Arc<SharedHeap>` for a mutator's lock-free fast
    /// path (MM-3). The mutator caches this at registration and bumps /
    /// sets start bits / checks poison through it without the heap lock.
    pub fn shared_handle(&self) -> Arc<SharedHeap> {
        Arc::clone(&self.shared)
    }

    /// Internal access to the start-bit bitmap slice. Used by the
    /// allocator helpers in `alloc.rs`.
    pub(crate) fn start_bits_slice(&self) -> &[AtomicU64] {
        &self.shared.start_bits
    }

    // -- Mark bitmap (sub-phase 5) ------------------------------------

    /// Test whether the cell at global index `cell_idx` is marked.
    /// Caller is responsible for passing an in-range index — sub-
    /// phase 5 is the mark pass itself; downstream evacuation
    /// will treat unmarked cells as garbage.
    pub fn is_marked(&self, cell_idx: usize) -> bool {
        let w = cell_idx / 64;
        let b = cell_idx % 64;
        debug_assert!(
            w < self.mark_bits.len(),
            "is_marked: cell {cell_idx} past end"
        );
        (self.mark_bits[w] >> b) & 1 != 0
    }

    /// Mark the cell at global index `cell_idx`. Returns the
    /// previous mark state — true if the cell was already marked
    /// (i.e., this is a re-visit and the caller should NOT recurse
    /// into the payload). Mark BFS uses this as the "have I seen
    /// this object?" gate.
    pub fn mark_cell(&mut self, cell_idx: usize) -> bool {
        let w = cell_idx / 64;
        let b = cell_idx % 64;
        let prev = (self.mark_bits[w] >> b) & 1 != 0;
        self.mark_bits[w] |= 1u64 << b;
        prev
    }

    /// Clear mark bits across every page in `target` generation.
    /// Called at the start of a mark cycle so the bitmap reflects
    /// only "alive in this cycle." Other generations' bits are
    /// preserved — useful when a full GC marks one generation at
    /// a time without losing prior survivors.
    pub fn clear_mark_bits_in_gen(&mut self, target: Generation) {
        // Collect page indices first to avoid borrowing `self`
        // mutably twice. Fast — n_pages is at most 32768.
        let pages: Vec<usize> = self
            .descs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| if d.generation == target { Some(i) } else { None })
            .collect();
        for page_idx in pages {
            self.clear_mark_bits_for_page(page_idx);
        }
    }

    /// Clear mark bits for a single page. The page's cells span
    /// global indices `page_idx * PAGE_SIZE_CELLS` to
    /// `(page_idx + 1) * PAGE_SIZE_CELLS`. PAGE_SIZE_CELLS = 8192
    /// = 128 mark-bitmap words, page-aligned in the bitmap, so
    /// clearing is a tight loop over 128 `u64` writes.
    fn clear_mark_bits_for_page(&mut self, page_idx: usize) {
        let first_word = page_idx * PAGE_SIZE_CELLS / 64;
        let words_per_page = PAGE_SIZE_CELLS / 64;
        for w in first_word..first_word + words_per_page {
            self.mark_bits[w] = 0;
        }
    }

    /// Read-only access to the raw mark bitmap. Used by the mark
    /// pass internals in `mark.rs`.
    pub(crate) fn mark_bits_slice(&self) -> &[u64] {
        &self.mark_bits
    }

    /// Count marked cells in `target` generation. Diagnostics
    /// helper for the mark-pass tests; not on any hot path.
    pub fn count_marked_in_gen(&self, target: Generation) -> usize {
        let mut count = 0;
        for (page_idx, d) in self.descs.iter().enumerate() {
            if d.generation != target {
                continue;
            }
            let first_word = page_idx * PAGE_SIZE_CELLS / 64;
            let words_per_page = PAGE_SIZE_CELLS / 64;
            for w in first_word..first_word + words_per_page {
                count += self.mark_bits[w].count_ones() as usize;
            }
        }
        count
    }
}

/// One-shot snapshot of `PageHeap` state. Produced by
/// [`PageHeap::stats`]. Every public diagnostic field on the heap
/// appears here, so a status endpoint / `(gc-stats)` Lisp form /
/// log line can call `stats()` once and render everything from
/// the result.
///
/// All bytes are exact (not rounded). Cohort counters reflect the
/// state RIGHT NOW, not at the end of the last cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GcStats {
    // -- Capacity ---------------------------------------------------------
    pub reserved_bytes: usize,
    pub committed_bytes: usize,
    pub page_count: usize,
    pub committed_pages: usize,

    // -- Generation occupancy --------------------------------------------
    pub g0_pages: usize,
    pub g1_pages: usize,
    pub tenured_pages: usize,
    pub free_pages: usize,
    pub g0_used_bytes: usize,
    pub g1_used_bytes: usize,
    pub tenured_used_bytes: usize,
    pub total_used_bytes: usize,

    // -- Trigger policy (sub-phase 10) -----------------------------------
    pub bytes_alloc_since_gc: usize,
    pub auto_gc_trigger_bytes: usize,
    pub gc_budget_min_bytes: usize,
    pub tenured_full_threshold_bps: u32,

    // -- Last-cycle telemetry --------------------------------------------
    pub last_mark_live_bytes: usize,
    pub last_mark_live_pages: usize,
    pub last_zero_live_pages_released: usize,
    pub last_pin_summary_objects: usize,
    pub last_pin_summary_cells: usize,

    // -- Cohort counters (sub-phase 8) -----------------------------------
    pub minors_since_g0_promote: u32,
    pub g0_promotes_since_g1_promote: u32,
}

impl GcStats {
    /// Render as a single-line key=value diagnostic string. Good
    /// for log emission. Pairs with `Display`.
    pub fn render(&self) -> String {
        format!(
            "reserved={} committed={} pages={}/{} g0={}p/{}b g1={}p/{}b tenured={}p/{}b free={}p \
             alloc_since_gc={} trigger={} budget_min={} tenured_thresh_bps={} \
             last_mark={}b/{}p zero_live_released={} pin_objs={} pin_cells={} \
             minors_since_promote={} g0_promotes_since_g1_promote={}",
            self.reserved_bytes, self.committed_bytes,
            self.committed_pages, self.page_count,
            self.g0_pages, self.g0_used_bytes,
            self.g1_pages, self.g1_used_bytes,
            self.tenured_pages, self.tenured_used_bytes,
            self.free_pages,
            self.bytes_alloc_since_gc, self.auto_gc_trigger_bytes,
            self.gc_budget_min_bytes, self.tenured_full_threshold_bps,
            self.last_mark_live_bytes, self.last_mark_live_pages,
            self.last_zero_live_pages_released,
            self.last_pin_summary_objects, self.last_pin_summary_cells,
            self.minors_since_g0_promote, self.g0_promotes_since_g1_promote,
        )
    }
}

impl std::fmt::Display for GcStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render())
    }
}

/// Errors from `commit_page` / `decommit_page`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitError {
    /// `idx >= page_count()`.
    OutOfRange(usize),
    /// The OS refused the commit (typically: page-file exhausted)
    /// or decommit (very rare; usually a programming bug — passing
    /// an address that wasn't part of the reservation).
    OsRefused,
}

impl std::fmt::Display for CommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommitError::OutOfRange(idx) => write!(f, "page index {idx} out of range"),
            CommitError::OsRefused => write!(f, "OS refused the commit/decommit"),
        }
    }
}

impl std::error::Error for CommitError {}

/// Run a GC-cycle closure, catching the `GcStallError` that
/// mid-evacuation-OOM panics carry (`std::panic::panic_any`).
/// Re-raises any other panic.
fn run_catching_oom<R, F>(f: F) -> Result<R, super::evac::GcError>
where
    F: FnOnce() -> R,
{
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    match result {
        Ok(value) => Ok(value),
        Err(payload) => {
            if let Some(stall) = payload.downcast_ref::<super::evac::GcStallError>() {
                Err(super::evac::GcError::MidEvacOom(stall.clone()))
            } else {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cap the reservation size in tests so we don't ask the OS for
    /// 2 GB just to verify the bookkeeping.
    fn small_heap() -> PageHeap<crate::lisp_layout::LispLayout> {
        // 1 MB = 16 pages. Plenty to exercise page indexing without
        // wasting VAD space across thousands of test runs.
        PageHeap::<crate::lisp_layout::LispLayout>::with_reservation(1024 * 1024)
    }

    #[test]
    fn fresh_heap_has_no_committed_pages() {
        let h = small_heap();
        assert_eq!(h.page_count(), 16);
        assert_eq!(h.committed_pages(), 0);
        // Box-backed (non-Windows) flips this — the cfg!(windows)
        // gate keeps the assertion meaningful on the platform where
        // commit semantics actually exist.
        #[cfg(windows)]
        for i in 0..h.page_count() {
            assert!(!h.is_committed(i), "page {i} should be uncommitted");
        }
    }

    #[test]
    fn commit_single_page_roundtrips() {
        let mut h = small_heap();
        h.commit_page(3).expect("commit page 3");
        assert!(h.is_committed(3));
        #[cfg(windows)]
        assert_eq!(h.committed_pages(), 1);

        // Write and read through the committed page to prove it
        // really is backed memory.
        let ptr = h.page_ptr(3);
        unsafe {
            ptr.write(0xAB);
            ptr.add(PAGE_SIZE_BYTES - 1).write(0xCD);
            assert_eq!(ptr.read(), 0xAB);
            assert_eq!(ptr.add(PAGE_SIZE_BYTES - 1).read(), 0xCD);
        }
    }

    #[test]
    fn commit_then_decommit() {
        let mut h = small_heap();
        h.commit_page(5).unwrap();
        assert!(h.is_committed(5));
        h.decommit_page(5).unwrap();
        // On Box-backed, decommit clears the bit but the memory
        // stays resident. On VirtualAlloc-backed, the page is
        // genuinely decommitted.
        assert!(!h.is_committed(5));
    }

    #[test]
    fn commit_is_idempotent() {
        let mut h = small_heap();
        h.commit_page(7).unwrap();
        h.commit_page(7).unwrap();
        h.commit_page(7).unwrap();
        // Still exactly one page logically committed.
        // (On non-Windows the counter starts at page_count() — skip
        // the assertion there.)
        #[cfg(windows)]
        assert_eq!(h.committed_pages(), 1);
        assert!(h.is_committed(7));
    }

    #[test]
    fn decommit_uncommitted_is_noop() {
        let mut h = small_heap();
        h.decommit_page(2).unwrap();
        assert!(!h.is_committed(2));
    }

    #[test]
    fn out_of_range_returns_error() {
        let mut h = small_heap();
        assert_eq!(
            h.commit_page(9999),
            Err(CommitError::OutOfRange(9999))
        );
        assert_eq!(
            h.decommit_page(9999),
            Err(CommitError::OutOfRange(9999))
        );
        // is_committed silently returns false for out-of-range —
        // matches the "no such page is committed" intuition rather
        // than panicking.
        assert!(!h.is_committed(9999));
    }

    #[test]
    fn page_of_arithmetic_round_trip() {
        let h = small_heap();
        // First byte of page 0, first byte of page 5, last byte of
        // page 5, first byte of page 6.
        let base = h.base_ptr();
        unsafe {
            assert_eq!(h.page_of(base), Some(0));
            assert_eq!(h.page_of(base.add(5 * PAGE_SIZE_BYTES)), Some(5));
            assert_eq!(
                h.page_of(base.add(6 * PAGE_SIZE_BYTES - 1)),
                Some(5),
                "last byte of page 5 is still in page 5"
            );
            assert_eq!(h.page_of(base.add(6 * PAGE_SIZE_BYTES)), Some(6));
            // Outside the reservation: None.
            assert_eq!(h.page_of(base.wrapping_sub(1)), None);
            assert_eq!(
                h.page_of(base.add(h.reserved_bytes())),
                None,
                "byte just past end is outside"
            );
        }
    }

    #[test]
    fn page_ptr_addresses_are_64kb_aligned() {
        let h = small_heap();
        for i in 0..h.page_count() {
            let p = h.page_ptr(i) as usize;
            assert_eq!(p % PAGE_SIZE_BYTES, 0, "page {i} not aligned");
        }
    }

    #[test]
    fn fresh_heap_has_only_free_descriptors() {
        let h = small_heap();
        for i in 0..h.page_count() {
            let d = h.desc(i);
            assert_eq!(d, PageDesc::FREE, "page {i} should start FREE");
        }
        // Every generation count is zero except `Free`.
        assert_eq!(h.count_pages_in_gen(Generation::G0), 0);
        assert_eq!(h.count_pages_in_gen(Generation::G1), 0);
        assert_eq!(h.count_pages_in_gen(Generation::Tenured), 0);
        assert_eq!(h.count_pages_in_gen(Generation::Free), h.page_count());
    }

    #[test]
    fn descriptor_mutation_round_trip() {
        let mut h = small_heap();
        {
            let d = h.desc_mut(4);
            d.generation = Generation::G0;
            d.kind = super::super::page_desc::PageKind::Cons;
            d.words_used = 1234;
            d.scan_start_offset = 16;
            d.age = 2;
        }
        let d = h.desc(4);
        assert_eq!(d.generation, Generation::G0);
        assert_eq!(d.kind, super::super::page_desc::PageKind::Cons);
        assert_eq!(d.words_used, 1234);
        assert_eq!(d.scan_start_offset, 16);
        assert_eq!(d.age, 2);
    }

    #[test]
    fn pages_in_gen_filters_correctly() {
        let mut h = small_heap();
        // Assign a few pages to G0, one to G1, leave the rest Free.
        h.desc_mut(0).generation = Generation::G0;
        h.desc_mut(3).generation = Generation::G0;
        h.desc_mut(7).generation = Generation::G1;
        h.desc_mut(10).generation = Generation::G0;

        let g0_pages: Vec<usize> = h.pages_in_gen(Generation::G0).collect();
        assert_eq!(g0_pages, vec![0, 3, 10], "G0 page list");
        let g1_pages: Vec<usize> = h.pages_in_gen(Generation::G1).collect();
        assert_eq!(g1_pages, vec![7], "G1 page list");
        let tenured_pages: Vec<usize> = h.pages_in_gen(Generation::Tenured).collect();
        assert!(tenured_pages.is_empty(), "no tenured pages");

        assert_eq!(h.count_pages_in_gen(Generation::G0), 3);
        assert_eq!(h.count_pages_in_gen(Generation::G1), 1);
        assert_eq!(h.count_pages_in_gen(Generation::Free), 12);
    }

    #[test]
    fn descs_slice_matches_page_count() {
        let h = small_heap();
        assert_eq!(h.descs().len(), h.page_count());
        assert_eq!(h.descs().len(), 16);
    }

    #[test]
    fn pin_byte_round_trip_through_desc_mut() {
        let mut h = small_heap();
        h.desc_mut(2).set_pin(0);
        h.desc_mut(2).set_pin(7);
        assert!(h.desc(2).has_pins());
        assert!(h.desc(2).is_pinned(0));
        assert!(h.desc(2).is_pinned(7));
        assert!(!h.desc(2).is_pinned(3));
        h.desc_mut(2).clear_pins();
        assert!(!h.desc(2).has_pins());
    }

    #[test]
    #[should_panic(expected = "PageHeap::desc")]
    fn desc_out_of_range_panics() {
        let h = small_heap();
        let _ = h.desc(9999);
    }

    // The `concurrent_commit_is_safe` test was deleted as part of
    // bug-fix #3 (commit/decommit race). After the signature
    // change to `&mut self`, the test's pattern (`Arc<PageHeap>`
    // shared across threads, each calling `commit_page` via a
    // shared ref) no longer compiles — and that's the point: the
    // borrow checker now enforces the exclusivity that the
    // test was probing for. The new
    // `commit_decommit_use_exclusive_signature` test below
    // verifies the new shape.

    #[test]
    fn commit_decommit_use_exclusive_signature() {
        // Regression test for bug #3 from the code review: both
        // commit_page and decommit_page now take &mut self.
        // Calling them through a mutable binding compiles; the
        // borrow checker would reject any attempt to share the
        // page heap across threads via Arc<PageHeap> + call these
        // methods. The new signature IS the safety property.
        let mut h = small_heap();
        h.commit_page(0).unwrap();
        h.decommit_page(0).unwrap();
        assert!(!h.is_committed(0));
    }
}
