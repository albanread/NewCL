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
//! 1 GB) divided into 64 KB pages. Each page is in one of two
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

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::heap_common::CardTable;

use super::alloc::{AllocRegion, PageStartBits};
use super::page_desc::{Generation, PageDesc, PageKind};

/// Size of a single page in bytes. 64 KB matches Windows'
/// VirtualAlloc allocation granularity (the smallest size that
/// VirtualAlloc will return as a separately-decommittable region).
pub const PAGE_SIZE_BYTES: usize = 64 * 1024;

/// Size of a page in cells (64-bit words).
pub const PAGE_SIZE_CELLS: usize = PAGE_SIZE_BYTES / 8;

/// Default reservation size: 2 GB → 32768 pages. Sized for a
/// long-running session with plenty of headroom for large-object
/// allocation (which needs contiguous free-page runs). Costs ~32 KB
/// of commit-bitmap storage and one entry in the OS VAD tree; no
/// physical RAM until pages are committed.
pub const DEFAULT_RESERVATION_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// The page-heap reservation.
///
/// `Send + Sync` because all interior mutability goes through atomic
/// loads/stores on `committed_bits` and through `commit_lock` for
/// the VirtualAlloc calls themselves.
pub struct PageHeap {
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
    /// Open allocation regions, one per (generation, kind). Indexed
    /// by `(generation_idx, kind_idx)` — see `region_index` for
    /// the encoding. Sub-phase 4 supports `Cons` and `Boxed`
    /// kinds across `G0`, `G1`, `Tenured` generations = 6
    /// regions; `Free` and `Large` get no region.
    alloc_regions: [[AllocRegion; 2]; 3],
    /// Global start-bit bitmap covering the whole reservation.
    /// Same encoding as `heap::Semispace::starts`: 2 bits per
    /// cell, packed into `AtomicU64` words. Pair `01` = boxed
    /// header start, `11` = cons start, `00` = not a start.
    ///
    /// 32 MB for the 1 GB default reservation (3% overhead).
    /// Mutators can cache an `Arc<[AtomicU64]>` handle (same as
    /// `StartBits` in `heap.rs`) and use the same atomic-OR fast
    /// path for marking starts.
    start_bits: PageStartBits,
    /// Mark bitmap covering the whole reservation. One bit per
    /// cell, packed 64 cells per `u64` word. Bit `c % 64` of
    /// word `c / 64` is set when cell `c` is the start of a
    /// reachable object on the most recent mark pass.
    ///
    /// 16 MB for the 1 GB default reservation. Plain `Box<[u64]>`,
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
    /// Soft card-marking table covering the WHOLE reservation
    /// (page-heap doesn't split into young/old address ranges the
    /// way the semispace does). One byte per `CARD_SIZE_BYTES`
    /// = 512 bytes; ~2 MB for the 1 GB default reservation.
    ///
    /// Sub-phase 9: mutator-side stores into older-than-G0 objects
    /// mark cards via `GcCoordinator::mark_card`. Minor GC scans
    /// dirty cards in G1/Tenured pages for cross-gen pointers into
    /// G0. The field exists from sub-phase 11a so the coordinator
    /// can wire its barrier through here; full minor-GC integration
    /// follows in sub-phase 11b.
    pub(super) cards: Arc<CardTable>,
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
}

enum Backing {
    /// Box-backed fallback for non-Windows or for tests that want
    /// a small, fully-committed reservation. All "pages" are always
    /// "committed"; `commit_page` and `decommit_page` are no-ops on
    /// this path.
    Boxed(Box<[u8]>),
    /// Windows `VirtualAlloc(MEM_RESERVE)` reservation. Pages are
    /// individually committed/decommitted as needed.
    #[cfg(windows)]
    Virtual {
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
        }
    }
}

#[cfg(windows)]
impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Virtual { base, .. } = self {
            // VirtualFree(addr, 0, MEM_RELEASE) drops the entire
            // reservation (committed + uncommitted). Size MUST be 0
            // for MEM_RELEASE.
            use windows::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
            unsafe {
                let _ = VirtualFree(*base as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}

impl PageHeap {
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
        // ~12 bytes × n_pages of allocation (192 KB for the 1 GB
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
        // 1 GB default reservation: 16384 × 8192 / 32 = 4M words
        // = 32 MB.
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
        let recycle_live_counts = vec![0u16; n_pages];
        // Card table covering the whole reservation. Same 512-byte
        // card granularity as the semispace heap so the IR-level
        // barrier shape is identical.
        let cards = Arc::new(CardTable::new(total_bytes));

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
                storage: Backing::Virtual {
                    base: base as *mut u8,
                    reserved_bytes: total_bytes,
                },
                n_pages,
                committed_bits,
                committed_count: AtomicUsize::new(0),
                commit_lock: Mutex::new(()),
                descs,
                alloc_regions,
                start_bits,
                mark_bits,
                pinned_cells,
                recycle_live_counts,
                recycle_live_counts_target: None,
                last_mark_live_cells: 0,
                last_mark_live_pages: 0,
                last_zero_live_pages_released: 0,
                minors_since_g0_promote: 0,
                g0_promotes_since_g1_promote: 0,
                cards,
                last_pin_summary: (0, 0),
                young_page_cap: n_pages,
            }
        }
        #[cfg(not(windows))]
        {
            // Box-backed. All pages "committed" by virtue of the
            // Rust allocator zeroing and committing the whole
            // allocation.
            let boxed = vec![0u8; total_bytes].into_boxed_slice();
            // Pre-fill the commit bitmap so `is_committed` returns
            // true uniformly — matches the production behaviour of
            // "yes this address is backed."
            for w in &committed_bits {
                w.store(u64::MAX, Ordering::Relaxed);
            }
            PageHeap {
                storage: Backing::Boxed(boxed),
                n_pages,
                committed_bits,
                committed_count: AtomicUsize::new(n_pages),
                commit_lock: Mutex::new(()),
                descs,
                alloc_regions,
                start_bits,
                mark_bits,
                pinned_cells,
                recycle_live_counts,
                recycle_live_counts_target: None,
                last_mark_live_cells: 0,
                last_mark_live_pages: 0,
                last_zero_live_pages_released: 0,
                minors_since_g0_promote: 0,
                g0_promotes_since_g1_promote: 0,
                cards,
                last_pin_summary: (0, 0),
                young_page_cap: n_pages,
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
    pub fn commit_page(&mut self, idx: usize) -> Result<(), CommitError> {
        if idx >= self.n_pages {
            return Err(CommitError::OutOfRange(idx));
        }
        // Idempotent — already committed.
        if self.is_committed(idx) {
            return Ok(());
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
        // Set the commit bit + bump the count.
        let word_idx = idx / 64;
        let bit = 1u64 << (idx % 64);
        self.committed_bits[word_idx].fetch_or(bit, Ordering::AcqRel);
        self.committed_count.fetch_add(1, Ordering::AcqRel);
        Ok(())
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
        let word_idx = idx / 64;
        let bit = 1u64 << (idx % 64);
        self.committed_bits[word_idx].fetch_and(!bit, Ordering::AcqRel);
        self.committed_count.fetch_sub(1, Ordering::AcqRel);
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
        Arc::clone(&self.start_bits)
    }

    /// Internal access to the start-bit bitmap slice. Used by the
    /// allocator helpers in `alloc.rs`.
    pub(crate) fn start_bits_slice(&self) -> &[AtomicU64] {
        &self.start_bits
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
        // mutably twice. Fast — n_pages is at most 16384.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Cap the reservation size in tests so we don't ask the OS for
    /// 1 GB just to verify the bookkeeping.
    fn small_heap() -> PageHeap {
        // 1 MB = 16 pages. Plenty to exercise page indexing without
        // wasting VAD space across thousands of test runs.
        PageHeap::with_reservation(1024 * 1024)
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
