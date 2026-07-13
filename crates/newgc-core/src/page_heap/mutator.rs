//! Multi-mutator front end.
//!
//! A [`GcCoordinator`] owns the heap behind an `Arc<Mutex<PageHeap>>` and
//! hands out [`Mutator`] handles; any number of threads can each hold a
//! handle and allocate concurrently.
//!
//! Current model (MM-3 … MM-7 have landed):
//! - **Per-thread lock-free TLABs** (MM-3): the fast path bumps a
//!   thread-local slab with no lock; only TLAB *refill* takes the heap
//!   mutex. The mutex still makes refill/collection mutually exclusive.
//! - **Safepoint protocol** (MM-4): any mutator can drive a
//!   stop-the-world collection ([`drive_collect`]); peers park at their
//!   next safepoint poll. The handshake (`epoch` / `world_running` /
//!   `is_acting_coordinator`, parked under `park_mutex` + `park_cv`)
//!   serializes drivers via `coord_mutex` and resumes the world on every
//!   exit path (including an OOM unwind) via `ResumeGuard`.
//! - **Per-mutator root snapshots** (MM-5): each mutator publishes its
//!   own roots into `roots_snapshot` at the safepoint; the driver visits
//!   **every** active mutator's snapshot (updated in place by the
//!   evacuator) in addition to the caller's `extra` closure. Callers must
//!   therefore supply only the driving thread's *extra* roots — NOT every
//!   thread's roots. Threads blocked in foreign code (`IN_NATIVE`, MM-6)
//!   publish their roots before leaving and are collected around.
//!
//! See `docs/MULTI_MUTATOR_DESIGN.md`.

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
#[cfg(feature = "conservative-pin")]
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::traits::HeapLayout;
use crate::word::Word;

use super::alloc::{set_cons_start_bit_at, set_start_bit_at};
use super::cycle::{CollectResult, FullCollectResult};
use super::evac::{GcError, PageEvacuator};
use super::page_desc::{Generation, PageKind};
use super::pin::PinHandle;
use super::shared::SharedHeap;
use super::space::{PageHeap, PAGE_SIZE_CELLS};

/// Initial TLAB refill size in cells (4 KB). Doubles each refill up to
/// `MAX_TLAB_CELLS`.
const INITIAL_TLAB_CELLS: usize = 512;
/// Max TLAB size in cells (one 64 KB page).
const MAX_TLAB_CELLS: usize = PAGE_SIZE_CELLS;

/// `(gen_idx, kind_idx)` into the per-mutator `tlabs` array. Mirrors
/// `PageHeap::region_index` (kept local since that one is private).
#[inline]
fn region_index(generation: Generation, kind: PageKind) -> (usize, usize) {
    let gi = match generation {
        Generation::G0 => 0,
        Generation::G1 => 1,
        Generation::Tenured => 2,
        Generation::Free => unreachable!("Free has no alloc region"),
    };
    let ki = match kind {
        PageKind::Cons => 0,
        PageKind::Boxed => 1,
        _ => unreachable!("only Cons/Boxed have TLABs"),
    };
    (gi, ki)
}

/// A thread-local allocation buffer: a slab carved from the heap that
/// the owning mutator bumps **lock-free**. One per `(gen, kind)`.
/// `Copy` so the `[[Tlab; 2]; 3]` array initializes cheaply.
#[derive(Copy, Clone)]
struct Tlab {
    /// First cell of the slab (null = empty, triggers refill).
    start: *mut u64,
    /// Next free cell. Bumped by the fast path.
    cursor: *mut u64,
    /// One past the last cell of the slab.
    end: *mut u64,
    /// Page this slab lives on (for `words_used` reconciliation).
    page_idx: usize,
    /// Cells reserved at refill (for reconciling the unused tail).
    reserved_cells: u32,
    /// Size to request at the next refill (dynamic 4 KB → 64 KB).
    next_refill_cells: u32,
}

impl Tlab {
    const fn empty() -> Self {
        Self {
            start: std::ptr::null_mut(),
            cursor: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            page_idx: 0,
            reserved_cells: 0,
            next_refill_cells: INITIAL_TLAB_CELLS as u32,
        }
    }

    #[inline]
    fn room_cells(&self) -> usize {
        if self.start.is_null() {
            0
        } else {
            (self.end as usize - self.cursor as usize) / 8
        }
    }
}

/// Stable identifier for a registered mutator (index into the
/// coordinator's registry). Used by later sprints (MM-4) to look up a
/// mutator's safepoint state; in MM-1 it only drives slot lifecycle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MutatorId(usize);

impl MutatorId {
    /// The registry slot index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Execution state for the native-call convention (design §4.6).
/// `IN_DYLAN` = running managed code; cooperates with safepoints by
/// polling. `IN_NATIVE` = blocked in foreign code that may run long; the
/// driver's wait loop skips it (it touches no managed heap while native,
/// so it is safe to collect *around* it instead of waiting on it).
const IN_DYLAN: u8 = 0;
const IN_NATIVE: u8 = 1;

/// Per-mutator metadata in the registry, shared between the owning
/// mutator and any collection driver. MM-4 adds the safepoint state;
/// MM-5 adds the published root snapshot; MM-6 adds the native-call
/// execution state.
struct MutatorInner {
    /// Last safepoint epoch this mutator has reached (parked at, or is
    /// driving). The driver waits for `last_epoch >= target`. Mutator
    /// stores Release; driver loads Acquire.
    last_epoch: AtomicU64,
    /// False once the mutator begins `Drop`. The driver drops an
    /// inactive mutator from its wait set (design B-1).
    is_active: AtomicBool,
    /// True while this mutator is driving the current STW cycle; the
    /// driver's wait loop skips itself (design B-2).
    is_acting_coordinator: AtomicBool,
    /// Root `Word`s the mutator published before reaching the safepoint.
    /// The collector visits + updates these in place; the mutator copies
    /// them back on resume (design §5). Never contended in practice (the
    /// owner is parked when the driver reads it).
    roots_snapshot: Mutex<Vec<Word>>,
    /// Native-call execution state (`IN_DYLAN` / `IN_NATIVE`, design
    /// §4.6). A mutator blocked in foreign code publishes `IN_NATIVE` so
    /// the driver's wait loop skips it instead of holding every GC hostage
    /// until the 10 s timeout. Mutator stores Release (after publishing
    /// roots + flushing TLABs); driver loads Acquire.
    state: AtomicU8,
    /// Conservative stack-scan window `[lo, hi)` published by the owning
    /// mutator (design §5.3, `conservative-pin` only). The driver unions
    /// every active mutator's window into the slice handed to
    /// `pin_pointers_in_ranges`, pinning pointer-shaped stack words so
    /// stack-resident copies the collector can't rewrite stay valid.
    /// `lo >= hi` (default `0, 0`) means "no window." Mutator stores
    /// Release; driver loads Acquire under the world-stopped barrier.
    #[cfg(feature = "conservative-pin")]
    stack_lo: AtomicUsize,
    #[cfg(feature = "conservative-pin")]
    stack_hi: AtomicUsize,
}

impl MutatorInner {
    fn new(current_epoch: u64) -> Self {
        Self {
            // A newcomer is "already at" the current epoch, so it isn't
            // waited on for a cycle it never participated in (design A-1).
            last_epoch: AtomicU64::new(current_epoch),
            is_active: AtomicBool::new(true),
            is_acting_coordinator: AtomicBool::new(false),
            roots_snapshot: Mutex::new(Vec::new()),
            state: AtomicU8::new(IN_DYLAN),
            #[cfg(feature = "conservative-pin")]
            stack_lo: AtomicUsize::new(0),
            #[cfg(feature = "conservative-pin")]
            stack_hi: AtomicUsize::new(0),
        }
    }
}

/// Shared mutator registry. A free slot is reused before the vector
/// grows, so `MutatorId`s stay small and dense.
struct Registry {
    slots: RwLock<Vec<Option<Arc<MutatorInner>>>>,
}

impl Registry {
    fn new() -> Self {
        Self {
            slots: RwLock::new(Vec::new()),
        }
    }

    fn register(&self, current_epoch: u64) -> (MutatorId, Arc<MutatorInner>) {
        let inner = Arc::new(MutatorInner::new(current_epoch));
        let mut slots = self.slots.write().unwrap();
        let id = match slots.iter().position(|s| s.is_none()) {
            Some(i) => {
                slots[i] = Some(Arc::clone(&inner));
                i
            }
            None => {
                slots.push(Some(Arc::clone(&inner)));
                slots.len() - 1
            }
        };
        (MutatorId(id), inner)
    }

    fn deregister(&self, id: MutatorId) {
        let mut slots = self.slots.write().unwrap();
        if let Some(slot) = slots.get_mut(id.0) {
            *slot = None;
        }
    }

    fn live_count(&self) -> usize {
        self.slots.read().unwrap().iter().filter(|s| s.is_some()).count()
    }

    /// Snapshot the currently-registered mutators' inner handles. The
    /// driver iterates this to wait for arrivals and to gather roots.
    fn snapshot(&self) -> Vec<Arc<MutatorInner>> {
        self.slots
            .read()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect()
    }
}

/// Owns the heap and hands out mutator handles. `Clone` is cheap (it
/// clones the inner `Arc`s) so each thread can hold its own coordinator
/// handle and register a mutator locally. `Send + Sync`.
pub struct GcCoordinator<L: HeapLayout> {
    heap: Arc<Mutex<PageHeap<L>>>,
    registry: Arc<Registry>,
}

impl<L: HeapLayout> Clone for GcCoordinator<L> {
    fn clone(&self) -> Self {
        Self {
            heap: Arc::clone(&self.heap),
            registry: Arc::clone(&self.registry),
        }
    }
}

impl<L: HeapLayout> GcCoordinator<L> {
    /// Build a coordinator over a heap with `young_bytes` + `old_bytes`
    /// (mirrors [`PageHeap::new`]).
    pub fn new(young_bytes: usize, old_bytes: usize) -> Self {
        Self::from_heap(PageHeap::<L>::new(young_bytes, old_bytes))
    }

    /// Build a coordinator over a heap reserving `reserved_bytes`
    /// (mirrors [`PageHeap::with_reservation`]).
    pub fn with_reservation(reserved_bytes: usize) -> Self {
        Self::from_heap(PageHeap::<L>::with_reservation(reserved_bytes))
    }

    fn from_heap(heap: PageHeap<L>) -> Self {
        Self {
            heap: Arc::new(Mutex::new(heap)),
            registry: Arc::new(Registry::new()),
        }
    }

    /// Register a mutator on the current thread. The returned
    /// [`Mutator`] is `!Send` — keep it on this thread. Any number of
    /// threads may register concurrently; registration serializes with
    /// an in-flight STW cycle via `coord_mutex` so a newcomer can't join
    /// mid-collection and escape the world-stop (design A-1/B-3).
    pub fn register_mutator(&self) -> Mutator<L> {
        // Cache the lock-free handles (one heap lock, at registration).
        let (shared, base_addr) = {
            let h = self.heap.lock().unwrap();
            (h.shared_handle(), h.base_ptr() as usize)
        };
        // Serialize with STW; register at the current (quiescent) epoch.
        // Scope the coord_mutex guard so it drops before `shared` moves
        // into the Mutator below.
        let (id, inner) = {
            let _coord = shared.coord_mutex.lock().unwrap();
            let current_epoch = shared.safepoint.epoch.load(Ordering::Acquire);
            self.registry.register(current_epoch)
        };
        Mutator {
            heap: Arc::clone(&self.heap),
            shared,
            base_addr,
            registry: Arc::clone(&self.registry),
            id,
            tlabs: [[Tlab::empty(); 2]; 3],
            tlab_refills: 0,
            _inner: inner,
            _not_send: PhantomData,
        }
    }

    /// Number of currently-registered (live) mutators.
    pub fn mutator_count(&self) -> usize {
        self.registry.live_count()
    }

    /// Set the per-arrival safepoint wait-timeout — the diagnostic
    /// backstop a driver uses when waiting on a single mutator to park
    /// (design §4.4). The default is 10 s; the protocol does not depend on
    /// it (a cooperating mutator parks long before it fires), so this only
    /// bounds how quickly a driver re-checks a stuck/non-cooperating
    /// thread. Affects every subsequent collection. Clamped to ≥ 1 ms so
    /// the driver's wait can't busy-spin.
    pub fn set_safepoint_timeout(&self, timeout: Duration) {
        let ms = timeout.as_millis().clamp(1, u64::MAX as u128) as u64;
        let shared = self.heap.lock().unwrap().shared_handle();
        shared
            .safepoint
            .wait_timeout_ms
            .store(ms, Ordering::Relaxed);
    }

    /// Current safepoint wait-timeout (diagnostic / test hook).
    pub fn safepoint_timeout(&self) -> Duration {
        let shared = self.heap.lock().unwrap().shared_handle();
        Duration::from_millis(shared.safepoint.wait_timeout_ms.load(Ordering::Relaxed))
    }

    /// Run a closure with exclusive `&mut PageHeap`. Locks the heap
    /// mutex for the duration — allocation by any mutator is excluded.
    /// Escape hatch for diagnostics/tests; collection should go through
    /// the `collect_*` wrappers.
    pub fn with_heap<R>(&self, f: impl FnOnce(&mut PageHeap<L>) -> R) -> R {
        f(&mut self.heap.lock().unwrap())
    }

    // -- Collection entries (MM-1: lock + delegate) ----------------------
    //
    // In MM-4 the trigger moves onto `Mutator` (the driver self-parks);
    // for MM-1, with no safepoints, collection is driven here and the
    // caller supplies roots via the closure exactly as the single-mutator
    // `PageHeap::collect_*` does. The heap mutex serializes collection
    // against allocation.

    /// Minor collection. The closure must visit every live root (across
    /// all mutators) — see the module soundness caveat.
    pub fn collect_minor<F>(&self, visit_roots: F) -> CollectResult
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        self.heap.lock().unwrap().collect_minor(visit_roots)
    }

    /// Major collection (G1→Tenured, then G0→G0).
    pub fn collect_major<F>(&self, visit_roots: F) -> CollectResult
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        self.heap.lock().unwrap().collect_major(visit_roots)
    }

    /// Full collection (force-promote + compact Tenured).
    pub fn collect_full<F>(&self, visit_roots: F) -> FullCollectResult
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        self.heap.lock().unwrap().collect_full(visit_roots)
    }

    /// Trigger-policy-driven collection (minor or major per heap state).
    pub fn collect_auto<F>(&self, visit_roots: F) -> CollectResult
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        self.heap.lock().unwrap().collect_auto(visit_roots)
    }

    /// Recoverable minor collection — `Err` on mid-evac OOM (and the
    /// heap is poisoned thereafter; see [`PageHeap::is_poisoned`]).
    pub fn try_collect_minor<F>(&self, visit_roots: F) -> Result<CollectResult, GcError>
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        self.heap.lock().unwrap().try_collect_minor(visit_roots)
    }

    /// Recoverable major collection.
    pub fn try_collect_major<F>(&self, visit_roots: F) -> Result<CollectResult, GcError>
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        self.heap.lock().unwrap().try_collect_major(visit_roots)
    }

    /// Recoverable auto collection.
    pub fn try_collect_auto<F>(&self, visit_roots: F) -> Result<CollectResult, GcError>
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        self.heap.lock().unwrap().try_collect_auto(visit_roots)
    }

    /// True if a previous `try_collect_*` poisoned the heap.
    pub fn is_poisoned(&self) -> bool {
        self.heap.lock().unwrap().is_poisoned()
    }
}

/// Per-thread allocation handle. `!Send + !Sync` — bound to the thread
/// that registered it. In MM-1 every operation locks the shared heap
/// mutex; MM-3 adds a lock-free TLAB fast path.
pub struct Mutator<L: HeapLayout> {
    heap: Arc<Mutex<PageHeap<L>>>,
    /// Lock-free shared state (start bits, poison flag, alloc counter).
    /// The bump fast path touches only this — no heap lock (MM-3).
    shared: Arc<SharedHeap>,
    /// Reservation base address, cached for global-cell-index math.
    base_addr: usize,
    registry: Arc<Registry>,
    id: MutatorId,
    /// Per-`(gen, kind)` thread-local allocation buffers.
    tlabs: [[Tlab; 2]; 3],
    /// Count of TLAB refills (each takes the heap lock once). Diagnostic
    /// — lets tests verify the bump fast path amortizes the lock.
    tlab_refills: u64,
    _inner: Arc<MutatorInner>,
    _not_send: PhantomData<*mut ()>,
}

impl<L: HeapLayout> Mutator<L> {
    /// This mutator's stable id.
    pub fn id(&self) -> MutatorId {
        self.id
    }

    /// Allocate a cons cell (2 cells) in `generation`. Lock-free bump
    /// in the common case; locks the heap only to refill an exhausted
    /// TLAB. Returns `None` on OOM or if the heap is poisoned.
    #[inline]
    pub fn try_alloc_cons_in(&mut self, generation: Generation) -> Option<NonNull<u64>> {
        self.bump(generation, PageKind::Cons, 2, /*is_cons=*/ true)
    }

    /// Allocate an `n_cells` boxed object (header + payload) in
    /// `generation`. Lock-free bump; refill on exhaustion.
    #[inline]
    pub fn try_alloc_boxed_in(
        &mut self,
        generation: Generation,
        n_cells: usize,
    ) -> Option<NonNull<u64>> {
        if n_cells == 0 || n_cells > PAGE_SIZE_CELLS {
            return None;
        }
        self.bump(generation, PageKind::Boxed, n_cells, /*is_cons=*/ false)
    }

    /// Allocate a large (≥ one page) object in `generation`. Large
    /// objects bypass TLABs and go through the central path under the
    /// heap lock.
    pub fn try_alloc_large(
        &mut self,
        n_cells: usize,
        generation: Generation,
    ) -> Option<NonNull<u64>> {
        self.heap.lock().unwrap().try_alloc_large(n_cells, generation)
    }

    /// The lock-free bump fast path. On a hit it advances the TLAB
    /// cursor, sets the object's start bit (atomic `fetch_or`), bumps
    /// the alloc counter (atomic), and returns — **no heap lock**. On a
    /// miss it refills (one heap lock) and retries.
    #[inline]
    fn bump(
        &mut self,
        generation: Generation,
        kind: PageKind,
        n_cells: usize,
        is_cons: bool,
    ) -> Option<NonNull<u64>> {
        // Poison check is lock-free (Acquire on the shared flag).
        if self.shared.poisoned.load(Ordering::Acquire) {
            return None;
        }
        let (gi, ki) = region_index(generation, kind);
        loop {
            // Fast path: room in the current TLAB.
            if self.tlabs[gi][ki].room_cells() >= n_cells {
                let ptr = self.tlabs[gi][ki].cursor;
                self.tlabs[gi][ki].cursor = unsafe { ptr.add(n_cells) };
                let cell_idx = (ptr as usize - self.base_addr) / 8;
                if is_cons {
                    set_cons_start_bit_at(&self.shared.start_bits, cell_idx);
                } else {
                    set_start_bit_at(&self.shared.start_bits, cell_idx);
                }
                self.shared
                    .bytes_alloc_since_gc
                    .fetch_add(n_cells * 8, Ordering::Relaxed);
                return Some(unsafe { NonNull::new_unchecked(ptr) });
            }
            // Slow path: refill and retry. If refill fails (OOM / cap /
            // poison), give up.
            if !self.refill(generation, kind, gi, ki, n_cells) {
                return None;
            }
        }
    }

    /// Refill the `(gi, ki)` TLAB with a fresh slab carved under the
    /// heap lock. Grows the request 4 KB → 64 KB across successive
    /// refills. Returns false on OOM / young-cap / poison.
    #[cold]
    fn refill(
        &mut self,
        generation: Generation,
        kind: PageKind,
        gi: usize,
        ki: usize,
        min_cells: usize,
    ) -> bool {
        let want = (self.tlabs[gi][ki].next_refill_cells as usize).max(min_cells);
        let slab = {
            let mut heap = self.heap.lock().unwrap();
            heap.reserve_tlab(generation, kind, min_cells, want)
        };
        self.tlab_refills += 1;
        match slab {
            Some((ptr, page_idx, cells)) => {
                let next = ((self.tlabs[gi][ki].next_refill_cells as usize) * 2)
                    .min(MAX_TLAB_CELLS) as u32;
                let start = ptr.as_ptr();
                self.tlabs[gi][ki] = Tlab {
                    start,
                    cursor: start,
                    end: unsafe { start.add(cells) },
                    page_idx,
                    reserved_cells: cells as u32,
                    next_refill_cells: next,
                };
                true
            }
            None => false,
        }
    }

    /// Clear every TLAB so the next allocation refills fresh. **Must be
    /// called before a collection** while this mutator holds live TLABs
    /// — the cursor would otherwise dangle if GC moved the TLAB's page.
    /// MM-4's safepoint protocol calls this automatically at park; a
    /// single-mutator client can call it explicitly before a collect.
    ///
    /// It deliberately does **not** reconcile `words_used`. `words_used`
    /// is a per-page high-water mark, and a page may carry TLAB slabs
    /// from *several* mutators; subtracting each slab's unused tail would
    /// collapse the watermark below a later slab's live objects, and the
    /// evacuator's `[0, words_used)` scan would then miss them. Leaving
    /// `words_used` at the carved watermark over-states live data by the
    /// unused tails, which is harmless: those cells are zeroed with no
    /// start bits, so the start-bit-driven walkers skip them, and the
    /// next evacuation rebuilds an exact `words_used` on the dest pages.
    pub fn flush_tlabs(&mut self) {
        for gi in 0..3 {
            for ki in 0..2 {
                self.tlabs[gi][ki] = Tlab::empty();
            }
        }
    }

    /// Number of TLAB refills this mutator has performed (each took the
    /// heap lock once). Diagnostic / test hook.
    pub fn tlab_refill_count(&self) -> u64 {
        self.tlab_refills
    }

    /// Card barrier — mark the card covering `slot_addr`.
    pub fn mark_card_at(&self, slot_addr: *const u8) {
        self.heap.lock().unwrap().mark_card_at(slot_addr);
    }

    /// Explicit FFI pin (MM-0). Keeps `w`'s target fixed until `unpin`.
    pub fn pin(&mut self, w: Word) -> PinHandle {
        self.heap.lock().unwrap().pin(w)
    }

    /// Release an explicit pin.
    pub fn unpin(&mut self, handle: PinHandle) {
        self.heap.lock().unwrap().unpin(handle);
    }

    // -- MM-4/MM-5: safepoint protocol -----------------------------------

    #[inline]
    fn inner(&self) -> &MutatorInner {
        &self._inner
    }

    /// Cooperative safepoint poll (design §4.2/§4.3). Cheap on the fast
    /// path: a single relaxed-vs-acquire epoch compare. If a collection
    /// has been requested, this **parks** the mutator — publishing
    /// `roots`, flushing its TLABs, and blocking until the world resumes
    /// — then copies the (possibly forwarded) roots back into `roots`.
    ///
    /// **Poll-site contract (§4.2):** `roots` must be the mutator's
    /// complete, consistent live-root set at this point — every live
    /// in-flight `Word`. A poll with a half-built root set lets the
    /// collector move an object the mutator still holds. The frontend
    /// owns this guarantee.
    pub fn poll_safepoint(&mut self, roots: &mut [Word]) {
        let global = self.shared.safepoint.epoch.load(Ordering::Acquire);
        if self.inner().last_epoch.load(Ordering::Relaxed) == global {
            return; // fast path: no safepoint pending
        }
        self.park(global, roots);
    }

    /// True when a peer has requested a stop-the-world and this
    /// mutator hasn't parked for it yet — i.e. the next
    /// `poll_safepoint` would take the cold path. Two atomic loads,
    /// no side effects.
    ///
    /// Lets a frontend skip the *preparation* for a poll (publishing
    /// its stack window, borrowing its root set) on the
    /// overwhelmingly common no-GC-pending path. Racing past a
    /// just-raised epoch is benign: the request is simply observed
    /// at the caller's next poll site, exactly as if the epoch had
    /// been raised one instruction later.
    #[inline]
    pub fn safepoint_pending(&self) -> bool {
        let global = self.shared.safepoint.epoch.load(Ordering::Acquire);
        self.inner().last_epoch.load(Ordering::Relaxed) != global
    }

    /// Cold path: park at `target` epoch until the world resumes.
    #[cold]
    fn park(&mut self, target: u64, roots: &mut [Word]) {
        // Publish roots + flush TLABs BEFORE announcing arrival, so the
        // driver (which reads them after observing our last_epoch) sees
        // consistent state. flush_tlabs reconciles + clears (the TLAB
        // page may be evacuated by this cycle); next alloc refills.
        *self.inner().roots_snapshot.lock().unwrap() = roots.to_vec();
        self.flush_tlabs();

        let sp = &self.shared.safepoint;
        // Announce + block under park_mutex (mutate-under-lock + notify
        // is the lost-wakeup-free condvar pattern).
        //
        // Straggler re-arm: a mutator that parked for epoch `cur` and is
        // still blocked when the *next* cycle begins (the driver bumped the
        // epoch and re-stopped the world before we observed the resume)
        // would otherwise be frozen at `last_epoch == cur` forever — the
        // new driver waits for `last_epoch >= new_target`, but we can only
        // advance by returning to a poll we never reach. So whenever we
        // observe the epoch has moved, re-publish `last_epoch` at the new
        // target and re-announce. This is sound: we've run no mutator code
        // since publishing our roots, and the prior cycle updated our
        // snapshot in place, so re-exposing it to the new cycle is correct.
        // The driver cannot skip a cycle past us (it blocks until we reach
        // each target), so our roots are visited on every cycle.
        let mut guard = sp.park_mutex.lock().unwrap();
        let mut cur = target;
        self.inner().last_epoch.store(cur, Ordering::Release);
        sp.park_cv.notify_all();
        loop {
            let global = sp.epoch.load(Ordering::Acquire);
            if global != cur {
                cur = global;
                self.inner().last_epoch.store(cur, Ordering::Release);
                sp.park_cv.notify_all();
                continue;
            }
            if sp.world_running.load(Ordering::Acquire) == 1 {
                break;
            }
            guard = sp.park_cv.wait(guard).unwrap();
        }
        drop(guard);

        // Resume: copy the collector-updated snapshot back into `roots`.
        let snap = self.inner().roots_snapshot.lock().unwrap();
        let n = roots.len().min(snap.len());
        roots[..n].copy_from_slice(&snap[..n]);
    }

    // -- MM-6: native-call boundary convention (design §4.6) -------------

    /// Call immediately **before** a foreign call that may block or run
    /// long (a message pump, blocking I/O, a lock, a GPU present). After
    /// this returns the thread **must not touch the managed heap** until
    /// [`leave_native`](Self::leave_native) returns.
    ///
    /// Publishes `roots` and flushes TLABs (so a concurrent collector can
    /// move the objects this thread holds and update them in place), then
    /// announces `IN_NATIVE` so a driver skips this thread instead of
    /// waiting out the 10 s timeout. Does **not** advance `last_epoch`:
    /// the wait predicate skips `IN_NATIVE` regardless of epoch, so the
    /// thread stays "arrived" across any number of cycles while blocked.
    ///
    /// **Not** an FFI pin: an object whose address is *passed into* the
    /// foreign call and dereferenced by it while we block is not protected
    /// — the foreign code holds a raw copy the in-place root update can't
    /// reach. Pin those explicitly with [`pin`](Self::pin) first (§5.4).
    pub fn enter_native(&mut self, roots: &[Word]) {
        // Publish roots + flush TLABs BEFORE the Release store of `state`,
        // so a collector that observes `IN_NATIVE` sees consistent state.
        *self.inner().roots_snapshot.lock().unwrap() = roots.to_vec();
        self.flush_tlabs();

        let sp = &self.shared.safepoint;
        // Announce IN_NATIVE under park_mutex + notify. A driver already
        // blocked waiting on us (it saw us IN_DYLAN with a stale epoch when
        // it built its wait set) wakes, re-checks, and drops us from the
        // wait set immediately — rather than stalling on the 10 s timeout,
        // which is the very hostage situation §4.6 exists to prevent.
        let _g = sp.park_mutex.lock().unwrap();
        self.inner().state.store(IN_NATIVE, Ordering::Release);
        sp.park_cv.notify_all();
    }

    /// Call immediately **after** the foreign call returns, before
    /// touching the managed heap again. Updates `roots` in place with the
    /// (possibly forwarded) values the collector wrote while we blocked.
    ///
    /// If a collection is in progress, blocks until the world resumes
    /// *before* flipping back to `IN_DYLAN` — so a returning thread never
    /// resumes heap access while the collector owns the heap (this mirrors
    /// the tail of [`park`](Self::park)).
    pub fn leave_native(&mut self, roots: &mut [Word]) {
        let sp = &self.shared.safepoint;
        {
            let mut guard = sp.park_mutex.lock().unwrap();
            while sp.world_running.load(Ordering::Acquire) == 0 {
                guard = sp.park_cv.wait(guard).unwrap();
            }
            // Re-enter at the current epoch (we owed no poll while native;
            // adopting the latest makes our next poll a fast no-op). Both
            // stores under park_mutex so a driver's stop (also taken under
            // the lock, §4.4) cannot interleave between them.
            self.inner()
                .last_epoch
                .store(sp.epoch.load(Ordering::Acquire), Ordering::Release);
            self.inner().state.store(IN_DYLAN, Ordering::Release);
        }
        // The snapshot now holds forwarded values; copy them back to the
        // caller's slots (same as the resume tail of `park`). TLABs are
        // empty (cleared at enter_native); the next alloc refills.
        let snap = self.inner().roots_snapshot.lock().unwrap();
        let n = roots.len().min(snap.len());
        roots[..n].copy_from_slice(&snap[..n]);
    }

    // -- MM-7: conservative stack pins across mutators (design §5.3) ------

    /// Publish this mutator's conservative stack-scan window `[lo, hi)`
    /// (design §5.3, `conservative-pin` builds only). During a collection
    /// the driver unions every active mutator's window and scans it for
    /// pointer-shaped `Word`s, pinning their targets so a stack-resident
    /// pointer the moving collector cannot rewrite stays valid.
    ///
    /// Call before reaching a safepoint, with bounds covering every stack
    /// slot that may hold a live `Word` (typically the thread's full
    /// stack span). `lo >= hi` clears the window. No-op in precise-only
    /// builds (compiled out with the `conservative-pin` feature).
    #[cfg(feature = "conservative-pin")]
    pub fn set_stack_range(&mut self, lo: usize, hi: usize) {
        self.inner().stack_lo.store(lo, Ordering::Release);
        self.inner().stack_hi.store(hi, Ordering::Release);
    }

    /// Drive a **minor** collection from this mutator's thread (design
    /// §2.5/§4.4). Self-parks (publishes `roots`, flushes TLABs, marks
    /// itself acting-coordinator), requests the safepoint, waits for
    /// every *other* active mutator to park, then collects with **all**
    /// mutators' published roots (plus the `extra` closure) visited in
    /// place, and resumes the world. `roots` is updated in place.
    ///
    /// Panics on mid-evac OOM (poisoning the heap) — same as
    /// `PageHeap::collect_minor`; the world is always resumed first.
    pub fn collect_minor<F>(&mut self, roots: &mut [Word], extra: F) -> CollectResult
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        // Conservative pins cover the generations a minor cycle moves: G0
        // always, and G1 when a cascade promotes G1→Tenured (§5.3).
        self.drive_collect(
            roots,
            |heap, visit| heap.collect_minor(visit),
            &[Generation::G0, Generation::G1],
            extra,
        )
    }

    /// Drive a **full** collection from this mutator's thread.
    pub fn collect_full<F>(&mut self, roots: &mut [Word], extra: F) -> FullCollectResult
    where
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        // A full cycle can move objects in every generation.
        self.drive_collect(
            roots,
            |heap, visit| heap.collect_full(visit),
            &[Generation::G0, Generation::G1, Generation::Tenured],
            extra,
        )
    }

    /// Shared driver body for `collect_minor` / `collect_full`. `run`
    /// invokes the chosen `PageHeap` collector with the combined root
    /// visitor; `pin_gens` are the generations to conservatively pin from
    /// the mutators' stack windows (`conservative-pin` only).
    fn drive_collect<R, C, F>(
        &mut self,
        roots: &mut [Word],
        run: C,
        pin_gens: &[Generation],
        mut extra: F,
    ) -> R
    where
        C: FnOnce(&mut PageHeap<L>, &mut dyn FnMut(&mut PageEvacuator<'_, L>)) -> R,
        F: FnMut(&mut PageEvacuator<'_, L>),
    {
        // (a) Self-park: publish our own roots + flush our TLABs, and
        //     mark ourselves the driver so the wait loop skips us.
        //
        // Announce `is_acting_coordinator` UNDER park_mutex + notify. This
        // becoming-a-coordinator transition falsifies an *active* driver's
        // wait predicate (`!is_acting_coordinator`): when several mutators
        // drive concurrently, one holds `coord_mutex` and waits on the
        // others, while a late driver sets this flag and blocks on
        // `coord_mutex`. Without the notify, the waiting driver isn't woken
        // to drop the late driver from its wait set and stalls until the
        // timeout fires (per-handoff), which under sustained concurrent
        // driving degrades to an apparent hang. This mirrors the
        // mutate-under-lock + notify discipline of `park` / `enter_native`
        // / `Drop`; publishing roots *before* the flag keeps a concurrent
        // collector's in-place visit of our snapshot consistent.
        *self.inner().roots_snapshot.lock().unwrap() = roots.to_vec();
        self.flush_tlabs();
        {
            let sp = &self.shared.safepoint;
            let _g = sp.park_mutex.lock().unwrap();
            self.inner().is_acting_coordinator.store(true, Ordering::Release);
            sp.park_cv.notify_all();
        }

        // Resume-the-world + clear-coordinator on every exit path
        // (including an OOM unwind), so other mutators can't get stuck.
        struct ResumeGuard {
            shared: Arc<SharedHeap>,
            inner: Arc<MutatorInner>,
        }
        impl Drop for ResumeGuard {
            fn drop(&mut self) {
                let sp = &self.shared.safepoint;
                // Set world_running + notify UNDER park_mutex: parked
                // mutators block in a plain `cv.wait` (no timeout), so the
                // resume must be published while holding the mutex or a
                // notify could be lost and a worker would hang forever.
                {
                    let _g = sp.park_mutex.lock().unwrap();
                    sp.world_running.store(1, Ordering::Release);
                    // Clear the coordinator flag UNDER park_mutex too — it is
                    // *set* under the lock at the top of `drive_collect`, so
                    // mirror that here. A peer driver woken by this same notify
                    // then observes a consistent (world-running,
                    // not-coordinator) state in one re-evaluation of its wait
                    // predicate, instead of racing a bare Release store that it
                    // can observe late and so wait out the full timeout.
                    self.inner.is_acting_coordinator.store(false, Ordering::Release);
                    sp.park_cv.notify_all();
                }
            }
        }

        let result = {
            // Serialize STW drivers + registration.
            let _coord = self.shared.coord_mutex.lock().unwrap();
            let sp = &self.shared.safepoint;
            // Publish the stop (epoch bump + world_running = 0) UNDER
            // park_mutex. A parked worker reads `epoch` then `world_running`
            // in its park loop while holding park_mutex; performing both
            // stores under the same lock prevents it from observing a torn
            // (stale-epoch, fresh-stop) state and re-sleeping at the old
            // epoch while we wait for it to reach the new target.
            let target = {
                let _g = sp.park_mutex.lock().unwrap();
                let t = sp.epoch.fetch_add(1, Ordering::AcqRel) + 1;
                self.inner().last_epoch.store(t, Ordering::Release);
                sp.world_running.store(0, Ordering::Release);
                t
            };

            let _resume = ResumeGuard {
                shared: Arc::clone(&self.shared),
                inner: Arc::clone(&self._inner),
            };

            // (c) Wait for every other active, non-coordinator mutator
            //     to reach `target`.
            let others = self.registry.snapshot();
            {
                let mut guard = sp.park_mutex.lock().unwrap();
                for m in &others {
                    // Skip ourselves (B-2), the departed (B-1), and threads
                    // blocked in foreign code (§4.6 — IN_NATIVE; they touch
                    // no managed heap, so we collect around them rather than
                    // wait). Their published snapshot is still visited below.
                    while m.is_active.load(Ordering::Acquire)
                        && !m.is_acting_coordinator.load(Ordering::Acquire)
                        && m.state.load(Ordering::Acquire) != IN_NATIVE
                        && m.last_epoch.load(Ordering::Acquire) < target
                    {
                        let budget = Duration::from_millis(
                            sp.wait_timeout_ms.load(Ordering::Relaxed),
                        );
                        guard = sp.park_cv.wait_timeout(guard, budget).unwrap().0;
                    }
                }
            }

            // (d) World stopped. Lock the heap.
            let mut heap = self.heap.lock().unwrap();

            // (d.1) Conservative stack pins (§5.3, `conservative-pin`):
            //       union every active mutator's published stack window and
            //       pin pointer-shaped words into the moving generations, so
            //       a stack-resident pointer the moving collector cannot
            //       rewrite stays valid. Runs before the evac reads the pin
            //       set; `others` already includes this driver's own slot,
            //       so the driver's window (if any) is covered too. Compiled
            //       out entirely in precise-only builds.
            #[cfg(feature = "conservative-pin")]
            {
                let mut ranges: Vec<(usize, usize)> = Vec::new();
                for m in &others {
                    if !m.is_active.load(Ordering::Acquire) {
                        continue;
                    }
                    let lo = m.stack_lo.load(Ordering::Acquire);
                    let hi = m.stack_hi.load(Ordering::Acquire);
                    if lo < hi {
                        ranges.push((lo, hi));
                    }
                }
                if !ranges.is_empty() {
                    for &g in pin_gens {
                        heap.pin_pointers_in_ranges(g, &ranges);
                    }
                }
            }
            #[cfg(not(feature = "conservative-pin"))]
            let _ = pin_gens;

            // (d.2) Collect, visiting the caller's extra roots plus every
            //       active mutator's published snapshot (updated in place by
            //       the evacuator).
            let r = run(&mut heap, &mut |evac| {
                extra(evac);
                for m in &others {
                    if !m.is_active.load(Ordering::Acquire) {
                        continue;
                    }
                    let mut snap = m.roots_snapshot.lock().unwrap();
                    for w in snap.iter_mut() {
                        evac.visit(w);
                    }
                }
            });
            drop(heap);
            r
            // _resume drops here: world resumes, coordinator flag cleared.
            // _coord drops: next cycle/registration may proceed.
        };

        // Copy our own (updated) snapshot back into `roots`.
        let snap = self.inner().roots_snapshot.lock().unwrap();
        let n = roots.len().min(snap.len());
        roots[..n].copy_from_slice(&snap[..n]);
        result
    }
}

impl<L: HeapLayout> Drop for Mutator<L> {
    fn drop(&mut self) {
        // STW-aware drop (design §2.1 / B-1): mark inactive so a driver
        // waiting on us drops us from its wait set, deregister the slot,
        // and wake any waiting driver. We touch no heap state (TLAB tail
        // is abandoned — harmless; the page carries no start bits there).
        // Mark inactive + wake any waiting driver UNDER park_mutex, so a
        // driver blocked on our (now-departing) slot can't miss the notify
        // and stall on the 10 s timeout before re-checking is_active.
        {
            let sp = &self.shared.safepoint;
            let _g = sp.park_mutex.lock().unwrap();
            self.inner().is_active.store(false, Ordering::Release);
            sp.park_cv.notify_all();
        }
        self.registry.deregister(self.id);
    }
}
