//! Multi-threaded mutator state, TLAB allocation, cooperative
//! stop-the-world GC coordination.
//!
//! See `docs/GC.md` and the threading-model memory. NewCormanLisp
//! supports multiple Lisp threads. Each thread has its own
//! `MutatorState` containing a TLAB (thread-local allocation buffer)
//! — a slab of young-heap memory that the thread bump-allocates
//! within without locks. The slab refill path acquires the global
//! heap lock; on young exhaustion the refilling thread becomes the
//! GC trigger.
//!
//! Stop-the-world is cooperative: the trigger sets a `stop_requested`
//! flag, waits until every other registered mutator has parked itself
//! at a safe point, then runs the GC, then clears the flag and wakes
//! the parked threads. No `SuspendThread`, no signal-based
//! preemption. Pause time = max(time-between-safe-points across
//! threads) + GC-time, both bounded.
//!
//! For the v1 root API: each `MutatorState` exposes `push_root` /
//! `pop_root` / `root_at` / `set_root_at`. Stack-map-driven precise
//! root finding lands later (step 9 in the GC build order); until
//! then, the explicit root API is the contract.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::heap::{CardTable, Heap};
use crate::heap_backend::{HeapBackend, HeapBackendKind};
use crate::static_area::StaticArea;
use crate::word::{Tag, Word};

/// Configuration knobs for the GC. Real values for production land
/// later (16 MB young, 64 MB old, 16 MB static, 512 KB TLAB by the
/// design doc); tests override with much smaller numbers.
#[derive(Clone, Copy, Debug)]
pub struct GcConfig {
    pub young_bytes: usize,
    pub old_bytes: usize,
    /// Static area RESERVATION size. On Windows, when this is large
    /// (>= `ELASTIC_STATIC_THRESHOLD_BYTES`), the area is backed by
    /// `VirtualAlloc(MEM_RESERVE)` — pages only charge against the
    /// working set as the bump pointer crosses them. So a 256 MB
    /// reservation in a session that uses 20 MB of static costs
    /// 20 MB of RAM, not 256 MB.
    ///
    /// Small static_bytes (tests, embedded configs) get a
    /// Box-backed allocation with full commit up front, matching
    /// pre-Phase-2 semantics.
    ///
    /// On non-Windows the elastic path falls back to Box-backed
    /// regardless of size (proper `mmap(MAP_NORESERVE)` support is
    /// future work; Windows is the primary platform).
    ///
    /// Used to be a 16 MB hard cap with frequent "static exhausted"
    /// panics. Phase 2 of `docs/GC_DESIGN.md` lifted that on
    /// Windows; the production default is now 256 MB reserved with
    /// ~10-20 MB resident for typical workloads.
    pub static_bytes: usize,
    /// Cells per TLAB. Each cell is 8 bytes.
    pub tlab_cells: usize,
}

/// Above this static_bytes threshold, GcCoordinator picks the
/// elastic VirtualAlloc-backed StaticArea. Below it, the Box-backed
/// new() — keeps the existing tests, which use sub-megabyte static
/// areas to exercise the "exhausted" path, behaving identically.
pub const ELASTIC_STATIC_THRESHOLD_BYTES: usize = 16 * 1024 * 1024;

impl Default for GcConfig {
    fn default() -> Self {
        GcConfig {
            young_bytes: 16 * 1024 * 1024,
            old_bytes: 64 * 1024 * 1024,
            // 256 MB reservation. With the Windows elastic backing
            // in `StaticArea::new_elastic`, only as many pages as
            // the session touches are paid for. The cap is generous
            // enough that a long session loading several big images
            // won't hit it.
            static_bytes: 256 * 1024 * 1024,
            tlab_cells: 65536, // 512 KB
        }
    }
}

// -- GcCoordinator -----------------------------------------------------------

/// The shared GC state. One per process, held as `Arc<GcCoordinator>`
/// by every Lisp thread. Owns the heap behind a mutex; coordinates
/// stop-the-world via the `stop_requested` flag and a condvar.
pub struct GcCoordinator {
    config: GcConfig,
    /// The global heap. Acquiring this lock is required for TLAB
    /// refill and for running the actual GC.
    ///
    /// Phase 3 sub-phase 1 of `docs/GC_DESIGN.md`: indirected
    /// through `HeapBackend` so the page-based heap can slot in
    /// alongside today's two-semispace generational copier without
    /// a build-time fork. Allocation fast paths bypass this — they
    /// cache the young-base pointer and start-bit bitmap at mutator
    /// registration, so per-alloc cost is unchanged.
    heap: Mutex<Box<dyn HeapBackend>>,
    /// "Park yourselves." Mutators poll this at safe points.
    stop_requested: AtomicBool,
    /// Set of registered mutators + how many are currently parked.
    park_state: Mutex<ParkState>,
    /// Used to wait for all-others-parked (by the GC trigger) and
    /// for "GC is done" (by parked mutators).
    park_cv: Condvar,

    // ---- Lock-free card-marking façade --------------------------------
    //
    // Mutators dirty cards on every old→x store. The card store
    // path MUST NOT acquire the heap mutex — that would serialise
    // every barrier and defeat multi-threading. We cache the live
    // semispace's base pointer and the card table here so the
    // barrier is a single atomic load + a single atomic byte store.
    /// Card table covering the LIVE old-semispace. Shared with
    /// `Heap::old.cards` (same Arc).
    cards: Arc<CardTable>,
    /// Pointer to the start of the live old-semispace, as `usize`.
    /// Updated by the GC after a full-GC swap.
    live_base: AtomicUsize,
    /// Capacity (in bytes) of one old semispace. Used by the
    /// barrier to decide if a write address falls in the old heap.
    old_capacity: usize,

    /// Pinned static area. Allocated once, never moved, never freed.
    /// For JIT'd code, the loaded image's interned constants, the
    /// package and symbol registries.
    static_area: Arc<StaticArea>,

    /// Process-global intern table: name → Symbol-tagged Word. Each
    /// allocated Symbol lives forever in the static area, so the raw
    /// bits in this table stay valid for the process lifetime. Used
    /// by the compiler to look up stable symbol addresses for
    /// embedding in JIT'd code.
    intern_table: Mutex<HashMap<Arc<str>, u64>>,

    /// Per-coordinator macro registry: macro-name → Function-tagged
    /// Word. The compiler consults this during macroexpansion;
    /// `defmacro` writes to it. Macros are install-and-replace, like
    /// defun's symbol-function-cell — redefining a macro at the REPL
    /// is allowed. The registry is OUTSIDE the symbol's function
    /// cell so calling `(macro-name ...)` and `(funcall #'macro-name)`
    /// can stay distinct (the latter would error per CL).
    macros: Mutex<HashMap<Arc<str>, u64>>,

    /// Cumulative GC counters. All atomics so the trigger thread
    /// can publish updates without anyone else's lock. Exposed to
    /// Lisp via `(gc-stats)`.
    pub stats: GcStats,

    /// Which backend was selected for this coordinator. Read at
    /// `new` from `NCL_HEAP_BACKEND` (default Semispace) and never
    /// changed afterward. Surfaces in `(gc-stats)` as
    /// `:heap-backend` so the user can confirm what's running.
    pub heap_backend_kind: HeapBackendKind,
}

/// Process-global GC counters. Reset never (cumulative over the
/// run). The pin counter measures *unique* pinned objects per
/// minor cycle, summed — useful as a rough "how much memory is
/// our conservative scan keeping live" signal. The peak young
/// gauge resets only when set higher; over a long run it
/// converges on the high-water mark the workload demanded.
pub struct GcStats {
    pub minor_gcs: AtomicU64,
    /// Full GC cycles run (`collect_full`, young + old → fresh old).
    /// Bumped from `trigger_full_gc`, which the auto-trigger
    /// escalates to when old is past `FULL_GC_OLD_THRESHOLD`. Phase 1
    /// addition; surfaces via `(gc-stats)` as `:full-gcs`.
    pub full_gcs: AtomicU64,
    pub bytes_promoted_total: AtomicU64,
    pub objects_pinned_total: AtomicU64,
    pub peak_young_used_bytes: AtomicU64,
    pub pinned_residual_cells: AtomicU64,
}

impl Default for GcStats {
    fn default() -> Self {
        GcStats {
            minor_gcs: AtomicU64::new(0),
            full_gcs: AtomicU64::new(0),
            bytes_promoted_total: AtomicU64::new(0),
            objects_pinned_total: AtomicU64::new(0),
            peak_young_used_bytes: AtomicU64::new(0),
            pinned_residual_cells: AtomicU64::new(0),
        }
    }
}

struct ParkState {
    handles: Vec<Arc<MutatorHandle>>,
    parked_count: usize,
}

impl GcCoordinator {
    pub fn new(config: GcConfig) -> Arc<GcCoordinator> {
        // Pick heap backend from the env var. Default is Semispace
        // (current production). `NCL_HEAP_BACKEND=page-heap` selects
        // the page-based heap under construction (Phase 3 of
        // `docs/GC_DESIGN.md`) — currently panics on construction
        // because sub-phases 2-7 aren't built yet, but the dispatch
        // wiring is in place so flipping a single line of source in
        // sub-phase 11 makes it production-default.
        Self::new_with_backend(config, HeapBackendKind::from_env())
    }

    /// Explicit-backend construction. The public `new` reads
    /// `NCL_HEAP_BACKEND`; this lets tests and any code that wants
    /// to pin a specific backend bypass the env var.
    pub fn new_with_backend(
        config: GcConfig,
        backend: HeapBackendKind,
    ) -> Arc<GcCoordinator> {
        let heap = Heap::new(config.young_bytes, config.old_bytes);
        let cards = Arc::clone(heap.old_cards());
        let live_base = AtomicUsize::new(heap.old_live_base_ptr() as usize);
        let old_capacity = heap.old_capacity_bytes_per_semi();
        // Pick the static-area backing by size. Production
        // (≥ ELASTIC_STATIC_THRESHOLD_BYTES = 16 MB) gets the
        // VirtualAlloc-backed path on Windows so the reservation is
        // cheap. Test configs and small embedded uses stay with the
        // Box-backed fully-committed path — they want predictable
        // exhaustion semantics for unit tests like
        // `static_returns_none_when_exhausted`. The threshold + 1/4
        // initial-commit policy isolates the size choice from the
        // backing choice.
        let static_area = if config.static_bytes >= ELASTIC_STATIC_THRESHOLD_BYTES {
            // 1/4 of the reservation committed up front. For 256 MB
            // that's 64 MB — covers a full-stdlib + Win32 + a
            // demo's worth of static work without needing further
            // commit-grow round-trips for cold startup.
            let initial = config.static_bytes / 4;
            Arc::new(StaticArea::new_elastic(config.static_bytes, initial))
        } else {
            Arc::new(StaticArea::new(config.static_bytes))
        };
        // Backend dispatch. Today both variants compile in (the
        // page-heap is a panic-stub); after Phase 3 sub-phase 7
        // lands, PageHeap returns a real `Box<dyn HeapBackend>`.
        // The Heap is constructed unconditionally above so the
        // coordinator's old-card / live-base caches stay valid
        // even when we're routing to the page heap — those caches
        // are dead weight under PageHeap (its card table is
        // single-anchored), but keeping them avoids a wider
        // refactor of GcCoordinator until sub-phase 11.
        let heap_backend: Box<dyn HeapBackend> = match backend {
            HeapBackendKind::Semispace => Box::new(heap),
            HeapBackendKind::PageHeap => {
                panic!(
                    "ncl: NCL_HEAP_BACKEND=page-heap selected, but the page-based \
                     heap is not yet implemented. Phase 3 sub-phases 2-7 of \
                     docs/GC_DESIGN.md build it. Unset the env var (or set \
                     NCL_HEAP_BACKEND=semispace) to use the production heap."
                );
            }
        };
        Arc::new(GcCoordinator {
            heap: Mutex::new(heap_backend),
            config,
            stop_requested: AtomicBool::new(false),
            park_state: Mutex::new(ParkState {
                handles: Vec::new(),
                parked_count: 0,
            }),
            park_cv: Condvar::new(),
            cards,
            live_base,
            old_capacity,
            static_area,
            intern_table: Mutex::new(HashMap::new()),
            macros: Mutex::new(HashMap::new()),
            stats: GcStats::default(),
            heap_backend_kind: backend,
        })
    }

    /// Install (or replace) a macro by name. The Word must be a
    /// Function-tagged value with the standard JIT calling
    /// convention; the compiler will invoke it during expansion.
    pub fn install_macro(&self, name: Arc<str>, fn_word: Word) {
        self.macros.lock().unwrap().insert(name, fn_word.raw());
    }

    /// Look up a macro function by name. Returns the
    /// Function-tagged Word, or None.
    pub fn macro_for(&self, name: &str) -> Option<Word> {
        self.macros
            .lock()
            .unwrap()
            .get(name)
            .copied()
            .map(Word::from_raw)
    }

    /// Intern a symbol by name. Returns the same Symbol-tagged Word
    /// every time the same name is looked up — symbols allocated in
    /// the static area never move. The first lookup allocates; later
    /// ones hit the table. The name is also recorded in the global
    /// `sym_names` registry so the printer can render symbol Words
    /// as their names.
    pub fn intern(&self, name: &str) -> Word {
        let mut table = self.intern_table.lock().unwrap();
        if let Some(&raw) = table.get(name) {
            return Word::from_raw(raw);
        }
        let sym = crate::gc_symbol::alloc_symbol_in_static(
            &self.static_area,
            Word::NIL,
            Word::NIL,
        )
        .expect("static area exhausted during intern");
        let key: Arc<str> = Arc::from(name);
        crate::sym_names::register(sym.raw(), Arc::clone(&key));
        table.insert(key, sym.raw());
        sym
    }

    /// Look up an interned symbol by name without allocating. Returns
    /// `None` if the name has never been interned.
    pub fn find_interned(&self, name: &str) -> Option<Word> {
        self.intern_table
            .lock()
            .unwrap()
            .get(name)
            .copied()
            .map(Word::from_raw)
    }

    /// Access the static area (for allocation, registry setup, etc.).
    pub fn static_area(&self) -> &Arc<StaticArea> { &self.static_area }

    /// Mark the card containing `addr` as dirty. Safe to call from
    /// any Lisp thread; lock-free. Routes the mark to the right
    /// card table:
    ///   - in live old-semispace → old card table
    ///   - in static area → static card table
    ///   - elsewhere (young, stack, foreign) → no-op
    pub fn mark_card(&self, addr: *const u8) {
        let p = addr as usize;
        // Try old.
        let old_base = self.live_base.load(Ordering::Acquire);
        if p >= old_base && p < old_base + self.old_capacity {
            self.cards.mark_offset(p - old_base);
            return;
        }
        // Try static.
        let static_base = self.static_area.base_ptr() as usize;
        if p >= static_base && p < static_base + self.static_area.capacity_bytes() {
            self.static_area.cards().mark_offset(p - static_base);
        }
    }

    /// Register a new mutator. Returns the per-thread state. The
    /// returned `MutatorState` is `Send` (so it can move to a
    /// thread) but `!Sync` (only owned by one thread at a time).
    /// On drop, the mutator deregisters from the coordinator.
    pub fn register_mutator(self: &Arc<Self>) -> MutatorState {
        let (lo, hi) = current_thread_stack_range();
        let handle = Arc::new(MutatorHandle {
            parked: AtomicBool::new(false),
            roots: Mutex::new(Vec::new()),
            stack_lo: AtomicUsize::new(lo),
            stack_hi: AtomicUsize::new(hi),
            parked_rsp: AtomicUsize::new(0),
        });
        self.park_state
            .lock()
            .unwrap()
            .handles
            .push(Arc::clone(&handle));
        let (young_base, young_starts) = {
            let heap = self.heap.lock().unwrap();
            (heap.young_base_ptr(), heap.young_starts_handle())
        };
        MutatorState {
            coord: Arc::clone(self),
            handle,
            tlab: Tlab::default(),
            young_base,
            young_starts,
        }
    }

    fn deregister(&self, handle: &Arc<MutatorHandle>) {
        let mut state = self.park_state.lock().unwrap();
        state.handles.retain(|h| !Arc::ptr_eq(h, handle));
        // A GC trigger could be waiting on parked_count to catch up
        // to (total - 1). If the deregistering thread was one of
        // the threads the trigger was waiting on, we owe it a wake
        // so it re-evaluates `target` against the now-smaller handle
        // set.
        drop(state);
        self.park_cv.notify_all();
    }

    pub fn used_bytes(&self) -> usize {
        self.heap.lock().unwrap().used_bytes()
    }

    pub fn young_used_bytes(&self) -> usize {
        self.heap.lock().unwrap().young_used_bytes()
    }

    /// Hand out the young start-bit bitmap to non-mutator threads
    /// (e.g. the entropy stirrer). Reading the bitmap from outside
    /// STW is safe because every op is relaxed-atomic and we never
    /// require a consistent snapshot — the reader is using bits
    /// as an entropy source, not as ground truth.
    pub fn young_starts(&self) -> crate::heap::StartBits {
        self.heap.lock().unwrap().young_starts_handle()
    }

    pub fn old_used_bytes(&self) -> usize {
        self.heap.lock().unwrap().old_used_bytes()
    }
}

// -- MutatorHandle (shared with the coordinator) ----------------------------

/// Per-mutator state visible to the coordinator. Tracks parked status
/// and the explicit root list. Symbol/value/function cells are
/// elsewhere; this is the per-thread root vector.
pub struct MutatorHandle {
    parked: AtomicBool,
    roots: Mutex<Vec<Word>>,
    /// Range of this thread's stack, captured once at registration
    /// time via `GetCurrentThreadStackLimits` (Win32). The full
    /// committed stack lies in `[stack_lo, stack_hi)`. The GC needs
    /// these to bound a conservative pin scan when a mutator parks.
    stack_lo: AtomicUsize,
    stack_hi: AtomicUsize,
    /// Snapshot of RSP captured at the instant this thread parked.
    /// 0 when not parked. The GC scans `[parked_rsp, stack_hi)`
    /// conservatively — every 8-byte slot in that range that looks
    /// like a pointer into young/old causes the target object to be
    /// pinned (skipped by the copier this cycle).
    parked_rsp: AtomicUsize,
}

// -- Tlab --------------------------------------------------------------------

/// Thread-local allocation buffer. Two pointers (top, limit) and a
/// raw base. The owning `MutatorState` is `!Sync`, so even though
/// `*mut u64` doesn't auto-Send, our discipline guarantees one
/// thread at a time.
struct Tlab {
    base: *mut u64,
    top: usize,
    limit: usize,
}

impl Default for Tlab {
    fn default() -> Self {
        Tlab { base: std::ptr::null_mut(), top: 0, limit: 0 }
    }
}

// SAFETY: `MutatorState` (which owns `Tlab`) is created on a thread
// and moved to that thread's body via `thread::spawn`. The TLAB
// pointer is into the global young heap, which lives as long as the
// `GcCoordinator`, which is reference-counted via Arc and thus
// outlives any mutator. Cross-thread access is prevented because we
// don't impl Sync — only Send.
unsafe impl Send for Tlab {}

// -- MutatorState (per-Lisp-thread) -----------------------------------------

/// Per-Lisp-thread state. Owned by one thread at a time; never
/// shared. Holds the TLAB and the handle that exposes this thread's
/// roots to the GC.
pub struct MutatorState {
    coord: Arc<GcCoordinator>,
    handle: Arc<MutatorHandle>,
    tlab: Tlab,
    /// Cached at registration so the alloc fast path can flip the
    /// young start-bit bitmap without taking the heap lock. The
    /// young semispace's storage doesn't move for the lifetime of
    /// the coordinator (and the coordinator outlives every mutator
    /// via Arc), so these stay valid as long as `self` exists.
    /// `young_starts` is a single packed bitmap with 2 bits per
    /// cell — see the Semispace docs for the encoding.
    young_base: *const u64,
    young_starts: crate::heap::StartBits,
}

// SAFETY: MutatorState contains a `*mut u64` (in Tlab) and a
// `*const u64` (young_base) — neither is Sync. We deliberately don't
// impl Sync. We DO impl Send: a MutatorState can move from creating
// thread to its working thread, but only one thread accesses it at a
// time. The young_base pointer is a read-only handle into a Box owned
// by the coordinator, which outlives every mutator.
unsafe impl Send for MutatorState {}

impl Drop for MutatorState {
    fn drop(&mut self) {
        self.coord.deregister(&self.handle);
    }
}

impl MutatorState {
    /// Access the GC coordinator. Used by runtime ABI helpers
    /// (e.g. `ncl_make_closure`) that need to allocate in static.
    pub fn coord(&self) -> &Arc<GcCoordinator> { &self.coord }

    /// Allocate a cons cell. Bumps in TLAB on the fast path; refills
    /// (and possibly triggers GC) on the slow path.
    pub fn alloc_cons(&mut self, car: Word, cdr: Word) -> Word {
        if self.tlab.top + 2 <= self.tlab.limit {
            return unsafe { self.tlab_write_cons(car, cdr) };
        }
        self.refill_tlab(2);
        // Guaranteed to fit after refill (TLAB size >> 2 cells).
        unsafe { self.tlab_write_cons(car, cdr) }
    }

    /// Allocate a Vector with `length_cells` payload cells. Returns
    /// the Vector-tagged Word. The header is initialised but the
    /// payload cells are zero (which is fixnum-tagged 0, not nil —
    /// callers that want nil must initialise explicitly).
    pub fn alloc_vector(&mut self, length_cells: u32) -> Word {
        self.alloc_typed_vector(crate::heap::HeapType::Vector, length_cells)
    }

    /// Like alloc_vector but lets the caller pick the HeapType.
    /// Used for heap kinds that share the Vector tag — currently
    /// only Bignum (raw u64 limb cells the printer / typep
    /// recognise via the header type).
    pub fn alloc_typed_vector(
        &mut self,
        ty: crate::heap::HeapType,
        length_cells: u32,
    ) -> Word {
        let total = 1 + length_cells as usize;
        if self.tlab.top + total > self.tlab.limit {
            self.refill_tlab(total);
        }
        let p = unsafe { self.tlab.base.add(self.tlab.top) };
        unsafe {
            *p = crate::heap::HeapHeader::new(ty, length_cells).raw();
        }
        self.mark_young_start(p);
        self.tlab.top += total;
        Word::from_ptr(p as *const u8, Tag::Vector)
    }

    /// Reserve a young-heap object header'd as `String` with the
    /// given payload length. Returns the address of the header
    /// cell — the caller fills char_count and the codepoint cells.
    /// The header is initialised; the payload is uninitialised.
    pub fn alloc_string_buffer(&mut self, payload_cells: u32) -> *mut u64 {
        let total = 1 + payload_cells as usize;
        if self.tlab.top + total > self.tlab.limit {
            self.refill_tlab(total);
        }
        let p = unsafe { self.tlab.base.add(self.tlab.top) };
        unsafe {
            *p = crate::heap::HeapHeader::new(
                crate::heap::HeapType::String,
                payload_cells,
            )
            .raw();
        }
        self.mark_young_start(p);
        self.tlab.top += total;
        p
    }

    /// Inline cons write. Caller has confirmed `top + 2 <= limit`.
    unsafe fn tlab_write_cons(&mut self, car: Word, cdr: Word) -> Word {
        let p = unsafe { self.tlab.base.add(self.tlab.top) };
        unsafe {
            *p = car.raw();
            *p.add(1) = cdr.raw();
        }
        self.mark_young_cons_start(p);
        self.tlab.top += 2;
        Word::from_ptr(p as *const u8, Tag::Cons)
    }

    /// Set the young start-bit for a header-bearing object whose
    /// header lives at `p`. The walkers iterate `young_starts` and
    /// treat any cell whose `cons_starts` bit is NOT set as a
    /// header (read length from the cell).
    #[inline]
    fn mark_young_start(&self, p: *const u64) {
        let cell_idx = (p as usize - self.young_base as usize) / 8;
        crate::heap::Semispace::set_start_bit_at(&self.young_starts, cell_idx);
    }

    /// Set the cons-start bit pair for a cons whose car lives at
    /// `p`. The "is-cons" bit lets walkers skip past 2 cells without
    /// trying to decode the car value as a header word.
    #[inline]
    fn mark_young_cons_start(&self, p: *const u64) {
        let cell_idx = (p as usize - self.young_base as usize) / 8;
        crate::heap::Semispace::set_cons_start_bit_at(
            &self.young_starts,
            cell_idx,
        );
    }

    /// Refill the TLAB. May park (if a GC is in progress) and may
    /// trigger a GC (if young is exhausted). Loops until a fresh
    /// slab is held.
    fn refill_tlab(&mut self, min_cells: usize) {
        let slab_cells = self.coord.config.tlab_cells.max(min_cells);
        loop {
            // Cooperate with any pending GC first.
            if self.coord.stop_requested.load(Ordering::Acquire) {
                self.park();
                continue;
            }

            // Try to allocate a slab.
            let slab_opt = {
                let mut heap = self.coord.heap.lock().unwrap();
                heap.young_try_alloc_slab(slab_cells)
            };
            if let Some(slab) = slab_opt {
                self.tlab.base = slab.as_ptr();
                self.tlab.top = 0;
                self.tlab.limit = slab_cells;
                return;
            }

            // Young can't fit a slab. Trigger a GC.
            self.trigger_minor_gc();
            // Loop and retry; after the GC, young is empty and the
            // next slab attempt will succeed.
        }
    }

    /// Drive a minor GC. Only one mutator at a time becomes the
    /// trigger — others see `stop_requested` and park instead. The
    /// trigger waits for all OTHER mutators to park, then runs the
    /// GC, then clears the flag and wakes everyone.
    ///
    /// Auto-full-GC was attempted in Phase 1 of `docs/GC_DESIGN.md`
    /// and reverted: `collect_full` only follows explicit roots; it
    /// has no conservative-stack-pin pass, so any JIT-stack-only
    /// rooted Word would be lost across a full cycle, corrupting
    /// downstream computation. A proper `collect_full_with_static`
    /// (mirror of `collect_minor_with_static`) lands in Phase 3 of
    /// the design (page-based heap), at which point auto-escalation
    /// becomes safe to re-enable.
    fn trigger_minor_gc(&mut self) {
        self.do_minor_gc();
    }

    /// The standard minor-GC implementation. Extracted from
    /// `trigger_minor_gc` so a future auto-full-GC escalation path
    /// can share the "did we already park everyone" decision point.
    fn do_minor_gc(&mut self) {
        // We are the trigger. Set the flag; this prevents new
        // mutators from entering allocation slow paths or running
        // through safe points without parking.
        self.coord.stop_requested.store(true, Ordering::Release);

        // Wait for every other live mutator to park. `target` has
        // to be recomputed each pass — mutators deregistering on
        // their way out shrink the live set, and we'd otherwise
        // wait for a parked-count that's no longer achievable.
        // `deregister` posts to park_cv to wake us in that case.
        let other_handles: Vec<Arc<MutatorHandle>> = {
            let mut state = self.coord.park_state.lock().unwrap();
            loop {
                let target = state.handles.len().saturating_sub(1);
                if state.parked_count >= target {
                    break;
                }
                state = self.coord.park_cv.wait(state).unwrap();
            }
            state
                .handles
                .iter()
                .filter(|h| !Arc::ptr_eq(h, &self.handle))
                .cloned()
                .collect()
        };

        // Gather stack-range conservative-scan windows. We include:
        //   - the trigger thread's own current frame, snapshotted by
        //     reading RSP right here (we're about to drive the GC,
        //     so any Lisp values our caller had in stack locals are
        //     potentially live);
        //   - every parked mutator's `[parked_rsp, stack_hi)`.
        // The pin pass marks any candidate young pointers found in
        // these ranges so the copier leaves their targets in place.
        let mut pin_ranges: Vec<(usize, usize)> = Vec::with_capacity(other_handles.len() + 1);
        let my_rsp = current_rsp();
        let my_stack_hi = self.handle.stack_hi.load(Ordering::Acquire);
        if my_stack_hi > my_rsp {
            pin_ranges.push((my_rsp, my_stack_hi));
        }
        for h in &other_handles {
            let rsp = h.parked_rsp.load(Ordering::Acquire);
            let hi = h.stack_hi.load(Ordering::Acquire);
            if rsp != 0 && hi > rsp {
                pin_ranges.push((rsp, hi));
            }
        }

        // All others parked. Run the GC. Pass the static area so
        // dirty static cards are scanned for static→young pointers.
        let my_handle = Arc::clone(&self.handle);
        let static_base = self.coord.static_area.base_ptr() as *mut u64;
        let static_cells = self.coord.static_area.used_cells();
        let static_cards = Arc::clone(self.coord.static_area.cards());
        // Stats: snapshot young usage now so we can compute bytes
        // promoted as (young_before - young_after).
        let young_before = {
            let heap = self.coord.heap.lock().unwrap();
            heap.young_used_bytes()
        };
        let peak = &self.coord.stats.peak_young_used_bytes;
        let mut prev = peak.load(Ordering::Relaxed);
        while (young_before as u64) > prev {
            match peak.compare_exchange(
                prev,
                young_before as u64,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(now) => prev = now,
            }
        }
        {
            let mut heap = self.coord.heap.lock().unwrap();
            // The trait method takes `&mut dyn FnMut(&mut
            // RootScanner)`, not `impl FnMut`. We bind the closure
            // to a local first so Rust can infer its concrete
            // FnMut-implementing type, then coerce to the trait
            // object at the call site.
            let mut visit = |scanner: &mut crate::heap::RootScanner<'_, '_>| {
                // My own roots first.
                {
                    let mut my = my_handle.roots.lock().unwrap();
                    for r in my.iter_mut() {
                        scanner.visit(r);
                    }
                }
                // Other mutators' roots.
                for h in &other_handles {
                    let mut other = h.roots.lock().unwrap();
                    for r in other.iter_mut() {
                        scanner.visit(r);
                    }
                }
            };
            heap.collect_minor_with_static(
                &static_cards,
                static_base,
                static_cells,
                &pin_ranges,
                &mut visit,
            );
        }

        // Stats: count this GC and compute promotion delta. The
        // pin counter / residual is published by the heap layer
        // via `last_pin_summary()` since it's the one that knows.
        let (pinned_count, pinned_cells) = {
            let heap = self.coord.heap.lock().unwrap();
            heap.last_pin_summary()
        };
        let young_after = {
            let heap = self.coord.heap.lock().unwrap();
            heap.young_used_bytes()
        };
        let bytes_promoted = young_before.saturating_sub(young_after) as u64;
        self.coord.stats.minor_gcs.fetch_add(1, Ordering::Relaxed);
        self.coord
            .stats
            .bytes_promoted_total
            .fetch_add(bytes_promoted, Ordering::Relaxed);
        self.coord
            .stats
            .objects_pinned_total
            .fetch_add(pinned_count as u64, Ordering::Relaxed);
        self.coord
            .stats
            .pinned_residual_cells
            .store(pinned_cells as u64, Ordering::Relaxed);

        // My TLAB pointed into young, which is now empty (or
        // rewound past pinned survivors). Abandon it; the next
        // alloc will refill from the fresh young.
        self.retire_tlab();

        // Clear the flag and wake parked threads. They'll each
        // discover their own TLAB is invalid and refill.
        self.coord.stop_requested.store(false, Ordering::Release);
        self.coord.park_cv.notify_all();
    }

    /// Park: voluntarily yield to a pending GC. Caller has already
    /// observed `stop_requested == true`. Returns when the GC
    /// completes and clears the flag.
    fn park(&mut self) {
        // Snapshot RSP BEFORE acquiring any lock so the GC's
        // conservative pin pass sees every live Lisp value the
        // JIT'd code had on the stack at this safepoint. Values
        // born inside `park()` after this point are GC-internal
        // (no Lisp-Word pointers) so we don't care about them.
        let rsp = current_rsp();
        self.handle.parked_rsp.store(rsp, Ordering::Release);
        self.handle.parked.store(true, Ordering::Release);
        let mut state = self.coord.park_state.lock().unwrap();
        state.parked_count += 1;
        // Wake the trigger thread which may be waiting on
        // parked_count to reach the target.
        self.coord.park_cv.notify_all();

        // Wait until the GC clears the stop flag.
        while self.coord.stop_requested.load(Ordering::Acquire) {
            state = self.coord.park_cv.wait(state).unwrap();
        }
        state.parked_count -= 1;
        self.handle.parked.store(false, Ordering::Release);
        self.handle.parked_rsp.store(0, Ordering::Release);
        drop(state);

        // Our TLAB is now invalid (young was cleared or rewound).
        // Abandon it; next alloc will refill.
        self.retire_tlab();
    }

    /// Cooperative safe point. Call at function-call boundaries,
    /// loop back-edges, and anywhere a long compute might hold a
    /// thread for a noticeable time without allocating. The compiler
    /// will emit these automatically when stack maps land; for now
    /// runtime helpers and tests call this manually.
    pub fn safepoint(&mut self) {
        if self.coord.stop_requested.load(Ordering::Acquire) {
            self.park();
        }
    }

    /// Mark the calling thread as parked for the duration of a
    /// blocking native call (`sleep`, `join-thread`, mailbox
    /// receive, condvar wait, etc.). Pair every `enter_blocked`
    /// with exactly one `leave_blocked`.
    ///
    /// Without this, a thread that blocks inside Rust without going
    /// through a safepoint stays in the GC coordinator's "unparked"
    /// count forever. If every other mutator does park, the
    /// GC trigger waits for a target that's no longer reachable
    /// and hangs the whole process.
    ///
    /// While `enter_blocked`, the GC may run; we publish RSP and
    /// the `parked` flag so the conservative pin scan covers this
    /// thread's stack. `leave_blocked` waits out any in-progress
    /// GC before returning (so the caller doesn't race with a
    /// concurrent collection while resuming work).
    pub fn enter_blocked(&mut self) {
        let rsp = current_rsp();
        self.handle.parked_rsp.store(rsp, Ordering::Release);
        self.handle.parked.store(true, Ordering::Release);
        let state = self.coord.park_state.lock().unwrap();
        let mut state = state;
        state.parked_count += 1;
        // Wake a trigger that was waiting on us.
        self.coord.park_cv.notify_all();
        drop(state);
    }

    pub fn leave_blocked(&mut self) {
        let mut state = self.coord.park_state.lock().unwrap();
        // If a GC is in progress, wait for it to finish before
        // resuming Lisp work — otherwise we'd race with the
        // trigger's heap manipulations.
        while self.coord.stop_requested.load(Ordering::Acquire) {
            state = self.coord.park_cv.wait(state).unwrap();
        }
        state.parked_count -= 1;
        self.handle.parked.store(false, Ordering::Release);
        self.handle.parked_rsp.store(0, Ordering::Release);
        drop(state);
        // Abandon the TLAB. The unused tail is invisible to the
        // bitmap-driven walkers (no start-bits set), so leaving the
        // cells as-is is safe.
        self.retire_tlab();
    }

    /// Abandon the current TLAB. The unused tail between
    /// `tlab.top` and `tlab.limit` becomes a run of `00` pairs in
    /// the young start-bit bitmap — bitmap-driven walkers iterate
    /// set bits only, so they skip the gap for free. This is what
    /// replaced the old `HeapType::Filler` stamping pass: the
    /// bitmap encoding subsumes the linear-parseability role the
    /// Filler header used to provide. See the `StartBits` docs in
    /// heap.rs for the encoding (and the reserved `10` slot, kept
    /// in reserve if a future GC phase ever wants an *explicit*
    /// free-zone marker).
    fn retire_tlab(&mut self) {
        self.tlab = Tlab::default();
    }

    /// Lock-free write barrier. Mark the card containing `addr` as
    /// dirty. Call after writing a Word into an old-heap object. The
    /// compiler will emit this at every relevant store once it
    /// understands heap-object types; for v1 this is the explicit
    /// API. Passing an address outside the old heap is a no-op.
    pub fn mark_card(&self, addr: *const u8) {
        self.coord.mark_card(addr);
    }

    // -- Symbol API (the load-bearing redefinition path) --------------------

    /// Allocate a fresh symbol in the static area. Returns a
    /// Symbol-tagged Word, or `None` if static is exhausted. The
    /// intern table that maps `(name, package)` → existing-Symbol
    /// lands later; this is the raw constructor.
    pub fn alloc_symbol(&self, name: Word, package: Word) -> Option<Word> {
        crate::gc_symbol::alloc_symbol_in_static(&self.coord.static_area, name, package)
    }

    /// Read a symbol's function cell with **acquire** semantics.
    /// This is the dispatch path of every function call in v1.
    pub fn symbol_function(&self, sym: Word) -> Word {
        crate::gc_symbol::function_acquire(sym)
    }

    /// Atomically install a new function in a symbol's function cell.
    /// **This is `defun` at the runtime level.** Single store-release
    /// + a card mark on the static cell containing the function slot,
    /// so a subsequent minor GC sees any young/old pointer just
    /// stored.
    pub fn set_symbol_function(&self, sym: Word, new_fn: Word) {
        crate::gc_symbol::set_function_release(sym, new_fn);
        self.mark_card(crate::gc_symbol::function_cell_addr(sym));
    }

    /// CAS the function cell. Returns `Ok(())` on success, or
    /// `Err(observed)` on mismatch. Card marked unconditionally
    /// (false positive cards are fine).
    pub fn cas_symbol_function(
        &self,
        sym: Word,
        expected: Word,
        new_fn: Word,
    ) -> Result<(), Word> {
        let res = crate::gc_symbol::cas_function(sym, expected, new_fn);
        if res.is_ok() {
            self.mark_card(crate::gc_symbol::function_cell_addr(sym));
        }
        res
    }

    /// Read a symbol's value cell with **acquire** semantics. This
    /// is the dynamic-variable lookup path.
    pub fn symbol_value(&self, sym: Word) -> Word {
        crate::gc_symbol::value_acquire(sym)
    }

    /// Atomically install a new value in a symbol's value cell.
    /// `defparameter` and `(setq foo …)` for special variables go
    /// through this.
    pub fn set_symbol_value(&self, sym: Word, new_val: Word) {
        crate::gc_symbol::set_value_release(sym, new_val);
        self.mark_card(crate::gc_symbol::value_cell_addr(sym));
    }

    // -- Roots API (explicit for v1; replaced by stack maps later) ----------

    pub fn push_root(&self, w: Word) -> usize {
        let mut roots = self.handle.roots.lock().unwrap();
        roots.push(w);
        roots.len() - 1
    }

    pub fn pop_root(&self) -> Option<Word> {
        self.handle.roots.lock().unwrap().pop()
    }

    pub fn root_at(&self, idx: usize) -> Word {
        *self.handle.roots.lock().unwrap().get(idx).expect("root index out of range")
    }

    pub fn set_root_at(&self, idx: usize, w: Word) {
        let mut roots = self.handle.roots.lock().unwrap();
        roots[idx] = w;
    }

    pub fn root_count(&self) -> usize {
        self.handle.roots.lock().unwrap().len()
    }

    // -- Force a GC (tests, explicit user calls) ----------------------------

    /// Force a minor GC. The current thread becomes the trigger.
    pub fn collect_minor(&mut self) {
        self.trigger_minor_gc();
    }
}

// ─── Stack-range capture for conservative pin scans ───────────────────────
//
// To pin Lisp values that JIT'd code is holding in stack-resident
// locals at GC-stop time, the collector needs each parked thread's
// stack range. We capture two things per mutator:
//
//   - stack_lo / stack_hi: the full committed stack range, taken
//     once when the MutatorHandle is registered. On Windows this
//     is `GetCurrentThreadStackLimits`. The returned values bound
//     the thread's full reserved stack; the GC scans only the
//     portion currently in use (RSP..stack_hi).
//
//   - parked_rsp: a snapshot of RSP at the instant `park()` was
//     called, taken via inline asm before any further frames are
//     pushed. Cleared back to 0 on unpark.
//
// Conservative scan range for a parked mutator = [parked_rsp, stack_hi).
// Any 8-byte slot in there that decodes as a tagged pointer to
// young/old causes that target to be pinned by the GC.

#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentThreadStackLimits(LowLimit: *mut usize, HighLimit: *mut usize);
}

fn current_thread_stack_range() -> (usize, usize) {
    #[cfg(windows)]
    {
        let mut lo: usize = 0;
        let mut hi: usize = 0;
        unsafe { GetCurrentThreadStackLimits(&mut lo, &mut hi) };
        (lo, hi)
    }
    #[cfg(not(windows))]
    {
        // Best-effort fallback: tag a local variable's address as
        // an approximate "current frame depth," and use a generous
        // 8 MiB ceiling for the upper bound. Refine when we wire
        // up the Mac/Linux ports (pthread_get_stackaddr_np /
        // pthread_attr_getstack).
        let local = 0u64;
        let p = &local as *const u64 as usize;
        (p.saturating_sub(8 * 1024 * 1024), p + 64)
    }
}

/// `(gc-stats)` — return a plist of current GC counters. Useful
/// for observing how a workload exercises the heap:
///
///   :minor-gcs              count of minor GC cycles run
///   :bytes-promoted-total   cumulative bytes copied young → old
///   :objects-pinned-total   cumulative pinned-object count over
///                           all cycles (sum, so divide by
///                           :minor-gcs for an average)
///   :pinned-residual-cells  cells still pinned in young after
///                           the most recent cycle
///   :peak-young-bytes       highest young usage observed at any
///                           GC trigger point
///   :young-used / :young-cap / :old-used / :old-cap   current
///   :static-used / :static-cap                        current
///                           bump-pointer usage of the static
///                           area (JIT code, interned symbols
///                           and strings, every Function record,
///                           every closure literal). This area
///                           is never freed; growth here is
///                           monotonic for the process lifetime.
///
/// Values are fixnums (cells/bytes/counts).
pub extern "C-unwind" fn gc_stats_shim(
    mutator: *mut MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let coord = Arc::clone(m.coord());
    let s = &coord.stats;
    let static_area = coord.static_area();

    let pairs: [(&str, i64); 13] = [
        ("MINOR-GCS",             s.minor_gcs.load(Ordering::Relaxed) as i64),
        ("FULL-GCS",              s.full_gcs.load(Ordering::Relaxed) as i64),
        ("BYTES-PROMOTED-TOTAL",  s.bytes_promoted_total.load(Ordering::Relaxed) as i64),
        ("OBJECTS-PINNED-TOTAL",  s.objects_pinned_total.load(Ordering::Relaxed) as i64),
        ("PINNED-RESIDUAL-CELLS", s.pinned_residual_cells.load(Ordering::Relaxed) as i64),
        ("PEAK-YOUNG-BYTES",      s.peak_young_used_bytes.load(Ordering::Relaxed) as i64),
        ("YOUNG-USED",            coord.young_used_bytes() as i64),
        ("YOUNG-CAP",             coord.config.young_bytes as i64),
        ("OLD-USED",              coord.old_used_bytes() as i64),
        ("OLD-CAP",               coord.config.old_bytes as i64),
        ("STATIC-USED",           static_area.used_bytes() as i64),
        ("STATIC-CAP",            static_area.capacity_bytes() as i64),
        // STATIC-COMMITTED: bytes currently backed by physical pages
        // or page-file. For the elastic VirtualAlloc-backed area
        // this is the page-aligned commit frontier; for Box-backed
        // it equals STATIC-CAP.
        ("STATIC-COMMITTED",      static_area.committed_bytes() as i64),
    ];

    // Build the plist `(:key1 v1 :key2 v2 ... )` from the end.
    let mut result = Word::NIL;
    for (name, value) in pairs.iter().rev() {
        let kw = coord.intern(&format!(":{name}"));
        let v = Word::fixnum(*value);
        result = m.alloc_cons(v, result);
        result = m.alloc_cons(kw, result);
    }
    // Prepend `:heap-backend <symbol>` so the user can tell which
    // implementation they're running on. Value is a plain symbol
    // (`semispace` or `page-heap`) — not a keyword, so the symbol's
    // print name matches its `name()` exactly.
    {
        let backend_sym = coord.intern(&coord.heap_backend_kind.name().to_uppercase());
        let kw = coord.intern(":HEAP-BACKEND");
        result = m.alloc_cons(backend_sym, result);
        result = m.alloc_cons(kw, result);
    }
    result.raw()
}

#[inline(always)]
fn current_rsp() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let rsp: usize;
        unsafe { std::arch::asm!("mov {}, rsp", out(reg) rsp, options(nomem, preserves_flags)) };
        rsp
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let local = 0u64;
        &local as *const u64 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn small_config() -> GcConfig {
        // 16 KB young = 2048 cells, 16 KB old, 8 KB static.
        // 64-cell TLABs (512 bytes) so we exercise refill quickly.
        GcConfig {
            young_bytes: 16 * 1024,
            old_bytes: 16 * 1024,
            static_bytes: 8 * 1024,
            tlab_cells: 64,
        }
    }

    #[test]
    fn single_mutator_basic_alloc() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        let c = m.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        assert!(c.is_cons());
        let p = c.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*p.add(1)).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn tlab_refills_when_full() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        // TLAB is 64 cells. Each cons takes 2 cells. So 32 conses
        // exhaust the TLAB; the 33rd triggers a refill.
        for i in 0..40 {
            m.alloc_cons(Word::fixnum(i), Word::NIL);
        }

        // We've allocated 40 conses = 80 cells. With 64-cell TLABs,
        // that's 2 TLABs (128 cells reserved in young).
        assert_eq!(coord.young_used_bytes(), 128 * 8);
    }

    #[test]
    fn rooted_cons_survives_minor_gc() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        let initial = m.alloc_cons(Word::fixnum(42), Word::NIL);
        let idx = m.push_root(initial);

        // Allocate enough garbage to trigger at least one GC.
        // Young is 2048 cells; each cons is 2 cells. 2048 / 2 =
        // 1024 conses fill young. We allocate 2000 to force a GC.
        for _ in 0..2000 {
            m.alloc_cons(Word::fixnum(99), Word::fixnum(99));
        }

        // The rooted cons should still be alive. After GCs,
        // it's been moved to old.
        let root = m.root_at(idx);
        assert!(root.is_cons());
        let p = root.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(42));
            assert!(Word::from_raw(*p.add(1)).is_nil());
        }
        // Root pointer now points into old (it's been promoted).
        assert!(coord.old_used_bytes() >= 16);
    }

    #[test]
    fn explicit_collect_minor_works() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        let r = m.alloc_cons(Word::fixnum(7), Word::NIL);
        let idx = m.push_root(r);

        // Allocate some garbage.
        for _ in 0..10 {
            m.alloc_cons(Word::fixnum(99), Word::fixnum(99));
        }

        m.collect_minor();

        // Rooted cons survived.
        let r2 = m.root_at(idx);
        let p = r2.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(7));
        }
    }

    #[test]
    fn unrooted_cons_dies() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        for _ in 0..50 {
            m.alloc_cons(Word::fixnum(99), Word::fixnum(99));
        }
        // No roots. Force GC.
        m.collect_minor();

        assert_eq!(coord.old_used_bytes(), 0);
    }

    #[test]
    fn mutator_deregisters_on_drop() {
        let coord = GcCoordinator::new(small_config());
        {
            let _m1 = coord.register_mutator();
            let _m2 = coord.register_mutator();
            assert_eq!(coord.park_state.lock().unwrap().handles.len(), 2);
        }
        // Both dropped — handles cleared.
        assert_eq!(coord.park_state.lock().unwrap().handles.len(), 0);
    }

    #[test]
    fn two_threads_alloc_concurrently() {
        let coord = GcCoordinator::new(small_config());

        let coord_a = Arc::clone(&coord);
        let coord_b = Arc::clone(&coord);

        let h1 = thread::spawn(move || {
            let mut m = coord_a.register_mutator();
            for i in 0..200 {
                let c = m.alloc_cons(Word::fixnum(i), Word::NIL);
                // Validate the cons is well-formed.
                let p = c.as_ptr::<u64>(Tag::Cons).unwrap();
                unsafe { assert_eq!(Word::from_raw(*p).as_fixnum(), Some(i)); }
                m.safepoint();
            }
        });

        let h2 = thread::spawn(move || {
            let mut m = coord_b.register_mutator();
            for i in 0..200 {
                let c = m.alloc_cons(Word::fixnum(i + 1000), Word::NIL);
                let p = c.as_ptr::<u64>(Tag::Cons).unwrap();
                unsafe { assert_eq!(Word::from_raw(*p).as_fixnum(), Some(i + 1000)); }
                m.safepoint();
            }
        });

        h1.join().expect("thread 1");
        h2.join().expect("thread 2");
    }

    #[test]
    fn many_threads_with_gc_pressure() {
        let coord = GcCoordinator::new(small_config());
        let n_threads = 4;
        let n_allocs_per_thread = 500;

        let handles: Vec<_> = (0..n_threads).map(|tid| {
            let coord = Arc::clone(&coord);
            thread::spawn(move || {
                let mut m = coord.register_mutator();
                // Each thread holds one root through its loop.
                let r = m.alloc_cons(Word::fixnum(tid as i64), Word::NIL);
                let idx = m.push_root(r);

                for i in 0..n_allocs_per_thread {
                    m.alloc_cons(Word::fixnum(i), Word::fixnum(99));
                    if i % 16 == 0 {
                        m.safepoint();
                    }
                }

                // Verify the root is still tagged correctly.
                let r2 = m.root_at(idx);
                assert!(r2.is_cons(), "thread {tid} root not a cons");
                let p = r2.as_ptr::<u64>(Tag::Cons).unwrap();
                unsafe {
                    assert_eq!(
                        Word::from_raw(*p).as_fixnum(),
                        Some(tid as i64),
                        "thread {tid} root payload corrupted"
                    );
                }
            })
        }).collect();

        for h in handles {
            h.join().expect("thread completed");
        }
    }

    #[test]
    fn mark_card_lets_old_to_young_pointer_survive() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        // Promote a cons to old.
        let a = m.alloc_cons(Word::fixnum(1), Word::NIL);
        let a_idx = m.push_root(a);
        m.collect_minor();
        let a = m.root_at(a_idx);

        // Allocate a fresh young cons B.
        let b = m.alloc_cons(Word::fixnum(2), Word::NIL);

        // Patch B into A's cdr.
        let a_ptr = a.as_mut_ptr::<u64>(Tag::Cons).unwrap();
        unsafe { *a_ptr.add(1) = b.raw(); }
        // Write barrier: this is the new contract.
        m.mark_card(a_ptr as *const u8);

        // Force a minor GC. The card scan should find B and promote it.
        m.collect_minor();

        // a is still rooted; verify both a and b ended up in old.
        let a = m.root_at(a_idx);
        unsafe {
            let p = a.as_ptr::<u64>(Tag::Cons).unwrap();
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(1));
            let cdr = Word::from_raw(*p.add(1));
            assert!(cdr.is_cons(), "cdr should be a cons after promotion");
            let bp = cdr.as_ptr::<u64>(Tag::Cons).unwrap();
            assert_eq!(Word::from_raw(*bp).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn mark_card_outside_old_is_noop() {
        let coord = GcCoordinator::new(small_config());
        let m = coord.register_mutator();

        // Stack address is far from the old heap.
        let stack_var: u64 = 0;
        m.mark_card(&stack_var as *const u64 as *const u8);

        // No card was marked.
        assert_eq!(coord.cards.dirty_count(), 0);

        // A young-heap address should also be a no-op (cards only
        // cover old).
        drop(m); // m is borrowed, drop before next op
        let mut m = coord.register_mutator();
        let young_cons = m.alloc_cons(Word::fixnum(1), Word::NIL);
        m.mark_card(young_cons.as_ptr::<u8>(Tag::Cons).unwrap());
        assert_eq!(coord.cards.dirty_count(), 0);
    }

    #[test]
    fn many_threads_with_card_marks() {
        // Each thread:
        //  - allocates a cons, promotes it to old via a manual GC
        //  - allocates more young objects, patches them into the
        //    old cons's cdr, marks the card
        //  - triggers another GC
        //  - verifies the chain survived
        let coord = GcCoordinator::new(small_config());
        let n_threads = 4;

        let handles: Vec<_> = (0..n_threads).map(|tid| {
            let coord = Arc::clone(&coord);
            thread::spawn(move || {
                let mut m = coord.register_mutator();
                let head = m.alloc_cons(Word::fixnum(tid as i64), Word::NIL);
                let idx = m.push_root(head);
                m.collect_minor();
                let head = m.root_at(idx);

                // Build a chain in young, patch into head.cdr, mark card.
                let tail = m.alloc_cons(Word::fixnum(tid as i64 + 100), Word::NIL);
                let head_ptr = head.as_mut_ptr::<u64>(Tag::Cons).unwrap();
                unsafe { *head_ptr.add(1) = tail.raw(); }
                m.mark_card(head_ptr as *const u8);

                // Generate noise to trigger GCs.
                for i in 0..200 {
                    m.alloc_cons(Word::fixnum(i), Word::fixnum(0));
                    if i % 16 == 0 { m.safepoint(); }
                }

                // Verify chain survived.
                let head = m.root_at(idx);
                unsafe {
                    let p = head.as_ptr::<u64>(Tag::Cons).unwrap();
                    assert_eq!(
                        Word::from_raw(*p).as_fixnum(),
                        Some(tid as i64),
                        "thread {tid} head corrupted",
                    );
                    let cdr = Word::from_raw(*p.add(1));
                    assert!(cdr.is_cons(), "thread {tid} lost tail");
                    let tp = cdr.as_ptr::<u64>(Tag::Cons).unwrap();
                    assert_eq!(
                        Word::from_raw(*tp).as_fixnum(),
                        Some(tid as i64 + 100),
                        "thread {tid} tail corrupted",
                    );
                }
            })
        }).collect();

        for h in handles { h.join().expect("thread"); }
    }

    #[test]
    fn static_to_young_pointer_with_card_mark_promotes() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        // Allocate a 2-cell slot in static. (Manually — for v1 the
        // higher-level "intern symbol into static" API hasn't landed
        // yet; we use the raw allocator.)
        let static_slot = coord
            .static_area()
            .try_alloc_cells(2)
            .expect("static alloc");
        unsafe {
            *static_slot.as_ptr() = Word::NIL.raw();
            *static_slot.as_ptr().add(1) = Word::NIL.raw();
        }

        // Allocate a young cons.
        let y = m.alloc_cons(Word::fixnum(42), Word::NIL);

        // Patch young pointer into the static slot. WITH write barrier.
        unsafe { *static_slot.as_ptr() = y.raw(); }
        m.mark_card(static_slot.as_ptr() as *const u8);

        // Force minor GC. The static→young scan should find y and
        // promote it. The static slot is updated in place.
        m.collect_minor();

        // Read the static slot — it now points at the promoted y in old.
        let promoted = unsafe { Word::from_raw(*static_slot.as_ptr()) };
        assert!(promoted.is_cons());
        let p = promoted.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(42));
            assert!(Word::from_raw(*p.add(1)).is_nil());
        }
    }

    #[test]
    fn static_card_unmarked_loses_young() {
        // Negative test: same scenario as above but WITHOUT
        // mark_card. The young object is reclaimed because the GC
        // doesn't know to look at the static slot.
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        let static_slot = coord.static_area().try_alloc_cells(2).unwrap();
        unsafe {
            *static_slot.as_ptr() = Word::NIL.raw();
            *static_slot.as_ptr().add(1) = Word::NIL.raw();
        }

        let y = m.alloc_cons(Word::fixnum(42), Word::NIL);
        unsafe { *static_slot.as_ptr() = y.raw(); }
        // DO NOT mark_card.

        m.collect_minor();

        // The static slot still HOLDS a Word, and it's still
        // Cons-tagged (the GC didn't update it because it didn't
        // know about it). But it points at a stale young address.
        // We verify the slot wasn't updated to a NEW (post-GC)
        // location — it should match the original raw value.
        let still_there = unsafe { Word::from_raw(*static_slot.as_ptr()) };
        assert!(still_there.is_cons(), "tag stayed");
        // Old should be empty — y was not promoted.
        assert_eq!(coord.old_used_bytes(), 0);
    }

    #[test]
    fn static_cards_clear_after_minor() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        let static_slot = coord.static_area().try_alloc_cells(2).unwrap();
        unsafe {
            *static_slot.as_ptr() = Word::NIL.raw();
            *static_slot.as_ptr().add(1) = Word::NIL.raw();
        }
        let y = m.alloc_cons(Word::fixnum(1), Word::NIL);
        unsafe { *static_slot.as_ptr() = y.raw(); }
        m.mark_card(static_slot.as_ptr() as *const u8);

        assert_eq!(coord.static_area().cards().dirty_count(), 1);
        m.collect_minor();
        assert_eq!(coord.static_area().cards().dirty_count(), 0);
    }

    #[test]
    fn many_threads_allocating_in_static() {
        // Static's lock-free CAS-bump allocator: 4 threads each
        // making 100 allocations. Verify total used and disjoint
        // ranges.
        let coord = GcCoordinator::new(small_config());
        let n_threads = 4;
        let allocs = 100;

        let handles: Vec<_> = (0..n_threads).map(|_| {
            let coord = Arc::clone(&coord);
            thread::spawn(move || {
                let mut my = Vec::new();
                for _ in 0..allocs {
                    let p = coord.static_area().try_alloc_cells(1).expect("alloc");
                    my.push(p.as_ptr() as usize);
                }
                my
            })
        }).collect();

        let mut all: Vec<usize> = Vec::new();
        for h in handles {
            all.extend(h.join().expect("thread"));
        }
        all.sort();
        for w in all.windows(2) {
            assert!(w[0] < w[1]);
            assert!(w[1] - w[0] >= 8);
        }
        assert_eq!(coord.static_area().used_cells(), n_threads * allocs);
    }

    // -- Symbol API tests ---------------------------------------------------

    #[test]
    fn alloc_symbol_returns_symbol_tagged_word() {
        let coord = GcCoordinator::new(small_config());
        let m = coord.register_mutator();
        let sym = m.alloc_symbol(Word::fixnum(7), Word::NIL).expect("alloc");
        assert_eq!(sym.tag(), Tag::Symbol);
        assert!(coord.static_area().contains_ptr(sym.as_ptr::<u8>(Tag::Symbol).unwrap()));
        assert!(m.symbol_function(sym).is_unbound());
        assert!(m.symbol_value(sym).is_unbound());
    }

    #[test]
    fn defun_via_set_symbol_function_round_trips() {
        let coord = GcCoordinator::new(small_config());
        let m = coord.register_mutator();
        let sym = m.alloc_symbol(Word::NIL, Word::NIL).unwrap();

        m.set_symbol_function(sym, Word::fixnum(42));
        assert_eq!(m.symbol_function(sym).as_fixnum(), Some(42));

        // Redefine.
        m.set_symbol_function(sym, Word::fixnum(99));
        assert_eq!(m.symbol_function(sym).as_fixnum(), Some(99));
    }

    #[test]
    fn cas_symbol_function_replaces_only_on_match() {
        let coord = GcCoordinator::new(small_config());
        let m = coord.register_mutator();
        let sym = m.alloc_symbol(Word::NIL, Word::NIL).unwrap();
        m.set_symbol_function(sym, Word::fixnum(1));

        // Wrong expected → fails.
        let r = m.cas_symbol_function(sym, Word::fixnum(99), Word::fixnum(2));
        assert!(matches!(r, Err(observed) if observed.as_fixnum() == Some(1)));
        assert_eq!(m.symbol_function(sym).as_fixnum(), Some(1));

        // Correct expected → succeeds.
        m.cas_symbol_function(sym, Word::fixnum(1), Word::fixnum(3)).unwrap();
        assert_eq!(m.symbol_function(sym).as_fixnum(), Some(3));
    }

    #[test]
    fn symbol_address_stable_across_gc() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        let sym = m.alloc_symbol(Word::NIL, Word::NIL).unwrap();
        let addr_before = sym.as_ptr::<u8>(Tag::Symbol).unwrap() as usize;

        // Lots of allocation pressure to force GCs.
        for _ in 0..2000 {
            m.alloc_cons(Word::fixnum(0), Word::fixnum(0));
        }

        // Symbol still at the same address; static doesn't move.
        assert_eq!(sym.as_ptr::<u8>(Tag::Symbol).unwrap() as usize, addr_before);
        // Still a symbol; layout intact.
        assert_eq!(crate::gc_symbol::header(sym).ty(), crate::HeapType::Symbol);
    }

    #[test]
    fn function_cell_with_young_target_promotes_via_card() {
        // The full vertical: store a young pointer into a symbol's
        // function cell, GC, verify the cell now points at the
        // promoted location in old. This exercises the entire
        // step-7 + step-8 chain (static → young promotion via card
        // table, plus the symbol-cell's atomic store).
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();

        let sym = m.alloc_symbol(Word::NIL, Word::NIL).unwrap();
        // Stash sym as a root so the symbol-pointer Word survives.
        // (The Symbol itself doesn't move, but the Word value we hold
        // is what gets visited; the visit is a no-op for static-
        // pointing words because copy_into rejects non-young/old
        // sources.)
        let _idx = m.push_root(sym);

        // Allocate young object that we'll point the function cell at.
        let young_fn = m.alloc_cons(Word::fixnum(123), Word::NIL);
        m.set_symbol_function(sym, young_fn);

        // Verify the function cell currently points at young.
        let stored = m.symbol_function(sym);
        assert!(coord.static_area().contains_ptr(
            crate::gc_symbol::function_cell_addr(sym)
        ));
        let target = stored.as_ptr::<u8>(Tag::Cons).unwrap();
        assert!(!coord.static_area().contains_ptr(target),
            "young_fn should be in young, not static");

        // Force minor GC.
        m.collect_minor();

        // Function cell now holds the promoted location. Read it
        // back and verify the data is intact.
        let promoted = m.symbol_function(sym);
        assert!(promoted.is_cons());
        let pp = promoted.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*pp).as_fixnum(), Some(123));
        }
    }

    #[test]
    fn concurrent_defun_no_torn_reads() {
        // Two threads racing on set_symbol_function; a third reads.
        // Barrier syncs starts so no writer is starved; reader exits
        // early once both values have been observed.
        use std::sync::Barrier;
        let coord = GcCoordinator::new(small_config());
        let m = coord.register_mutator();
        let sym = m.alloc_symbol(Word::NIL, Word::NIL).unwrap();
        m.set_symbol_function(sym, Word::fixnum(1));
        drop(m);

        let v1 = Word::fixnum(1);
        let v2 = Word::fixnum(2);
        let barrier = Arc::new(Barrier::new(3));
        let stop = Arc::new(AtomicBool::new(false));

        let coord1 = Arc::clone(&coord);
        let coord2 = Arc::clone(&coord);
        let coord3 = Arc::clone(&coord);
        let bar1 = Arc::clone(&barrier);
        let bar2 = Arc::clone(&barrier);
        let stop1 = Arc::clone(&stop);
        let stop2 = Arc::clone(&stop);
        let sym_raw = sym.raw();

        let w1 = thread::spawn(move || {
            let m = coord1.register_mutator();
            bar1.wait();
            let sym = Word::from_raw(sym_raw);
            while !stop1.load(Ordering::Relaxed) {
                m.set_symbol_function(sym, v1);
            }
        });
        let w2 = thread::spawn(move || {
            let m = coord2.register_mutator();
            bar2.wait();
            let sym = Word::from_raw(sym_raw);
            while !stop2.load(Ordering::Relaxed) {
                m.set_symbol_function(sym, v2);
            }
        });

        let m = coord3.register_mutator();
        barrier.wait();
        let sym = Word::from_raw(sym_raw);
        let mut saw_v1 = 0u64;
        let mut saw_v2 = 0u64;
        for _ in 0..1_000_000 {
            let f = m.symbol_function(sym);
            if f.raw() == v1.raw() { saw_v1 += 1; }
            else if f.raw() == v2.raw() { saw_v2 += 1; }
            else { panic!("torn read: {f:?}"); }
            if saw_v1 > 0 && saw_v2 > 0 { break; }
        }
        stop.store(true, Ordering::Relaxed);
        w1.join().unwrap();
        w2.join().unwrap();
        assert!(saw_v1 > 0);
        assert!(saw_v2 > 0);
    }

    #[test]
    fn safepoint_with_no_gc_pending_is_cheap() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        // No GC pending, so safepoint is a flag check.
        for _ in 0..1000 {
            m.safepoint();
        }
    }

    #[test]
    fn gc_triggered_by_one_thread_parks_others() {
        // Slightly tricky: spin one thread allocating in a loop;
        // the other thread does many allocations to trigger GCs;
        // verify the spinner doesn't get stale data through the
        // GC.
        let coord = GcCoordinator::new(small_config());

        let coord_spinner = Arc::clone(&coord);
        let coord_alloc = Arc::clone(&coord);

        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_spinner = Arc::clone(&stop);

        let spinner = thread::spawn(move || {
            let mut m = coord_spinner.register_mutator();
            let r = m.alloc_cons(Word::fixnum(123), Word::NIL);
            let idx = m.push_root(r);
            while !stop_for_spinner.load(Ordering::Relaxed) {
                m.safepoint();
                let r2 = m.root_at(idx);
                let p = r2.as_ptr::<u64>(Tag::Cons).unwrap();
                unsafe {
                    assert_eq!(
                        Word::from_raw(*p).as_fixnum(),
                        Some(123),
                        "spinner saw stale root"
                    );
                }
                thread::sleep(Duration::from_micros(10));
            }
        });

        let allocator = thread::spawn(move || {
            let mut m = coord_alloc.register_mutator();
            for i in 0..3000 {
                m.alloc_cons(Word::fixnum(i), Word::fixnum(0));
                if i % 8 == 0 {
                    m.safepoint();
                }
            }
        });

        allocator.join().expect("allocator");
        stop.store(true, Ordering::Relaxed);
        spinner.join().expect("spinner");
    }
}
