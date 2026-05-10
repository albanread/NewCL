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

use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::heap::Heap;
use crate::word::{Tag, Word};

/// Configuration knobs for the GC. Real values for production land
/// later (16 MB young, 64 MB old, 512 KB TLAB by the design doc);
/// tests override with much smaller numbers.
#[derive(Clone, Copy, Debug)]
pub struct GcConfig {
    pub young_bytes: usize,
    pub old_bytes: usize,
    /// Cells per TLAB. Each cell is 8 bytes.
    pub tlab_cells: usize,
}

impl Default for GcConfig {
    fn default() -> Self {
        GcConfig {
            young_bytes: 16 * 1024 * 1024,
            old_bytes: 64 * 1024 * 1024,
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
    heap: Mutex<Heap>,
    /// "Park yourselves." Mutators poll this at safe points.
    stop_requested: AtomicBool,
    /// Set of registered mutators + how many are currently parked.
    park_state: Mutex<ParkState>,
    /// Used to wait for all-others-parked (by the GC trigger) and
    /// for "GC is done" (by parked mutators).
    park_cv: Condvar,
}

struct ParkState {
    handles: Vec<Arc<MutatorHandle>>,
    parked_count: usize,
}

impl GcCoordinator {
    pub fn new(config: GcConfig) -> Arc<GcCoordinator> {
        Arc::new(GcCoordinator {
            heap: Mutex::new(Heap::new(config.young_bytes, config.old_bytes)),
            config,
            stop_requested: AtomicBool::new(false),
            park_state: Mutex::new(ParkState {
                handles: Vec::new(),
                parked_count: 0,
            }),
            park_cv: Condvar::new(),
        })
    }

    /// Register a new mutator. Returns the per-thread state. The
    /// returned `MutatorState` is `Send` (so it can move to a
    /// thread) but `!Sync` (only owned by one thread at a time).
    /// On drop, the mutator deregisters from the coordinator.
    pub fn register_mutator(self: &Arc<Self>) -> MutatorState {
        let handle = Arc::new(MutatorHandle {
            parked: AtomicBool::new(false),
            roots: Mutex::new(Vec::new()),
        });
        self.park_state
            .lock()
            .unwrap()
            .handles
            .push(Arc::clone(&handle));
        MutatorState {
            coord: Arc::clone(self),
            handle,
            tlab: Tlab::default(),
        }
    }

    fn deregister(&self, handle: &Arc<MutatorHandle>) {
        let mut state = self.park_state.lock().unwrap();
        state.handles.retain(|h| !Arc::ptr_eq(h, handle));
    }

    pub fn used_bytes(&self) -> usize {
        self.heap.lock().unwrap().used_bytes()
    }

    pub fn young_used_bytes(&self) -> usize {
        self.heap.lock().unwrap().young_used_bytes()
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
}

// SAFETY: MutatorState contains a `*mut u64` (in Tlab) which is not
// Sync. We deliberately don't impl Sync. We DO impl Send: a
// MutatorState can move from creating thread to its working thread,
// but only one thread accesses it at a time.
unsafe impl Send for MutatorState {}

impl Drop for MutatorState {
    fn drop(&mut self) {
        self.coord.deregister(&self.handle);
    }
}

impl MutatorState {
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

    /// Inline cons write. Caller has confirmed `top + 2 <= limit`.
    unsafe fn tlab_write_cons(&mut self, car: Word, cdr: Word) -> Word {
        let p = unsafe { self.tlab.base.add(self.tlab.top) };
        unsafe {
            *p = car.raw();
            *p.add(1) = cdr.raw();
        }
        self.tlab.top += 2;
        Word::from_ptr(p as *const u8, Tag::Cons)
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
    fn trigger_minor_gc(&mut self) {
        // We are the trigger. Set the flag; this prevents new
        // mutators from entering allocation slow paths or running
        // through safe points without parking.
        self.coord.stop_requested.store(true, Ordering::Release);

        // Wait for every other mutator to park.
        let other_handles: Vec<Arc<MutatorHandle>> = {
            let mut state = self.coord.park_state.lock().unwrap();
            let total = state.handles.len();
            // We are not parked; we're total - 1 from done.
            let target = total.saturating_sub(1);
            while state.parked_count < target {
                state = self.coord.park_cv.wait(state).unwrap();
            }
            // Snapshot of other handles for root scanning.
            state
                .handles
                .iter()
                .filter(|h| !Arc::ptr_eq(h, &self.handle))
                .cloned()
                .collect()
        };

        // All others parked. Run the GC.
        let my_handle = Arc::clone(&self.handle);
        {
            let mut heap = self.coord.heap.lock().unwrap();
            heap.collect_minor(|scanner| {
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
            });
        }

        // My TLAB pointed into young, which is now empty.
        // Reset so the next refill attempt reads a fresh slab.
        self.tlab = Tlab::default();

        // Clear the flag and wake parked threads. They'll each
        // discover their own TLAB is invalid and refill.
        self.coord.stop_requested.store(false, Ordering::Release);
        self.coord.park_cv.notify_all();
    }

    /// Park: voluntarily yield to a pending GC. Caller has already
    /// observed `stop_requested == true`. Returns when the GC
    /// completes and clears the flag.
    fn park(&mut self) {
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
        drop(state);

        // Our TLAB is now invalid (young was cleared).
        self.tlab = Tlab::default();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn small_config() -> GcConfig {
        // 16 KB young = 2048 cells, 16 KB old. 64-cell TLABs (512
        // bytes) so we can exercise refill quickly.
        GcConfig {
            young_bytes: 16 * 1024,
            old_bytes: 16 * 1024,
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
