//! `SharedHeap` — the lock-free, atomically-accessed slice of the heap.
//!
//! Sprint MM-2 of the multi-mutator plan (`docs/MULTI_MUTATOR_DESIGN.md`
//! §2.0). These are the fields a mutator must touch **without** taking
//! the heap lock: the start-bit bitmap and card table (already atomic),
//! the poison flag, and the allocation counter. Extracting them into a
//! separate `Arc`-shared struct is the prerequisite for:
//!
//!   1. the lock-free TLAB fast path (MM-3) — bump + set-start-bit +
//!      poison-check + alloc-counter without locking, and
//!   2. the soundness of the collector's `&mut PageHeap` while mutators
//!      are parked — a mutator holds `Arc<SharedHeap>`, never a bare
//!      `&PageHeap`, so the two can't alias.
//!
//! MM-2 is a **pure refactor**: `PageHeap` now reaches these fields
//! through `self.shared`, with identical single-threaded behavior. The
//! `poisoned` flag becomes `AtomicBool` and `bytes_alloc_since_gc`
//! becomes `AtomicUsize`; everything still runs under the heap mutex
//! today, so the orderings (Acquire/Release on poison, Relaxed on the
//! counter) are conservative-correct and future-proof for MM-3.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize};
use std::sync::{Arc, Condvar, Mutex};

use crate::heap_common::CardTable;

use super::alloc::PageStartBits;

/// Cooperative stop-the-world rendezvous (MM-4; design §4). A collection
/// driver bumps `epoch` and flips `world_running` to 0; every other
/// mutator notices at its next `poll_safepoint`, parks (publishing its
/// roots + flushing its TLABs), advances its own `last_epoch` to the
/// target, and blocks on `park_cv` until `world_running` is 1 again.
/// There is **no global `parked_count`** — the driver waits on each
/// mutator's own `last_epoch` (design B-1).
pub struct Safepoint {
    /// Bumped (under `coord_mutex`) to request a safepoint. A mutator
    /// whose `last_epoch < epoch` owes a park.
    pub(crate) epoch: AtomicU64,
    /// 1 = world running; 0 = stopped, parked mutators must block.
    pub(crate) world_running: AtomicU8,
    /// Guards the condvar wait/notify; also the mutex the driver holds
    /// while polling per-mutator `last_epoch`.
    pub(crate) park_mutex: Mutex<()>,
    /// Mutators wait here for resume; the driver waits here for arrivals.
    pub(crate) park_cv: Condvar,
    /// Per-arrival `wait_timeout` budget, in milliseconds (default
    /// `DEFAULT_SAFEPOINT_TIMEOUT_MS`). A diagnostic backstop only — the
    /// protocol does not rely on it; it bounds how long a driver waits on
    /// a single non-cooperating mutator before re-checking. Settable via
    /// `GcCoordinator::set_safepoint_timeout`. Relaxed: read once per
    /// wait, set rarely; exact cross-thread freshness doesn't matter.
    pub(crate) wait_timeout_ms: AtomicU64,
}

/// Default safepoint per-arrival wait budget (10 s).
pub(crate) const DEFAULT_SAFEPOINT_TIMEOUT_MS: u64 = 10_000;

impl Safepoint {
    fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            world_running: AtomicU8::new(1),
            park_mutex: Mutex::new(()),
            park_cv: Condvar::new(),
            wait_timeout_ms: AtomicU64::new(DEFAULT_SAFEPOINT_TIMEOUT_MS),
        }
    }
}

/// Lock-free shared heap state. Cloned (via `Arc`) into `PageHeap` and,
/// from MM-3 on, into every `Mutator`. Not generic over the layout —
/// none of these fields depend on `L`.
pub struct SharedHeap {
    /// Set once a `try_collect_*` aborts on mid-evacuation OOM. Once
    /// poisoned, allocation refuses and further `try_collect_*` calls
    /// short-circuit. Acquire load / Release store.
    pub(crate) poisoned: AtomicBool,
    /// Bytes the mutator has allocated since the last collection. Drives
    /// `should_collect`. Relaxed — it's a heuristic trigger, and exact
    /// cross-thread freshness isn't required.
    pub(crate) bytes_alloc_since_gc: AtomicUsize,
    /// Global start-bit bitmap (2 bits/cell). Already atomic; mutators
    /// set starts via `fetch_or(Relaxed)`.
    pub(crate) start_bits: PageStartBits,
    /// Soft card table over the whole reservation. Atomic interior;
    /// `mark_card_at` is a Relaxed byte store.
    pub(crate) cards: Arc<CardTable>,
    /// Stop-the-world rendezvous (MM-4).
    pub(crate) safepoint: Safepoint,
    /// Serializes collection drivers and mutator registration, so only
    /// one STW cycle runs at a time and a newcomer can't join mid-cycle
    /// (design §2.2, §4.4).
    pub(crate) coord_mutex: Mutex<()>,
}

impl SharedHeap {
    /// Build the shared state for a fresh heap.
    pub(crate) fn new(start_bits: PageStartBits, cards: Arc<CardTable>) -> Self {
        Self {
            poisoned: AtomicBool::new(false),
            bytes_alloc_since_gc: AtomicUsize::new(0),
            start_bits,
            cards,
            safepoint: Safepoint::new(),
            coord_mutex: Mutex::new(()),
        }
    }
}
