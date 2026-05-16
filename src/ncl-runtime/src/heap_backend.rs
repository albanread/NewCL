//! Heap-backend abstraction.
//!
//! Phase 3 sub-phase 1 of `docs/GC_DESIGN.md` introduces this trait
//! so the GC coordinator can talk to either:
//!
//!   - The current `Heap` (two-semispace generational copying GC,
//!     `heap.rs`), or
//!   - The forthcoming `PageHeap` (page-based mark-evacuate with
//!     three generations, sub-phases 2-10) once it lands.
//!
//! The trait surface is **exactly** the set of methods that
//! `mutator.rs` already calls on `Heap`. No new functionality is
//! introduced in this sub-phase — only a layer of indirection so a
//! second implementation can slot in side-by-side via
//! `GcConfig::backend`.
//!
//! ## Why a trait and not a typedef
//!
//! The migration plan (`docs/GC_DESIGN.md` §5b) calls for
//! side-by-side coexistence of the old and new heap implementations
//! for ~3-4 weeks. That requires runtime dispatch (a `Box<dyn
//! HeapBackend>` held by `GcCoordinator`) — a typedef alias would
//! force a build-time pick between the two and we'd lose easy
//! rollback. Dynamic dispatch costs one vtable indirection per
//! coordinator-mediated call; allocation fast paths bypass this
//! entirely via `MutatorState` caching the young-heap base pointer
//! and start-bit bitmap at registration time.
//!
//! ## `collect_minor_with_static` and `RootScanner`
//!
//! The minor-GC entry point takes a callback that receives a
//! `RootScanner` to visit each per-mutator root word. `RootScanner`
//! currently lives in `heap.rs` as a thin enum over the semispace's
//! `MinorState` / `FullState`. The page heap will need a different
//! internal scanner state, but it can still expose the same
//! `RootScanner` API by mirroring the enum's variants. Sub-phase 7
//! either keeps this shape or introduces a `RootVisitor` trait if
//! the duplication becomes painful.
//!
//! ## Backend-agnostic `dynamic_*` surface
//!
//! Sub-phase 6.5 of the design doc adds `dynamic_used_bytes`,
//! `dynamic_capacity_bytes`, `dynamic_base_ptr`, `dynamic_cards`,
//! and `dynamic_contains` to the trait. Each has a default
//! implementation that delegates to the existing semispace-shaped
//! `young_*` / `old_*` methods, so the production `Heap` gets the
//! new surface for free. The forthcoming `PageHeap` will override
//! them with backend-native logic (one big card table, single
//! reservation range, aggregate usage across all generations).
//!
//! The semispace-specific `young_*` / `old_*` methods are
//! `#[deprecated]` so call sites get a hint to migrate. Sub-phase
//! 12 deletes them entirely; until then the production semispace
//! impl and the gc-stats path keep using them.

use std::ptr::NonNull;
use std::sync::Arc;

use crate::heap::{CardTable, RootScanner, StartBits};

/// Which heap implementation to instantiate. Picked at coordinator
/// construction; cannot be swapped on a live session.
///
/// Production default is `Semispace` (the original NCL heap,
/// `heap.rs`). The `PageHeap` variant exists as scaffolding for
/// Phase 3 sub-phases 2-7 of `docs/GC_DESIGN.md`; selecting it
/// today panics on coordinator construction because the
/// implementation isn't built yet.
///
/// Switch via the env var `NCL_HEAP_BACKEND`:
///
///   NCL_HEAP_BACKEND=semispace   (default — explicit form)
///   NCL_HEAP_BACKEND=page-heap   (will panic until Phase 3 lands)
///
/// Reading is once-per-coordinator at startup; flipping mid-session
/// has no effect. Unknown values fall back to `Semispace` with a
/// stderr warning so a typo doesn't silently mis-select.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapBackendKind {
    /// Two-semispace generational copying GC. `heap.rs` —
    /// production-tested since project inception.
    Semispace,
    /// Page-based mark-evacuate with three generations and per-page
    /// pin bitmaps. **Under construction** as Phase 3 of the GC
    /// design doc; pick at your peril until sub-phase 7 lands.
    PageHeap,
}

impl Default for HeapBackendKind {
    fn default() -> Self {
        HeapBackendKind::Semispace
    }
}

impl HeapBackendKind {
    /// Resolve from the `NCL_HEAP_BACKEND` env var. Unknown values
    /// trigger a stderr warning and fall back to the default
    /// (semispace).
    pub fn from_env() -> Self {
        match std::env::var("NCL_HEAP_BACKEND") {
            Ok(s) => match s.to_ascii_lowercase().as_str() {
                "semispace" | "" => HeapBackendKind::Semispace,
                "page-heap" | "page_heap" | "pageheap" => HeapBackendKind::PageHeap,
                other => {
                    eprintln!(
                        "ncl: warning: NCL_HEAP_BACKEND={other:?} unrecognised; \
                         using semispace. Known: semispace, page-heap."
                    );
                    HeapBackendKind::Semispace
                }
            },
            Err(_) => HeapBackendKind::default(),
        }
    }

    /// Human-readable label for `(gc-stats)` / diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            HeapBackendKind::Semispace => "semispace",
            HeapBackendKind::PageHeap => "page-heap",
        }
    }
}

/// The minimal interface every heap implementation must expose to
/// the GC coordinator and any code that needs to inspect the heap
/// from outside the runtime (notably `(gc-stats)`).
///
/// Two surfaces coexist:
///   - `dynamic_*` methods — backend-agnostic. Default impls
///     forward to the semispace shape; the page heap will override
///     them with its own logic.
///   - `young_*` / `old_*` methods — semispace-shaped. Required
///     today (the semispace impl uses them), `#[deprecated]` so
///     new code calls `dynamic_*` instead. Sub-phase 12 removes
///     them.
pub trait HeapBackend: Send + 'static {
    // -- Capacity / usage queries (constant-time, `(gc-stats)` path) ---

    #[deprecated(note = "use dynamic_used_bytes; sub-phase 12 removes this")]
    fn young_used_bytes(&self) -> usize;
    #[deprecated(note = "use dynamic_used_bytes; sub-phase 12 removes this")]
    fn old_used_bytes(&self) -> usize;
    fn used_bytes(&self) -> usize;
    #[deprecated(note = "use dynamic_capacity_bytes; sub-phase 12 removes this")]
    fn young_capacity_bytes(&self) -> usize;
    #[deprecated(note = "use dynamic_capacity_bytes; sub-phase 12 removes this")]
    fn old_capacity_bytes(&self) -> usize;
    #[deprecated(note = "semispace-specific; sub-phase 12 removes this")]
    fn old_capacity_bytes_per_semi(&self) -> usize;

    // -- Mutator-facing fast-path setup --------------------------------

    /// Reserve a TLAB slab in the nursery. Returns `None` if young
    /// is exhausted (which the coordinator handles by triggering a
    /// minor GC and retrying).
    fn young_try_alloc_slab(&mut self, cells: usize) -> Option<NonNull<u64>>;
    /// Base pointer of young's cell storage. Mutators cache this at
    /// registration so the alloc fast path can compute cell indices
    /// without taking the heap mutex.
    fn young_base_ptr(&self) -> *const u64;
    /// Lock-free handle to the start-bit bitmap. Used by the alloc
    /// fast path to mark cell starts as objects are bumped in.
    fn young_starts_handle(&self) -> StartBits;

    // -- Range tests for tags / conservative scan ---------------------

    #[deprecated(note = "use dynamic_contains; sub-phase 12 removes this")]
    fn young_contains(&self, ptr: *const u8) -> bool;
    #[deprecated(note = "use dynamic_contains; sub-phase 12 removes this")]
    fn old_contains(&self, ptr: *const u8) -> bool;

    // -- Card-marking machinery (Inter-generational pointer tracking) -

    #[deprecated(note = "use dynamic_cards; sub-phase 12 removes this")]
    fn old_cards(&self) -> &Arc<CardTable>;
    #[deprecated(note = "use dynamic_base_ptr; sub-phase 12 removes this")]
    fn old_live_base_ptr(&self) -> *const u8;

    // -- Backend-agnostic `dynamic_*` surface (sub-phase 6.5) ----------
    //
    // Default impls forward to the semispace shape. PageHeap will
    // override with backend-native logic when it lands at sub-phase
    // 11.

    /// Total bytes used across the entire dynamic heap (all
    /// generations combined). For semispace = young + old; for the
    /// page heap = sum of `words_used * 8` across non-Free pages.
    fn dynamic_used_bytes(&self) -> usize {
        #[allow(deprecated)]
        { self.young_used_bytes() + self.old_used_bytes() }
    }

    /// Total dynamic-heap address-space capacity in bytes. For
    /// semispace = young_cap + old_cap; for the page heap = the
    /// whole reservation.
    fn dynamic_capacity_bytes(&self) -> usize {
        #[allow(deprecated)]
        { self.young_capacity_bytes() + self.old_capacity_bytes() }
    }

    /// Base pointer of the dynamic-heap region — anchor for card-
    /// table arithmetic. For semispace this is `old.live`'s base
    /// (today's barrier anchor); for the page heap this becomes
    /// the reservation base.
    fn dynamic_base_ptr(&self) -> *const u8 {
        #[allow(deprecated)]
        { self.old_live_base_ptr() }
    }

    /// Card table covering the dynamic heap. For semispace this is
    /// `old.live`'s; for the page heap (sub-phase 9) this becomes
    /// a single table covering the whole reservation.
    fn dynamic_cards(&self) -> &Arc<CardTable> {
        #[allow(deprecated)]
        { self.old_cards() }
    }

    /// Whether `ptr` falls within the dynamic heap. For semispace
    /// = `young_contains(ptr) || old_contains(ptr)`; for the page
    /// heap = single reservation-range check.
    fn dynamic_contains(&self, ptr: *const u8) -> bool {
        #[allow(deprecated)]
        { self.young_contains(ptr) || self.old_contains(ptr) }
    }

    // -- GC entry points ----------------------------------------------

    /// Run a minor GC. `pin_stack_ranges` is a slice of
    /// `(rsp, stack_hi)` pairs — one per mutator stack window the
    /// conservative pinner should scan. `static_cards` /
    /// `static_base` / `static_cells` give the GC the static-area
    /// dirty-card slice to scan for static→young pointers.
    ///
    /// `visit_roots` is invoked once and should walk every
    /// per-mutator explicit-root Word.
    fn collect_minor_with_static(
        &mut self,
        static_cards: &CardTable,
        static_base: *mut u64,
        static_cells: usize,
        pin_stack_ranges: &[(usize, usize)],
        visit_roots: &mut dyn FnMut(&mut RootScanner<'_, '_>),
    );

    /// Per-cycle pin summary `(n_objects, n_cells)` from the last
    /// `collect_minor_with_static`. Surfaced via `(gc-stats)` as
    /// `:objects-pinned-total` / `:pinned-residual-cells`.
    fn last_pin_summary(&self) -> (usize, usize);
}
