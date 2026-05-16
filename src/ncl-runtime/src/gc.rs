//! GC backend selection — build-time, not runtime.
//!
//! Exactly one of the `gc-semispace` / `gc-page-heap` Cargo features
//! must be enabled (`gc-semispace` is the default). This module
//! re-exports the chosen concrete heap type as `gc::Heap`. Code that
//! holds the GC by-value writes `gc::Heap`; the compiler picks the
//! right implementation per build.
//!
//! ## Rationale
//!
//! An earlier attempt (sub-phase 1 of `docs/GC_DESIGN.md`) routed
//! the coordinator through a `Box<dyn HeapBackend>` so both heaps
//! could coexist at runtime via `NCL_HEAP_BACKEND`. That trait
//! ended up with a tail of `#[deprecated]` semispace-specific
//! methods plus `dynamic_*` papered-over abstractions — fighting
//! the MANIFESTO's "no speculative layers" rule. Switching to
//! build-time selection lets each backend expose its own native
//! API; the cost is "to switch GCs you rebuild," which for a Lisp
//! IDE (rather than a long-running server) is fine.
//!
//! ## Switching
//!
//! ```bash
//! cargo build                                           # gc-semispace (default)
//! cargo build --no-default-features --features gc-page-heap
//! cargo test  --no-default-features --features gc-page-heap
//! ```
//!
//! CI runs both feature combinations so neither rots.

#[cfg(all(feature = "gc-semispace", feature = "gc-page-heap"))]
compile_error!(
    "ncl-runtime: features `gc-semispace` and `gc-page-heap` are mutually \
     exclusive. Pass `--no-default-features` when enabling `gc-page-heap`."
);

#[cfg(not(any(feature = "gc-semispace", feature = "gc-page-heap")))]
compile_error!(
    "ncl-runtime: no GC backend selected. Enable either `gc-semispace` \
     (default) or `gc-page-heap`."
);

/// Display name for the active backend. Surfaces in `(gc-stats)` as
/// `:heap-backend`.
#[cfg(feature = "gc-semispace")]
pub const ACTIVE_BACKEND_NAME: &str = "semispace";
#[cfg(feature = "gc-page-heap")]
pub const ACTIVE_BACKEND_NAME: &str = "page-heap";

/// The concrete heap type for this build. Both backends expose the
/// inherent methods that `GcCoordinator` calls — no trait required.
#[cfg(feature = "gc-semispace")]
pub use crate::heap::Heap;
#[cfg(feature = "gc-page-heap")]
pub use crate::page_heap::PageHeap as Heap;

/// The active backend's root scanner. The visitor-pattern surface
/// matches across both backends: `visit(&mut Word)` reads the slot,
/// possibly evacuates the referenced object, and rewrites the slot
/// with the post-evac Word.
#[cfg(feature = "gc-semispace")]
pub use crate::heap::RootScanner;
#[cfg(feature = "gc-page-heap")]
pub use crate::page_heap::scanner::RootScanner;
