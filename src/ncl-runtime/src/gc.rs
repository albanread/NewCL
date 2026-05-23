//! GC backend — page-heap.
//!
//! The semispace GC has been removed. Page-heap is the one and only
//! backend. `gc::Heap` and `gc::RootScanner` alias the page-heap
//! concrete types; all callers compile against these names unchanged.

#[cfg(not(feature = "gc-page-heap"))]
compile_error!("ncl-runtime: the gc-page-heap feature must be enabled.");

/// Display name for the active backend. Surfaces in `(gc-stats)` as
/// `:heap-backend`.
pub const ACTIVE_BACKEND_NAME: &str = "page-heap";

/// The concrete heap type for this build.
pub use crate::page_heap::PageHeap as Heap;

/// The active backend's root scanner.
pub use crate::page_heap::scanner::RootScanner;
