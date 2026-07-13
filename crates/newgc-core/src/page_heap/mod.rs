//! Page-based dynamic-space heap. Phase 3 of `docs/GC_DESIGN.md`.
//!
//! Under construction. Selected at runtime via
//! `NCL_HEAP_BACKEND=page-heap`. As of this commit only sub-phase 2
//! (raw page reservation + commit) is built; sub-phase 3+ add page
//! descriptors, object allocation, mark, evacuation, and the rest.
//! Selecting the page-heap backend today still panics from
//! `GcCoordinator::new_with_backend` — wiring lands in sub-phase 11.
//!
//! Layout target:
//!
//! ```text
//!   reservation  ┌────────────────────────────────────────────┐
//!                │  page 0   page 1   page 2   …   page N-1   │
//!                │  64 KB    64 KB    64 KB        64 KB      │
//!                └────────────────────────────────────────────┘
//!                ↑
//!                base_ptr()
//! ```
//!
//! Total reservation: 1 GB by default. Address space is reserved
//! lazily; physical pages are committed only when a page is first
//! used. Decommitting a page returns its physical backing to the
//! OS without releasing the address range — the page can be
//! re-committed later at the same address. This is the kernel
//! primitive that lets a moving GC reclaim memory without
//! fragmenting the reservation.

pub mod alloc;
pub mod coordinator_api;
pub mod cycle;
pub mod evac;
pub mod mark;
pub mod mutator;
pub mod page_desc;
pub mod pin;
pub mod scanner;
pub mod shared;
pub mod space;

pub use alloc::{AllocRegion, PageStartBits};
pub use cycle::{CollectResult, FullCollectResult, G0_PROMOTION_THRESHOLD, G1_PROMOTION_THRESHOLD};
pub use evac::{EvacResult, PageEvacuator};
pub use mutator::{GcCoordinator, Mutator, MutatorId};
pub use page_desc::{Generation, PageDesc, PageKind};
pub use pin::PinHandle;
pub use shared::SharedHeap;
pub use space::{PageHeap, PAGE_SIZE_BYTES, PAGE_SIZE_CELLS};
