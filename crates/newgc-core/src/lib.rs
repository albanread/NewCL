//! NewGC — page-based mark-evacuate generational garbage collector.
//!
//! Phase 1: a near-verbatim lift of NewCormanLisp's `page_heap` module
//! plus the shared `word` / `heap_common` types. Builds and tests
//! standalone. Coordinator-API integration (NCL's `coordinator_api.rs`)
//! is deliberately omitted — that file is the language-runtime
//! bind point and lives downstream of this crate.
//!
//! Phase 2 will extract `HeapWord` and `ObjectShape` traits so the
//! GC engine can serve more than one language runtime without
//! re-importing this code wholesale.

pub mod crash;
pub mod heap_common;
pub mod lisp_layout;
pub mod page_heap;
pub mod tiny_layout;
pub mod traits;
pub mod word;

pub use lisp_layout::LispLayout;
pub use page_heap::space::GcStats;
pub use tiny_layout::TinyLayout;
pub use traits::{HeapLayout, ObjectLayout, PointerKind, WordKind};

pub use heap_common::{
    CardTable, GcBit, HeapHeader, HeapType, MAX_OBJECT_CELLS, StartBits,
    CARD_SIZE_BYTES, CARD_SIZE_CELLS,
};
pub use word::{Tag, Word, FIXNUM_MAX, FIXNUM_MIN, PAYLOAD_MASK, TAG_BITS, TAG_MASK};
pub use page_heap::{
    AllocRegion, CollectResult, EvacResult, FullCollectResult, GcCoordinator, Generation,
    Mutator, MutatorId, PageDesc, PageEvacuator, PageHeap, PageKind, PageStartBits, PinHandle,
    SharedHeap, G0_PROMOTION_THRESHOLD, G1_PROMOTION_THRESHOLD, PAGE_SIZE_BYTES, PAGE_SIZE_CELLS,
};
