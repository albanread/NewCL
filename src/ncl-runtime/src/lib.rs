//! Resident runtime: value representation, symbols and packages,
//! features, GC (later), FFI (later), and the iGui shim (later).
//!
//! OS-specific code lives only inside the iGui submodule (when it
//! lands). The rest is OS-independent Rust — see MANIFESTO.md,
//! "Design values" #2.
//!
//! For function-redefinition semantics see MANIFESTO.md, "Note:
//! function redefinition and dispatch": global function cells are
//! atomically swapped, retirement machinery is reserved for orphaned
//! compiled code only.

pub mod abi;
pub mod gc_symbol;
pub mod heap;
pub mod mutator;
pub mod printer;
pub mod stack_map;
pub mod static_area;
pub mod symbol;
pub mod universe;
pub mod value;
pub mod word;

pub use abi::{ncl_alloc_cons, ncl_car, ncl_cdr};
pub use printer::format_word;

pub use heap::{
    CardTable, GcBit, Heap, HeapHeader, HeapType, CARD_SIZE_BYTES, CARD_SIZE_CELLS,
    MAX_OBJECT_CELLS,
};
pub use mutator::{GcConfig, GcCoordinator, MutatorHandle, MutatorState};
pub use stack_map::{LiveSlot, ParkedFrame, StackMap, StackMapEntry, walk_parked_frame};
pub use static_area::StaticArea;
pub use symbol::{Package, Symbol, Visibility};
pub use universe::{pkg, universe, Universe};
pub use value::{Cons, FfiBlock, Value};
pub use word::{Tag, Word, FIXNUM_MAX, FIXNUM_MIN, PAYLOAD_MASK, TAG_BITS, TAG_MASK};
