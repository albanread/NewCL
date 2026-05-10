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
pub mod file_sys;
pub mod format;
pub mod gc_function;
pub mod gc_string;
pub mod gc_symbol;
pub mod heap;
pub mod mutator;
pub mod printer;
pub mod stack_map;
pub mod static_area;
pub mod sym_names;
pub mod symbol;
pub mod universe;
pub mod value;
pub mod word;

pub use abi::{
    append_shim, close_stream_shim, delete_file_shim, error_shim, file_exists_shim,
    file_length_shim, file_position_shim, format_shim, handler_case_shim,
    ncl_alloc_cons, ncl_apply, ncl_build_rest_list, ncl_call, ncl_car, ncl_cdr,
    ncl_equal, ncl_funcall, ncl_length, ncl_load_function, ncl_load_value,
    ncl_make_closure, ncl_set_car, ncl_set_cdr, ncl_store_value, ncl_string_char,
    ncl_string_eq, ncl_string_set, open_append_file_shim, open_input_file_shim,
    open_output_file_shim, read_char_from_shim, read_line_shim, write_string_to_shim,
    NclCondition,
};
pub use printer::{format_word, format_word_aesthetic};

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
