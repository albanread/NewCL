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

pub mod symbol;
pub mod universe;
pub mod value;
pub mod word;

pub use symbol::{Package, Symbol, Visibility};
pub use universe::{pkg, universe, Universe};
pub use value::{Cons, FfiBlock, Value};
pub use word::{Tag, Word, FIXNUM_MAX, FIXNUM_MIN, PAYLOAD_MASK, TAG_BITS, TAG_MASK};
