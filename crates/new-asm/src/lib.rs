//! Shared ASM-procedure support for the NewLang family.
//!
//! Provides types and string-building helpers consumed by each language's
//! `*-ir` and `*-llvm` crates. No inkwell dependency — the actual LLVM
//! calls stay in the language crate.

pub mod substitute;
pub mod types;

pub use substitute::build_module_asm_string;
pub use types::{AsmParam, AsmProc, AsmRetType, AsmType};
