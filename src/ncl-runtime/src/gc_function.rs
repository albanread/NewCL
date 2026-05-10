//! GC-managed Function layout.
//!
//! A `Function` is the heap object a Symbol's function cell points
//! at when the symbol has a defun'd binding. Allocated in the static
//! area (immortal — defun'd functions don't die in v1; later, the
//! loader's retirement-with-quiescent-epoch story handles the case
//! where a redefinition orphans a Function whose old code is still
//! on a live stack).
//!
//! Layout (4 cells = 32 bytes):
//!
//! ```text
//!   cell 0   HeapHeader      type=Function, length=3
//!   cell 1   code_ptr        raw machine-code pointer (NOT a Word)
//!   cell 2   arity           required-argument count, as u64
//!   cell 3   name            Word — the Symbol this was bound to
//! ```
//!
//! The code pointer is the address of JIT'd native code with
//! signature `fn(*mut MutatorState, *const u64 /* args */, u64 /* n */) -> u64`.
//! Calling a function dispatches through this signature so every
//! call site has the same shape regardless of arity.

use std::ptr::NonNull;

use crate::heap::{HeapHeader, HeapType};
use crate::static_area::StaticArea;
use crate::word::{Tag, Word};

pub const CODE_PTR_OFFSET: usize = 1;
pub const ARITY_OFFSET: usize = 2;
pub const NAME_OFFSET: usize = 3;
pub const FUNCTION_PAYLOAD_CELLS: u32 = 3;

/// Allocate a fresh Function in the static area. The code_ptr is a
/// raw machine-code address; the JIT keeps the underlying engine
/// alive so this pointer stays valid for the process lifetime.
pub fn alloc_function_in_static(
    static_area: &StaticArea,
    code_ptr: usize,
    arity: u32,
    name: Word,
) -> Option<Word> {
    let header_ptr =
        static_area.try_alloc_with_header(HeapType::Function, FUNCTION_PAYLOAD_CELLS)?;
    let p = header_ptr.as_ptr() as *mut u64;
    unsafe {
        *p.add(CODE_PTR_OFFSET) = code_ptr as u64;
        *p.add(ARITY_OFFSET) = arity as u64;
        *p.add(NAME_OFFSET) = name.raw();
    }
    Some(Word::from_ptr(p as *const u8, Tag::Function))
}

/// Read the code pointer from a Function-tagged Word.
pub fn code_ptr(fn_word: Word) -> usize {
    let p = fn_ptr(fn_word);
    unsafe { *p.add(CODE_PTR_OFFSET) as usize }
}

pub fn arity(fn_word: Word) -> u32 {
    let p = fn_ptr(fn_word);
    unsafe { *p.add(ARITY_OFFSET) as u32 }
}

pub fn name(fn_word: Word) -> Word {
    let p = fn_ptr(fn_word);
    Word::from_raw(unsafe { *p.add(NAME_OFFSET) })
}

fn fn_ptr(fn_word: Word) -> *mut u64 {
    debug_assert_eq!(fn_word.tag(), Tag::Function);
    fn_word.as_mut_ptr::<u64>(Tag::Function).expect("function ptr")
}

/// Pointer to the code-pointer cell — used by the GC to scan if
/// we ever start tracking JIT code via the GC.
pub fn code_ptr_cell_addr(fn_word: Word) -> *const u8 {
    let p = fn_ptr(fn_word);
    unsafe { p.add(CODE_PTR_OFFSET) as *const u8 }
}

/// Header read for tests / diagnostics.
pub fn header(fn_word: Word) -> HeapHeader {
    let p = fn_ptr(fn_word);
    HeapHeader::from_raw(unsafe { *p })
}

/// SAFETY: `code` must be a function pointer compatible with
/// `extern "C" fn(*mut MutatorState, *const u64, u64) -> u64`,
/// `args` must hold `n_args` valid `Word`s. Used by the dispatch
/// helper in `abi.rs`.
pub type LispCodeFn = unsafe extern "C" fn(
    mutator: *mut crate::mutator::MutatorState,
    args: *const u64,
    n_args: u64,
) -> u64;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn fresh_static() -> Arc<StaticArea> {
        Arc::new(StaticArea::new(64 * 1024))
    }

    #[test]
    fn function_layout() {
        let s = fresh_static();
        let f = alloc_function_in_static(&s, 0xdeadbeef, 2, Word::NIL).unwrap();
        assert_eq!(f.tag(), Tag::Function);
        assert_eq!(header(f).ty(), HeapType::Function);
        assert_eq!(header(f).length_cells(), FUNCTION_PAYLOAD_CELLS);
        assert_eq!(code_ptr(f), 0xdeadbeef);
        assert_eq!(arity(f), 2);
        assert!(name(f).is_nil());
    }
}
