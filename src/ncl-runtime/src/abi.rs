//! C ABI surface for JIT'd Lisp code.
//!
//! Compiled Lisp code calls back into the runtime through these
//! `extern "C"` functions. They take and return `u64` (raw `Word`
//! bits) so the function signatures match what LLVM IR can express
//! without any extra translation layer.
//!
//! All functions in this module are entry points from JIT'd code
//! and live for the process lifetime — never reorder, never
//! rename. The compiler's emitter looks them up by name.

use crate::mutator::MutatorState;
use crate::word::{Tag, Word};

/// Allocate a cons cell. JIT'd `(cons car cdr)` lowers to a call
/// here.
///
/// SAFETY: `mutator` must be a valid pointer to a `MutatorState`
/// owned by the calling thread. The Lisp-thread discipline ensures
/// this: the driver passes the mutator pointer when invoking the
/// entry function, and the entry function threads it through every
/// call.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_alloc_cons(mutator: *mut MutatorState, car: u64, cdr: u64) -> u64 {
    let m = unsafe { &mut *mutator };
    let car_w = Word::from_raw(car);
    let cdr_w = Word::from_raw(cdr);
    m.alloc_cons(car_w, cdr_w).raw()
}

/// Read the car field of a cons cell. JIT'd `(car x)` lowers to
/// untag-and-load directly without calling here, but this is also
/// exposed for use from FFI / debugger / tests.
///
/// SAFETY: `cons` must be a valid Cons-tagged `Word`.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_car(cons: u64) -> u64 {
    let w = Word::from_raw(cons);
    let p = w.as_ptr::<u64>(Tag::Cons).expect("ncl_car called on non-cons");
    unsafe { *p }
}

/// Read the cdr field of a cons cell. As `ncl_car` but at offset 1.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_cdr(cons: u64) -> u64 {
    let w = Word::from_raw(cons);
    let p = w.as_ptr::<u64>(Tag::Cons).expect("ncl_cdr called on non-cons");
    unsafe { *p.add(1) }
}

/// Load a global symbol's value cell with acquire semantics.
/// JIT'd code calls this when it reads a bare symbol that wasn't
/// resolved to a local at compile time. Panics on unbound — the
/// condition system that should turn this into a proper Lisp
/// error lands later.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_load_value(sym_word: u64) -> u64 {
    let sym = Word::from_raw(sym_word);
    let value = crate::gc_symbol::value_acquire(sym);
    if value.is_unbound() {
        let name = crate::sym_names::lookup(sym_word)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{sym_word:#x}"));
        panic!("unbound variable: {name}");
    }
    value.raw()
}

/// Store a value into a global symbol's value cell with release
/// semantics. JIT'd `(setq …)` and `(defparameter …)` lower to a
/// call here. Returns the stored value (CL's `setq` returns the
/// last value assigned).
#[unsafe(no_mangle)]
pub extern "C" fn ncl_store_value(
    mutator: *mut crate::mutator::MutatorState,
    sym_word: u64,
    new_value: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let sym = Word::from_raw(sym_word);
    m.set_symbol_value(sym, Word::from_raw(new_value));
    new_value
}

/// Dispatch a function call through a Symbol's function cell.
/// JIT'd `(name arg1 arg2 ...)` lowers to a call here.
///
/// Loads the symbol's function cell with acquire semantics, follows
/// the Function pointer to get the code address and the closure
/// env, and calls
///   `code(mutator, env, args, n_args)`.
/// Panics if the cell is unbound.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_call(
    mutator: *mut crate::mutator::MutatorState,
    sym_word: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    let sym = Word::from_raw(sym_word);
    let fn_value = crate::gc_symbol::function_acquire(sym);
    if fn_value.is_unbound() {
        let name = crate::sym_names::lookup(sym_word)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{sym_word:#x}"));
        panic!("ncl_call: unbound function: {name}");
    }
    if fn_value.tag() != Tag::Function {
        panic!("ncl_call: function cell is not a Function: {fn_value:?}");
    }
    let env = crate::gc_function::env(fn_value);
    let code = crate::gc_function::code_ptr(fn_value);
    let f: crate::gc_function::LispCodeFn =
        unsafe { std::mem::transmute(code) };
    unsafe { f(mutator, env.raw(), args, n_args) }
}

/// Call a Function value directly (no symbol lookup). Used by
/// `funcall` and by code that has a first-class Function value.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_funcall(
    mutator: *mut crate::mutator::MutatorState,
    fn_word: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    let f_word = Word::from_raw(fn_word);
    if f_word.tag() != Tag::Function {
        panic!("funcall: not a function: {f_word:?}");
    }
    let env = crate::gc_function::env(f_word);
    let code = crate::gc_function::code_ptr(f_word);
    let f: crate::gc_function::LispCodeFn =
        unsafe { std::mem::transmute(code) };
    unsafe { f(mutator, env.raw(), args, n_args) }
}

/// Allocate a closure: a Function in static with the given code,
/// arity, and a freshly allocated env Vector containing the
/// captured values. JIT'd `(lambda …)` evaluates each capture
/// expression in outer scope, packs the values into a stack
/// buffer, and calls here to materialise the function value.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_make_closure(
    mutator: *mut crate::mutator::MutatorState,
    code_ptr: u64,
    arity: u64,
    captures: *const u64,
    n_captures: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let env_word = if n_captures == 0 {
        Word::NIL
    } else {
        let vec = m.alloc_vector(n_captures as u32);
        // Vec layout: cell 0 is header, cells 1..=n are payload.
        let p = vec
            .as_mut_ptr::<u64>(Tag::Vector)
            .expect("vector ptr");
        unsafe {
            for i in 0..n_captures {
                *p.add(1 + i as usize) = *captures.add(i as usize);
            }
        }
        vec
    };
    // Lambdas live in static for now (with the env-pointer card
    // marked if env is in young — see below).
    let coord = m.coord();
    let fn_word = crate::gc_function::alloc_function_in_static(
        coord.static_area(),
        code_ptr as usize,
        arity as u32,
        Word::NIL, // anonymous lambdas have no name
        env_word,
    )
    .expect("static area exhausted in lambda creation");
    // If env is a young-heap Vector, the Function in static now
    // contains a static→young pointer. Mark the card.
    if !env_word.is_nil() {
        let env_cell_addr = unsafe {
            (fn_word.as_ptr::<u8>(Tag::Function).unwrap() as *const u8)
                .add(crate::gc_function::ENV_OFFSET * 8)
        };
        coord.mark_card(env_cell_addr);
    }
    fn_word.raw()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutator::{GcConfig, GcCoordinator};

    fn small_config() -> GcConfig {
        GcConfig {
            young_bytes: 16 * 1024,
            old_bytes: 16 * 1024,
            static_bytes: 8 * 1024,
            tlab_cells: 64,
        }
    }

    #[test]
    fn alloc_cons_via_abi_returns_cons_tagged_word() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let result = ncl_alloc_cons(&mut m as *mut _, Word::fixnum(1).raw(), Word::fixnum(2).raw());
        let w = Word::from_raw(result);
        assert!(w.is_cons());
        assert_eq!(ncl_car(result), Word::fixnum(1).raw());
        assert_eq!(ncl_cdr(result), Word::fixnum(2).raw());
    }
}
