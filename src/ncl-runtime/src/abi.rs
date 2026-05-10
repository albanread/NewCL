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

/// Length of a String or proper list (in codepoints / cons cells).
/// JIT'd `(length …)` lowers to a call here. Polymorphic on tag.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_length(w: u64) -> u64 {
    let word = Word::from_raw(w);
    if word.is_nil() {
        return Word::fixnum(0).raw();
    }
    match word.tag() {
        Tag::String => Word::fixnum(crate::gc_string::char_count(word) as i64).raw(),
        Tag::Cons => {
            // Walk the cons spine.
            let mut cur = word;
            let mut count: i64 = 0;
            while !cur.is_nil() {
                if cur.tag() != Tag::Cons {
                    panic!("length: improper list");
                }
                let p = cur.as_ptr::<u64>(Tag::Cons).expect("cons");
                count += 1;
                cur = Word::from_raw(unsafe { *p.add(1) });
            }
            Word::fixnum(count).raw()
        }
        _ => panic!("length: not a sequence: {word:?}"),
    }
}

/// Structural equality. Recurses through cons cells, treats
/// strings codepoint-by-codepoint, falls back to eq for atoms.
/// JIT'd `(equal a b)` lowers to a call here.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_equal(a: u64, b: u64) -> u64 {
    if equal_recursive(Word::from_raw(a), Word::from_raw(b)) {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

fn equal_recursive(a: Word, b: Word) -> bool {
    // Fast path: same Word bits = trivially equal (covers eq case
    // for symbols, fixnums, immediates, and identical pointers).
    if a.raw() == b.raw() {
        return true;
    }
    match (a.tag(), b.tag()) {
        (Tag::Cons, Tag::Cons) => {
            let pa = a.as_ptr::<u64>(Tag::Cons).expect("cons");
            let pb = b.as_ptr::<u64>(Tag::Cons).expect("cons");
            let car_a = Word::from_raw(unsafe { *pa });
            let car_b = Word::from_raw(unsafe { *pb });
            if !equal_recursive(car_a, car_b) {
                return false;
            }
            let cdr_a = Word::from_raw(unsafe { *pa.add(1) });
            let cdr_b = Word::from_raw(unsafe { *pb.add(1) });
            equal_recursive(cdr_a, cdr_b)
        }
        (Tag::String, Tag::String) => crate::gc_string::string_eq(a, b),
        // For any other combination — different tags, or non-
        // structured atom types where we already know the bits
        // differ — they're not equal.
        _ => false,
    }
}

/// String equality. Both args must be Tag::String. Returns Word::T
/// or Word::NIL.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_string_eq(a: u64, b: u64) -> u64 {
    let wa = Word::from_raw(a);
    let wb = Word::from_raw(b);
    if wa.tag() != Tag::String || wb.tag() != Tag::String {
        panic!("string=: arguments must be strings");
    }
    if crate::gc_string::string_eq(wa, wb) {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

/// Read the i-th codepoint of a string as a character Word.
/// `(char s i)` and `(aref s i)` both lower here for strings.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_string_char(s: u64, idx: u64) -> u64 {
    let ws = Word::from_raw(s);
    if ws.tag() != Tag::String {
        panic!("string-char: argument must be a string");
    }
    let n = crate::gc_string::char_count(ws);
    let i = idx as usize;
    if i >= n {
        panic!("string-char: index {i} out of bounds for length {n}");
    }
    let cp = crate::gc_string::codepoint_at(ws, i);
    let c = char::from_u32(cp).expect("invalid codepoint in string");
    Word::char(c).raw()
}

/// Load a Symbol's function cell with acquire semantics. JIT'd
/// `#'name` (i.e. `(function name)`) lowers to a call here. Used
/// to pass defun'd functions as first-class values to higher-order
/// code. Panics on unbound — distinct from unbound *variable*
/// because here we know it's a function lookup.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_load_function(sym_word: u64) -> u64 {
    let sym = Word::from_raw(sym_word);
    let fn_value = crate::gc_symbol::function_acquire(sym);
    if fn_value.is_unbound() {
        let name = crate::sym_names::lookup(sym_word)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{sym_word:#x}"));
        panic!("undefined function: {name}");
    }
    fn_value.raw()
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

/// Mutate the car field of a cons cell. JIT'd `(setf (car x) v)`
/// lowers to a call here. Returns the new value (per setf
/// semantics — the *cell* gets the new value, but the form
/// evaluates to the value, not the cons).
///
/// The card containing the modified cell is marked so the GC's
/// inter-generational scan picks up any old→young pointer this
/// store may have created.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_set_car(
    mutator: *mut crate::mutator::MutatorState,
    cons: u64,
    new_value: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let w = Word::from_raw(cons);
    let p = w
        .as_mut_ptr::<u64>(Tag::Cons)
        .expect("ncl_set_car called on non-cons");
    unsafe { *p = new_value };
    m.mark_card(p as *const u8);
    new_value
}

/// Mutate the cdr field of a cons cell. As `ncl_set_car` but at
/// offset 1.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_set_cdr(
    mutator: *mut crate::mutator::MutatorState,
    cons: u64,
    new_value: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let w = Word::from_raw(cons);
    let p = w
        .as_mut_ptr::<u64>(Tag::Cons)
        .expect("ncl_set_cdr called on non-cons");
    let slot = unsafe { p.add(1) };
    unsafe { *slot = new_value };
    m.mark_card(slot as *const u8);
    new_value
}

/// `(apply fn prefix-arg-1 ... prefix-arg-N tail-list)` —
/// call `fn` with the prefix args followed by the spread elements
/// of `tail-list`. The compiler emits a call here with the prefix
/// already packed into a stack buffer; we walk the list to count
/// it, allocate one combined args buffer, copy the prefix in,
/// splat the list, and dispatch through `ncl_funcall`.
///
/// Empty prefix (just `(apply f lst)`) is supported by passing
/// `n_prefix = 0` and a null `prefix` pointer.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_apply(
    mutator: *mut crate::mutator::MutatorState,
    fn_word: u64,
    prefix: *const u64,
    n_prefix: u64,
    tail_list: u64,
) -> u64 {
    let f_word = Word::from_raw(fn_word);
    if f_word.tag() != Tag::Function {
        panic!("apply: not a function: {f_word:?}");
    }

    // Walk tail_list to count its length, then build a combined
    // args buffer of n_prefix + tail_len Words.
    let mut tail_len: usize = 0;
    let mut cur = Word::from_raw(tail_list);
    while !cur.is_nil() {
        if cur.tag() != Tag::Cons {
            panic!("apply: tail argument must be a proper list, got {cur:?}");
        }
        tail_len += 1;
        let p = cur.as_ptr::<u64>(Tag::Cons).expect("cons");
        cur = Word::from_raw(unsafe { *p.add(1) });
    }

    let total = n_prefix as usize + tail_len;
    let mut buf: Vec<u64> = Vec::with_capacity(total);
    for i in 0..n_prefix {
        buf.push(unsafe { *prefix.add(i as usize) });
    }
    let mut cur = Word::from_raw(tail_list);
    while !cur.is_nil() {
        let p = cur.as_ptr::<u64>(Tag::Cons).expect("cons");
        buf.push(unsafe { *p });
        cur = Word::from_raw(unsafe { *p.add(1) });
    }

    let env = crate::gc_function::env(f_word);
    let code = crate::gc_function::code_ptr(f_word);
    let f: crate::gc_function::LispCodeFn =
        unsafe { std::mem::transmute(code) };
    unsafe { f(mutator, env.raw(), buf.as_ptr(), total as u64) }
}

/// Build a freshly-allocated list of `args[start..n_args]` in
/// order. The variadic function-entry prologue calls this to bind
/// `&rest` parameters. If `n_args <= start`, returns NIL.
///
/// Allocation goes through the mutator's TLAB, so the resulting
/// list lives in the young heap and participates in normal GC.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_build_rest_list(
    mutator: *mut crate::mutator::MutatorState,
    args: *const u64,
    start: u64,
    n_args: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let mut result = Word::NIL;
    if n_args > start {
        // Cons right-to-left so the list ends up in order.
        let mut i = n_args;
        while i > start {
            i -= 1;
            let arg_w = Word::from_raw(unsafe { *args.add(i as usize) });
            result = m.alloc_cons(arg_w, result);
        }
    }
    result.raw()
}

/// Mutate the i-th codepoint of a string. JIT'd
/// `(setf (aref s i) c)` and `(setf (char s i) c)` lower here.
/// Returns the character word that was stored.
///
/// No write barrier: strings hold scalar codepoints, so a string
/// store can never create an old→young pointer.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_string_set(s: u64, idx: u64, ch: u64) -> u64 {
    let ws = Word::from_raw(s);
    if ws.tag() != Tag::String {
        panic!("(setf (aref string ...)): argument is not a string: {ws:?}");
    }
    let n = crate::gc_string::char_count(ws);
    let i = idx as usize;
    if i >= n {
        panic!("(setf (aref s {i}) ...): index out of bounds for length {n}");
    }
    let cw = Word::from_raw(ch);
    let cp = cw
        .as_char()
        .unwrap_or_else(|| panic!("(setf (aref string ...)): value is not a character: {cw:?}"))
        as u32;
    crate::gc_string::set_char_at(ws, i, cp);
    ch
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
