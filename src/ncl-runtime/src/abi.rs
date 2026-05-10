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
pub extern "C-unwind" fn ncl_call(
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
pub extern "C-unwind" fn ncl_funcall(
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

// ---- Conditions: error / handler-case --------------------------------------
//
// The control-transfer mechanism is intentionally NOT panic-based.
// Rust panics work cleanly between Rust frames, but propagating
// them through LLVM-MCJIT-emitted Lisp frames on Windows requires
// SEH .pdata tables that MCJIT doesn't reliably register — the
// process aborts with code 0xe06d7363 ("C++ exception not caught")
// when the unwinder hits a JIT frame.
//
// Instead we use a thread-local condition slot + handler-depth
// counter:
//
//   - `error` checks the depth. If zero (no handler installed),
//     it prints the message and aborts. If non-zero, it stores
//     the condition in the slot and returns NIL — allowing the
//     JIT frames above to drain naturally up the stack.
//   - `handler-case` increments the depth, clears the slot, runs
//     the body, decrements the depth, then inspects the slot. If
//     set, it dispatches to the handler.
//
// Limitation: between `error` and the matching handler-case, the
// JIT code unwinds via *normal returns*. If those frames perform
// trapping operations on the body's return value (e.g.
// `(car (error ...))` would call `car` on the NIL return) the
// trap fires before the handler sees the condition. In practice
// `error` is overwhelmingly the last thing in its branch, so this
// is rarely observable. A proper unwinding mechanism (ORC JIT or
// custom SEH registration) lands later.

use std::cell::Cell;

thread_local! {
    /// The condition value set by `error`, cleared by
    /// `handler-case` on entry, read by `handler-case` on exit.
    static CONDITION_SLOT: Cell<u64> = const { Cell::new(0) };
    /// Set if a condition has been signalled and not yet handled.
    /// Distinct from CONDITION_SLOT==0 because nil is a legal
    /// condition value.
    static CONDITION_PENDING: Cell<bool> = const { Cell::new(false) };
    /// Nesting depth of active `handler-case` invocations.
    /// `error` aborts the process when this is zero.
    static HANDLER_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Convenience wrapper kept around for FFI symmetry — exposes the
/// condition payload type to callers that want to manipulate it
/// programmatically. Currently empty since the pending-condition
/// state lives in thread-locals.
#[derive(Debug, Clone)]
pub struct NclCondition {
    pub value: u64,
}

/// `(error condition-or-message)` — signal a condition. With a
/// matching `handler-case` on the stack, returns NIL after stashing
/// the condition; the unwind happens through normal stack returns.
/// With no handler active, prints the message and aborts the
/// process — matches CL's "unhandled condition enters the
/// debugger; here we just abort" simplification.
pub extern "C-unwind" fn error_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    let cond = if n_args < 1 {
        Word::NIL.raw()
    } else {
        unsafe { *args }
    };
    if HANDLER_DEPTH.with(|d| d.get()) == 0 {
        // No handler — render the condition and abort.
        let w = Word::from_raw(cond);
        let msg = crate::printer::format_word_aesthetic(w);
        eprintln!("unhandled condition: {msg}");
        std::process::abort();
    }
    CONDITION_SLOT.with(|s| s.set(cond));
    CONDITION_PENDING.with(|p| p.set(true));
    Word::NIL.raw()
}

/// `(%handler-case body-thunk handler-fn)` — internal primitive
/// behind the `handler-case` macro. Calls `body-thunk` with no
/// args; if a condition was signalled during the body, invokes
/// `handler-fn` with the condition as its single argument and
/// returns the handler's value. Otherwise returns the body's
/// result.
pub extern "C-unwind" fn handler_case_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("handler-case expects 2 args (body-thunk, handler-fn), got {n_args}");
    }
    let body_word = Word::from_raw(unsafe { *args });
    let handler_word = Word::from_raw(unsafe { *args.add(1) });
    if body_word.tag() != Tag::Function || handler_word.tag() != Tag::Function {
        panic!("handler-case: both args must be functions");
    }

    // Save the outer handler's pending state — we want a handler
    // entered while another is unhandled to start clean and not
    // accidentally catch the outer's condition.
    let saved_slot = CONDITION_SLOT.with(|s| s.get());
    let saved_pending = CONDITION_PENDING.with(|p| p.replace(false));
    HANDLER_DEPTH.with(|d| d.set(d.get() + 1));

    let body_result = {
        let env = crate::gc_function::env(body_word);
        let code = crate::gc_function::code_ptr(body_word);
        let f: crate::gc_function::LispCodeFn =
            unsafe { std::mem::transmute(code) };
        unsafe { f(mutator, env.raw(), std::ptr::null(), 0) }
    };

    HANDLER_DEPTH.with(|d| d.set(d.get() - 1));
    let was_pending = CONDITION_PENDING.with(|p| p.replace(saved_pending));
    let cond = CONDITION_SLOT.with(|s| s.replace(saved_slot));

    if was_pending {
        let env = crate::gc_function::env(handler_word);
        let code = crate::gc_function::code_ptr(handler_word);
        let f: crate::gc_function::LispCodeFn =
            unsafe { std::mem::transmute(code) };
        unsafe { f(mutator, env.raw(), &cond as *const u64, 1) }
    } else {
        body_result
    }
}

// ---- Loop / return ---------------------------------------------------------
//
// `loop` and `return` use the same thread-local-flag mechanism as
// `error` / `handler-case` because we can't unwind through JIT
// frames on Windows. `%native-loop` calls a thunk repeatedly,
// checking `LOOP_BREAK_PENDING` after each iteration; `%loop-
// return` sets the flag.
//
// Same limitation as error: code between `(return v)` and the end
// of the iteration body still runs, since the JIT frames don't
// know about the flag and just continue. Putting `return` in a
// terminal position (the last form of a cond/case clause)
// sidesteps the issue, matching idiomatic CL anyway.

thread_local! {
    static LOOP_BREAK_PENDING: Cell<bool> = const { Cell::new(false) };
    static LOOP_BREAK_VALUE: Cell<u64> = const { Cell::new(0) };
}

/// `(%native-loop thunk)` — call `thunk` with no args repeatedly
/// until `(%loop-return v)` is signalled. Returns `v`.
///
/// Saves and restores the outer LOOP_BREAK state so nested loops
/// each have their own break.
pub extern "C-unwind" fn native_loop_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("%native-loop expects 1 arg (thunk), got {n_args}");
    }
    let thunk = Word::from_raw(unsafe { *args });
    if thunk.tag() != Tag::Function {
        panic!("%native-loop: arg must be a function");
    }

    let saved_pending = LOOP_BREAK_PENDING.with(|p| p.replace(false));
    let saved_value = LOOP_BREAK_VALUE.with(|v| v.replace(0));

    let result = loop {
        let env = crate::gc_function::env(thunk);
        let code = crate::gc_function::code_ptr(thunk);
        let f: crate::gc_function::LispCodeFn =
            unsafe { std::mem::transmute(code) };
        let _ = unsafe { f(mutator, env.raw(), std::ptr::null(), 0) };

        if LOOP_BREAK_PENDING.with(|p| p.get()) {
            break LOOP_BREAK_VALUE.with(|v| v.get());
        }
    };

    LOOP_BREAK_PENDING.with(|p| p.set(saved_pending));
    LOOP_BREAK_VALUE.with(|v| v.set(saved_value));
    result
}

/// `(%loop-return value)` — signal that the current loop should
/// exit with `value`. Returns NIL; the JIT frames between this
/// call and the matching `%native-loop` see NIL and drain via
/// normal returns.
pub extern "C-unwind" fn loop_return_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    let value = if n_args >= 1 {
        unsafe { *args }
    } else {
        Word::NIL.raw()
    };
    LOOP_BREAK_VALUE.with(|v| v.set(value));
    LOOP_BREAK_PENDING.with(|p| p.set(true));
    Word::NIL.raw()
}

// ---- File I/O shims --------------------------------------------------------
//
// Each shim has the standard JIT calling convention so it can be
// installed in a symbol's function cell via the install_native
// mechanism. Internally, all dispatch to the higher-level
// functions in `file_sys`.

fn arg(args: *const u64, i: u64) -> Word {
    Word::from_raw(unsafe { *args.add(i as usize) })
}

fn arg_fixnum(args: *const u64, i: u64) -> Option<i64> {
    arg(args, i).as_fixnum()
}

/// (open-input-file path) → handle (fixnum) or 0
pub extern "C-unwind" fn open_input_file_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("open-input-file: expected 1 arg (path), got {n_args}");
    }
    let h = crate::file_sys::open_file(arg(args, 0), crate::file_sys::Mode::Input);
    Word::fixnum(h).raw()
}

/// (open-output-file path) → handle (truncates existing)
pub extern "C-unwind" fn open_output_file_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("open-output-file: expected 1 arg (path), got {n_args}");
    }
    let h = crate::file_sys::open_file(arg(args, 0), crate::file_sys::Mode::Output);
    Word::fixnum(h).raw()
}

/// (open-append-file path) → handle (creates or appends)
pub extern "C-unwind" fn open_append_file_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("open-append-file: expected 1 arg (path), got {n_args}");
    }
    let h = crate::file_sys::open_file(arg(args, 0), crate::file_sys::Mode::Append);
    Word::fixnum(h).raw()
}

/// (close-stream handle) → t
pub extern "C-unwind" fn close_stream_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("close-stream: expected 1 arg (handle), got {n_args}");
    }
    let h = arg_fixnum(args, 0).unwrap_or(0);
    crate::file_sys::close_file(h);
    Word::T.raw()
}

/// (read-line handle) → string or nil (EOF)
pub extern "C-unwind" fn read_line_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("read-line: expected 1 arg (handle), got {n_args}");
    }
    let h = arg_fixnum(args, 0).unwrap_or(0);
    match crate::file_sys::read_line(h) {
        Some(s) => {
            let m = unsafe { &mut *mutator };
            crate::gc_string::alloc_string_in_young(m, &s).raw()
        }
        None => Word::NIL.raw(),
    }
}

/// (read-char-from handle) → char or nil (EOF)
pub extern "C-unwind" fn read_char_from_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("read-char-from: expected 1 arg (handle), got {n_args}");
    }
    let h = arg_fixnum(args, 0).unwrap_or(0);
    match crate::file_sys::read_char(h) {
        Some(c) => Word::char(c).raw(),
        None => Word::NIL.raw(),
    }
}

/// (write-string-to handle string) → string
pub extern "C-unwind" fn write_string_to_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("write-string-to: expected 2 args (handle, string), got {n_args}");
    }
    let h = arg_fixnum(args, 0).unwrap_or(0);
    let s_word = arg(args, 1);
    if s_word.tag() != Tag::String {
        panic!("write-string-to: second arg must be a string, got {s_word:?}");
    }
    let s: String = crate::gc_string::chars_of(s_word).collect();
    crate::file_sys::write_string(h, &s);
    s_word.raw()
}

/// (file-position handle) → fixnum or -1
pub extern "C-unwind" fn file_position_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("file-position: expected 1 arg (handle), got {n_args}");
    }
    let h = arg_fixnum(args, 0).unwrap_or(0);
    Word::fixnum(crate::file_sys::file_position(h)).raw()
}

/// (file-length handle) → fixnum or -1
pub extern "C-unwind" fn file_length_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("file-length: expected 1 arg (handle), got {n_args}");
    }
    let h = arg_fixnum(args, 0).unwrap_or(0);
    Word::fixnum(crate::file_sys::file_length(h)).raw()
}

/// (file-exists path) → t or nil
pub extern "C-unwind" fn file_exists_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("file-exists: expected 1 arg (path), got {n_args}");
    }
    if crate::file_sys::file_exists(arg(args, 0)) {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

/// (delete-file path) → t or nil
pub extern "C-unwind" fn delete_file_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("delete-file: expected 1 arg (path), got {n_args}");
    }
    if crate::file_sys::delete_file(arg(args, 0)) {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

/// Lisp-callable shim for `append`. Binary append — concatenates
/// two lists; the second list's tail is shared (not copied). Used
/// by backquote-splicing macros, so it has to be available before
/// the user-Lisp stdlib loads. Variadic CL `append` lands when
/// `&rest` argument unpacking does.
pub extern "C-unwind" fn append_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("append: expected 2 args, got {n_args}");
    }
    let a = Word::from_raw(unsafe { *args });
    let b = Word::from_raw(unsafe { *args.add(1) });
    let m = unsafe { &mut *mutator };

    // Walk `a`, collecting cars; then cons onto `b` right-to-left.
    let mut cars: Vec<Word> = Vec::new();
    let mut cur = a;
    while !cur.is_nil() {
        if cur.tag() != Tag::Cons {
            panic!("append: first argument must be a proper list, got {cur:?}");
        }
        let p = cur.as_ptr::<u64>(Tag::Cons).expect("cons");
        cars.push(Word::from_raw(unsafe { *p }));
        cur = Word::from_raw(unsafe { *p.add(1) });
    }

    let mut result = b;
    for car in cars.iter().rev() {
        result = m.alloc_cons(*car, result);
    }
    result.raw()
}

/// Lisp-callable shim for `format`. Has the standard JIT function
/// signature so it can be installed in a Symbol's function cell
/// just like a defun'd function — making `format` a first-class
/// function (callable via `#'`, `apply`, `funcall`).
///
/// Arity is "at least 2" (dest + control); subsequent args become
/// the rest list, which `run_format` consumes one-by-one as
/// directives are processed.
pub extern "C-unwind" fn format_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args < 2 {
        panic!("format: expected at least 2 args (dest, control), got {n_args}");
    }
    let dest = Word::from_raw(unsafe { *args });
    let ctrl = Word::from_raw(unsafe { *args.add(1) });

    // Build the rest list from args[2..] (right-to-left).
    let m = unsafe { &mut *mutator };
    let mut rest = Word::NIL;
    let mut i = n_args;
    while i > 2 {
        i -= 1;
        let arg = Word::from_raw(unsafe { *args.add(i as usize) });
        rest = m.alloc_cons(arg, rest);
    }

    crate::format::run_format(m, dest, ctrl, rest).raw()
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
pub extern "C-unwind" fn ncl_apply(
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
