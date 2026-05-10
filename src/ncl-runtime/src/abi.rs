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
/// resolved to a local at compile time. On an unbound cell, signals
/// a Lisp condition (catchable with `handler-case`); aborts the
/// process if no handler is installed.
#[unsafe(no_mangle)]
pub extern "C-unwind" fn ncl_load_value(
    mutator: *mut crate::mutator::MutatorState,
    sym_word: u64,
) -> u64 {
    let sym = Word::from_raw(sym_word);
    let value = crate::gc_symbol::value_acquire(sym);
    if value.is_unbound() {
        let name = crate::sym_names::lookup(sym_word)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{sym_word:#x}"));
        return signal_condition_string(
            mutator,
            &format!("unbound variable: {name}"),
        );
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
        Tag::Vector => Word::fixnum(vector_payload_len(word) as i64).raw(),
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

/// Read a Vector-tagged Word's element count from its header.
/// SAFETY: `v` must be a valid Vector-tagged Word.
fn vector_payload_len(v: Word) -> u32 {
    let p = v.as_ptr::<u64>(Tag::Vector).expect("vector_payload_len: not a vector");
    let header = crate::heap::HeapHeader::from_raw(unsafe { *p });
    header.length_cells()
}

/// Read element `i` of a vector. SAFETY: `v` must be a valid
/// Vector-tagged Word and `i` must be in bounds.
fn vector_cell(v: Word, i: u32) -> u64 {
    let p = v.as_ptr::<u64>(Tag::Vector).expect("vector_cell: not a vector");
    unsafe { *p.add(1 + i as usize) }
}

/// Write element `i` of a vector and card-mark via the mutator
/// for the next GC's old-to-young scan. SAFETY: as `vector_cell`,
/// plus `mutator` must be the live MutatorState for this thread.
fn set_vector_cell(
    mutator: *mut crate::mutator::MutatorState,
    v: Word,
    i: u32,
    val: u64,
) {
    let p = v.as_mut_ptr::<u64>(Tag::Vector).expect("set_vector_cell: not a vector");
    let slot = unsafe { p.add(1 + i as usize) };
    unsafe { *slot = val };
    let m = unsafe { &mut *mutator };
    m.mark_card(slot as *const u8);
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
/// code. Signals on unbound — same condition shape as
/// `ncl_load_value` and `ncl_call`'s unbound case.
#[unsafe(no_mangle)]
pub extern "C-unwind" fn ncl_load_function(
    mutator: *mut crate::mutator::MutatorState,
    sym_word: u64,
) -> u64 {
    let sym = Word::from_raw(sym_word);
    let fn_value = crate::gc_symbol::function_acquire(sym);
    if fn_value.is_unbound() {
        let name = crate::sym_names::lookup(sym_word)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{sym_word:#x}"));
        return signal_condition_string(
            mutator,
            &format!("undefined function: {name}"),
        );
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
        return signal_condition_string(
            mutator,
            &format!("undefined function: {name}"),
        );
    }
    if fn_value.tag() != Tag::Function {
        return signal_condition_string(
            mutator,
            &format!("not a function: {fn_value:?}"),
        );
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

/// Signal a condition with the given message. Allocates a String
/// in the calling thread's young heap, then stashes it in the
/// condition slot if a handler is installed; otherwise prints and
/// aborts. Used by `ncl_call` / `ncl_load_value` / `ncl_load_function`
/// to turn unbound-symbol panics into catchable Lisp conditions.
pub fn signal_condition_string(
    mutator: *mut crate::mutator::MutatorState,
    msg: &str,
) -> u64 {
    if HANDLER_DEPTH.with(|d| d.get()) == 0 {
        eprintln!("unhandled condition: {msg}");
        std::process::abort();
    }
    let m = unsafe { &mut *mutator };
    let cond = crate::gc_string::alloc_string_in_young(m, msg);
    CONDITION_SLOT.with(|s| s.set(cond.raw()));
    CONDITION_PENDING.with(|p| p.set(true));
    Word::NIL.raw()
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
    // Also flip the unified abort flag so the call-site
    // instrumentation propagates this condition out of the body
    // immediately rather than waiting for natural return.
    ABORT_PENDING.with(|p| p.set(true));
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
        // We're consuming this condition — clear the unified
        // abort flag so the call-site instrumentation in the
        // handler body doesn't see a stale pending.
        ABORT_PENDING.with(|p| p.set(false));
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
        let body_value = unsafe { f(mutator, env.raw(), std::ptr::null(), 0) };

        // Block return / condition signal in flight — exit the
        // loop and propagate. Without this, a (return-from N V)
        // from inside a loop body would set ABORT_PENDING, the
        // body's call-site instrumentation would early-return,
        // and the loop would just keep iterating (we'd never
        // observe the abort or the value).
        if ABORT_PENDING.with(|p| p.get()) {
            break body_value;
        }
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

// ───────────────────────────────────────────────────────────────────
// Foundation utilities for CLOS port: gensym, eql, typep.
//
// These are deliberately small. `gensym` and `eql` are essentially
// one-liners on top of existing primitives; `typep` dispatches on
// the Word tag plus a few keyword type names. Together they're the
// first prerequisite chunk for porting Closette.
// ───────────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic counter for gensym names. Reset would defeat the
/// purpose (the symbols are interned by their generated name) so
/// we never reset it. AtomicU64 is sufficient — 2^64 gensyms is
/// not a realistic concern.
static GENSYM_COUNTER: AtomicU64 = AtomicU64::new(0);

/// `(gensym)` / `(gensym prefix)` — return a freshly interned
/// symbol whose name is `prefix` followed by an unused decimal
/// integer. Default prefix is "G". CL's gensym returns an
/// uninterned symbol; we don't have uninterned symbols yet, so
/// we intern a name guaranteed never to collide with anything
/// the user has typed (the counter only goes up).
pub extern "C-unwind" fn gensym_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    let prefix = if n_args >= 1 {
        let w = Word::from_raw(unsafe { *args });
        if w.tag() == Tag::String {
            crate::gc_string::chars_of(w).collect::<String>()
        } else {
            "G".to_string()
        }
    } else {
        "G".to_string()
    };
    let n = GENSYM_COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = format!("{prefix}{n}");
    let m = unsafe { &mut *mutator };
    m.coord().intern(&name).raw()
}

/// `(car x)` — Lisp-callable shim. CAR is a special form
/// lowering to Expr::Car; this shim covers `#'car` for use
/// with funcall / map / :key etc.
pub extern "C-unwind" fn car_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("car: expected 1 arg, got {n_args}");
    }
    let w = Word::from_raw(unsafe { *args });
    if w.is_nil() {
        return Word::NIL.raw();
    }
    ncl_car(unsafe { *args })
}

/// `(cdr x)` — sibling shim for car. NIL is a fixed point.
pub extern "C-unwind" fn cdr_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("cdr: expected 1 arg, got {n_args}");
    }
    let w = Word::from_raw(unsafe { *args });
    if w.is_nil() {
        return Word::NIL.raw();
    }
    ncl_cdr(unsafe { *args })
}

/// `(cons a b)` — Lisp-callable shim.
pub extern "C-unwind" fn cons_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("cons: expected 2 args, got {n_args}");
    }
    let car = unsafe { *args };
    let cdr = unsafe { *args.add(1) };
    ncl_alloc_cons(mutator, car, cdr)
}

/// `(equal a b)` — Lisp-callable shim around `ncl_equal`. The
/// EQUAL special form lowers directly to Expr::Equal; this shim
/// is needed for `#'equal` (taking the function as a value, e.g.
/// `(member item lst :test #'equal)`).
pub extern "C-unwind" fn equal_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("equal: expected 2 args, got {n_args}");
    }
    let a = unsafe { *args };
    let b = unsafe { *args.add(1) };
    ncl_equal(a, b)
}

/// `(eq a b)` — Lisp-callable shim. EQ is also a special form;
/// this is for `#'eq`.
pub extern "C-unwind" fn eq_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("eq: expected 2 args, got {n_args}");
    }
    let a = unsafe { *args };
    let b = unsafe { *args.add(1) };
    if a == b { Word::T.raw() } else { Word::NIL.raw() }
}

/// `(eql a b)` — true if the two values are the same object, or
/// (eventually) the same number / character of the same type and
/// value. For the data types currently supported (fixnums, chars,
/// symbols, NIL, T, immediate Words, plus identity-compared
/// strings/cons/functions), `eql` is exactly `eq` because:
///
///   - Fixnums and chars are stored fully in the Word's bits, so
///     two equal-valued ones compare bit-equal.
///   - Symbols are interned, so equal-named ones share a Word.
///   - NIL / T / unbound markers are unique constant Words.
///   - Strings / conses / functions get identity (CL allows that).
///
/// When floats and bignums land, this shim has to grow value
/// comparisons for those — that's the only behavioral change.
pub extern "C-unwind" fn eql_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("eql: expected 2 args, got {n_args}");
    }
    let a = unsafe { *args };
    let b = unsafe { *args.add(1) };
    if a == b {
        Word::T.raw()
    } else {
        Word::NIL.raw()
    }
}

/// `(typep object type-spec)` — true if `object` belongs to the
/// CL type named by `type-spec`. `type-spec` is a symbol; compound
/// type specs like `(integer 0 100)` are not yet supported.
///
/// Recognised type names (case-insensitive comparison via the
/// printer name): T, NIL, NULL, ATOM, CONS, LIST, SYMBOL, KEYWORD,
/// FIXNUM, INTEGER, NUMBER, RATIONAL, REAL, STRING, SIMPLE-STRING,
/// CHARACTER, FUNCTION, VECTOR, SIMPLE-VECTOR, ARRAY, SEQUENCE,
/// HASH-TABLE, STANDARD-OBJECT.
///
/// Types we don't yet have storage for (vectors, arrays, hash
/// tables, standard objects) always return NIL; they'll start
/// returning T once the relevant chunks land.
pub extern "C-unwind" fn typep_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("typep: expected 2 args (object type), got {n_args}");
    }
    let obj = Word::from_raw(unsafe { *args });
    let type_w = Word::from_raw(unsafe { *args.add(1) });
    if type_w.tag() != Tag::Symbol {
        // Compound specs — `(integer ...)`, `(or ...)`, etc.
        // — fall through as "unsupported, returns NIL" for now.
        return Word::NIL.raw();
    }
    let name = match crate::sym_names::lookup(type_w.raw()) {
        Some(n) => n,
        None => return Word::NIL.raw(),
    };
    let matches = match name.as_ref() {
        "T" => true,
        "NIL" | "NULL" => obj.is_nil(),
        "ATOM" => obj.tag() != Tag::Cons,
        "CONS" => obj.tag() == Tag::Cons,
        "LIST" => obj.is_nil() || obj.tag() == Tag::Cons,
        "SYMBOL" => obj.tag() == Tag::Symbol || obj.is_nil(),
        "KEYWORD" => is_keyword(obj),
        "FIXNUM" | "INTEGER" | "RATIONAL" | "REAL" | "NUMBER" => {
            obj.tag() == Tag::Fixnum
        }
        "STRING" | "SIMPLE-STRING" => obj.tag() == Tag::String,
        "CHARACTER" => obj.as_char().is_some(),
        "FUNCTION" => obj.tag() == Tag::Function,
        "VECTOR" | "SIMPLE-VECTOR" | "ARRAY" => obj.tag() == Tag::Vector,
        "SEQUENCE" => {
            obj.is_nil()
                || obj.tag() == Tag::Cons
                || obj.tag() == Tag::String
                || obj.tag() == Tag::Vector
        }
        "HASH-TABLE" => false, // future
        "STANDARD-OBJECT" => false, // future
        _ => false,
    };
    if matches { Word::T.raw() } else { Word::NIL.raw() }
}

/// Keywords are symbols whose printer name starts with `:` (the
/// reader interns them with the leading colon).
fn is_keyword(w: Word) -> bool {
    if w.tag() != Tag::Symbol {
        return false;
    }
    crate::sym_names::lookup(w.raw())
        .map(|n| n.starts_with(':'))
        .unwrap_or(false)
}

// ───────────────────────────────────────────────────────────────────
// Multiple values — thread-local "extra return values" buffer.
//
// Per CL, a function that returns normally produces one value, but
// `(values v1 v2 ... vN)` lets it return N. The receiving site
// (multiple-value-bind / multiple-value-list) reads them.
//
// Storage: a thread-local Vec<u64>. The compiler enforces the
// invariant that, after every function call, the slot contains
// exactly the called function's return values:
//   - If the function's tail expression was `(values ...)`,
//     `Expr::Values` wrote the full set.
//   - Otherwise the tail-position transform wraps the body in
//     `Expr::EnsureSingleMv`, which writes [primary] at exit.
//
// Without that invariant, a subroutine call inside a function would
// pollute the slot, and the outer multiple-value-bind would observe
// stale data. With the invariant, the slot is always fresh.
//
// JIT-callable. Lowered from `Expr::Values` and `Expr::EnsureSingleMv`.
// ───────────────────────────────────────────────────────────────────

use std::cell::RefCell;

thread_local! {
    static MV_VALUES: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// `(%mv-clear)` — empty the multi-value slot. The
/// multiple-value-bind / multiple-value-list macros call this
/// before evaluating their form so that, if the form turns out to
/// be a non-function expression (constant, variable lookup, native
/// shim call), the slot is observably empty and the macro falls
/// back to `[primary]`. JIT-compiled function calls overwrite the
/// slot via EnsureSingleMv or Expr::Values; native shim calls
/// don't, hence the need for an explicit pre-clear.
pub extern "C-unwind" fn mv_clear_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    _args: *const u64,
    _n_args: u64,
) -> u64 {
    MV_VALUES.with(|cell| cell.borrow_mut().clear());
    Word::NIL.raw()
}

/// Write `[v]` into the multi-value slot. Used by the
/// EnsureSingleMv epilogue around any function body whose tail
/// expression isn't `(values ...)`.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_set_mv_single(v: u64) {
    MV_VALUES.with(|cell| {
        let mut b = cell.borrow_mut();
        b.clear();
        b.push(v);
    });
}

/// Write `args[0..n]` into the multi-value slot. Used by
/// `Expr::Values` for `(values v1 v2 ... vN)` in tail position.
///
/// SAFETY: `args` must point to at least `n` valid `u64`s.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_set_mv_many(args: *const u64, n: u64) {
    MV_VALUES.with(|cell| {
        let mut b = cell.borrow_mut();
        b.clear();
        for i in 0..n {
            b.push(unsafe { *args.add(i as usize) });
        }
    });
}

/// `(multiple-value-list-of primary)` — return the multi-value slot
/// as a fresh Lisp list. If the slot is somehow empty (shouldn't
/// happen given the invariant, but defensive), fall back to
/// `(primary)`. Used inside the `multiple-value-bind` /
/// `multiple-value-list` macro expansions, called immediately
/// after the form being inspected.
pub extern "C-unwind" fn multiple_value_list_of_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!(
            "multiple-value-list-of: expected 1 arg (primary), got {n_args}"
        );
    }
    let primary = unsafe { *args };
    let m = unsafe { &mut *mutator };
    let snapshot: Vec<u64> = MV_VALUES.with(|cell| cell.borrow().clone());
    let source = if snapshot.is_empty() {
        vec![primary]
    } else {
        snapshot
    };
    let mut acc = Word::NIL;
    for &v in source.iter().rev() {
        acc = m.alloc_cons(Word::from_raw(v), acc);
    }
    acc.raw()
}

// ───────────────────────────────────────────────────────────────────
// block / return-from — named non-local exits.
//
// CL `(block NAME body)` establishes a named exit point;
// `(return-from NAME val)` returns VAL from the matching block.
//
// Implementation mirrors the existing error / handler-case flag
// mechanism: a thread-local "abort pending" slot per active
// block. return-from sets the slot; block on its way out checks
// the slot and either returns the captured value (matching
// name) or leaves the slot set so a surrounding block (or
// no-block-at-all panic) sees it.
//
// Trade-off shared with handler-case: the flag doesn't abort
// the body's execution. Code after a (return-from N V) call
// runs to completion before block reads the flag. For tail-
// position return-from this is fine; for non-tail uses inside
// a do-loop with code after the loop, the trailing code runs.
// (See setf-getf* in Closette for the worst-case shape.)
// Promoting to call-site instrumentation lands when handler-
// case grows the same.
// ───────────────────────────────────────────────────────────────────

struct BlockFrame {
    name: u64,
    value: u64,
}

thread_local! {
    /// Stack of active block frames in dynamic-extent order.
    /// Innermost is at the back. Each (block N body) pushes a
    /// frame on entry and pops on exit; (return-from N val)
    /// walks the stack for matching name, sets value on the
    /// frame, and signals "abort pending."
    static BLOCK_STACK: RefCell<Vec<BlockFrame>> = const { RefCell::new(Vec::new()) };

    /// Name (raw symbol Word) of the block return-from is
    /// targeting. Only meaningful when ABORT_PENDING is true
    /// AND the abort is a block return (not a condition signal).
    static BLOCK_TARGET: Cell<u64> = const { Cell::new(0) };
}

// Unified non-local exit flag. Set by either `error` or
// `return-from`; cleared by whichever (handler-case / native-
// block) ends up consuming it. The JIT instruments every Lisp
// function call to check this — if set, the calling function
// returns immediately without executing further forms,
// propagating the abort up to its consumer.
//
// This is the call-site instrumentation that makes (return-from
// …) actually abort the rest of the body, not just set a flag
// that's checked at block boundary. Same fix applies to (error
// …): code after `(error 'foo)` no longer runs.
thread_local! {
    pub(crate) static ABORT_PENDING: Cell<bool> = const { Cell::new(false) };
}

/// JIT-callable check. Returns 1 if a non-local exit is
/// pending; 0 otherwise. Lowered as an inline call after every
/// Expr::Call / Funcall / Apply.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_abort_pending() -> i32 {
    if ABORT_PENDING.with(|c| c.get()) {
        1
    } else {
        0
    }
}

/// `(%native-block NAME THUNK)` — the primitive behind the
/// `block` macro. Pushes a frame, invokes the body thunk, then
/// inspects the abort-pending flag: if set and targeting our
/// name, consume it and return the frame's value; otherwise
/// pass the body's value through (or leave the abort pending
/// for a surrounding block to catch).
pub extern "C-unwind" fn native_block_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("%native-block: expected 2 args (name thunk), got {n_args}");
    }
    let name = unsafe { *args };
    let thunk = Word::from_raw(unsafe { *args.add(1) });
    if thunk.tag() != Tag::Function {
        panic!("%native-block: thunk must be a function, got {thunk:?}");
    }

    BLOCK_STACK.with(|s| {
        s.borrow_mut().push(BlockFrame {
            name,
            value: Word::NIL.raw(),
        });
    });

    let body_result = {
        let env = crate::gc_function::env(thunk);
        let code = crate::gc_function::code_ptr(thunk);
        let f: crate::gc_function::LispCodeFn =
            unsafe { std::mem::transmute(code) };
        unsafe { f(mutator, env.raw(), std::ptr::null(), 0) }
    };

    // Pop our frame and decide what to do based on the abort flag.
    let our_frame_value = BLOCK_STACK.with(|s| {
        s.borrow_mut()
            .pop()
            .expect("BLOCK_STACK underflow at native-block exit")
            .value
    });

    let pending = ABORT_PENDING.with(|p| p.get());
    if pending {
        // Abort might be ours (a return-from targeting our name)
        // or somebody else's (a condition, or a return-from
        // targeting an outer block). BLOCK_TARGET == 0 means it's
        // not a block abort at all (it's a condition).
        let target = BLOCK_TARGET.with(|t| t.get());
        if target != 0 && target == name {
            // The abort targets us — consume it and return the
            // value the matching return-from stashed on our frame.
            ABORT_PENDING.with(|p| p.set(false));
            BLOCK_TARGET.with(|t| t.set(0));
            return our_frame_value;
        }
        // Not our abort — leave the flag pending so a surrounding
        // block (or handler-case for a condition) catches it.
        // Drop body_result.
    }
    body_result
}

/// `(%return-from NAME VALUE)` — find the topmost matching block
/// on BLOCK_STACK, write VALUE into its frame, set the abort
/// flag. Returns NIL (the body keeps running until natural exit;
/// see module docstring for the trade-off). Panics if no
/// matching block is currently active.
pub extern "C-unwind" fn return_from_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("%return-from: expected 2 args (name value), got {n_args}");
    }
    let name = unsafe { *args };
    let val = unsafe { *args.add(1) };

    let found = BLOCK_STACK.with(|s| {
        let mut stack = s.borrow_mut();
        let pos = stack.iter().rposition(|f| f.name == name);
        match pos {
            Some(i) => {
                stack[i].value = val;
                true
            }
            None => false,
        }
    });
    if !found {
        let sym_name = crate::sym_names::lookup(name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        panic!("return-from: no enclosing block named {sym_name}");
    }
    ABORT_PENDING.with(|p| p.set(true));
    BLOCK_TARGET.with(|t| t.set(name));
    Word::NIL.raw()
}

// ───────────────────────────────────────────────────────────────────
// symbol-function — read / write / unbind a symbol's function cell.
//
// Closette installs JIT-generated discriminating functions into
// each generic function's symbol cell via (setf (symbol-function
// name) ...). These shims expose the existing function-cell API
// (gc_symbol::function_acquire / mutator::set_symbol_function) to
// Lisp.
// ───────────────────────────────────────────────────────────────────

/// `(intern name)` — intern NAME (a string) as a symbol and
/// return it. NAME is used verbatim — it's not upcased or
/// touched. Defstruct uses this from its macro expansion to
/// build constructor / accessor / setter symbol names from the
/// struct's name.
pub extern "C-unwind" fn intern_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("intern: expected 1 arg (string), got {n_args}");
    }
    let w = Word::from_raw(unsafe { *args });
    if w.tag() != Tag::String {
        panic!("intern: argument must be a string, got {w:?}");
    }
    let name: String = crate::gc_string::chars_of(w).collect();
    let m = unsafe { &mut *mutator };
    m.coord().intern(&name).raw()
}

/// `(symbol-function sym)` — return the function bound to SYM, or
/// signal an error (panic for now) if the cell is unbound.
pub extern "C-unwind" fn symbol_function_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("symbol-function: expected 1 arg (symbol), got {n_args}");
    }
    let sym = Word::from_raw(unsafe { *args });
    if sym.tag() != Tag::Symbol {
        panic!("symbol-function: argument must be a symbol, got {sym:?}");
    }
    let f = crate::gc_symbol::function_acquire(sym);
    if f.is_unbound() {
        let name = crate::sym_names::lookup(sym.raw())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        panic!("symbol-function: undefined function: {name}");
    }
    f.raw()
}

/// `(%set-symbol-function sym fn)` — install FN as the function
/// bound to SYM. The setf-symbol-function lowering rewrites
/// `(setf (symbol-function s) f)` into a call to this. Returns
/// the new function value.
pub extern "C-unwind" fn set_symbol_function_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        panic!("%set-symbol-function: expected 2 args (sym fn), got {n_args}");
    }
    let sym = Word::from_raw(unsafe { *args });
    let new_fn = Word::from_raw(unsafe { *args.add(1) });
    if sym.tag() != Tag::Symbol {
        panic!("%set-symbol-function: first arg must be a symbol, got {sym:?}");
    }
    if new_fn.tag() != Tag::Function {
        panic!(
            "%set-symbol-function: second arg must be a function, got {new_fn:?}"
        );
    }
    let m = unsafe { &mut *mutator };
    m.set_symbol_function(sym, new_fn);
    new_fn.raw()
}

/// `(fmakunbound sym)` — clear SYM's function cell. Returns SYM.
pub extern "C-unwind" fn fmakunbound_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("fmakunbound: expected 1 arg (symbol), got {n_args}");
    }
    let sym = Word::from_raw(unsafe { *args });
    if sym.tag() != Tag::Symbol {
        panic!("fmakunbound: argument must be a symbol, got {sym:?}");
    }
    let m = unsafe { &mut *mutator };
    m.set_symbol_function(sym, Word::UNBOUND);
    sym.raw()
}

/// `(fboundp sym)` — return T if SYM has a function cell bound,
/// NIL otherwise.
pub extern "C-unwind" fn fboundp_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("fboundp: expected 1 arg (symbol), got {n_args}");
    }
    let sym = Word::from_raw(unsafe { *args });
    if sym.tag() != Tag::Symbol {
        return Word::NIL.raw();
    }
    let f = crate::gc_symbol::function_acquire(sym);
    if f.is_unbound() {
        Word::NIL.raw()
    } else {
        Word::T.raw()
    }
}

// ───────────────────────────────────────────────────────────────────
// Hash function — bit-mix on a Word's raw bits.
//
// The Lisp hash-table layer is built on top of Vectors and cons
// cells (see core.lisp). The only piece that genuinely has to be
// in Rust is a fast hash that takes a Word and produces a
// non-negative fixnum the bucket index calculation can mod.
//
// For the EQ / EQL cases (the only ones Closette and the GUI
// demos actually need), hashing the raw bits is correct because
// two values that are EQ have identical Word bits. Symbols are
// interned (so equal-named ones share a Word), fixnums and chars
// are immediate, T/NIL are unique.
//
// EQUAL hash tables on strings and conses would need a
// content-hash; that's deferred until something needs it.
// ───────────────────────────────────────────────────────────────────

/// `(%word-hash w)` — return a non-negative fixnum hash of the
/// raw Word bits. Uses a SplitMix64-style finaliser; one shift +
/// two multiplies. Caller mods by bucket count.
pub extern "C-unwind" fn word_hash_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("%word-hash: expected 1 arg, got {n_args}");
    }
    let w = unsafe { *args };
    // SplitMix64 finaliser.
    let mut h = w.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    // Fixnums are 61 bits signed; mask to 60 bits to guarantee a
    // non-negative result that fits.
    let positive = (h & ((1u64 << 60) - 1)) as i64;
    Word::fixnum(positive).raw()
}

// ───────────────────────────────────────────────────────────────────
// Vectors — make-array, vector, aref, svref, (setf aref).
//
// CL's `make-array` is heavily overloaded (multidimensional, fill
// pointers, displaced arrays, element-type, adjustable). We support
// the simple-vector subset here — that's what Closette and the GUI
// demos need. If a richer surface is required later, `make-array`'s
// shim can grow keyword args without touching the underlying
// vector heap object.
// ───────────────────────────────────────────────────────────────────

/// `(make-array dim &key initial-element initial-contents)`.
/// `dim` is a fixnum length (multidimensional shapes deferred —
/// reject lists for now). `initial-element` fills every cell;
/// without it, cells are NIL. `initial-contents` (a list) copies
/// list elements into positions; if shorter than `dim`, trailing
/// positions stay NIL (or the initial-element).
pub extern "C-unwind" fn make_array_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args == 0 {
        panic!("make-array: expected at least 1 arg (dimension)");
    }
    let dim_w = Word::from_raw(unsafe { *args });
    let n = match dim_w.as_fixnum() {
        Some(n) if n >= 0 => n as u32,
        _ => panic!(
            "make-array: dimension must be a non-negative fixnum, got {dim_w:?}"
        ),
    };
    // Scan keyword args. Only :initial-element and :initial-contents
    // recognised. Unknown keywords are silently ignored, matching
    // the ergonomic-but-permissive style of the existing shims.
    let mut init_element: Word = Word::NIL;
    let mut init_contents: Option<Word> = None;
    let mut i = 1u64;
    while i + 1 < n_args {
        let key = Word::from_raw(unsafe { *args.add(i as usize) });
        let val = Word::from_raw(unsafe { *args.add((i + 1) as usize) });
        if let Some(name) = crate::sym_names::lookup(key.raw()) {
            match name.as_ref() {
                ":INITIAL-ELEMENT" => init_element = val,
                ":INITIAL-CONTENTS" => init_contents = Some(val),
                _ => {}
            }
        }
        i += 2;
    }
    let m = unsafe { &mut *mutator };
    let v = m.alloc_vector(n);
    // Initialise. alloc_vector zero-fills (zero == fixnum 0 != NIL),
    // so we always do an explicit init here.
    for idx in 0..n {
        set_vector_cell(mutator, v, idx, init_element.raw());
    }
    if let Some(list) = init_contents {
        let mut cur = list;
        let mut idx = 0u32;
        while !cur.is_nil() && idx < n {
            if cur.tag() != Tag::Cons {
                panic!("make-array :initial-contents must be a proper list");
            }
            let p = cur.as_ptr::<u64>(Tag::Cons).unwrap();
            let elem = unsafe { *p };
            set_vector_cell(mutator, v, idx, elem);
            cur = Word::from_raw(unsafe { *p.add(1) });
            idx += 1;
        }
    }
    v.raw()
}

/// `(vector e1 e2 ... eN)` — construct a fresh vector containing
/// the given elements in order. Same length as the call's arg
/// count.
pub extern "C-unwind" fn vector_shim(
    mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let v = m.alloc_vector(n_args as u32);
    for i in 0..n_args {
        let elem = unsafe { *args.add(i as usize) };
        set_vector_cell(mutator, v, i as u32, elem);
    }
    v.raw()
}

/// `(svref v i)` and `(aref v i)` for vectors. JIT-callable from
/// the polymorphic AREF lowering. Bounds-checked; out-of-range
/// indices panic.
///
/// SAFETY: arguments come from JIT'd code; both must be valid
/// Words. `v` must be Vector- or String-tagged for the dispatch
/// to make sense.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_aref_generic(v: u64, i: u64) -> u64 {
    let vw = Word::from_raw(v);
    let iw = Word::from_raw(i);
    let idx = match iw.as_fixnum() {
        Some(n) if n >= 0 => n as usize,
        _ => panic!("aref: index must be a non-negative fixnum, got {iw:?}"),
    };
    match vw.tag() {
        Tag::Vector => {
            let n = vector_payload_len(vw) as usize;
            if idx >= n {
                panic!("aref: index {idx} out of bounds for vector of length {n}");
            }
            vector_cell(vw, idx as u32)
        }
        // ncl_string_char takes a raw (untagged) index.
        Tag::String => crate::abi::ncl_string_char(v, idx as u64),
        _ => panic!("aref: not a sequence: {vw:?}"),
    }
}

/// `(setf (aref v i) val)` for vectors and strings. Polymorphic
/// dispatch. Returns `val`. Card-marks for vectors (strings have
/// their own GC story).
#[unsafe(no_mangle)]
pub extern "C" fn ncl_aset_generic(
    mutator: *mut crate::mutator::MutatorState,
    v: u64,
    i: u64,
    val: u64,
) -> u64 {
    let vw = Word::from_raw(v);
    let iw = Word::from_raw(i);
    let idx = match iw.as_fixnum() {
        Some(n) if n >= 0 => n as usize,
        _ => panic!("(setf aref): index must be a non-negative fixnum, got {iw:?}"),
    };
    match vw.tag() {
        Tag::Vector => {
            let n = vector_payload_len(vw) as usize;
            if idx >= n {
                panic!(
                    "(setf aref): index {idx} out of bounds for vector of length {n}"
                );
            }
            set_vector_cell(mutator, vw, idx as u32, val);
            val
        }
        // ncl_string_set takes a raw (untagged) index.
        Tag::String => crate::abi::ncl_string_set(v, idx as u64, val),
        _ => panic!("(setf aref): not a sequence: {vw:?}"),
    }
}

/// `ncl_lookup_keyword(args, key_start, n_args, keyword) -> value`.
///
/// Scan `args[key_start..n_args]` in steps of 2. If `args[i]`
/// matches `keyword`, return `args[i+1]`. If no match (or the
/// trailing slot has no following value), return `Word::UNBOUND`
/// — the IR's `KeyArg` lowering branches on that to evaluate the
/// default form.
///
/// JIT-callable. The compiler emits a call to this for every `&key`
/// parameter at function entry.
///
/// SAFETY: `args` must point to at least `n_args` valid `u64`s.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_lookup_keyword(
    args: *const u64,
    key_start: u64,
    n_args: u64,
    keyword: u64,
) -> u64 {
    let mut i = key_start;
    while i + 1 < n_args {
        let here = unsafe { *args.add(i as usize) };
        if here == keyword {
            return unsafe { *args.add((i + 1) as usize) };
        }
        i += 2;
    }
    Word::UNBOUND.raw()
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
