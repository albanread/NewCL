//! LLVM bindings + JIT.
//!
//! Phase 2: hand-built `jit_three`/`jit_add` (smoke tests).
//! Phase 3: `jit_eval` for arithmetic.
//! Cons/Car/Cdr extension: tagged Word, callback into runtime via
//! `ncl_alloc_cons`.
//! Eq/If: control flow with phi nodes.
//! Node 2 (this file): `jit_compile_function` produces a code
//! pointer for a parameterised function. The unified function
//! signature for ALL JIT'd Lisp code is now
//!   `fn(mutator: *mut MutatorState, args: *const u64, n_args: u64) -> u64`
//! so dispatch through `ncl_call` is uniform regardless of arity.
//! `jit_eval` calls the entry function with `(mutator, null, 0)`.

use std::sync::Mutex;

use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::{Linkage, Module};
use inkwell::attributes::AttributeLoc;
use inkwell::values::{FunctionValue, IntValue};

use ncl_ir::Expr;
use ncl_runtime::{
    ncl_abort_pending, ncl_alloc_cons, ncl_apply, ncl_aref_generic, ncl_aset_generic,
    ncl_build_rest_list, ncl_call, ncl_equal, ncl_funcall,
    ncl_length, ncl_load_function, ncl_load_value, ncl_lookup_keyword,
    ncl_make_closure, ncl_set_car,
    ncl_set_cdr, ncl_set_mv_many, ncl_set_mv_single, ncl_store_value,
    ncl_string_char, ncl_string_eq, ncl_string_set,
    MutatorState, Tag, Word,
};

// We leak each compilation's Context + Module + Engine so the
// JIT'd code stays valid for the process lifetime. A real loader
// would track these for retirement (see MANIFESTO.md, "The
// loader") but that machinery isn't wired through yet. We store
// addresses as usize so the static is Sync (raw pointers aren't).
static KEEP_ALIVE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn keep_forever<T: 'static>(t: T) -> *mut T {
    let p = Box::into_raw(Box::new(t));
    KEEP_ALIVE.lock().unwrap().push(p as usize);
    p
}

// -- Phase 2 smoke functions ------------------------------------------------

pub fn jit_three() -> Result<i64, String> {
    let context = Context::create();
    let module = context.create_module("ncl_smoke_three");
    let builder = context.create_builder();

    let i64_t = context.i64_type();
    let fn_type = i64_t.fn_type(&[], false);
    let function = module.add_function("three", fn_type, None);
    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);
    let three = i64_t.const_int(3, false);
    builder
        .build_return(Some(&three))
        .map_err(|e| format!("build_return: {e}"))?;

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| format!("create_jit_execution_engine: {e}"))?;

    type ThreeFn = unsafe extern "C" fn() -> i64;
    let jit_fn: JitFunction<ThreeFn> = unsafe {
        engine
            .get_function("three")
            .map_err(|e| format!("get_function: {e}"))?
    };
    Ok(unsafe { jit_fn.call() })
}

pub fn jit_add(a: i64, b: i64) -> Result<i64, String> {
    let context = Context::create();
    let module = context.create_module("ncl_smoke_add");
    let builder = context.create_builder();

    let i64_t = context.i64_type();
    let fn_type = i64_t.fn_type(&[i64_t.into(), i64_t.into()], false);
    let function = module.add_function("add", fn_type, None);
    let entry = context.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let lhs = function.get_nth_param(0).unwrap().into_int_value();
    let rhs = function.get_nth_param(1).unwrap().into_int_value();
    let sum = builder
        .build_int_add(lhs, rhs, "sum")
        .map_err(|e| format!("build_int_add: {e}"))?;
    builder
        .build_return(Some(&sum))
        .map_err(|e| format!("build_return: {e}"))?;

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| format!("create_jit_execution_engine: {e}"))?;

    type AddFn = unsafe extern "C" fn(i64, i64) -> i64;
    let jit_fn: JitFunction<AddFn> = unsafe {
        engine
            .get_function("add")
            .map_err(|e| format!("get_function: {e}"))?
    };
    Ok(unsafe { jit_fn.call(a, b) })
}

// -- Public JIT API ---------------------------------------------------------

/// JIT-compile and run an `Expr` tree as a top-level form. Builds
/// an entry function with the unified Lisp signature and calls it
/// with `(mutator, NIL, null, 0)` — top-level forms have no
/// parameters and no closure environment. Returns the result as
/// a tagged `Word`.
pub fn jit_eval(expr: &Expr, mutator: *mut MutatorState) -> Result<Word, String> {
    let code = build_lisp_function(expr, 0)?;
    type EntryFn =
        unsafe extern "C" fn(*mut MutatorState, u64, *const u64, u64) -> u64;
    let f: EntryFn = unsafe { std::mem::transmute(code) };
    Ok(Word::from_raw(unsafe {
        f(mutator, Word::NIL.raw(), std::ptr::null(), 0)
    }))
}

/// JIT-compile a Lisp function with `arity` positional parameters.
/// Returns the raw machine-code address of the entry. Used by
/// `defun` to install a Function in a Symbol's function cell.
///
/// The returned pointer is a function with signature
///   `extern "C" fn(*mut MutatorState, env: u64, *const u64, u64) -> u64`
/// and is valid for the process lifetime. The `env` slot is the
/// closure's captured-env Vector (or NIL for non-closures).
pub fn jit_compile_function(arity: u32, body: &Expr) -> Result<usize, String> {
    build_lisp_function(body, arity)
}

// -- Implementation ---------------------------------------------------------

fn build_lisp_function(body: &Expr, arity: u32) -> Result<usize, String> {
    let context_ptr = keep_forever(Context::create());
    let context: &'static Context = unsafe { &*context_ptr };

    let module = context.create_module("ncl_fn");
    let builder = context.create_builder();

    let i64_t = context.i64_type();
    let ptr_t = context.ptr_type(AddressSpace::default());

    // Unified Lisp function signature:
    //   fn(mutator: ptr, env: i64, args: ptr, n_args: i64) -> i64
    let fn_type = i64_t.fn_type(
        &[ptr_t.into(), i64_t.into(), ptr_t.into(), i64_t.into()],
        false,
    );
    let function = module.add_function("the_fn", fn_type, None);
    // `uwtable` tells the backend to emit unwind tables for this
    // function. On Windows we need them so a Rust panic raised in
    // a runtime helper (e.g. `error_shim`) can unwind back through
    // the JIT frame to the matching `handler-case`. Without this
    // the unwinder hits the JIT frame and the panic escapes to the
    // OS (MSVC SEH 0xe06d7363 = "C++ exception not caught").
    let kind_id = inkwell::attributes::Attribute::get_named_enum_kind_id("uwtable");
    let attr = context.create_enum_attribute(kind_id, 1);
    function.add_attribute(AttributeLoc::Function, attr);
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let helpers = declare_runtime_helpers(context, &module);

    let mut locals: Vec<IntValue<'_>> = Vec::new();
    let result = emit_expr(
        context,
        &builder,
        &function,
        &helpers,
        arity,
        &mut locals,
        body,
    )?;
    builder
        .build_return(Some(&result))
        .map_err(|e| format!("build_return: {e}"))?;

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| format!("create_jit_execution_engine: {e}"))?;
    register_runtime_helpers(&engine, &helpers);

    let addr = engine
        .get_function_address("the_fn")
        .map_err(|e| format!("get_function_address: {e}"))?;

    let _ = keep_forever(module);
    let _ = keep_forever(engine);

    Ok(addr)
}

struct Helpers<'ctx> {
    alloc_cons: FunctionValue<'ctx>,
    call_fn: FunctionValue<'ctx>,
    funcall_fn: FunctionValue<'ctx>,
    make_closure: FunctionValue<'ctx>,
    load_value: FunctionValue<'ctx>,
    load_function: FunctionValue<'ctx>,
    store_value: FunctionValue<'ctx>,
    length: FunctionValue<'ctx>,
    equal: FunctionValue<'ctx>,
    string_eq: FunctionValue<'ctx>,
    string_char: FunctionValue<'ctx>,
    set_car: FunctionValue<'ctx>,
    set_cdr: FunctionValue<'ctx>,
    string_set: FunctionValue<'ctx>,
    build_rest_list: FunctionValue<'ctx>,
    apply: FunctionValue<'ctx>,
    lookup_keyword: FunctionValue<'ctx>,
    set_mv_single: FunctionValue<'ctx>,
    set_mv_many: FunctionValue<'ctx>,
    aref_generic: FunctionValue<'ctx>,
    aset_generic: FunctionValue<'ctx>,
    abort_pending: FunctionValue<'ctx>,
}

fn declare_runtime_helpers<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
) -> Helpers<'ctx> {
    let i64_t = context.i64_type();
    let ptr_t = context.ptr_type(AddressSpace::default());

    let alloc_cons_type =
        i64_t.fn_type(&[ptr_t.into(), i64_t.into(), i64_t.into()], false);
    let alloc_cons = module.add_function(
        "ncl_alloc_cons",
        alloc_cons_type,
        Some(Linkage::External),
    );

    // ncl_call(mutator, sym_word, args_ptr, n_args) -> u64
    let call_type = i64_t.fn_type(
        &[ptr_t.into(), i64_t.into(), ptr_t.into(), i64_t.into()],
        false,
    );
    let call_fn = module.add_function("ncl_call", call_type, Some(Linkage::External));

    // ncl_load_value(mutator, sym_word) -> u64
    let load_value_type = i64_t.fn_type(&[ptr_t.into(), i64_t.into()], false);
    let load_value = module.add_function(
        "ncl_load_value",
        load_value_type,
        Some(Linkage::External),
    );

    // ncl_store_value(mutator, sym_word, new_value) -> u64
    let store_value_type =
        i64_t.fn_type(&[ptr_t.into(), i64_t.into(), i64_t.into()], false);
    let store_value = module.add_function(
        "ncl_store_value",
        store_value_type,
        Some(Linkage::External),
    );

    // ncl_funcall(mutator, fn_word, args_ptr, n_args) -> u64
    let funcall_type = i64_t.fn_type(
        &[ptr_t.into(), i64_t.into(), ptr_t.into(), i64_t.into()],
        false,
    );
    let funcall_fn =
        module.add_function("ncl_funcall", funcall_type, Some(Linkage::External));

    // ncl_make_closure(mutator, code_ptr, arity, captures, n_caps) -> u64
    let make_closure_type = i64_t.fn_type(
        &[
            ptr_t.into(),
            i64_t.into(),
            i64_t.into(),
            ptr_t.into(),
            i64_t.into(),
        ],
        false,
    );
    let make_closure = module.add_function(
        "ncl_make_closure",
        make_closure_type,
        Some(Linkage::External),
    );

    // ncl_load_function(mutator, sym_word) -> u64
    let load_function_type = i64_t.fn_type(&[ptr_t.into(), i64_t.into()], false);
    let load_function = module.add_function(
        "ncl_load_function",
        load_function_type,
        Some(Linkage::External),
    );

    // ncl_length(w) -> u64
    let unary_u64_type = i64_t.fn_type(&[i64_t.into()], false);
    let length =
        module.add_function("ncl_length", unary_u64_type, Some(Linkage::External));

    // ncl_equal(a, b) -> u64
    let binary_u64_type = i64_t.fn_type(&[i64_t.into(), i64_t.into()], false);
    let equal =
        module.add_function("ncl_equal", binary_u64_type, Some(Linkage::External));

    // ncl_string_eq(a, b) -> u64
    let string_eq = module.add_function(
        "ncl_string_eq",
        binary_u64_type,
        Some(Linkage::External),
    );

    // ncl_string_char(s, i) -> u64  (i is a tagged fixnum)
    let string_char = module.add_function(
        "ncl_string_char",
        binary_u64_type,
        Some(Linkage::External),
    );

    // ncl_set_car(mutator, cons, value) -> u64
    let mutator_binary_type =
        i64_t.fn_type(&[ptr_t.into(), i64_t.into(), i64_t.into()], false);
    let set_car = module.add_function(
        "ncl_set_car",
        mutator_binary_type,
        Some(Linkage::External),
    );
    let set_cdr = module.add_function(
        "ncl_set_cdr",
        mutator_binary_type,
        Some(Linkage::External),
    );

    // ncl_string_set(s, idx, ch) -> u64  (idx is raw, not tagged)
    let string_set_type =
        i64_t.fn_type(&[i64_t.into(), i64_t.into(), i64_t.into()], false);
    let string_set = module.add_function(
        "ncl_string_set",
        string_set_type,
        Some(Linkage::External),
    );

    // ncl_build_rest_list(mutator, args_ptr, start, n_args) -> u64
    let build_rest_list_type = i64_t.fn_type(
        &[ptr_t.into(), ptr_t.into(), i64_t.into(), i64_t.into()],
        false,
    );
    let build_rest_list = module.add_function(
        "ncl_build_rest_list",
        build_rest_list_type,
        Some(Linkage::External),
    );

    // ncl_apply(mutator, fn_word, prefix_ptr, n_prefix, tail_list) -> u64
    let apply_type = i64_t.fn_type(
        &[
            ptr_t.into(),
            i64_t.into(),
            ptr_t.into(),
            i64_t.into(),
            i64_t.into(),
        ],
        false,
    );
    let apply =
        module.add_function("ncl_apply", apply_type, Some(Linkage::External));

    // ncl_lookup_keyword(args_ptr, key_start, n_args, keyword) -> u64
    let lookup_keyword_type = i64_t.fn_type(
        &[ptr_t.into(), i64_t.into(), i64_t.into(), i64_t.into()],
        false,
    );
    let lookup_keyword = module.add_function(
        "ncl_lookup_keyword",
        lookup_keyword_type,
        Some(Linkage::External),
    );

    // ncl_set_mv_single(value) -> ()
    let set_mv_single_type = context.void_type().fn_type(&[i64_t.into()], false);
    let set_mv_single = module.add_function(
        "ncl_set_mv_single",
        set_mv_single_type,
        Some(Linkage::External),
    );

    // ncl_set_mv_many(args_ptr, n) -> ()
    let set_mv_many_type =
        context.void_type().fn_type(&[ptr_t.into(), i64_t.into()], false);
    let set_mv_many = module.add_function(
        "ncl_set_mv_many",
        set_mv_many_type,
        Some(Linkage::External),
    );

    // ncl_aref_generic(v, i) -> u64
    let aref_generic_type =
        i64_t.fn_type(&[i64_t.into(), i64_t.into()], false);
    let aref_generic = module.add_function(
        "ncl_aref_generic",
        aref_generic_type,
        Some(Linkage::External),
    );

    // ncl_aset_generic(mutator, v, i, val) -> u64
    let aset_generic_type = i64_t.fn_type(
        &[ptr_t.into(), i64_t.into(), i64_t.into(), i64_t.into()],
        false,
    );
    let aset_generic = module.add_function(
        "ncl_aset_generic",
        aset_generic_type,
        Some(Linkage::External),
    );

    // ncl_abort_pending() -> i32  — call-site check for the
    // non-local exit flag. Lowered after every Lisp call.
    let i32_t = context.i32_type();
    let abort_pending_type = i32_t.fn_type(&[], false);
    let abort_pending = module.add_function(
        "ncl_abort_pending",
        abort_pending_type,
        Some(Linkage::External),
    );

    Helpers {
        alloc_cons,
        call_fn,
        funcall_fn,
        make_closure,
        load_value,
        load_function,
        store_value,
        length,
        equal,
        string_eq,
        string_char,
        set_car,
        set_cdr,
        string_set,
        build_rest_list,
        apply,
        lookup_keyword,
        set_mv_single,
        set_mv_many,
        aref_generic,
        aset_generic,
        abort_pending,
    }
}

fn register_runtime_helpers(engine: &ExecutionEngine<'_>, helpers: &Helpers<'_>) {
    engine.add_global_mapping(&helpers.alloc_cons, ncl_alloc_cons as *const () as usize);
    engine.add_global_mapping(&helpers.call_fn, ncl_call as *const () as usize);
    engine.add_global_mapping(&helpers.funcall_fn, ncl_funcall as *const () as usize);
    engine.add_global_mapping(&helpers.make_closure, ncl_make_closure as *const () as usize);
    engine.add_global_mapping(&helpers.load_value, ncl_load_value as *const () as usize);
    engine.add_global_mapping(&helpers.load_function, ncl_load_function as *const () as usize);
    engine.add_global_mapping(&helpers.store_value, ncl_store_value as *const () as usize);
    engine.add_global_mapping(&helpers.length, ncl_length as *const () as usize);
    engine.add_global_mapping(&helpers.equal, ncl_equal as *const () as usize);
    engine.add_global_mapping(&helpers.string_eq, ncl_string_eq as *const () as usize);
    engine.add_global_mapping(&helpers.string_char, ncl_string_char as *const () as usize);
    engine.add_global_mapping(&helpers.set_car, ncl_set_car as *const () as usize);
    engine.add_global_mapping(&helpers.set_cdr, ncl_set_cdr as *const () as usize);
    engine.add_global_mapping(&helpers.string_set, ncl_string_set as *const () as usize);
    engine.add_global_mapping(&helpers.build_rest_list, ncl_build_rest_list as *const () as usize);
    engine.add_global_mapping(&helpers.apply, ncl_apply as *const () as usize);
    engine.add_global_mapping(&helpers.lookup_keyword, ncl_lookup_keyword as *const () as usize);
    engine.add_global_mapping(&helpers.set_mv_single, ncl_set_mv_single as *const () as usize);
    engine.add_global_mapping(&helpers.set_mv_many, ncl_set_mv_many as *const () as usize);
    engine.add_global_mapping(&helpers.aref_generic, ncl_aref_generic as *const () as usize);
    engine.add_global_mapping(&helpers.aset_generic, ncl_aset_generic as *const () as usize);
    engine.add_global_mapping(&helpers.abort_pending, ncl_abort_pending as *const () as usize);
}

/// Emit the post-call abort check. Calls ncl_abort_pending; if
/// non-zero, branches to a fresh "early return" block that
/// returns NIL; otherwise positions at a fresh "continue" block
/// where the caller's IR can keep building. The caller's
/// pre-check `result` IntValue stays SSA-accessible in the
/// continue block (LLVM dominance lets us still use it).
///
/// This is the call-site instrumentation that makes (return-from
/// …) and (error …) actually abort the rest of the body, not
/// just set a flag the next block boundary will pick up. Same
/// flag mechanism, plumbed through every Lisp call.
fn emit_post_call_abort_check<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    function: &FunctionValue<'ctx>,
    helpers: &Helpers<'ctx>,
) -> Result<(), String> {
    let i32_t = context.i32_type();
    let i64_t = context.i64_type();
    let call = builder
        .build_call(helpers.abort_pending, &[], "abort_check")
        .map_err(|e| format!("call abort_pending: {e}"))?;
    let pending = call
        .try_as_basic_value()
        .unwrap_basic()
        .into_int_value();
    let zero = i32_t.const_int(0, false);
    let is_pending = builder
        .build_int_compare(IntPredicate::NE, pending, zero, "is_abort")
        .map_err(|e| format!("icmp: {e}"))?;
    let exit_bb = context.append_basic_block(*function, "abort_exit");
    let cont_bb = context.append_basic_block(*function, "abort_cont");
    builder
        .build_conditional_branch(is_pending, exit_bb, cont_bb)
        .map_err(|e| format!("cond br: {e}"))?;
    // Early-return block: NIL is a placeholder — the value is
    // never observed because every function in the abort chain
    // also returns early; the matching block / handler-case at
    // the outer end reads the real value from the abort state.
    builder.position_at_end(exit_bb);
    let nil = i64_t.const_int(Word::NIL.raw(), false);
    builder
        .build_return(Some(&nil))
        .map_err(|e| format!("early ret: {e}"))?;
    // Resume normal IR from the continue block.
    builder.position_at_end(cont_bb);
    Ok(())
}

/// Convert an i1 comparison result to a tagged Word (T or NIL).
fn emit_bool_select<'ctx>(
    builder: &Builder<'ctx>,
    cmp: IntValue<'ctx>,
    i64_t: inkwell::types::IntType<'ctx>,
) -> Result<IntValue<'ctx>, String> {
    let t = i64_t.const_int(Word::T.raw(), false);
    let nil = i64_t.const_int(Word::NIL.raw(), false);
    builder
        .build_select(cmp, t, nil, "bool_result")
        .map_err(|e| format!("select: {e}"))
        .map(|v| v.into_int_value())
}

/// Emit a binary integer comparison, return Word::T or Word::NIL.
#[allow(clippy::too_many_arguments)]
fn emit_cmp<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    function: &FunctionValue<'ctx>,
    helpers: &Helpers<'ctx>,
    arity: u32,
    locals: &mut Vec<IntValue<'ctx>>,
    a: &Expr,
    b: &Expr,
    pred: IntPredicate,
) -> Result<IntValue<'ctx>, String> {
    let lhs = emit_expr(context, builder, function, helpers, arity, locals, a)?;
    let rhs = emit_expr(context, builder, function, helpers, arity, locals, b)?;
    let cmp = builder
        .build_int_compare(pred, lhs, rhs, "cmp")
        .map_err(|e| format!("icmp: {e}"))?;
    let i64_t = context.i64_type();
    emit_bool_select(builder, cmp, i64_t)
}

/// Tag-equality predicate.
#[allow(clippy::too_many_arguments)]
fn emit_tag_eq<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    function: &FunctionValue<'ctx>,
    helpers: &Helpers<'ctx>,
    arity: u32,
    locals: &mut Vec<IntValue<'ctx>>,
    x: &Expr,
    tag: Tag,
) -> Result<IntValue<'ctx>, String> {
    let i64_t = context.i64_type();
    let v = emit_expr(context, builder, function, helpers, arity, locals, x)?;
    let mask = i64_t.const_int(0b111, false);
    let tag_bits = builder
        .build_and(v, mask, "tag_bits")
        .map_err(|e| format!("and: {e}"))?;
    let expected = i64_t.const_int(tag as u64, false);
    let cmp = builder
        .build_int_compare(IntPredicate::EQ, tag_bits, expected, "tag_eq")
        .map_err(|e| format!("icmp: {e}"))?;
    emit_bool_select(builder, cmp, i64_t)
}

fn emit_expr<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    function: &FunctionValue<'ctx>,
    helpers: &Helpers<'ctx>,
    arity: u32,
    locals: &mut Vec<IntValue<'ctx>>,
    expr: &Expr,
) -> Result<IntValue<'ctx>, String> {
    let i64_t = context.i64_type();
    let ptr_t = context.ptr_type(AddressSpace::default());
    match expr {
        Expr::Const(n) => Ok(i64_t.const_int(Word::fixnum(*n).raw(), false)),
        Expr::Word(w) => Ok(i64_t.const_int(*w, false)),
        Expr::Nil => Ok(i64_t.const_int(Word::NIL.raw(), false)),
        Expr::True => Ok(i64_t.const_int(Word::T.raw(), false)),
        Expr::Local(idx) => {
            locals
                .get(*idx)
                .copied()
                .ok_or_else(|| format!("Local({idx}) out of range — only {} locals in scope", locals.len()))
        }
        Expr::Progn(forms) => {
            if forms.is_empty() {
                return Ok(i64_t.const_int(Word::NIL.raw(), false));
            }
            let mut last = i64_t.const_int(Word::NIL.raw(), false);
            for f in forms {
                last = emit_expr(context, builder, function, helpers, arity, locals, f)?;
            }
            Ok(last)
        }
        Expr::Let { bindings, body } => {
            let saved = locals.len();
            for binding in bindings {
                // Evaluate in CURRENT locals (outer scope) — let's
                // parallel-binding semantics. The binding doesn't
                // see itself or sibling bindings.
                let v = emit_expr(
                    context, builder, function, helpers, arity, locals, binding,
                )?;
                locals.push(v);
            }
            let result = emit_expr(
                context, builder, function, helpers, arity, locals, body,
            )?;
            locals.truncate(saved);
            Ok(result)
        }
        Expr::Param(idx) => {
            if *idx as u32 >= arity {
                return Err(format!(
                    "Param({idx}) out of range for arity {arity}"
                ));
            }
            // args_ptr is param 2 (mutator=0, env=1, args=2, n_args=3).
            let args_ptr = function
                .get_nth_param(2)
                .unwrap()
                .into_pointer_value();
            let i = i64_t.const_int(*idx as u64, false);
            let elem_ptr = unsafe {
                builder
                    .build_in_bounds_gep(i64_t, args_ptr, &[i], "arg_ptr")
                    .map_err(|e| format!("gep arg: {e}"))?
            };
            let val = builder
                .build_load(i64_t, elem_ptr, "arg")
                .map_err(|e| format!("load arg: {e}"))?;
            Ok(val.into_int_value())
        }
        Expr::BindRest(start) => {
            // ncl_build_rest_list(mutator, args_ptr, start, n_args)
            let mutator_arg = function.get_nth_param(0).unwrap();
            let args_ptr = function.get_nth_param(2).unwrap();
            let n_args = function.get_nth_param(3).unwrap();
            let start_const = i64_t.const_int(*start as u64, false);
            let call = builder
                .build_call(
                    helpers.build_rest_list,
                    &[
                        mutator_arg.into(),
                        args_ptr.into(),
                        start_const.into(),
                        n_args.into(),
                    ],
                    "rest",
                )
                .map_err(|e| format!("build_call build_rest_list: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::OptArg { idx, default } => {
            // if (n_args > idx) args[idx] else <default>
            let n_args = function.get_nth_param(3).unwrap().into_int_value();
            let args_ptr = function.get_nth_param(2).unwrap().into_pointer_value();
            let idx_const = i64_t.const_int(*idx as u64, false);
            let cond = builder
                .build_int_compare(
                    IntPredicate::UGT,
                    n_args,
                    idx_const,
                    "opt_present",
                )
                .map_err(|e| format!("opt cmp: {e}"))?;
            let then_bb = context.append_basic_block(*function, "opt_supplied");
            let else_bb = context.append_basic_block(*function, "opt_default");
            let cont_bb = context.append_basic_block(*function, "opt_cont");
            builder
                .build_conditional_branch(cond, then_bb, else_bb)
                .map_err(|e| format!("opt br: {e}"))?;
            // then: load args[idx]
            builder.position_at_end(then_bb);
            let elem_ptr = unsafe {
                builder
                    .build_in_bounds_gep(i64_t, args_ptr, &[idx_const], "opt_arg_ptr")
                    .map_err(|e| format!("opt gep: {e}"))?
            };
            let supplied = builder
                .build_load(i64_t, elem_ptr, "opt_arg")
                .map_err(|e| format!("opt load: {e}"))?
                .into_int_value();
            let then_end = builder.get_insert_block().unwrap();
            builder
                .build_unconditional_branch(cont_bb)
                .map_err(|e| format!("opt then br: {e}"))?;
            // else: lower the default expression
            builder.position_at_end(else_bb);
            let defaulted = emit_expr(
                context, builder, function, helpers, arity, locals, default,
            )?;
            let else_end = builder.get_insert_block().unwrap();
            builder
                .build_unconditional_branch(cont_bb)
                .map_err(|e| format!("opt else br: {e}"))?;
            // continuation: phi
            builder.position_at_end(cont_bb);
            let phi = builder
                .build_phi(i64_t, "opt_phi")
                .map_err(|e| format!("opt phi: {e}"))?;
            phi.add_incoming(&[(&supplied, then_end), (&defaulted, else_end)]);
            Ok(phi.as_basic_value().into_int_value())
        }
        Expr::KeyArg { keyword_word, key_start, default } => {
            // v = ncl_lookup_keyword(args, key_start, n_args, keyword)
            // if (v == UNBOUND) <default> else v
            let args_ptr = function.get_nth_param(2).unwrap();
            let n_args = function.get_nth_param(3).unwrap();
            let key_start_const = i64_t.const_int(*key_start as u64, false);
            let kw_const = i64_t.const_int(*keyword_word, false);
            let call = builder
                .build_call(
                    helpers.lookup_keyword,
                    &[
                        args_ptr.into(),
                        key_start_const.into(),
                        n_args.into(),
                        kw_const.into(),
                    ],
                    "key_lookup",
                )
                .map_err(|e| format!("key lookup call: {e}"))?;
            let raw = call.try_as_basic_value().unwrap_basic().into_int_value();
            let unbound = i64_t.const_int(Word::UNBOUND.raw(), false);
            let cond = builder
                .build_int_compare(
                    IntPredicate::EQ,
                    raw,
                    unbound,
                    "key_missing",
                )
                .map_err(|e| format!("key cmp: {e}"))?;
            let then_bb = context.append_basic_block(*function, "key_default");
            let else_bb = context.append_basic_block(*function, "key_supplied");
            let cont_bb = context.append_basic_block(*function, "key_cont");
            builder
                .build_conditional_branch(cond, then_bb, else_bb)
                .map_err(|e| format!("key br: {e}"))?;
            // then: lower the default expression
            builder.position_at_end(then_bb);
            let defaulted = emit_expr(
                context, builder, function, helpers, arity, locals, default,
            )?;
            let then_end = builder.get_insert_block().unwrap();
            builder
                .build_unconditional_branch(cont_bb)
                .map_err(|e| format!("key then br: {e}"))?;
            // else: use the supplied raw value
            builder.position_at_end(else_bb);
            let else_end = builder.get_insert_block().unwrap();
            builder
                .build_unconditional_branch(cont_bb)
                .map_err(|e| format!("key else br: {e}"))?;
            // continuation: phi
            builder.position_at_end(cont_bb);
            let phi = builder
                .build_phi(i64_t, "key_phi")
                .map_err(|e| format!("key phi: {e}"))?;
            phi.add_incoming(&[(&defaulted, then_end), (&raw, else_end)]);
            Ok(phi.as_basic_value().into_int_value())
        }
        Expr::Values(vals) => {
            // Evaluate each val, store into a stack-alloca'd buffer,
            // call ncl_set_mv_many, return vals[0] (or NIL).
            if vals.is_empty() {
                let nil = i64_t.const_int(Word::NIL.raw(), false);
                let n = i64_t.const_int(0, false);
                let null_ptr = ptr_t.const_null();
                builder
                    .build_call(
                        helpers.set_mv_many,
                        &[null_ptr.into(), n.into()],
                        "set_mv_zero",
                    )
                    .map_err(|e| format!("call set_mv_many: {e}"))?;
                return Ok(nil);
            }
            let n = vals.len() as u64;
            let n_const = i64_t.const_int(n, false);
            let buf = builder
                .build_array_alloca(i64_t, n_const, "mv_buf")
                .map_err(|e| format!("alloca mv_buf: {e}"))?;
            let mut lowered_vals = Vec::with_capacity(vals.len());
            for (i, v) in vals.iter().enumerate() {
                let lv = emit_expr(context, builder, function, helpers, arity, locals, v)?;
                let idx = i64_t.const_int(i as u64, false);
                let slot = unsafe {
                    builder
                        .build_in_bounds_gep(i64_t, buf, &[idx], "mv_slot")
                        .map_err(|e| format!("gep mv_slot: {e}"))?
                };
                builder
                    .build_store(slot, lv)
                    .map_err(|e| format!("store mv_slot: {e}"))?;
                lowered_vals.push(lv);
            }
            builder
                .build_call(
                    helpers.set_mv_many,
                    &[buf.into(), n_const.into()],
                    "set_mv",
                )
                .map_err(|e| format!("call set_mv_many: {e}"))?;
            Ok(lowered_vals[0])
        }
        Expr::EnsureSingleMv(primary) => {
            let v = emit_expr(context, builder, function, helpers, arity, locals, primary)?;
            builder
                .build_call(helpers.set_mv_single, &[v.into()], "ensure_single")
                .map_err(|e| format!("call set_mv_single: {e}"))?;
            Ok(v)
        }
        Expr::ClosureRef(idx) => {
            // env (function param 1) is a Vector-tagged Word. Untag,
            // skip the header (cell 0), read cell idx+1.
            let env_word = function.get_nth_param(1).unwrap().into_int_value();
            let mask = i64_t.const_int(!0b111u64, false);
            let untagged = builder
                .build_and(env_word, mask, "untag_env")
                .map_err(|e| format!("and: {e}"))?;
            let ptr = builder
                .build_int_to_ptr(untagged, ptr_t, "env_ptr")
                .map_err(|e| format!("int_to_ptr: {e}"))?;
            // Skip header (cell 0), index idx+1.
            let i = i64_t.const_int((*idx as u64) + 1, false);
            let cell_ptr = unsafe {
                builder
                    .build_gep(i64_t, ptr, &[i], "env_cell_ptr")
                    .map_err(|e| format!("gep env: {e}"))?
            };
            let val = builder
                .build_load(i64_t, cell_ptr, "env_val")
                .map_err(|e| format!("load env: {e}"))?;
            Ok(val.into_int_value())
        }
        Expr::Lambda { arity: lam_arity, body, captures } => {
            // 1. JIT-compile the body as a separate function.
            //    Recursive call to build_lisp_function.
            let code_addr = build_lisp_function(body, *lam_arity)?;
            // 2. Evaluate each capture expression in CURRENT scope.
            let cap_vals: Vec<IntValue> = captures
                .iter()
                .map(|c| emit_expr(context, builder, function, helpers, arity, locals, c))
                .collect::<Result<_, _>>()?;
            // 3. Stack-alloc array for the captured values.
            let n = cap_vals.len();
            let arr_type = i64_t.array_type(n.max(1) as u32);
            let arr_alloca = builder
                .build_alloca(arr_type, "cap_buf")
                .map_err(|e| format!("alloca: {e}"))?;
            for (i, val) in cap_vals.iter().enumerate() {
                let elem = unsafe {
                    builder
                        .build_in_bounds_gep(
                            arr_type,
                            arr_alloca,
                            &[
                                i64_t.const_int(0, false),
                                i64_t.const_int(i as u64, false),
                            ],
                            "cap_slot",
                        )
                        .map_err(|e| format!("gep cap: {e}"))?
                };
                builder
                    .build_store(elem, *val)
                    .map_err(|e| format!("store cap: {e}"))?;
            }
            // 4. Call ncl_make_closure(mutator, code, arity, caps, n).
            let mutator_arg = function.get_nth_param(0).unwrap();
            let code_const = i64_t.const_int(code_addr as u64, false);
            let arity_const = i64_t.const_int(*lam_arity as u64, false);
            let n_const = i64_t.const_int(n as u64, false);
            let call = builder
                .build_call(
                    helpers.make_closure,
                    &[
                        mutator_arg.into(),
                        code_const.into(),
                        arity_const.into(),
                        arr_alloca.into(),
                        n_const.into(),
                    ],
                    "lambda",
                )
                .map_err(|e| format!("build_call make_closure: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::Funcall { fn_expr, args } => {
            let fn_val =
                emit_expr(context, builder, function, helpers, arity, locals, fn_expr)?;
            let arg_vals: Vec<IntValue> = args
                .iter()
                .map(|a| emit_expr(context, builder, function, helpers, arity, locals, a))
                .collect::<Result<_, _>>()?;
            let n = arg_vals.len();
            let arr_type = i64_t.array_type(n.max(1) as u32);
            let arr_alloca = builder
                .build_alloca(arr_type, "funcall_args")
                .map_err(|e| format!("alloca: {e}"))?;
            for (i, val) in arg_vals.iter().enumerate() {
                let elem = unsafe {
                    builder
                        .build_in_bounds_gep(
                            arr_type,
                            arr_alloca,
                            &[
                                i64_t.const_int(0, false),
                                i64_t.const_int(i as u64, false),
                            ],
                            "fc_slot",
                        )
                        .map_err(|e| format!("gep fc: {e}"))?
                };
                builder
                    .build_store(elem, *val)
                    .map_err(|e| format!("store fc: {e}"))?;
            }
            let mutator_arg = function.get_nth_param(0).unwrap();
            let n_const = i64_t.const_int(n as u64, false);
            let call = builder
                .build_call(
                    helpers.funcall_fn,
                    &[
                        mutator_arg.into(),
                        fn_val.into(),
                        arr_alloca.into(),
                        n_const.into(),
                    ],
                    "funcall_result",
                )
                .map_err(|e| format!("build_call funcall: {e}"))?;
            let result = call.try_as_basic_value().unwrap_basic().into_int_value();
            emit_post_call_abort_check(context, builder, function, helpers)?;
            Ok(result)
        }
        Expr::Apply { fn_expr, prefix, tail } => {
            let fn_val =
                emit_expr(context, builder, function, helpers, arity, locals, fn_expr)?;
            let prefix_vals: Vec<IntValue> = prefix
                .iter()
                .map(|a| emit_expr(context, builder, function, helpers, arity, locals, a))
                .collect::<Result<_, _>>()?;
            let tail_val =
                emit_expr(context, builder, function, helpers, arity, locals, tail)?;
            let n = prefix_vals.len();
            // Allocate even when n=0, but use a single-slot dummy
            // — the runtime ignores prefix when n_prefix=0, so the
            // pointer doesn't matter. Pass the address anyway, it's
            // simpler than branching on emptiness.
            let arr_type = i64_t.array_type(n.max(1) as u32);
            let arr_alloca = builder
                .build_alloca(arr_type, "apply_prefix")
                .map_err(|e| format!("alloca: {e}"))?;
            for (i, val) in prefix_vals.iter().enumerate() {
                let elem = unsafe {
                    builder
                        .build_in_bounds_gep(
                            arr_type,
                            arr_alloca,
                            &[
                                i64_t.const_int(0, false),
                                i64_t.const_int(i as u64, false),
                            ],
                            "ap_slot",
                        )
                        .map_err(|e| format!("gep ap: {e}"))?
                };
                builder
                    .build_store(elem, *val)
                    .map_err(|e| format!("store ap: {e}"))?;
            }
            let mutator_arg = function.get_nth_param(0).unwrap();
            let n_const = i64_t.const_int(n as u64, false);
            let call = builder
                .build_call(
                    helpers.apply,
                    &[
                        mutator_arg.into(),
                        fn_val.into(),
                        arr_alloca.into(),
                        n_const.into(),
                        tail_val.into(),
                    ],
                    "apply_result",
                )
                .map_err(|e| format!("build_call apply: {e}"))?;
            let result = call.try_as_basic_value().unwrap_basic().into_int_value();
            emit_post_call_abort_check(context, builder, function, helpers)?;
            Ok(result)
        }
        Expr::Add(a, b) => {
            let lhs = emit_expr(context, builder, function, helpers, arity, locals, a)?;
            let rhs = emit_expr(context, builder, function, helpers, arity, locals, b)?;
            builder
                .build_int_add(lhs, rhs, "add")
                .map_err(|e| format!("build_int_add: {e}"))
        }
        Expr::Sub(a, b) => {
            let lhs = emit_expr(context, builder, function, helpers, arity, locals, a)?;
            let rhs = emit_expr(context, builder, function, helpers, arity, locals, b)?;
            builder
                .build_int_sub(lhs, rhs, "sub")
                .map_err(|e| format!("build_int_sub: {e}"))
        }
        Expr::Mul(a, b) => {
            let lhs = emit_expr(context, builder, function, helpers, arity, locals, a)?;
            let rhs = emit_expr(context, builder, function, helpers, arity, locals, b)?;
            let three = i64_t.const_int(3, false);
            let rhs_untagged = builder
                .build_right_shift(rhs, three, true, "untag_rhs")
                .map_err(|e| format!("ashr: {e}"))?;
            builder
                .build_int_mul(lhs, rhs_untagged, "mul")
                .map_err(|e| format!("build_int_mul: {e}"))
        }
        Expr::Truncate(a, b) => {
            // Untag both, sdiv, retag.
            let lhs = emit_expr(context, builder, function, helpers, arity, locals, a)?;
            let rhs = emit_expr(context, builder, function, helpers, arity, locals, b)?;
            let three = i64_t.const_int(3, false);
            let lhs_u = builder
                .build_right_shift(lhs, three, true, "untag_lhs")
                .map_err(|e| format!("ashr: {e}"))?;
            let rhs_u = builder
                .build_right_shift(rhs, three, true, "untag_rhs")
                .map_err(|e| format!("ashr: {e}"))?;
            let q = builder
                .build_int_signed_div(lhs_u, rhs_u, "trunc")
                .map_err(|e| format!("build_int_signed_div: {e}"))?;
            builder
                .build_left_shift(q, three, "trunc_tag")
                .map_err(|e| format!("shl: {e}"))
        }
        Expr::Rem(a, b) => {
            // srem of two tagged fixnums returns an already-tagged
            // fixnum: (a<<3) srem (b<<3) = (a rem b) << 3.
            let lhs = emit_expr(context, builder, function, helpers, arity, locals, a)?;
            let rhs = emit_expr(context, builder, function, helpers, arity, locals, b)?;
            builder
                .build_int_signed_rem(lhs, rhs, "rem")
                .map_err(|e| format!("build_int_signed_rem: {e}"))
        }
        Expr::Cons(car, cdr) => {
            let car_val = emit_expr(context, builder, function, helpers, arity, locals, car)?;
            let cdr_val = emit_expr(context, builder, function, helpers, arity, locals, cdr)?;
            let mutator_arg = function.get_nth_param(0).unwrap();
            let call = builder
                .build_call(
                    helpers.alloc_cons,
                    &[mutator_arg.into(), car_val.into(), cdr_val.into()],
                    "cons",
                )
                .map_err(|e| format!("build_call alloc_cons: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::Car(x) => {
            let cons_val = emit_expr(context, builder, function, helpers, arity, locals, x)?;
            let mask = i64_t.const_int(!0b111u64, false);
            let untagged = builder
                .build_and(cons_val, mask, "untag_cons")
                .map_err(|e| format!("and: {e}"))?;
            let ptr = builder
                .build_int_to_ptr(untagged, ptr_t, "as_ptr")
                .map_err(|e| format!("int_to_ptr: {e}"))?;
            let loaded = builder
                .build_load(i64_t, ptr, "car")
                .map_err(|e| format!("load car: {e}"))?;
            Ok(loaded.into_int_value())
        }
        Expr::Cdr(x) => {
            let cons_val = emit_expr(context, builder, function, helpers, arity, locals, x)?;
            let mask = i64_t.const_int(!0b111u64, false);
            let untagged = builder
                .build_and(cons_val, mask, "untag_cons")
                .map_err(|e| format!("and: {e}"))?;
            let ptr = builder
                .build_int_to_ptr(untagged, ptr_t, "as_ptr")
                .map_err(|e| format!("int_to_ptr: {e}"))?;
            let one = i64_t.const_int(1, false);
            let cdr_ptr = unsafe {
                builder
                    .build_gep(i64_t, ptr, &[one], "cdr_ptr")
                    .map_err(|e| format!("gep cdr: {e}"))?
            };
            let loaded = builder
                .build_load(i64_t, cdr_ptr, "cdr")
                .map_err(|e| format!("load cdr: {e}"))?;
            Ok(loaded.into_int_value())
        }
        Expr::Eq(a, b) => emit_cmp(context, builder, function, helpers, arity, locals, a, b, IntPredicate::EQ),
        Expr::Lt(a, b) => emit_cmp(context, builder, function, helpers, arity, locals, a, b, IntPredicate::SLT),
        Expr::Gt(a, b) => emit_cmp(context, builder, function, helpers, arity, locals, a, b, IntPredicate::SGT),
        Expr::Le(a, b) => emit_cmp(context, builder, function, helpers, arity, locals, a, b, IntPredicate::SLE),
        Expr::Ge(a, b) => emit_cmp(context, builder, function, helpers, arity, locals, a, b, IntPredicate::SGE),
        Expr::NumEq(a, b) => emit_cmp(context, builder, function, helpers, arity, locals, a, b, IntPredicate::EQ),
        Expr::IsNull(x) => {
            let v = emit_expr(context, builder, function, helpers, arity, locals, x)?;
            let nil = i64_t.const_int(Word::NIL.raw(), false);
            let cmp = builder
                .build_int_compare(IntPredicate::EQ, v, nil, "is_null")
                .map_err(|e| format!("icmp: {e}"))?;
            emit_bool_select(builder, cmp, i64_t)
        }
        Expr::IsCons(x) => emit_tag_eq(context, builder, function, helpers, arity, locals, x, Tag::Cons),
        Expr::IsAtom(x) => {
            let v = emit_expr(context, builder, function, helpers, arity, locals, x)?;
            let mask = i64_t.const_int(0b111, false);
            let tag_bits = builder
                .build_and(v, mask, "tag_bits")
                .map_err(|e| format!("and: {e}"))?;
            let cons_tag = i64_t.const_int(Tag::Cons as u64, false);
            let is_cons = builder
                .build_int_compare(IntPredicate::EQ, tag_bits, cons_tag, "is_cons")
                .map_err(|e| format!("icmp: {e}"))?;
            let true_const = context.bool_type().const_int(1, false);
            let is_atom = builder
                .build_xor(is_cons, true_const, "is_atom")
                .map_err(|e| format!("xor: {e}"))?;
            emit_bool_select(builder, is_atom, i64_t)
        }
        Expr::IsListp(x) => {
            let v = emit_expr(context, builder, function, helpers, arity, locals, x)?;
            let nil = i64_t.const_int(Word::NIL.raw(), false);
            let is_nil = builder
                .build_int_compare(IntPredicate::EQ, v, nil, "is_nil")
                .map_err(|e| format!("icmp: {e}"))?;
            let mask = i64_t.const_int(0b111, false);
            let tag_bits = builder
                .build_and(v, mask, "tag_bits")
                .map_err(|e| format!("and: {e}"))?;
            let cons_tag = i64_t.const_int(Tag::Cons as u64, false);
            let is_cons = builder
                .build_int_compare(IntPredicate::EQ, tag_bits, cons_tag, "is_cons")
                .map_err(|e| format!("icmp: {e}"))?;
            let either = builder
                .build_or(is_nil, is_cons, "is_listp")
                .map_err(|e| format!("or: {e}"))?;
            emit_bool_select(builder, either, i64_t)
        }
        Expr::If(cond, then_branch, else_branch) => {
            let cond_val = emit_expr(context, builder, function, helpers, arity, locals, cond)?;
            let nil_word = i64_t.const_int(Word::NIL.raw(), false);
            let is_truthy = builder
                .build_int_compare(IntPredicate::NE, cond_val, nil_word, "is_truthy")
                .map_err(|e| format!("icmp: {e}"))?;

            let then_block = context.append_basic_block(*function, "then");
            let else_block = context.append_basic_block(*function, "else");
            let merge_block = context.append_basic_block(*function, "merge");

            builder
                .build_conditional_branch(is_truthy, then_block, else_block)
                .map_err(|e| format!("br: {e}"))?;

            builder.position_at_end(then_block);
            let then_val = emit_expr(context, builder, function, helpers, arity, locals, then_branch)?;
            let then_end = builder.get_insert_block().unwrap();
            builder
                .build_unconditional_branch(merge_block)
                .map_err(|e| format!("br: {e}"))?;

            builder.position_at_end(else_block);
            let else_val = emit_expr(context, builder, function, helpers, arity, locals, else_branch)?;
            let else_end = builder.get_insert_block().unwrap();
            builder
                .build_unconditional_branch(merge_block)
                .map_err(|e| format!("br: {e}"))?;

            builder.position_at_end(merge_block);
            let phi = builder
                .build_phi(i64_t, "if_result")
                .map_err(|e| format!("phi: {e}"))?;
            phi.add_incoming(&[(&then_val, then_end), (&else_val, else_end)]);
            Ok(phi.as_basic_value().into_int_value())
        }
        Expr::LoadGlobal(sym_word) => {
            let mutator_arg = function.get_nth_param(0).unwrap();
            let sym_const = i64_t.const_int(*sym_word, false);
            let call = builder
                .build_call(
                    helpers.load_value,
                    &[mutator_arg.into(), sym_const.into()],
                    "load_value",
                )
                .map_err(|e| format!("build_call load_value: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::LoadFunction(sym_word) => {
            let mutator_arg = function.get_nth_param(0).unwrap();
            let sym_const = i64_t.const_int(*sym_word, false);
            let call = builder
                .build_call(
                    helpers.load_function,
                    &[mutator_arg.into(), sym_const.into()],
                    "load_fn",
                )
                .map_err(|e| format!("build_call load_function: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::Length(x) => {
            let v = emit_expr(context, builder, function, helpers, arity, locals, x)?;
            let call = builder
                .build_call(helpers.length, &[v.into()], "length")
                .map_err(|e| format!("build_call length: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::Equal(a, b) => {
            let va = emit_expr(context, builder, function, helpers, arity, locals, a)?;
            let vb = emit_expr(context, builder, function, helpers, arity, locals, b)?;
            let call = builder
                .build_call(helpers.equal, &[va.into(), vb.into()], "equal")
                .map_err(|e| format!("build_call equal: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::StringEq(a, b) => {
            let va = emit_expr(context, builder, function, helpers, arity, locals, a)?;
            let vb = emit_expr(context, builder, function, helpers, arity, locals, b)?;
            let call = builder
                .build_call(helpers.string_eq, &[va.into(), vb.into()], "str_eq")
                .map_err(|e| format!("build_call string_eq: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::StringChar(s, i) => {
            let vs = emit_expr(context, builder, function, helpers, arity, locals, s)?;
            let vi_tagged = emit_expr(context, builder, function, helpers, arity, locals, i)?;
            // The fixnum is tagged (n << 3); ncl_string_char expects
            // the raw index, so untag with arithmetic shift right.
            let three = i64_t.const_int(3, false);
            let untagged = builder
                .build_right_shift(vi_tagged, three, true, "untag_idx")
                .map_err(|e| format!("ashr: {e}"))?;
            let call = builder
                .build_call(
                    helpers.string_char,
                    &[vs.into(), untagged.into()],
                    "str_char",
                )
                .map_err(|e| format!("build_call string_char: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::SetCar(cons, value) => {
            let vc = emit_expr(context, builder, function, helpers, arity, locals, cons)?;
            let vv = emit_expr(context, builder, function, helpers, arity, locals, value)?;
            let mutator_arg = function.get_nth_param(0).unwrap();
            let call = builder
                .build_call(
                    helpers.set_car,
                    &[mutator_arg.into(), vc.into(), vv.into()],
                    "set_car",
                )
                .map_err(|e| format!("build_call set_car: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::SetCdr(cons, value) => {
            let vc = emit_expr(context, builder, function, helpers, arity, locals, cons)?;
            let vv = emit_expr(context, builder, function, helpers, arity, locals, value)?;
            let mutator_arg = function.get_nth_param(0).unwrap();
            let call = builder
                .build_call(
                    helpers.set_cdr,
                    &[mutator_arg.into(), vc.into(), vv.into()],
                    "set_cdr",
                )
                .map_err(|e| format!("build_call set_cdr: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::SetChar { s, idx, ch } => {
            let vs = emit_expr(context, builder, function, helpers, arity, locals, s)?;
            let vi_tagged = emit_expr(context, builder, function, helpers, arity, locals, idx)?;
            let vch = emit_expr(context, builder, function, helpers, arity, locals, ch)?;
            // Index is a tagged fixnum; ncl_string_set wants raw.
            let three = i64_t.const_int(3, false);
            let untagged = builder
                .build_right_shift(vi_tagged, three, true, "untag_idx")
                .map_err(|e| format!("ashr: {e}"))?;
            let call = builder
                .build_call(
                    helpers.string_set,
                    &[vs.into(), untagged.into(), vch.into()],
                    "str_set",
                )
                .map_err(|e| format!("build_call string_set: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::Aref(v, i) => {
            // ncl_aref_generic(v_word, i_word) — both tagged.
            let vv = emit_expr(context, builder, function, helpers, arity, locals, v)?;
            let vi = emit_expr(context, builder, function, helpers, arity, locals, i)?;
            let call = builder
                .build_call(
                    helpers.aref_generic,
                    &[vv.into(), vi.into()],
                    "aref",
                )
                .map_err(|e| format!("build_call aref_generic: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::SetAref { v, idx, val } => {
            // ncl_aset_generic(mutator, v_word, i_word, val_word).
            let mutator_arg = function.get_nth_param(0).unwrap();
            let vv = emit_expr(context, builder, function, helpers, arity, locals, v)?;
            let vi = emit_expr(context, builder, function, helpers, arity, locals, idx)?;
            let vval = emit_expr(context, builder, function, helpers, arity, locals, val)?;
            let call = builder
                .build_call(
                    helpers.aset_generic,
                    &[mutator_arg.into(), vv.into(), vi.into(), vval.into()],
                    "aset",
                )
                .map_err(|e| format!("build_call aset_generic: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::StoreGlobal { sym_word, value } => {
            let val =
                emit_expr(context, builder, function, helpers, arity, locals, value)?;
            let mutator_arg = function.get_nth_param(0).unwrap();
            let sym_const = i64_t.const_int(*sym_word, false);
            let call = builder
                .build_call(
                    helpers.store_value,
                    &[mutator_arg.into(), sym_const.into(), val.into()],
                    "store_value",
                )
                .map_err(|e| format!("build_call store_value: {e}"))?;
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
        }
        Expr::Call { sym_word, args } => {
            // Evaluate each argument first.
            let arg_vals: Vec<IntValue> = args
                .iter()
                .map(|a| emit_expr(context, builder, function, helpers, arity, locals, a))
                .collect::<Result<_, _>>()?;
            let n = arg_vals.len();

            // Stack-allocate an [N x i64] for the arg array.
            let arr_type = i64_t.array_type(n.max(1) as u32);
            let arr_alloca = builder
                .build_alloca(arr_type, "call_args")
                .map_err(|e| format!("alloca: {e}"))?;

            // Store each evaluated arg into its slot.
            for (i, val) in arg_vals.iter().enumerate() {
                let elem_ptr = unsafe {
                    builder
                        .build_in_bounds_gep(
                            arr_type,
                            arr_alloca,
                            &[
                                i64_t.const_int(0, false),
                                i64_t.const_int(i as u64, false),
                            ],
                            "arg_slot",
                        )
                        .map_err(|e| format!("gep slot: {e}"))?
                };
                builder
                    .build_store(elem_ptr, *val)
                    .map_err(|e| format!("store: {e}"))?;
            }

            // Call ncl_call(mutator, sym_word, args_ptr, n_args).
            let mutator_arg = function.get_nth_param(0).unwrap();
            let sym_const = i64_t.const_int(*sym_word, false);
            let n_const = i64_t.const_int(n as u64, false);
            let call = builder
                .build_call(
                    helpers.call_fn,
                    &[
                        mutator_arg.into(),
                        sym_const.into(),
                        arr_alloca.into(),
                        n_const.into(),
                    ],
                    "call_result",
                )
                .map_err(|e| format!("build_call ncl_call: {e}"))?;
            let result = call.try_as_basic_value().unwrap_basic().into_int_value();
            emit_post_call_abort_check(context, builder, function, helpers)?;
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ncl_runtime::{GcConfig, GcCoordinator, Tag};

    fn small_config() -> GcConfig {
        GcConfig {
            young_bytes: 16 * 1024,
            old_bytes: 16 * 1024,
            static_bytes: 8 * 1024,
            tlab_cells: 64,
        }
    }

    #[test]
    fn three_returns_three() {
        assert_eq!(jit_three().expect("jit_three"), 3);
    }

    #[test]
    fn three_is_repeatable() {
        for _ in 0..3 {
            assert_eq!(jit_three().expect("jit_three"), 3);
        }
    }

    #[test]
    fn add_passes_args_correctly() {
        assert_eq!(jit_add(2, 3).expect("jit_add"), 5);
    }

    fn eval_with_fresh_mutator(expr: &Expr) -> Word {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        jit_eval(expr, &mut m as *mut _).unwrap()
    }

    #[test]
    fn eval_const() {
        assert_eq!(eval_with_fresh_mutator(&Expr::Const(42)).as_fixnum(), Some(42));
    }

    #[test]
    fn eval_add_returns_tagged_fixnum() {
        let e = Expr::add(Expr::Const(1), Expr::Const(2));
        assert_eq!(eval_with_fresh_mutator(&e).as_fixnum(), Some(3));
    }

    #[test]
    fn eval_cons_returns_cons_tagged() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let e = Expr::cons(Expr::Const(1), Expr::Const(2));
        let result = jit_eval(&e, &mut m as *mut _).unwrap();
        assert!(result.is_cons());
        let p = result.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*p.add(1)).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn eval_car_extracts_car() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let e = Expr::car(Expr::cons(Expr::Const(1), Expr::Const(2)));
        let result = jit_eval(&e, &mut m as *mut _).unwrap();
        assert_eq!(result.as_fixnum(), Some(1));
    }

    // -- Function compilation -----------------------------------------------

    #[test]
    fn compile_and_call_identity() {
        // (lambda (x) x) called with arg=42 returns 42.
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let body = Expr::Param(0);
        let code = jit_compile_function(1, &body).unwrap();
        type Fn1 = unsafe extern "C" fn(*mut MutatorState, u64, *const u64, u64) -> u64;
        let f: Fn1 = unsafe { std::mem::transmute(code) };
        let arg = Word::fixnum(42).raw();
        let r = unsafe { f(&mut m as *mut _, Word::NIL.raw(), &arg as *const u64, 1) };
        assert_eq!(Word::from_raw(r).as_fixnum(), Some(42));
    }

    #[test]
    fn compile_and_call_double() {
        // (lambda (x) (+ x x)) called with arg=21 returns 42.
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let body = Expr::add(Expr::Param(0), Expr::Param(0));
        let code = jit_compile_function(1, &body).unwrap();
        type Fn1 = unsafe extern "C" fn(*mut MutatorState, u64, *const u64, u64) -> u64;
        let f: Fn1 = unsafe { std::mem::transmute(code) };
        let arg = Word::fixnum(21).raw();
        let r = unsafe { f(&mut m as *mut _, Word::NIL.raw(), &arg as *const u64, 1) };
        assert_eq!(Word::from_raw(r).as_fixnum(), Some(42));
    }
}
