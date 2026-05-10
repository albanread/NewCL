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
use inkwell::values::{FunctionValue, IntValue};

use ncl_ir::Expr;
use ncl_runtime::{ncl_alloc_cons, ncl_call, MutatorState, Tag, Word};

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
/// with `(mutator, null, 0)` — top-level forms have no parameters.
/// Returns the result as a tagged `Word`.
pub fn jit_eval(expr: &Expr, mutator: *mut MutatorState) -> Result<Word, String> {
    let code = build_lisp_function(expr, 0)?;
    type EntryFn = unsafe extern "C" fn(*mut MutatorState, *const u64, u64) -> u64;
    let f: EntryFn = unsafe { std::mem::transmute(code) };
    Ok(Word::from_raw(unsafe { f(mutator, std::ptr::null(), 0) }))
}

/// JIT-compile a Lisp function with `arity` positional parameters.
/// Returns the raw machine-code address of the entry. Used by
/// `defun` to install a Function in a Symbol's function cell.
///
/// The returned pointer is a function with signature
///   `extern "C" fn(*mut MutatorState, *const u64, u64) -> u64`
/// and is valid for the process lifetime.
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

    let fn_type = i64_t.fn_type(&[ptr_t.into(), ptr_t.into(), i64_t.into()], false);
    let function = module.add_function("the_fn", fn_type, None);
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

    Helpers { alloc_cons, call_fn }
}

fn register_runtime_helpers(engine: &ExecutionEngine<'_>, helpers: &Helpers<'_>) {
    engine.add_global_mapping(
        &helpers.alloc_cons,
        ncl_alloc_cons as *const () as usize,
    );
    engine.add_global_mapping(
        &helpers.call_fn,
        ncl_call as *const () as usize,
    );
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
            // args_ptr is param 1. Load i64 at args_ptr[idx].
            let args_ptr = function
                .get_nth_param(1)
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
            Ok(call.try_as_basic_value().unwrap_basic().into_int_value())
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
        type Fn1 = unsafe extern "C" fn(*mut MutatorState, *const u64, u64) -> u64;
        let f: Fn1 = unsafe { std::mem::transmute(code) };
        let arg = Word::fixnum(42).raw();
        let r = unsafe { f(&mut m as *mut _, &arg as *const u64, 1) };
        assert_eq!(Word::from_raw(r).as_fixnum(), Some(42));
    }

    #[test]
    fn compile_and_call_double() {
        // (lambda (x) (+ x x)) called with arg=21 returns 42.
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let body = Expr::add(Expr::Param(0), Expr::Param(0));
        let code = jit_compile_function(1, &body).unwrap();
        type Fn1 = unsafe extern "C" fn(*mut MutatorState, *const u64, u64) -> u64;
        let f: Fn1 = unsafe { std::mem::transmute(code) };
        let arg = Word::fixnum(21).raw();
        let r = unsafe { f(&mut m as *mut _, &arg as *const u64, 1) };
        assert_eq!(Word::from_raw(r).as_fixnum(), Some(42));
    }
}
