//! LLVM bindings + JIT.
//!
//! Phase 2: hand-built `jit_three`/`jit_add` to prove the toolchain
//! works.
//! Phase 3: `jit_eval` takes an `Expr` tree, lowers to LLVM IR,
//! JITs, runs.
//! Cons/Car/Cdr extension: compiled code now operates on **tagged
//! `Word`s** (low 3 bits classify), threads a `*mut MutatorState`
//! through the entry function, and calls back into `ncl_alloc_cons`
//! when allocating. Fixnum arithmetic still works as plain i64
//! ops because shifting both operands left by 3 keeps add/sub
//! associative — the SBCL/CCL trick.
//!
//! All interaction with `inkwell` and `llvm-sys` is encapsulated
//! here so the rest of the workspace stays LLVM-version-agnostic.

use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::{Linkage, Module};
use inkwell::values::{FunctionValue, IntValue};

use ncl_ir::Expr;
use ncl_runtime::{ncl_alloc_cons, MutatorState, Word};

// -- Phase 2 smoke functions (kept for the existing test suite) -------------

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

// -- Phase 3 / cons-extension JIT ------------------------------------------

/// JIT-compile and run an `Expr` tree. Returns the result as a
/// tagged `Word`. The mutator pointer is threaded through the
/// entry function so `cons` allocations can call back into the
/// runtime. Pass any valid `&mut MutatorState`; non-allocating
/// expressions ignore it.
pub fn jit_eval(expr: &Expr, mutator: *mut MutatorState) -> Result<Word, String> {
    let context = Context::create();
    let module = context.create_module("ncl_eval");
    let builder = context.create_builder();

    let i64_t = context.i64_type();
    let ptr_t = context.ptr_type(AddressSpace::default());

    // Entry function: fn entry(mutator: *mut MutatorState) -> Word
    let entry_fn_type = i64_t.fn_type(&[ptr_t.into()], false);
    let entry_fn = module.add_function("entry", entry_fn_type, None);
    let entry_block = context.append_basic_block(entry_fn, "entry");
    builder.position_at_end(entry_block);

    // Declare runtime helpers as external functions.
    let helpers = declare_runtime_helpers(&context, &module);

    let result = emit_expr(&context, &builder, &entry_fn, &helpers, expr)?;
    builder
        .build_return(Some(&result))
        .map_err(|e| format!("build_return: {e}"))?;

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| format!("create_jit_execution_engine: {e}"))?;

    // Map "ncl_alloc_cons" -> the actual Rust function pointer.
    register_runtime_helpers(&engine, &helpers);

    type EntryFn = unsafe extern "C" fn(*mut MutatorState) -> u64;
    let jit_fn: JitFunction<EntryFn> = unsafe {
        engine
            .get_function("entry")
            .map_err(|e| format!("get_function: {e}"))?
    };
    Ok(Word::from_raw(unsafe { jit_fn.call(mutator) }))
}

// -- Helpers -----------------------------------------------------------------

struct Helpers<'ctx> {
    /// `extern "C" fn(mutator: *mut MutatorState, car: u64, cdr: u64) -> u64`
    alloc_cons: FunctionValue<'ctx>,
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
    Helpers { alloc_cons }
}

fn register_runtime_helpers(engine: &ExecutionEngine<'_>, helpers: &Helpers<'_>) {
    engine.add_global_mapping(&helpers.alloc_cons, ncl_alloc_cons as *const () as usize);
}

fn emit_expr<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    function: &FunctionValue<'ctx>,
    helpers: &Helpers<'ctx>,
    expr: &Expr,
) -> Result<IntValue<'ctx>, String> {
    let i64_t = context.i64_type();
    match expr {
        Expr::Const(n) => {
            // Tag the fixnum: shift left by 3 (zero tag bits = fixnum).
            // SBCL/CCL trick: tagged fixnums add and subtract directly.
            Ok(i64_t.const_int(Word::fixnum(*n).raw(), false))
        }
        Expr::Nil => Ok(i64_t.const_int(Word::NIL.raw(), false)),
        Expr::Add(a, b) => {
            let lhs = emit_expr(context, builder, function, helpers, a)?;
            let rhs = emit_expr(context, builder, function, helpers, b)?;
            // (a << 3) + (b << 3) = (a + b) << 3 — still tagged.
            builder
                .build_int_add(lhs, rhs, "add")
                .map_err(|e| format!("build_int_add: {e}"))
        }
        Expr::Sub(a, b) => {
            let lhs = emit_expr(context, builder, function, helpers, a)?;
            let rhs = emit_expr(context, builder, function, helpers, b)?;
            builder
                .build_int_sub(lhs, rhs, "sub")
                .map_err(|e| format!("build_int_sub: {e}"))
        }
        Expr::Mul(a, b) => {
            let lhs = emit_expr(context, builder, function, helpers, a)?;
            let rhs = emit_expr(context, builder, function, helpers, b)?;
            // Untag rhs first: (a << 3) * b = (a * b) << 3.
            let three = i64_t.const_int(3, false);
            let rhs_untagged = builder
                .build_right_shift(rhs, three, true /* arithmetic */, "untag_rhs")
                .map_err(|e| format!("ashr: {e}"))?;
            builder
                .build_int_mul(lhs, rhs_untagged, "mul")
                .map_err(|e| format!("build_int_mul: {e}"))
        }
        Expr::Cons(car, cdr) => {
            let car_val = emit_expr(context, builder, function, helpers, car)?;
            let cdr_val = emit_expr(context, builder, function, helpers, cdr)?;
            // The mutator pointer is the entry function's first param.
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
            let cons_val = emit_expr(context, builder, function, helpers, x)?;
            // Untag: word & ~7. For Cons (tag 001) the tag-clear can
            // also be `word - 1` but `& ~7` is the general form.
            let mask = i64_t.const_int(!0b111u64, false);
            let untagged = builder
                .build_and(cons_val, mask, "untag_cons")
                .map_err(|e| format!("and: {e}"))?;
            let ptr_t = context.ptr_type(AddressSpace::default());
            let ptr = builder
                .build_int_to_ptr(untagged, ptr_t, "as_ptr")
                .map_err(|e| format!("int_to_ptr: {e}"))?;
            let loaded = builder
                .build_load(i64_t, ptr, "car")
                .map_err(|e| format!("load car: {e}"))?;
            Ok(loaded.into_int_value())
        }
        Expr::Cdr(x) => {
            let cons_val = emit_expr(context, builder, function, helpers, x)?;
            let mask = i64_t.const_int(!0b111u64, false);
            let untagged = builder
                .build_and(cons_val, mask, "untag_cons")
                .map_err(|e| format!("and: {e}"))?;
            let ptr_t = context.ptr_type(AddressSpace::default());
            let ptr = builder
                .build_int_to_ptr(untagged, ptr_t, "as_ptr")
                .map_err(|e| format!("int_to_ptr: {e}"))?;
            // GEP to cell index 1 (cdr is at +8 bytes from the cons start).
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
        assert_eq!(jit_add(-7, 7).expect("jit_add"), 0);
        assert_eq!(jit_add(i64::MAX, 0).expect("jit_add"), i64::MAX);
    }

    fn eval_with_fresh_mutator(expr: &Expr) -> Word {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        jit_eval(expr, &mut m as *mut _).unwrap()
    }

    #[test]
    fn eval_const() {
        assert_eq!(eval_with_fresh_mutator(&Expr::Const(42)).as_fixnum(), Some(42));
        assert_eq!(eval_with_fresh_mutator(&Expr::Const(-7)).as_fixnum(), Some(-7));
        assert_eq!(eval_with_fresh_mutator(&Expr::Const(0)).as_fixnum(), Some(0));
    }

    #[test]
    fn eval_nil() {
        assert!(eval_with_fresh_mutator(&Expr::Nil).is_nil());
    }

    #[test]
    fn eval_add_returns_tagged_fixnum() {
        let e = Expr::add(Expr::Const(1), Expr::Const(2));
        assert_eq!(eval_with_fresh_mutator(&e).as_fixnum(), Some(3));
    }

    #[test]
    fn eval_sub() {
        let e = Expr::sub(Expr::Const(10), Expr::Const(3));
        assert_eq!(eval_with_fresh_mutator(&e).as_fixnum(), Some(7));
    }

    #[test]
    fn eval_mul_with_untag() {
        let e = Expr::mul(Expr::Const(6), Expr::Const(7));
        assert_eq!(eval_with_fresh_mutator(&e).as_fixnum(), Some(42));
    }

    #[test]
    fn eval_negative_results() {
        let e = Expr::sub(Expr::Const(5), Expr::Const(10));
        assert_eq!(eval_with_fresh_mutator(&e).as_fixnum(), Some(-5));
    }

    #[test]
    fn eval_nested_arithmetic() {
        // (* (+ 1 2) (- 10 4)) = 18
        let e = Expr::mul(
            Expr::add(Expr::Const(1), Expr::Const(2)),
            Expr::sub(Expr::Const(10), Expr::Const(4)),
        );
        assert_eq!(eval_with_fresh_mutator(&e).as_fixnum(), Some(18));
    }

    #[test]
    fn eval_cons_returns_cons_tagged() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        // (cons 1 2)
        let e = Expr::cons(Expr::Const(1), Expr::Const(2));
        let result = jit_eval(&e, &mut m as *mut _).unwrap();
        assert!(result.is_cons());
        // car == 1, cdr == 2
        let p = result.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(1));
            assert_eq!(Word::from_raw(*p.add(1)).as_fixnum(), Some(2));
        }
    }

    #[test]
    fn eval_car_extracts_car() {
        // (car (cons 1 2)) = 1
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let e = Expr::car(Expr::cons(Expr::Const(1), Expr::Const(2)));
        let result = jit_eval(&e, &mut m as *mut _).unwrap();
        assert_eq!(result.as_fixnum(), Some(1));
    }

    #[test]
    fn eval_cdr_extracts_cdr() {
        // (cdr (cons 1 2)) = 2
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let e = Expr::cdr(Expr::cons(Expr::Const(1), Expr::Const(2)));
        let result = jit_eval(&e, &mut m as *mut _).unwrap();
        assert_eq!(result.as_fixnum(), Some(2));
    }

    #[test]
    fn eval_nested_cons() {
        // (cons (cons 1 2) (cons 3 4))
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let e = Expr::cons(
            Expr::cons(Expr::Const(1), Expr::Const(2)),
            Expr::cons(Expr::Const(3), Expr::Const(4)),
        );
        let result = jit_eval(&e, &mut m as *mut _).unwrap();
        assert!(result.is_cons());
        // car is itself a cons
        let p = result.as_ptr::<u64>(Tag::Cons).unwrap();
        let car = Word::from_raw(unsafe { *p });
        let cdr = Word::from_raw(unsafe { *p.add(1) });
        assert!(car.is_cons());
        assert!(cdr.is_cons());
    }

    #[test]
    fn eval_car_of_cdr_of_list() {
        // (car (cdr (cons 1 (cons 2 (cons 3 nil)))))  = 2
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let list = Expr::cons(
            Expr::Const(1),
            Expr::cons(
                Expr::Const(2),
                Expr::cons(Expr::Const(3), Expr::Nil),
            ),
        );
        let e = Expr::car(Expr::cdr(list));
        let result = jit_eval(&e, &mut m as *mut _).unwrap();
        assert_eq!(result.as_fixnum(), Some(2));
    }

    #[test]
    fn eval_cons_with_arithmetic_args() {
        // (cons (+ 1 2) (* 3 4)) = (3 . 12)
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let e = Expr::cons(
            Expr::add(Expr::Const(1), Expr::Const(2)),
            Expr::mul(Expr::Const(3), Expr::Const(4)),
        );
        let result = jit_eval(&e, &mut m as *mut _).unwrap();
        let p = result.as_ptr::<u64>(Tag::Cons).unwrap();
        unsafe {
            assert_eq!(Word::from_raw(*p).as_fixnum(), Some(3));
            assert_eq!(Word::from_raw(*p.add(1)).as_fixnum(), Some(12));
        }
    }
}
