//! LLVM bindings + JIT.
//!
//! Phase 2: hand-built `jit_three` and `jit_add` to prove the
//! toolchain works.
//! Phase 3: `jit_eval` takes a `ncl_ir::Expr` tree, lowers it to
//! LLVM IR, JITs the result, and returns the i64 value.
//!
//! All interaction with `inkwell` and `llvm-sys` is encapsulated
//! here so the rest of the workspace stays LLVM-version-agnostic.

use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::values::IntValue;

use ncl_ir::Expr;

/// JIT-compile and run a hand-written `fn three() -> i64 { 3 }`.
/// Phase 2 smoke test; should always return `3`.
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

/// JIT-compile and run a hand-written `fn add(a: i64, b: i64) -> i64`.
/// Phase 2 smoke test for argument passing through the JIT calling
/// convention.
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

/// JIT-compile and run an `Expr` tree. The result is the i64 value
/// the expression evaluates to.
///
/// Phase 3: this is the bridge between the typed IR and machine
/// code. The full pipeline (read source → lower to Value → lower to
/// Expr → JIT) is wired up by `ncl_compiler::eval_str`.
pub fn jit_eval(expr: &Expr) -> Result<i64, String> {
    let context = Context::create();
    let module = context.create_module("ncl_eval");
    let builder = context.create_builder();

    let i64_t = context.i64_type();
    let fn_type = i64_t.fn_type(&[], false);
    let function = module.add_function("entry", fn_type, None);
    let entry_block = context.append_basic_block(function, "entry");
    builder.position_at_end(entry_block);

    let result = emit_expr(&context, &builder, expr)?;
    builder
        .build_return(Some(&result))
        .map_err(|e| format!("build_return: {e}"))?;

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| format!("create_jit_execution_engine: {e}"))?;

    type EntryFn = unsafe extern "C" fn() -> i64;
    let jit_fn: JitFunction<EntryFn> = unsafe {
        engine
            .get_function("entry")
            .map_err(|e| format!("get_function: {e}"))?
    };
    Ok(unsafe { jit_fn.call() })
}

fn emit_expr<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    expr: &Expr,
) -> Result<IntValue<'ctx>, String> {
    let i64_t = context.i64_type();
    match expr {
        Expr::Const(n) => {
            // `const_int` takes u64 but reinterprets the bit
            // pattern as i64 in our context (we use signed
            // integer types throughout).
            Ok(i64_t.const_int(*n as u64, false))
        }
        Expr::Add(a, b) => {
            let lhs = emit_expr(context, builder, a)?;
            let rhs = emit_expr(context, builder, b)?;
            builder
                .build_int_add(lhs, rhs, "add")
                .map_err(|e| format!("build_int_add: {e}"))
        }
        Expr::Sub(a, b) => {
            let lhs = emit_expr(context, builder, a)?;
            let rhs = emit_expr(context, builder, b)?;
            builder
                .build_int_sub(lhs, rhs, "sub")
                .map_err(|e| format!("build_int_sub: {e}"))
        }
        Expr::Mul(a, b) => {
            let lhs = emit_expr(context, builder, a)?;
            let rhs = emit_expr(context, builder, b)?;
            builder
                .build_int_mul(lhs, rhs, "mul")
                .map_err(|e| format!("build_int_mul: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // -- jit_eval tests -----------------------------------------------------

    #[test]
    fn eval_const() {
        assert_eq!(jit_eval(&Expr::Const(42)).unwrap(), 42);
        assert_eq!(jit_eval(&Expr::Const(-7)).unwrap(), -7);
        assert_eq!(jit_eval(&Expr::Const(0)).unwrap(), 0);
    }

    #[test]
    fn eval_add() {
        let e = Expr::add(Expr::Const(1), Expr::Const(2));
        assert_eq!(jit_eval(&e).unwrap(), 3);
    }

    #[test]
    fn eval_sub() {
        let e = Expr::sub(Expr::Const(10), Expr::Const(3));
        assert_eq!(jit_eval(&e).unwrap(), 7);
    }

    #[test]
    fn eval_mul() {
        let e = Expr::mul(Expr::Const(6), Expr::Const(7));
        assert_eq!(jit_eval(&e).unwrap(), 42);
    }

    #[test]
    fn eval_nested() {
        // (* (+ 1 2) (- 10 4)) = 3 * 6 = 18
        let e = Expr::mul(
            Expr::add(Expr::Const(1), Expr::Const(2)),
            Expr::sub(Expr::Const(10), Expr::Const(4)),
        );
        assert_eq!(jit_eval(&e).unwrap(), 18);
    }

    #[test]
    fn eval_negative_results() {
        // 5 - 10 = -5
        let e = Expr::sub(Expr::Const(5), Expr::Const(10));
        assert_eq!(jit_eval(&e).unwrap(), -5);
    }
}
