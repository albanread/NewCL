//! Lisp compiler: lowers `Value` (from the reader) through `Expr`
//! (in `ncl-ir`) down to LLVM IR (via `ncl-llvm`).
//!
//! Phase 3 starts with arithmetic on fixnums — `(+ 1 2)`, `(* 3 4)`,
//! `(- 10 (+ 1 2))`. Symbols, lambdas, and the symbol-cell dispatch
//! path arrive in follow-up phases.

use std::sync::Arc;

use ncl_runtime::{Symbol, Value};

pub mod lower;

pub use lower::{lower, CompileError};

/// End-to-end: take a Value (from `ncl_reader::read_one`), lower
/// to `Expr`, JIT-compile and run, return the result as `i64`.
pub fn eval_value(v: &Value) -> Result<i64, EvalError> {
    let expr = lower(v).map_err(EvalError::Compile)?;
    ncl_llvm::jit_eval(&expr).map_err(EvalError::Jit)
}

/// End-to-end: take a source string, read one form, eval, return
/// the result. Used by the driver's `--eval` flag and by tests.
pub fn eval_str(src: &str) -> Result<i64, EvalError> {
    let v = ncl_reader::read_one(src).map_err(|e| {
        EvalError::Read(format!("{:?}", e.kind))
    })?;
    eval_value(&v)
}

#[derive(Debug)]
pub enum EvalError {
    Read(String),
    Compile(CompileError),
    Jit(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Read(s) => write!(f, "read error: {s}"),
            EvalError::Compile(e) => write!(f, "compile error: {e:?}"),
            EvalError::Jit(s) => write!(f, "jit error: {s}"),
        }
    }
}

impl std::error::Error for EvalError {}

// Re-export for convenience.
pub fn symbol_name(sym: &Arc<Symbol>) -> &str { &sym.name }

#[cfg(test)]
mod end_to_end_tests {
    use super::*;

    /// The Phase 3 milestone. Source string in, integer out.
    #[test]
    fn the_milestone_one_plus_two_equals_three() {
        assert_eq!(eval_str("(+ 1 2)").unwrap(), 3);
    }

    #[test]
    fn integer_literal_evaluates_to_itself() {
        assert_eq!(eval_str("42").unwrap(), 42);
        assert_eq!(eval_str("-7").unwrap(), -7);
        assert_eq!(eval_str("0").unwrap(), 0);
    }

    #[test]
    fn arithmetic_combinations_eval_correctly() {
        assert_eq!(eval_str("(+ 1 2)").unwrap(), 3);
        assert_eq!(eval_str("(- 10 3)").unwrap(), 7);
        assert_eq!(eval_str("(* 6 7)").unwrap(), 42);
        assert_eq!(eval_str("(- 5 10)").unwrap(), -5);
    }

    #[test]
    fn nested_arithmetic_evals_correctly() {
        // (* (+ 1 2) (- 10 4)) = 3 * 6 = 18
        assert_eq!(eval_str("(* (+ 1 2) (- 10 4))").unwrap(), 18);
        // ((1 + 2 + 3) * (10 - 7)) = 6 * 3 = 18
        assert_eq!(eval_str("(* (+ 1 2 3) (- 10 7))").unwrap(), 18);
    }

    #[test]
    fn variadic_arithmetic_left_folds() {
        assert_eq!(eval_str("(+ 1 2 3 4 5)").unwrap(), 15);
        assert_eq!(eval_str("(* 2 3 4)").unwrap(), 24);
        assert_eq!(eval_str("(- 100 1 2 3 4)").unwrap(), 90);
    }

    #[test]
    fn nullary_arithmetic_uses_identity() {
        assert_eq!(eval_str("(+)").unwrap(), 0);
        assert_eq!(eval_str("(*)").unwrap(), 1);
    }

    #[test]
    fn unary_minus_negates() {
        assert_eq!(eval_str("(- 5)").unwrap(), -5);
        assert_eq!(eval_str("(- 0)").unwrap(), 0);
    }

    #[test]
    fn factorial_5_via_unrolled_multiplication() {
        // (* 1 2 3 4 5) = 120
        assert_eq!(eval_str("(* 1 2 3 4 5)").unwrap(), 120);
    }

    #[test]
    fn unknown_form_returns_compile_error() {
        let r = eval_str("(undefined-fn 1 2)");
        assert!(matches!(r, Err(EvalError::Compile(CompileError::NotImplemented(_)))));
    }

    #[test]
    fn unparseable_source_returns_read_error() {
        let r = eval_str("(unbalanced");
        assert!(matches!(r, Err(EvalError::Read(_))));
    }
}
