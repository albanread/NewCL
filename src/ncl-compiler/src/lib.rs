//! Lisp compiler: lowers `Value` (from the reader) through `Expr`
//! (in `ncl-ir`) down to LLVM IR (via `ncl-llvm`).
//!
//! Phase 3 starts with arithmetic on fixnums — `(+ 1 2)`, `(* 3 4)`,
//! `(- 10 (+ 1 2))`. Symbols, lambdas, and the symbol-cell dispatch
//! path arrive in follow-up phases.

use std::sync::Arc;

use ncl_runtime::{format_word, GcConfig, GcCoordinator, MutatorState, Symbol, Value, Word};

pub mod lower;

pub use lower::{lower, CompileError};

/// Default GC config for the embedded `eval_str` path. Tests and
/// applications that want different sizes can use the lower-level
/// `eval_value_with_mutator` API.
fn default_gc_config() -> GcConfig {
    GcConfig::default()
}

/// End-to-end: take a Value, lower to Expr, JIT-compile and run on
/// the given mutator, return the result as a tagged `Word`.
pub fn eval_value_with_mutator(
    v: &Value,
    mutator: &mut MutatorState,
) -> Result<Word, EvalError> {
    let expr = lower(v).map_err(EvalError::Compile)?;
    ncl_llvm::jit_eval(&expr, mutator as *mut _).map_err(EvalError::Jit)
}

/// Convenience: read one form from source, eval, return the printed
/// representation. Creates a fresh GC coordinator + mutator each
/// call — fine for one-shot --eval, wasteful for a REPL (where a
/// shared coordinator should be reused).
pub fn eval_str(src: &str) -> Result<String, EvalError> {
    let v = ncl_reader::read_one(src).map_err(|e| {
        EvalError::Read(format!("{:?}", e.kind))
    })?;
    let coord = GcCoordinator::new(default_gc_config());
    let mut m = coord.register_mutator();
    let result = eval_value_with_mutator(&v, &mut m)?;
    Ok(format_word(result))
}

/// Like `eval_str` but returns the raw `Word` instead of formatting
/// it. Used by tests that want to assert on tag/value structure.
pub fn eval_str_raw(src: &str) -> Result<Word, EvalError> {
    let v = ncl_reader::read_one(src).map_err(|e| {
        EvalError::Read(format!("{:?}", e.kind))
    })?;
    let coord = GcCoordinator::new(default_gc_config());
    let mut m = coord.register_mutator();
    eval_value_with_mutator(&v, &mut m)
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

    /// The milestone, now returning a printed value.
    #[test]
    fn the_milestone_one_plus_two_equals_three() {
        assert_eq!(eval_str("(+ 1 2)").unwrap(), "3");
    }

    #[test]
    fn integer_literal_evaluates_to_itself() {
        assert_eq!(eval_str("42").unwrap(), "42");
        assert_eq!(eval_str("-7").unwrap(), "-7");
        assert_eq!(eval_str("0").unwrap(), "0");
    }

    #[test]
    fn nil_evaluates_to_nil() {
        assert_eq!(eval_str("nil").unwrap(), "nil");
        assert_eq!(eval_str("()").unwrap(), "nil");
    }

    #[test]
    fn arithmetic_combinations_eval_correctly() {
        assert_eq!(eval_str("(+ 1 2)").unwrap(), "3");
        assert_eq!(eval_str("(- 10 3)").unwrap(), "7");
        assert_eq!(eval_str("(* 6 7)").unwrap(), "42");
        assert_eq!(eval_str("(- 5 10)").unwrap(), "-5");
    }

    #[test]
    fn nested_arithmetic_evals_correctly() {
        assert_eq!(eval_str("(* (+ 1 2) (- 10 4))").unwrap(), "18");
        assert_eq!(eval_str("(* (+ 1 2 3) (- 10 7))").unwrap(), "18");
    }

    #[test]
    fn variadic_arithmetic_left_folds() {
        assert_eq!(eval_str("(+ 1 2 3 4 5)").unwrap(), "15");
        assert_eq!(eval_str("(* 2 3 4)").unwrap(), "24");
        assert_eq!(eval_str("(- 100 1 2 3 4)").unwrap(), "90");
    }

    #[test]
    fn nullary_arithmetic_uses_identity() {
        assert_eq!(eval_str("(+)").unwrap(), "0");
        assert_eq!(eval_str("(*)").unwrap(), "1");
    }

    #[test]
    fn unary_minus_negates() {
        assert_eq!(eval_str("(- 5)").unwrap(), "-5");
        assert_eq!(eval_str("(- 0)").unwrap(), "0");
    }

    #[test]
    fn factorial_5_via_unrolled_multiplication() {
        assert_eq!(eval_str("(* 1 2 3 4 5)").unwrap(), "120");
    }

    #[test]
    fn cons_creates_a_pair() {
        assert_eq!(eval_str("(cons 1 2)").unwrap(), "(1 . 2)");
        assert_eq!(eval_str("(cons 1 nil)").unwrap(), "(1)");
    }

    #[test]
    fn car_and_cdr_extract() {
        assert_eq!(eval_str("(car (cons 1 2))").unwrap(), "1");
        assert_eq!(eval_str("(cdr (cons 1 2))").unwrap(), "2");
    }

    #[test]
    fn proper_list_via_nested_cons() {
        // (cons 1 (cons 2 (cons 3 nil))) = (1 2 3)
        assert_eq!(
            eval_str("(cons 1 (cons 2 (cons 3 nil)))").unwrap(),
            "(1 2 3)",
        );
    }

    #[test]
    fn cadr_extracts_second_element() {
        // (car (cdr ...)) is `cadr`. Useful pattern.
        assert_eq!(
            eval_str("(car (cdr (cons 1 (cons 2 (cons 3 nil)))))").unwrap(),
            "2",
        );
    }

    #[test]
    fn cons_arguments_can_be_arbitrary_expressions() {
        // (cons (+ 1 2) (* 3 4)) = (3 . 12)
        assert_eq!(eval_str("(cons (+ 1 2) (* 3 4))").unwrap(), "(3 . 12)");
    }

    #[test]
    fn nested_cons_yields_nested_list() {
        // (cons (cons 1 2) (cons 3 4)) = ((1 . 2) 3 . 4)
        assert_eq!(
            eval_str("(cons (cons 1 2) (cons 3 4))").unwrap(),
            "((1 . 2) 3 . 4)",
        );
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
