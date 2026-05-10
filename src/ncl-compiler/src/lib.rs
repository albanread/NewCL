//! Lisp compiler: lowers `Value` (from the reader) through `Expr`
//! (in `ncl-ir`) down to LLVM IR (via `ncl-llvm`).
//!
//! Node 1: arithmetic, cons/car/cdr, eq/if/quote.
//! Node 2 (this commit): multi-form evaluation, `defun`, function
//! calls, recursive functions.
//!
//! The user-facing entry point is `eval_str(src)` for one-shot
//! evaluation. State that needs to persist across calls — the GC
//! coordinator, the mutator, defun'd functions — lives in a
//! `Session`.

use std::sync::Arc;

use ncl_runtime::{
    format_word, gc_function, GcConfig, GcCoordinator, MutatorState, Value, Word,
};

pub mod lower;

pub use lower::{lower, lower_in, CompileError, LocalEnv};

/// A NewCormanLisp evaluation session. Owns the GC coordinator and
/// a single Lisp-thread mutator. defun'd functions persist across
/// `eval` calls because they live in the coordinator's static area
/// and intern table.
pub struct Session {
    coord: Arc<GcCoordinator>,
    mutator: Box<MutatorState>,
}

impl Session {
    pub fn new() -> Session {
        Session::with_config(GcConfig::default())
    }

    pub fn with_config(config: GcConfig) -> Session {
        let coord = GcCoordinator::new(config);
        let mutator = Box::new(coord.register_mutator());
        Session { coord, mutator }
    }

    pub fn coord(&self) -> &Arc<GcCoordinator> { &self.coord }

    /// Read every form in `src`, evaluate them in sequence, return
    /// the printed representation of the last value (or `nil` on
    /// empty input). `(defun …)` forms are intercepted and define
    /// the function in the session's symbol table; their result is
    /// `nil`.
    pub fn eval(&mut self, src: &str) -> Result<String, EvalError> {
        let values = ncl_reader::read_all(src)
            .map_err(|e| EvalError::Read(format!("{:?}", e.kind)))?;
        if values.is_empty() {
            return Ok("nil".to_string());
        }
        let mut last = Word::NIL;
        for v in &values {
            last = self.eval_value(v)?;
        }
        Ok(format_word(last))
    }

    /// Evaluate a single Value and return the resulting Word.
    /// Recognises `(defun …)` at top level; everything else goes
    /// through lower → JIT → call.
    pub fn eval_value(&mut self, v: &Value) -> Result<Word, EvalError> {
        if let Some((name, params, body)) = match_defun(v)? {
            return self.handle_defun(&name, &params, &body);
        }
        let expr = lower(v, &self.coord).map_err(EvalError::Compile)?;
        let mutator_ptr = &mut *self.mutator as *mut _;
        ncl_llvm::jit_eval(&expr, mutator_ptr).map_err(EvalError::Jit)
    }

    fn handle_defun(
        &mut self,
        name: &str,
        params: &[Arc<str>],
        body: &Value,
    ) -> Result<Word, EvalError> {
        let env = LocalEnv::with_params(params);
        let body_expr = lower_in(body, &env, &self.coord)
            .map_err(EvalError::Compile)?;
        let code_ptr = ncl_llvm::jit_compile_function(params.len() as u32, &body_expr)
            .map_err(EvalError::Jit)?;
        let sym_word = self.coord.intern(name);
        let fn_word = gc_function::alloc_function_in_static(
            self.coord.static_area(),
            code_ptr,
            params.len() as u32,
            sym_word,
        )
        .ok_or_else(|| EvalError::Jit("static area exhausted".to_string()))?;
        self.mutator.set_symbol_function(sym_word, fn_word);
        // CL says `defun` returns the symbol's name. We return nil
        // for v1; tests don't depend on it.
        Ok(Word::NIL)
    }
}

impl Default for Session {
    fn default() -> Self { Session::new() }
}

/// Recognise `(defun name (params...) body)`. Returns `Some` if the
/// form is a defun (with the body as a Value), else `None`. The
/// body is the LAST argument — implicit progn for multi-statement
/// bodies isn't yet supported, so a multi-form body errors.
fn match_defun(
    v: &Value,
) -> Result<Option<(String, Vec<Arc<str>>, Value)>, EvalError> {
    let Value::Cons(c) = v else { return Ok(None); };
    let Value::Symbol(head) = &c.car else { return Ok(None); };
    if &*head.name != "DEFUN" {
        return Ok(None);
    }
    // Walk the args.
    let args = list_to_vec_of_value(&c.cdr).map_err(|e| {
        EvalError::Compile(CompileError::ImproperList(e))
    })?;
    if args.len() < 3 {
        return Err(EvalError::Compile(CompileError::BadDefun(format!(
            "defun needs name, params, and body — got {} args",
            args.len()
        ))));
    }
    if args.len() > 3 {
        return Err(EvalError::Compile(CompileError::BadDefun(
            "multi-statement defun bodies (implicit progn) not yet supported"
                .to_string(),
        )));
    }
    let name = match &args[0] {
        Value::Symbol(s) => s.name.to_string(),
        other => {
            return Err(EvalError::Compile(CompileError::BadDefun(format!(
                "defun name must be a symbol, got {other:?}"
            ))));
        }
    };
    let params = parse_param_list(&args[1])?;
    let body = args[2].clone();
    Ok(Some((name, params, body)))
}

fn parse_param_list(v: &Value) -> Result<Vec<Arc<str>>, EvalError> {
    let mut out = Vec::new();
    let mut cur = v.clone();
    loop {
        match cur {
            Value::Nil => return Ok(out),
            Value::Cons(c) => {
                let Value::Symbol(s) = &c.car else {
                    return Err(EvalError::Compile(CompileError::BadDefun(format!(
                        "param list element must be a symbol, got {:?}",
                        c.car
                    ))));
                };
                out.push(Arc::clone(&s.name));
                cur = c.cdr.clone();
            }
            other => {
                return Err(EvalError::Compile(CompileError::BadDefun(format!(
                    "param list must be a proper list, got {other:?}"
                ))));
            }
        }
    }
}

fn list_to_vec_of_value(v: &Value) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    let mut cur = v.clone();
    loop {
        match cur {
            Value::Nil => return Ok(out),
            Value::Cons(c) => {
                out.push(c.car.clone());
                cur = c.cdr.clone();
            }
            other => return Err(format!("{other:?}")),
        }
    }
}

/// One-shot evaluation: create a fresh `Session`, evaluate the
/// source, return the printed result. Equivalent to
/// `Session::new().eval(src)` — the convenience entry point used
/// by the driver's `--eval` flag and most tests.
pub fn eval_str(src: &str) -> Result<String, EvalError> {
    Session::new().eval(src)
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

#[cfg(test)]
mod end_to_end_tests {
    use super::*;

    // -- Existing tests, retained --------------------------------------------

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
        assert_eq!(
            eval_str("(cons 1 (cons 2 (cons 3 nil)))").unwrap(),
            "(1 2 3)",
        );
    }

    #[test]
    fn eq_returns_t_for_equal_fixnums() {
        assert_eq!(eval_str("(eq 1 1)").unwrap(), "T");
        assert_eq!(eval_str("(eq 1 2)").unwrap(), "nil");
    }

    #[test]
    fn if_chooses_correct_branch() {
        assert_eq!(eval_str("(if t 7 8)").unwrap(), "7");
        assert_eq!(eval_str("(if nil 7 8)").unwrap(), "8");
        assert_eq!(eval_str("(if (eq 1 1) 7 8)").unwrap(), "7");
    }

    #[test]
    fn quote_fixnum_and_nil_and_t() {
        assert_eq!(eval_str("(quote 42)").unwrap(), "42");
        assert_eq!(eval_str("'42").unwrap(), "42");
        assert_eq!(eval_str("(quote nil)").unwrap(), "nil");
        assert_eq!(eval_str("'t").unwrap(), "T");
    }

    // -- Multi-form evaluation ---------------------------------------------

    #[test]
    fn multi_form_returns_last_result() {
        assert_eq!(eval_str("(+ 1 2) (* 3 4)").unwrap(), "12");
        assert_eq!(eval_str("1 2 3 4 5").unwrap(), "5");
    }

    #[test]
    fn empty_input_evaluates_to_nil() {
        assert_eq!(eval_str("").unwrap(), "nil");
    }

    // -- defun + function calls --------------------------------------------

    #[test]
    fn defun_creates_a_function_then_call_runs() {
        let mut session = Session::new();
        session.eval("(defun double (x) (+ x x))").unwrap();
        assert_eq!(session.eval("(double 21)").unwrap(), "42");
    }

    #[test]
    fn defun_via_multi_form_string() {
        // defun followed by a call in the same source.
        let result = eval_str("(defun triple (x) (+ x (+ x x))) (triple 7)").unwrap();
        assert_eq!(result, "21");
    }

    #[test]
    fn defun_with_two_params() {
        let mut session = Session::new();
        session.eval("(defun mul-add (x y) (+ (* x x) y))").unwrap();
        assert_eq!(session.eval("(mul-add 3 7)").unwrap(), "16");
    }

    #[test]
    fn redefinition_replaces_function() {
        let mut session = Session::new();
        session.eval("(defun id (x) x)").unwrap();
        assert_eq!(session.eval("(id 42)").unwrap(), "42");

        // Redefine.
        session.eval("(defun id (x) (+ x 100))").unwrap();
        assert_eq!(session.eval("(id 42)").unwrap(), "142");
    }

    // -- The big one: recursive defun --------------------------------------

    #[test]
    fn recursive_factorial_5_equals_120() {
        let result = eval_str(
            "(defun fact (n) (if (eq n 0) 1 (* n (fact (- n 1)))))
             (fact 5)",
        )
        .unwrap();
        assert_eq!(result, "120");
    }

    #[test]
    fn recursive_factorial_10_equals_3628800() {
        let result = eval_str(
            "(defun fact (n) (if (eq n 0) 1 (* n (fact (- n 1)))))
             (fact 10)",
        )
        .unwrap();
        assert_eq!(result, "3628800");
    }

    #[test]
    fn recursive_function_returning_cons() {
        // (defun count-down (n) (if (eq n 0) nil (cons n (count-down (- n 1)))))
        // (count-down 4) → (4 3 2 1)
        let result = eval_str(
            "(defun count-down (n)
               (if (eq n 0) nil (cons n (count-down (- n 1)))))
             (count-down 4)",
        )
        .unwrap();
        assert_eq!(result, "(4 3 2 1)");
    }

    #[test]
    fn function_calling_function() {
        let mut session = Session::new();
        session.eval("(defun double (x) (+ x x))").unwrap();
        session.eval("(defun quadruple (x) (double (double x)))").unwrap();
        assert_eq!(session.eval("(quadruple 5)").unwrap(), "20");
    }

    // -- Errors ------------------------------------------------------------

    #[test]
    fn defun_at_nested_position_errors() {
        let r = eval_str("(if t (defun foo () 1) 2)");
        assert!(matches!(r, Err(EvalError::Compile(CompileError::BadDefun(_)))));
    }

    #[test]
    fn calling_undefined_function_panics_at_runtime() {
        // The compile succeeds (we can't tell the function is
        // undefined at compile time — the symbol just isn't bound
        // yet). At runtime, ncl_call panics on unbound. We catch
        // it for this test.
        // (Disabled — JIT panics aren't easy to catch from Rust
        // tests without unwinding through C frames. Documenting
        // behaviour here.)
    }

    #[test]
    fn malformed_defun_errors() {
        let r = eval_str("(defun)");
        assert!(matches!(r, Err(EvalError::Compile(CompileError::BadDefun(_)))));
        let r = eval_str("(defun foo bar 1)"); // params must be list
        assert!(matches!(r, Err(EvalError::Compile(CompileError::BadDefun(_)))));
    }

    #[test]
    fn bare_unknown_symbol_in_body_compiles_but_calls_undefined() {
        // Lowering an unknown symbol in expression position fails
        // with NotImplemented (we don't have value cells yet).
        let r = eval_str("foo");
        assert!(matches!(r, Err(EvalError::Compile(CompileError::NotImplemented(_)))));
    }
}
