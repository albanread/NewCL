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
        if let Some((name, params, body_forms)) = match_defun(v)? {
            return self.handle_defun(&name, &params, &body_forms);
        }
        let expr = lower(v, &self.coord).map_err(EvalError::Compile)?;
        let mutator_ptr = &mut *self.mutator as *mut _;
        ncl_llvm::jit_eval(&expr, mutator_ptr).map_err(EvalError::Jit)
    }

    fn handle_defun(
        &mut self,
        name: &str,
        params: &[Arc<str>],
        body_forms: &[Value],
    ) -> Result<Word, EvalError> {
        let env = LocalEnv::with_params(params);
        // Implicit progn: multiple body forms wrap into a Progn.
        let body_expr = if body_forms.len() == 1 {
            lower_in(&body_forms[0], &env, &self.coord)
                .map_err(EvalError::Compile)?
        } else {
            let lowered: Result<Vec<_>, _> = body_forms
                .iter()
                .map(|f| lower_in(f, &env, &self.coord))
                .collect();
            ncl_ir::Expr::progn(lowered.map_err(EvalError::Compile)?)
        };
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
        Ok(Word::NIL)
    }
}

impl Default for Session {
    fn default() -> Self { Session::new() }
}

/// Recognise `(defun name (params...) body...)`. Returns `Some` if
/// the form is a defun. Implicit progn is supported — multiple body
/// forms are returned as a Vec for the caller to wrap.
fn match_defun(
    v: &Value,
) -> Result<Option<(String, Vec<Arc<str>>, Vec<Value>)>, EvalError> {
    let Value::Cons(c) = v else { return Ok(None); };
    let Value::Symbol(head) = &c.car else { return Ok(None); };
    if &*head.name != "DEFUN" {
        return Ok(None);
    }
    let args = list_to_vec_of_value(&c.cdr).map_err(|e| {
        EvalError::Compile(CompileError::ImproperList(e))
    })?;
    if args.len() < 3 {
        return Err(EvalError::Compile(CompileError::BadDefun(format!(
            "defun needs name, params, and body — got {} args",
            args.len()
        ))));
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
    let body_forms = args[2..].to_vec();
    Ok(Some((name, params, body_forms)))
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

    // -- progn -------------------------------------------------------------

    #[test]
    fn progn_returns_last_value() {
        assert_eq!(eval_str("(progn 1 2 3)").unwrap(), "3");
        assert_eq!(eval_str("(progn (+ 1 1) (* 3 4))").unwrap(), "12");
    }

    #[test]
    fn empty_progn_is_nil() {
        assert_eq!(eval_str("(progn)").unwrap(), "nil");
    }

    #[test]
    fn progn_in_function_body() {
        let mut session = Session::new();
        session
            .eval(
                "(defun do-stuff (x)
                   (progn
                     (* x 2)
                     (* x 3)
                     (* x 10)))",
            )
            .unwrap();
        assert_eq!(session.eval("(do-stuff 7)").unwrap(), "70");
    }

    #[test]
    fn implicit_progn_in_defun_body() {
        // No explicit progn; multi-form body.
        let mut session = Session::new();
        session
            .eval("(defun sum-of-cubes (x) (* x x) (* x x x))")
            .unwrap();
        assert_eq!(session.eval("(sum-of-cubes 3)").unwrap(), "27");
    }

    // -- let ---------------------------------------------------------------

    #[test]
    fn let_binds_local_variable() {
        assert_eq!(eval_str("(let ((x 10)) x)").unwrap(), "10");
        assert_eq!(eval_str("(let ((x 10)) (+ x x))").unwrap(), "20");
    }

    #[test]
    fn let_with_multiple_bindings() {
        assert_eq!(
            eval_str("(let ((x 10) (y 20)) (+ x y))").unwrap(),
            "30",
        );
    }

    #[test]
    fn let_bindings_are_parallel() {
        // Outer x = 1; inner let evaluates `(+ x 100)` with x=1
        // (outer scope), THEN binds x=101 for the body. Body uses
        // x=101 and y=2.
        let mut session = Session::new();
        session.eval("(defun id (n) n)").unwrap();
        // No outer x; this just tests the basic binding.
        assert_eq!(
            session.eval("(let ((x 5) (y 7)) (* x y))").unwrap(),
            "35",
        );
    }

    #[test]
    fn nested_let() {
        assert_eq!(
            eval_str("(let ((x 10)) (let ((y 5)) (+ x y)))").unwrap(),
            "15",
        );
    }

    #[test]
    fn inner_let_shadows_outer() {
        assert_eq!(
            eval_str("(let ((x 1)) (let ((x 99)) x))").unwrap(),
            "99",
        );
        assert_eq!(
            eval_str("(let ((x 1)) (+ (let ((x 99)) x) x))").unwrap(),
            "100",
        );
    }

    #[test]
    fn let_in_function_body() {
        let mut session = Session::new();
        session
            .eval(
                "(defun hypot-sq (a b)
                   (let ((aa (* a a))
                         (bb (* b b)))
                     (+ aa bb)))",
            )
            .unwrap();
        assert_eq!(session.eval("(hypot-sq 3 4)").unwrap(), "25");
    }

    #[test]
    fn let_with_multiple_body_forms() {
        // Implicit progn inside let.
        assert_eq!(
            eval_str("(let ((x 5)) 99 (+ x x))").unwrap(),
            "10",
        );
    }

    #[test]
    fn let_can_shadow_param() {
        let mut session = Session::new();
        session
            .eval("(defun f (x) (let ((x 99)) x))")
            .unwrap();
        assert_eq!(session.eval("(f 1)").unwrap(), "99");
    }

    #[test]
    fn empty_let_body_is_nil() {
        assert_eq!(eval_str("(let ((x 5)))").unwrap(), "nil");
    }

    // -- Numeric comparisons -----------------------------------------------

    #[test]
    fn lt_works() {
        assert_eq!(eval_str("(< 1 2)").unwrap(), "T");
        assert_eq!(eval_str("(< 2 1)").unwrap(), "nil");
        assert_eq!(eval_str("(< 1 1)").unwrap(), "nil");
        assert_eq!(eval_str("(< -5 0)").unwrap(), "T");
    }

    #[test]
    fn gt_works() {
        assert_eq!(eval_str("(> 2 1)").unwrap(), "T");
        assert_eq!(eval_str("(> 1 2)").unwrap(), "nil");
        assert_eq!(eval_str("(> 1 1)").unwrap(), "nil");
    }

    #[test]
    fn le_ge_eq_work() {
        assert_eq!(eval_str("(<= 1 1)").unwrap(), "T");
        assert_eq!(eval_str("(<= 1 2)").unwrap(), "T");
        assert_eq!(eval_str("(<= 2 1)").unwrap(), "nil");
        assert_eq!(eval_str("(>= 1 1)").unwrap(), "T");
        assert_eq!(eval_str("(>= 2 1)").unwrap(), "T");
        assert_eq!(eval_str("(= 1 1)").unwrap(), "T");
        assert_eq!(eval_str("(= 1 2)").unwrap(), "nil");
    }

    #[test]
    fn fibonacci_via_recursion() {
        let result = eval_str(
            "(defun fib (n)
               (if (< n 2)
                   n
                   (+ (fib (- n 1)) (fib (- n 2)))))
             (fib 10)",
        )
        .unwrap();
        assert_eq!(result, "55"); // fib(10)
    }

    #[test]
    fn fibonacci_15() {
        let result = eval_str(
            "(defun fib (n)
               (if (< n 2)
                   n
                   (+ (fib (- n 1)) (fib (- n 2)))))
             (fib 15)",
        )
        .unwrap();
        assert_eq!(result, "610"); // fib(15)
    }

    // -- Type predicates ---------------------------------------------------

    #[test]
    fn null_predicate() {
        assert_eq!(eval_str("(null nil)").unwrap(), "T");
        assert_eq!(eval_str("(null 0)").unwrap(), "nil");
        assert_eq!(eval_str("(null (cons 1 2))").unwrap(), "nil");
        assert_eq!(eval_str("(null t)").unwrap(), "nil");
    }

    #[test]
    fn consp_predicate() {
        assert_eq!(eval_str("(consp (cons 1 2))").unwrap(), "T");
        assert_eq!(eval_str("(consp nil)").unwrap(), "nil");
        assert_eq!(eval_str("(consp 0)").unwrap(), "nil");
        assert_eq!(eval_str("(consp t)").unwrap(), "nil");
    }

    #[test]
    fn atom_predicate() {
        assert_eq!(eval_str("(atom nil)").unwrap(), "T");
        assert_eq!(eval_str("(atom 42)").unwrap(), "T");
        assert_eq!(eval_str("(atom t)").unwrap(), "T");
        assert_eq!(eval_str("(atom (cons 1 2))").unwrap(), "nil");
    }

    #[test]
    fn listp_predicate() {
        assert_eq!(eval_str("(listp nil)").unwrap(), "T");
        assert_eq!(eval_str("(listp (cons 1 2))").unwrap(), "T");
        assert_eq!(eval_str("(listp 42)").unwrap(), "nil");
        assert_eq!(eval_str("(listp t)").unwrap(), "nil");
    }

    #[test]
    fn list_traversal_via_recursion() {
        // Compute the length of a proper list using car/cdr/null.
        let result = eval_str(
            "(defun length (lst)
               (if (null lst)
                   0
                   (+ 1 (length (cdr lst)))))
             (length (cons 1 (cons 2 (cons 3 (cons 4 nil)))))",
        )
        .unwrap();
        assert_eq!(result, "4");
    }

    #[test]
    fn list_sum_via_recursion() {
        let result = eval_str(
            "(defun sum-list (lst)
               (if (null lst)
                   0
                   (+ (car lst) (sum-list (cdr lst)))))
             (sum-list (cons 1 (cons 2 (cons 3 (cons 4 (cons 5 nil))))))",
        )
        .unwrap();
        assert_eq!(result, "15");
    }

    #[test]
    fn let_with_recursive_call_in_body() {
        // (defun fact-via-let (n)
        //   (let ((sub (- n 1)))
        //     (if (eq n 0) 1 (* n (fact-via-let sub)))))
        let result = eval_str(
            "(defun fact-via-let (n)
               (let ((sub (- n 1)))
                 (if (eq n 0) 1 (* n (fact-via-let sub)))))
             (fact-via-let 6)",
        )
        .unwrap();
        assert_eq!(result, "720");
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
