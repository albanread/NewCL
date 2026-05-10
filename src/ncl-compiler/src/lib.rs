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
            Word::NIL, // defun'd functions have no closure env
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

    // -- lambda + closures ------------------------------------------------

    #[test]
    fn lambda_no_capture_via_funcall() {
        assert_eq!(eval_str("(funcall (lambda (x) (+ x 1)) 41)").unwrap(), "42");
        assert_eq!(eval_str("(funcall (lambda (x y) (* x y)) 6 7)").unwrap(), "42");
    }

    #[test]
    fn lambda_zero_args() {
        assert_eq!(eval_str("(funcall (lambda () 42))").unwrap(), "42");
    }

    #[test]
    fn lambda_can_be_assigned_and_called() {
        let mut session = Session::new();
        session.eval("(defparameter *square* (lambda (x) (* x x)))").unwrap();
        assert_eq!(session.eval("(funcall *square* 9)").unwrap(), "81");
    }

    #[test]
    fn closure_captures_outer_param() {
        // The classic "make-adder" pattern. Verifies closure capture
        // of a function parameter.
        let result = eval_str(
            "(defun make-adder (n) (lambda (x) (+ x n)))
             (funcall (make-adder 5) 10)",
        )
        .unwrap();
        assert_eq!(result, "15");
    }

    #[test]
    fn closure_captures_let_local() {
        let result = eval_str(
            "(let ((n 100))
               (funcall (lambda (x) (+ x n)) 5))",
        )
        .unwrap();
        assert_eq!(result, "105");
    }

    #[test]
    fn higher_order_compose() {
        let result = eval_str(
            "(defun compose (f g)
               (lambda (x) (funcall f (funcall g x))))
             (funcall (compose (lambda (x) (* x x))
                               (lambda (x) (+ x 1)))
                      4)",
        )
        .unwrap();
        // compose(square, succ)(4) = square(succ(4)) = square(5) = 25
        assert_eq!(result, "25");
    }

    #[test]
    fn map_list_with_lambda() {
        // The first higher-order list operation.
        let result = eval_str(
            "(defun map-list (f lst)
               (if (null lst)
                   nil
                   (cons (funcall f (car lst)) (map-list f (cdr lst)))))
             (map-list (lambda (x) (* x x)) '(1 2 3 4 5))",
        )
        .unwrap();
        assert_eq!(result, "(1 4 9 16 25)");
    }

    #[test]
    fn closure_captures_multiple_outer_vars() {
        let result = eval_str(
            "(defun make-affine (m b)
               (lambda (x) (+ (* m x) b)))
             (funcall (make-affine 3 7) 10)",
        )
        .unwrap();
        // 3 * 10 + 7 = 37
        assert_eq!(result, "37");
    }

    #[test]
    fn nested_closures_inner_captures_outer_lambda_param() {
        // (lambda (x) (lambda (y) (+ x y))) — inner lambda captures
        // outer lambda's param x.
        let result = eval_str(
            "(funcall (funcall (lambda (x) (lambda (y) (+ x y))) 10) 5)",
        )
        .unwrap();
        assert_eq!(result, "15");
    }

    #[test]
    fn closure_used_recursively_via_caller() {
        // A function that takes a function and applies it n times.
        let result = eval_str(
            "(defun apply-n (f x n)
               (if (eq n 0) x (apply-n f (funcall f x) (- n 1))))
             (apply-n (lambda (x) (* x 2)) 1 10)",
        )
        .unwrap();
        // 1 * 2^10 = 1024
        assert_eq!(result, "1024");
    }

    // -- #' (function) — first-class defun'd function values ------------

    #[test]
    fn function_quote_loads_defun_function() {
        let mut session = Session::new();
        session.eval("(defun square (x) (* x x))").unwrap();
        // #'square reads the function cell.
        // Use funcall to invoke it.
        assert_eq!(session.eval("(funcall #'square 9)").unwrap(), "81");
    }

    #[test]
    fn function_quote_long_form() {
        let mut session = Session::new();
        session.eval("(defun cube (x) (* x x x))").unwrap();
        assert_eq!(session.eval("(funcall (function cube) 3)").unwrap(), "27");
    }

    #[test]
    fn map_list_with_defun_via_function_quote() {
        let result = eval_str(
            "(defun square (x) (* x x))
             (defun map-list (f lst)
               (if (null lst) nil
                   (cons (funcall f (car lst)) (map-list f (cdr lst)))))
             (map-list #'square '(1 2 3 4 5))",
        )
        .unwrap();
        assert_eq!(result, "(1 4 9 16 25)");
    }

    #[test]
    fn compose_defun_with_lambda() {
        let result = eval_str(
            "(defun double (x) (+ x x))
             (defun succ (x) (+ x 1))
             (defun compose (f g)
               (lambda (x) (funcall f (funcall g x))))
             (funcall (compose #'double #'succ) 5)",
        )
        .unwrap();
        // compose(double, succ)(5) = double(succ(5)) = double(6) = 12
        assert_eq!(result, "12");
    }

    #[test]
    fn function_quote_redefinition_visibility() {
        let mut session = Session::new();
        session.eval("(defun id (x) x)").unwrap();
        // #'id loads at the time of evaluation. Storing it now
        // captures the CURRENT id; later redefinition is NOT seen
        // by an already-captured #' value.
        session.eval("(defparameter *f* #'id)").unwrap();
        session.eval("(defun id (x) (+ x 100))").unwrap();
        // *f* still refers to the old id (its function-cell value
        // at the time of #'id).
        // Wait — actually no. #'id returns the Function-tagged
        // Word, which IS in the symbol's function cell. The cell
        // got overwritten by the redefinition. The old Function
        // object is now garbage.
        // Hmm, actually the previous Function is in static (never
        // collected). The symbol's cell now points at the new
        // Function. *f* still points at the OLD Function. So
        // (funcall *f* 5) calls the old id which returns 5.
        assert_eq!(session.eval("(funcall *f* 5)").unwrap(), "5");
        // (id 5) goes through the symbol cell — calls the new id.
        assert_eq!(session.eval("(id 5)").unwrap(), "105");
    }

    #[test]
    fn closure_filter_with_predicate() {
        let result = eval_str(
            "(defun filter (pred lst)
               (cond ((null lst) nil)
                     ((funcall pred (car lst))
                      (cons (car lst) (filter pred (cdr lst))))
                     (t (filter pred (cdr lst)))))
             (filter (lambda (x) (> x 3)) '(1 5 2 6 3 7))",
        )
        .unwrap();
        assert_eq!(result, "(5 6 7)");
    }

    // -- Strings -----------------------------------------------------------

    #[test]
    fn ascii_string_round_trip() {
        assert_eq!(eval_str(r#""hello""#).unwrap(), r#""hello""#);
        assert_eq!(eval_str(r#""""#).unwrap(), r#""""#);
        assert_eq!(eval_str(r#""a""#).unwrap(), r#""a""#);
    }

    #[test]
    fn unicode_string_round_trip() {
        // Codepoints preserved end-to-end.
        assert_eq!(eval_str(r#""café""#).unwrap(), r#""café""#);
        assert_eq!(eval_str(r#""日本""#).unwrap(), r#""日本""#);
        assert_eq!(eval_str(r#""🦀""#).unwrap(), r#""🦀""#);
    }

    #[test]
    fn string_length_in_codepoints() {
        assert_eq!(eval_str(r#"(length "hello")"#).unwrap(), "5");
        assert_eq!(eval_str(r#"(length "")"#).unwrap(), "0");
        // Codepoints, NOT bytes — 日本 is 2 codepoints (not 6 UTF-8 bytes).
        assert_eq!(eval_str(r#"(length "日本")"#).unwrap(), "2");
        // 🦀 is 1 codepoint (U+1F980, outside BMP).
        assert_eq!(eval_str(r#"(length "🦀")"#).unwrap(), "1");
    }

    #[test]
    fn length_polymorphic_on_lists() {
        // (length '(a b c)) — 3 cons cells.
        assert_eq!(eval_str(r#"(length '(a b c))"#).unwrap(), "3");
        assert_eq!(eval_str(r#"(length nil)"#).unwrap(), "0");
        assert_eq!(eval_str(r#"(length '(1))"#).unwrap(), "1");
    }

    #[test]
    fn string_eq_works() {
        assert_eq!(eval_str(r#"(string= "foo" "foo")"#).unwrap(), "T");
        assert_eq!(eval_str(r#"(string= "foo" "bar")"#).unwrap(), "nil");
        assert_eq!(eval_str(r#"(string= "" "")"#).unwrap(), "T");
        assert_eq!(eval_str(r#"(string= "café" "café")"#).unwrap(), "T");
    }

    #[test]
    fn char_aref_on_string() {
        // (char s i) reads the i-th codepoint as a character.
        assert_eq!(eval_str(r#"(char "hello" 0)"#).unwrap(), "#\\h");
        assert_eq!(eval_str(r#"(char "hello" 4)"#).unwrap(), "#\\o");
        // aref is the same for strings.
        assert_eq!(eval_str(r#"(aref "hello" 1)"#).unwrap(), "#\\e");
        // Unicode: 4-byte codepoints work.
        assert_eq!(eval_str(r#"(char "café" 3)"#).unwrap(), "#\\é");
        assert_eq!(eval_str(r#"(char "🦀x" 0)"#).unwrap(), "#\\🦀");
    }

    #[test]
    fn string_in_list_prints_correctly() {
        // Strings in proper lists print as elements.
        assert_eq!(eval_str(r#"(list "a" "b" "c")"#).unwrap(), r#"("a" "b" "c")"#);
    }

    #[test]
    fn quoted_string_literal() {
        // '"hello" reads as (quote "hello") and evaluates to "hello".
        assert_eq!(eval_str(r#"'"hello""#).unwrap(), r#""hello""#);
    }

    #[test]
    fn quoted_list_with_strings() {
        assert_eq!(
            eval_str(r#"'("hello" "world")"#).unwrap(),
            r#"("hello" "world")"#,
        );
    }

    #[test]
    fn defparameter_holds_string() {
        let mut session = Session::new();
        session.eval(r#"(defparameter *greeting* "hello")"#).unwrap();
        assert_eq!(session.eval("*greeting*").unwrap(), r#""hello""#);
        assert_eq!(
            session.eval(r#"(string= *greeting* "hello")"#).unwrap(),
            "T",
        );
    }

    #[test]
    fn string_with_escapes_round_trips() {
        // "she said \"hi\"" — the inner quotes need escaping in the
        // printed form too.
        assert_eq!(
            eval_str(r#""she said \"hi\"""#).unwrap(),
            r#""she said \"hi\"""#,
        );
        // Backslash escapes itself.
        assert_eq!(eval_str(r#""back\\slash""#).unwrap(), r#""back\\slash""#);
    }

    #[test]
    fn strings_are_not_eq_even_when_equal() {
        // Each "foo" literal allocates fresh static storage; two
        // distinct strings with the same content are NOT eq.
        assert_eq!(eval_str(r#"(eq "foo" "foo")"#).unwrap(), "nil");
        // string= is the right predicate for content equality.
        assert_eq!(eval_str(r#"(string= "foo" "foo")"#).unwrap(), "T");
    }

    #[test]
    fn function_can_take_string_arg() {
        let mut session = Session::new();
        session
            .eval(r#"(defun greet (name) (string= name "alice"))"#)
            .unwrap();
        assert_eq!(session.eval(r#"(greet "alice")"#).unwrap(), "T");
        assert_eq!(session.eval(r#"(greet "bob")"#).unwrap(), "nil");
    }

    // -- equal: recursive structural equality ------------------------------

    #[test]
    fn equal_on_fixnums() {
        // Same as eq for fixnums.
        assert_eq!(eval_str("(equal 1 1)").unwrap(), "T");
        assert_eq!(eval_str("(equal 1 2)").unwrap(), "nil");
        assert_eq!(eval_str("(equal 0 0)").unwrap(), "T");
    }

    #[test]
    fn equal_on_nil_and_t() {
        assert_eq!(eval_str("(equal nil nil)").unwrap(), "T");
        assert_eq!(eval_str("(equal t t)").unwrap(), "T");
        assert_eq!(eval_str("(equal nil t)").unwrap(), "nil");
    }

    #[test]
    fn equal_on_symbols() {
        assert_eq!(eval_str("(equal 'foo 'foo)").unwrap(), "T");
        assert_eq!(eval_str("(equal 'foo 'bar)").unwrap(), "nil");
    }

    #[test]
    fn equal_on_lists() {
        // equal recurses through cons cells where eq would not.
        assert_eq!(eval_str("(equal '(1 2 3) '(1 2 3))").unwrap(), "T");
        assert_eq!(eval_str("(equal '(1 2 3) '(1 2 4))").unwrap(), "nil");
        assert_eq!(eval_str("(equal '(1 2) '(1 2 3))").unwrap(), "nil");
        // Two distinct list literals — eq says no, equal says yes.
        assert_eq!(eval_str("(eq '(1 2 3) '(1 2 3))").unwrap(), "nil");
    }

    #[test]
    fn equal_on_nested_lists() {
        assert_eq!(
            eval_str("(equal '(1 (2 3)) '(1 (2 3)))").unwrap(),
            "T",
        );
        assert_eq!(
            eval_str("(equal '(1 (2 3)) '(1 (2 4)))").unwrap(),
            "nil",
        );
        assert_eq!(
            eval_str("(equal '((a b) (c d)) '((a b) (c d)))").unwrap(),
            "T",
        );
    }

    #[test]
    fn equal_on_strings() {
        // equal compares strings by content (like string=).
        assert_eq!(eval_str(r#"(equal "foo" "foo")"#).unwrap(), "T");
        assert_eq!(eval_str(r#"(equal "foo" "bar")"#).unwrap(), "nil");
        assert_eq!(eval_str(r#"(equal "" "")"#).unwrap(), "T");
        assert_eq!(eval_str(r#"(equal "café" "café")"#).unwrap(), "T");
    }

    #[test]
    fn equal_mixed_types() {
        // Different types are never equal.
        assert_eq!(eval_str(r#"(equal 1 "1")"#).unwrap(), "nil");
        assert_eq!(eval_str(r#"(equal '(1) 1)"#).unwrap(), "nil");
        assert_eq!(eval_str(r#"(equal nil 0)"#).unwrap(), "nil");
        assert_eq!(eval_str(r#"(equal 'foo "foo")"#).unwrap(), "nil");
    }

    #[test]
    fn equal_lists_of_strings() {
        assert_eq!(
            eval_str(r#"(equal '("a" "b") '("a" "b"))"#).unwrap(),
            "T",
        );
        assert_eq!(
            eval_str(r#"(equal '("a" "b") '("a" "c"))"#).unwrap(),
            "nil",
        );
    }

    #[test]
    fn equal_in_function_body() {
        let mut session = Session::new();
        session
            .eval("(defun same (a b) (equal a b))")
            .unwrap();
        assert_eq!(
            session.eval("(same '(1 2 3) '(1 2 3))").unwrap(),
            "T",
        );
        assert_eq!(
            session.eval(r#"(same "hi" "hi")"#).unwrap(),
            "T",
        );
        assert_eq!(
            session.eval("(same '(1 2) '(1 3))").unwrap(),
            "nil",
        );
    }

    // -- setf: generalised assignment --------------------------------------

    #[test]
    fn setf_on_symbol_acts_like_setq() {
        let mut session = Session::new();
        session.eval("(defparameter *x* 0)").unwrap();
        session.eval("(setf *x* 42)").unwrap();
        assert_eq!(session.eval("*x*").unwrap(), "42");
    }

    #[test]
    fn setf_on_symbol_returns_value() {
        let mut session = Session::new();
        session.eval("(defparameter *x* 0)").unwrap();
        // setf evaluates to the assigned value, like setq.
        assert_eq!(session.eval("(setf *x* 99)").unwrap(), "99");
    }

    #[test]
    fn setf_car_mutates_cons() {
        let mut session = Session::new();
        session.eval("(defparameter *p* (cons 1 2))").unwrap();
        session.eval("(setf (car *p*) 99)").unwrap();
        assert_eq!(session.eval("(car *p*)").unwrap(), "99");
        // cdr unchanged.
        assert_eq!(session.eval("(cdr *p*)").unwrap(), "2");
    }

    #[test]
    fn setf_cdr_mutates_cons() {
        let mut session = Session::new();
        session.eval("(defparameter *p* (cons 1 2))").unwrap();
        session.eval("(setf (cdr *p*) 99)").unwrap();
        assert_eq!(session.eval("(car *p*)").unwrap(), "1");
        assert_eq!(session.eval("(cdr *p*)").unwrap(), "99");
    }

    #[test]
    fn setf_first_and_rest_aliases() {
        let mut session = Session::new();
        session.eval("(defparameter *p* (cons 1 2))").unwrap();
        session.eval("(setf (first *p*) 10)").unwrap();
        session.eval("(setf (rest *p*) 20)").unwrap();
        assert_eq!(session.eval("(car *p*)").unwrap(), "10");
        assert_eq!(session.eval("(cdr *p*)").unwrap(), "20");
    }

    #[test]
    fn setf_returns_new_value_for_cons() {
        // CL: setf returns the value, not the modified container.
        let mut session = Session::new();
        session.eval("(defparameter *p* (cons 1 2))").unwrap();
        assert_eq!(session.eval("(setf (car *p*) 7)").unwrap(), "7");
        assert_eq!(session.eval("(setf (cdr *p*) 8)").unwrap(), "8");
    }

    #[test]
    fn setf_car_on_nested_list() {
        let mut session = Session::new();
        session
            .eval("(defparameter *p* (cons 1 (cons 2 (cons 3 nil))))")
            .unwrap();
        // Mutate the second cell's car.
        session.eval("(setf (car (cdr *p*)) 99)").unwrap();
        assert_eq!(session.eval("*p*").unwrap(), "(1 99 3)");
    }

    #[test]
    fn setf_cdr_can_create_dotted_pair() {
        let mut session = Session::new();
        session.eval("(defparameter *p* (cons 1 nil))").unwrap();
        session.eval("(setf (cdr *p*) 2)").unwrap();
        assert_eq!(session.eval("*p*").unwrap(), "(1 . 2)");
    }

    #[test]
    fn setf_aref_mutates_string() {
        let mut session = Session::new();
        session.eval(r#"(defparameter *s* "hello")"#).unwrap();
        session.eval(r#"(setf (aref *s* 0) #\H)"#).unwrap();
        assert_eq!(session.eval("*s*").unwrap(), r#""Hello""#);
    }

    #[test]
    fn setf_char_mutates_string() {
        let mut session = Session::new();
        session.eval(r#"(defparameter *s* "world")"#).unwrap();
        session.eval(r#"(setf (char *s* 4) #\!)"#).unwrap();
        assert_eq!(session.eval("*s*").unwrap(), r#""worl!""#);
    }

    #[test]
    fn setf_string_returns_char() {
        let mut session = Session::new();
        session.eval(r#"(defparameter *s* "abc")"#).unwrap();
        assert_eq!(
            session.eval(r#"(setf (aref *s* 1) #\X)"#).unwrap(),
            r#"#\X"#,
        );
    }

    #[test]
    fn setf_string_unicode() {
        // Set a Unicode codepoint in a string.
        let mut session = Session::new();
        session.eval(r#"(defparameter *s* "cafe")"#).unwrap();
        session.eval(r#"(setf (aref *s* 3) #\é)"#).unwrap();
        assert_eq!(session.eval("*s*").unwrap(), r#""café""#);
    }

    #[test]
    fn setf_in_function_body() {
        let mut session = Session::new();
        session
            .eval("(defun set-head (p v) (setf (car p) v))")
            .unwrap();
        session.eval("(defparameter *q* (cons 0 0))").unwrap();
        session.eval("(set-head *q* 42)").unwrap();
        assert_eq!(session.eval("(car *q*)").unwrap(), "42");
    }

    #[test]
    fn setf_unsupported_place_errors() {
        // (setf 5 6) — not a place.
        let r = eval_str("(setf 5 6)");
        assert!(r.is_err(), "expected error for non-place setf, got {r:?}");
    }

    #[test]
    fn setf_unknown_form_errors() {
        // No (foo …) setf-expander wired in. Fails to compile.
        let r = eval_str("(setf (foo x) 1)");
        assert!(matches!(
            r,
            Err(EvalError::Compile(CompileError::NotImplemented(_))),
        ));
    }

    // -- Mutable lexical bindings ------------------------------------------

    #[test]
    fn setq_local_let_binding() {
        // (let ((x 0)) (setq x 7) x) — the simplest case.
        assert_eq!(eval_str("(let ((x 0)) (setq x 7) x)").unwrap(), "7");
    }

    #[test]
    fn setf_local_let_binding() {
        assert_eq!(eval_str("(let ((x 1)) (setf x 99) x)").unwrap(), "99");
    }

    #[test]
    fn mutated_let_binding_starts_at_init() {
        // The init value is observable before mutation.
        assert_eq!(
            eval_str("(let ((x 10)) (let ((y x)) (setq x 99) y))").unwrap(),
            "10",
        );
    }

    #[test]
    fn mutated_let_in_function() {
        let mut session = Session::new();
        session
            .eval(
                "(defun count-to (n) \
                   (let ((i 0)) \
                     (if (= i n) i \
                       (progn (setq i n) i))))",
            )
            .unwrap();
        assert_eq!(session.eval("(count-to 5)").unwrap(), "5");
    }

    #[test]
    fn nested_let_shadows_outer_mutation() {
        // Inner setq targets inner x; outer x is untouched.
        assert_eq!(
            eval_str(
                "(let ((x 1)) \
                   (let ((x 100)) (setq x 200)) \
                   x)"
            )
            .unwrap(),
            "1",
        );
    }

    #[test]
    fn nested_let_can_mutate_outer() {
        // Inner scope binds y but not x; setq targets outer x.
        assert_eq!(
            eval_str(
                "(let ((x 1)) \
                   (let ((y 2)) (setq x 99)) \
                   x)"
            )
            .unwrap(),
            "99",
        );
    }

    #[test]
    fn closure_captures_and_mutates() {
        // The make-counter pattern: lambda captures a let-binding
        // and mutates it. Each call increments and returns the new
        // value.
        let mut session = Session::new();
        session
            .eval(
                "(defun make-counter () \
                   (let ((n 0)) \
                     (lambda () (setf n (+ n 1)) n)))",
            )
            .unwrap();
        session.eval("(defparameter *c* (make-counter))").unwrap();
        assert_eq!(session.eval("(funcall *c*)").unwrap(), "1");
        assert_eq!(session.eval("(funcall *c*)").unwrap(), "2");
        assert_eq!(session.eval("(funcall *c*)").unwrap(), "3");
    }

    #[test]
    fn each_counter_has_its_own_state() {
        // Two counters from the same factory share no state — each
        // gets its own boxed cell.
        let mut session = Session::new();
        session
            .eval(
                "(defun make-counter () \
                   (let ((n 0)) \
                     (lambda () (setf n (+ n 1)) n)))",
            )
            .unwrap();
        session.eval("(defparameter *c1* (make-counter))").unwrap();
        session.eval("(defparameter *c2* (make-counter))").unwrap();
        assert_eq!(session.eval("(funcall *c1*)").unwrap(), "1");
        assert_eq!(session.eval("(funcall *c1*)").unwrap(), "2");
        assert_eq!(session.eval("(funcall *c2*)").unwrap(), "1");
        assert_eq!(session.eval("(funcall *c1*)").unwrap(), "3");
    }

    #[test]
    fn closure_reads_outer_mutations() {
        // The outer scope mutates n; the captured lambda sees the
        // new value.
        let mut session = Session::new();
        session
            .eval(
                "(defun make-pair () \
                   (let ((n 0)) \
                     (cons (lambda () n) \
                           (lambda (v) (setf n v)))))",
            )
            .unwrap();
        session.eval("(defparameter *p* (make-pair))").unwrap();
        assert_eq!(session.eval("(funcall (car *p*))").unwrap(), "0");
        session.eval("(funcall (cdr *p*) 42)").unwrap();
        assert_eq!(session.eval("(funcall (car *p*))").unwrap(), "42");
    }

    #[test]
    fn unmutated_let_still_unboxed() {
        // No setq in body — lowering takes the cheap path. We can't
        // assert "no cons allocated" from outside but the test
        // exercises the non-boxed path for coverage.
        assert_eq!(
            eval_str("(let ((a 1) (b 2)) (+ a b))").unwrap(),
            "3",
        );
    }

    #[test]
    fn setq_of_param_still_errors() {
        // Mutable function parameters aren't wired yet — boxing the
        // param at function entry is future work.
        let r = eval_str("((lambda (x) (setq x 1) x) 0)");
        assert!(matches!(
            r,
            Err(EvalError::Compile(CompileError::NotImplemented(_))),
        ));
    }

    #[test]
    fn setq_unbound_local_falls_through_to_global() {
        // No local x. setq targets global *g*. (Defparameter then setq.)
        let mut session = Session::new();
        session.eval("(defparameter *g* 0)").unwrap();
        session.eval("(setq *g* 5)").unwrap();
        assert_eq!(session.eval("*g*").unwrap(), "5");
    }

    // -- defparameter / setq / global value cells --------------------------

    #[test]
    fn defparameter_then_read() {
        let mut session = Session::new();
        session.eval("(defparameter *foo* 42)").unwrap();
        assert_eq!(session.eval("*foo*").unwrap(), "42");
    }

    #[test]
    fn defparameter_returns_value() {
        // Like setq, defparameter returns the assigned value.
        assert_eq!(eval_str("(defparameter *x* 99)").unwrap(), "99");
    }

    #[test]
    fn defparameter_overrides_existing() {
        let result = eval_str(
            "(defparameter *foo* 1)
             (defparameter *foo* 99)
             *foo*",
        )
        .unwrap();
        assert_eq!(result, "99");
    }

    #[test]
    fn setq_assigns() {
        let result = eval_str(
            "(defparameter *x* 1)
             (setq *x* 100)
             *x*",
        )
        .unwrap();
        assert_eq!(result, "100");
    }

    #[test]
    fn setq_returns_assigned_value() {
        let mut session = Session::new();
        session.eval("(defparameter *x* 0)").unwrap();
        assert_eq!(session.eval("(setq *x* 42)").unwrap(), "42");
    }

    #[test]
    fn function_can_modify_global_via_setq() {
        let result = eval_str(
            "(defparameter *counter* 0)
             (defun bump () (setq *counter* (+ *counter* 1)))
             (bump) (bump) (bump)
             *counter*",
        )
        .unwrap();
        assert_eq!(result, "3");
    }

    #[test]
    fn global_value_visible_inside_function() {
        let result = eval_str(
            "(defparameter *base* 10)
             (defun add-base (x) (+ x *base*))
             (add-base 7)",
        )
        .unwrap();
        assert_eq!(result, "17");
    }

    #[test]
    fn local_shadows_global() {
        let result = eval_str(
            "(defparameter *x* 100)
             (defun f (x) x)
             (f 7)",
        )
        .unwrap();
        assert_eq!(result, "7");
    }

    #[test]
    fn let_local_shadows_global() {
        let result = eval_str(
            "(defparameter *x* 100)
             (let ((*x* 5)) *x*)",
        )
        .unwrap();
        assert_eq!(result, "5");
    }

    #[test]
    fn setq_of_param_errors() {
        // Mutable function parameters not yet supported. (Mutable
        // let-locals are — those are tested in the dedicated section
        // above.)
        let r = eval_str(
            "(defun f (x) (setq x 99) x)
             (f 1)",
        );
        assert!(matches!(r, Err(EvalError::Compile(CompileError::NotImplemented(_)))));
    }

    #[test]
    fn quoted_symbol_with_setq() {
        // (setq 'foo 1) should be an error — first arg must be an
        // unquoted symbol literal. But our reader produces `'foo`
        // as `(quote foo)` which is a Cons, not a Symbol — so
        // setq's symbol-check fails with NotImplemented.
        let r = eval_str("(setq 'foo 1)");
        assert!(matches!(r, Err(EvalError::Compile(CompileError::NotImplemented(_)))));
    }

    // -- list, quoted symbols, quoted lists --------------------------------

    #[test]
    fn list_builds_proper_lists() {
        assert_eq!(eval_str("(list)").unwrap(), "nil");
        assert_eq!(eval_str("(list 1)").unwrap(), "(1)");
        assert_eq!(eval_str("(list 1 2 3)").unwrap(), "(1 2 3)");
        assert_eq!(
            eval_str("(list (+ 1 1) (* 3 4) (- 10 1))").unwrap(),
            "(2 12 9)",
        );
    }

    #[test]
    fn quoted_symbol_prints_as_name() {
        assert_eq!(eval_str("'foo").unwrap(), "FOO");
        assert_eq!(eval_str("(quote bar)").unwrap(), "BAR");
        // Case-folding: lowercase source becomes upper-case symbol.
        assert_eq!(eval_str("'Hello").unwrap(), "HELLO");
    }

    #[test]
    fn quoted_symbols_are_eq_when_same_name() {
        // Interning means two `'foo` references resolve to the same
        // Word — `eq` returns T.
        assert_eq!(eval_str("(eq 'foo 'foo)").unwrap(), "T");
        assert_eq!(eval_str("(eq 'foo 'bar)").unwrap(), "nil");
    }

    #[test]
    fn quoted_list_literal() {
        assert_eq!(eval_str("'(1 2 3)").unwrap(), "(1 2 3)");
        assert_eq!(eval_str("'(a b c)").unwrap(), "(A B C)");
        assert_eq!(eval_str("'(1 . 2)").unwrap(), "(1 . 2)");
        assert_eq!(eval_str("'((1 2) (3 4))").unwrap(), "((1 2) (3 4))");
    }

    #[test]
    fn quoted_list_with_mixed_atoms() {
        // Mix fixnums, symbols, nested lists.
        assert_eq!(
            eval_str("'(name 42 (a b) nil)").unwrap(),
            "(NAME 42 (A B) nil)",
        );
    }

    #[test]
    fn quoted_lists_share_static_storage() {
        // Two references to '(1 2 3) intern as the same symbol-
        // table entries, but each `quote` form allocates its own
        // cons chain (we don't yet share). They are distinct cons
        // cells, so eq is nil.
        assert_eq!(eval_str("(eq '(1 2) '(1 2))").unwrap(), "nil");
    }

    #[test]
    fn cond_with_quoted_symbol_branches() {
        let result = eval_str(
            "(defun classify (n)
               (cond ((< n 0) 'negative)
                     ((= n 0) 'zero)
                     (t 'positive)))
             (list (classify -3) (classify 0) (classify 5))",
        )
        .unwrap();
        assert_eq!(result, "(NEGATIVE ZERO POSITIVE)");
    }

    #[test]
    fn member_via_recursion() {
        // (defun member (x lst) ...)  classic CL pattern, manual
        // implementation since `member` isn't a builtin yet.
        let result = eval_str(
            "(defun my-member (x lst)
               (cond ((null lst) nil)
                     ((eq x (car lst)) lst)
                     (t (my-member x (cdr lst)))))
             (my-member 'b '(a b c d))",
        )
        .unwrap();
        // Returns the tail starting with the match.
        assert_eq!(result, "(B C D)");
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

    // -- not / and / or / cond ---------------------------------------------

    #[test]
    fn not_inverts_truthy_and_nil() {
        assert_eq!(eval_str("(not nil)").unwrap(), "T");
        assert_eq!(eval_str("(not t)").unwrap(), "nil");
        assert_eq!(eval_str("(not 0)").unwrap(), "nil"); // 0 is truthy in CL
        assert_eq!(eval_str("(not (cons 1 2))").unwrap(), "nil");
    }

    #[test]
    fn and_returns_last_or_nil() {
        // CL: (and) → t
        assert_eq!(eval_str("(and)").unwrap(), "T");
        // (and x) → x
        assert_eq!(eval_str("(and 5)").unwrap(), "5");
        // (and a b) → b if a is non-nil
        assert_eq!(eval_str("(and t 7)").unwrap(), "7");
        // (and a b) → nil if a is nil
        assert_eq!(eval_str("(and nil 7)").unwrap(), "nil");
        // Multi-arg
        assert_eq!(eval_str("(and 1 2 3 4 5)").unwrap(), "5");
        assert_eq!(eval_str("(and 1 2 nil 4 5)").unwrap(), "nil");
    }

    #[test]
    fn or_returns_first_non_nil_or_nil() {
        assert_eq!(eval_str("(or)").unwrap(), "nil");
        assert_eq!(eval_str("(or 5)").unwrap(), "5");
        assert_eq!(eval_str("(or nil 7)").unwrap(), "7");
        assert_eq!(eval_str("(or 7 nil)").unwrap(), "7");
        assert_eq!(eval_str("(or nil nil 9)").unwrap(), "9");
        assert_eq!(eval_str("(or nil nil nil)").unwrap(), "nil");
    }

    #[test]
    fn or_short_circuits_on_first_truthy() {
        // First non-nil wins. (- 5 5) is fixnum 0, which is TRUTHY
        // in CL — only nil is false — so this returns 0, not the
        // cons. Tests that 0 is truthy AND that or short-circuits.
        assert_eq!(eval_str("(or (- 5 5) (cons 1 2))").unwrap(), "0");
        // With a real nil first, the cons is reached.
        assert_eq!(eval_str("(or nil (cons 1 2))").unwrap(), "(1 . 2)");
    }

    #[test]
    fn cond_picks_first_matching_clause() {
        assert_eq!(eval_str("(cond (t 1))").unwrap(), "1");
        assert_eq!(eval_str("(cond (nil 1) (t 2))").unwrap(), "2");
        assert_eq!(eval_str("(cond (nil 1) (nil 2) (t 3))").unwrap(), "3");
        assert_eq!(eval_str("(cond ((eq 1 2) 10) ((eq 1 1) 20))").unwrap(), "20");
    }

    #[test]
    fn cond_with_no_match_returns_nil() {
        assert_eq!(eval_str("(cond (nil 1) (nil 2))").unwrap(), "nil");
    }

    #[test]
    fn cond_implicit_progn_in_clause() {
        // Multi-form body in a clause uses implicit progn.
        assert_eq!(eval_str("(cond (t 1 2 3))").unwrap(), "3");
    }

    #[test]
    fn boolean_combinations() {
        // (and (or nil 5) (not nil) 7) → 7
        assert_eq!(eval_str("(and (or nil 5) (not nil) 7)").unwrap(), "7");
        // (or (and t nil) 99) → 99
        assert_eq!(eval_str("(or (and t nil) 99)").unwrap(), "99");
    }

    #[test]
    fn cond_with_recursion() {
        // Classic FizzBuzz-style multi-branch. Just two branches
        // for now since we don't have mod yet — return "low",
        // "mid", "high" via fixnums 1/2/3.
        let result = eval_str(
            "(defun classify (n)
               (cond ((< n 0) -1)
                     ((= n 0) 0)
                     ((< n 10) 1)
                     ((< n 100) 2)
                     (t 3)))
             (cons (classify -5)
                   (cons (classify 0)
                         (cons (classify 7)
                               (cons (classify 42)
                                     (cons (classify 1000) nil)))))",
        )
        .unwrap();
        assert_eq!(result, "(-1 0 1 2 3)");
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

    // (Previous "bare unknown symbol fails to compile" test removed
    // — bare globals now lower to LoadGlobal, which panics at
    // runtime when unbound. The panic crosses an FFI boundary and
    // is messy to catch from a unit test, so the behaviour is
    // documented in the commit log instead. Real condition handling
    // arrives later.)
}
