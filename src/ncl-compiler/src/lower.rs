//! Lowering: `ncl_runtime::Value` → `ncl_ir::Expr`.
//!
//! Phase 3a recognised fixnum literals and three arithmetic
//! operators. Cons/car/cdr added the first allocating forms.
//! Eq/if/quote added the first conditional. Now defun and function
//! calls land — `lower` takes a `LocalEnv` (parameter scope) and a
//! `GcCoordinator` (for symbol interning) so it can resolve symbol
//! references and emit calls through the symbol's function cell.
//!
//! Top-level forms are lowered with `LocalEnv::empty()`. Function
//! bodies are lowered with a `LocalEnv` that maps each parameter
//! name to its index. A bare symbol resolves locals first, then
//! special-cases `T`; anything else lowering as a function call
//! against an unknown head goes through symbol intern + `Expr::Call`.

use std::sync::Arc;

use ncl_ir::Expr;
use ncl_runtime::{GcCoordinator, Value};

/// A binding kind in the lexical environment. Parameters live in
/// the function's args array (Param(idx)); let-bound locals live
/// in a stack-vec the emitter manages (Local(idx)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    Param(usize),
    Local(usize),
}

/// Lexical environment for lowering. Tracks names and their
/// binding kinds, in order of introduction. Lookups walk in
/// reverse so inner shadows outer.
#[derive(Debug, Clone, Default)]
pub struct LocalEnv {
    bindings: Vec<(Arc<str>, Binding)>,
    /// Number of `Local` bindings currently in scope. Used by `let`
    /// to assign the next index when extending the env.
    local_count: usize,
}

impl LocalEnv {
    pub fn empty() -> LocalEnv { LocalEnv::default() }

    pub fn with_params(names: &[Arc<str>]) -> LocalEnv {
        let bindings = names
            .iter()
            .enumerate()
            .map(|(i, n)| (Arc::clone(n), Binding::Param(i)))
            .collect();
        LocalEnv { bindings, local_count: 0 }
    }

    pub fn find(&self, name: &str) -> Option<Binding> {
        self.bindings
            .iter()
            .rev()
            .find(|(n, _)| &**n == name)
            .map(|(_, b)| *b)
    }

    /// Push a new local binding. Used during `let` lowering.
    pub fn push_local(&mut self, name: Arc<str>) -> usize {
        let idx = self.local_count;
        self.bindings.push((name, Binding::Local(idx)));
        self.local_count += 1;
        idx
    }

    /// Snapshot the current binding count and local count, for
    /// `let` to roll back on scope exit.
    pub fn checkpoint(&self) -> (usize, usize) {
        (self.bindings.len(), self.local_count)
    }

    pub fn restore(&mut self, cp: (usize, usize)) {
        self.bindings.truncate(cp.0);
        self.local_count = cp.1;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompileError {
    NotImplemented(String),
    BadArity { head: String, expected: &'static str, got: usize },
    ImproperList(String),
    /// `(defun)` form at non-top-level, or malformed.
    BadDefun(String),
}

/// Lower a top-level Value to an Expr (no parameter scope).
pub fn lower(v: &Value, coord: &Arc<GcCoordinator>) -> Result<Expr, CompileError> {
    lower_in(v, &LocalEnv::empty(), coord)
}

/// Lower a Value in the given parameter environment.
pub fn lower_in(
    v: &Value,
    env: &LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    let mut env = env.clone();
    lower_in_mut(v, &mut env, coord)
}

/// Internal lowering with a mutable env (so `let` can extend it
/// in place and roll back at scope exit).
fn lower_in_mut(
    v: &Value,
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    match v {
        Value::Fixnum(n) => Ok(Expr::Const(*n)),
        Value::Nil => Ok(Expr::Nil),
        Value::Symbol(s) => {
            // Local first (param or let-bound), then T.
            if let Some(b) = env.find(&s.name) {
                Ok(match b {
                    Binding::Param(i) => Expr::Param(i),
                    Binding::Local(i) => Expr::Local(i),
                })
            } else if &*s.name == "T" {
                Ok(Expr::True)
            } else {
                Err(CompileError::NotImplemented(format!(
                    "global value reference: {}",
                    s.name
                )))
            }
        }
        Value::Cons(_) => lower_call_in_mut(v, env, coord),
        other => Err(CompileError::NotImplemented(format!("{other:?}"))),
    }
}

/// Lower a quoted form `(quote x)`. Most kinds need symbol
/// resolution or compile-time heap allocation; for v1 we support
/// only the literal kinds that map cleanly to existing IR.
fn lower_quoted(v: &Value) -> Result<Expr, CompileError> {
    match v {
        Value::Fixnum(n) => Ok(Expr::Const(*n)),
        Value::Nil => Ok(Expr::Nil),
        Value::Symbol(s) if &*s.name == "T" => Ok(Expr::True),
        other => Err(CompileError::NotImplemented(format!(
            "quoted {other:?}"
        ))),
    }
}

fn lower_call_in_mut(
    v: &Value,
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    let items = list_to_vec(v)?;
    let head_name = match items.first() {
        Some(Value::Symbol(s)) => s.name.to_string(),
        Some(other) => {
            return Err(CompileError::NotImplemented(format!(
                "non-symbol head: {other:?}"
            )));
        }
        None => return Err(CompileError::NotImplemented("empty list".into())),
    };
    let args = &items[1..];

    match head_name.as_str() {
        "+" => fold_arithmetic(&head_name, args, env, coord, 0, Expr::add),
        "*" => fold_arithmetic(&head_name, args, env, coord, 1, Expr::mul),
        "QUOTE" => {
            if args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "1",
                    got: args.len(),
                });
            }
            lower_quoted(&args[0])
        }
        "EQ" => {
            if args.len() != 2 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "2",
                    got: args.len(),
                });
            }
            Ok(Expr::eq(
                lower_in_mut(&args[0], env, coord)?,
                lower_in_mut(&args[1], env, coord)?,
            ))
        }
        "IF" => match args.len() {
            2 => Ok(Expr::if_(
                lower_in_mut(&args[0], env, coord)?,
                lower_in_mut(&args[1], env, coord)?,
                Expr::Nil,
            )),
            3 => Ok(Expr::if_(
                lower_in_mut(&args[0], env, coord)?,
                lower_in_mut(&args[1], env, coord)?,
                lower_in_mut(&args[2], env, coord)?,
            )),
            n => Err(CompileError::BadArity {
                head: head_name,
                expected: "2 or 3",
                got: n,
            }),
        },
        "CONS" => {
            if args.len() != 2 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "2",
                    got: args.len(),
                });
            }
            Ok(Expr::cons(
                lower_in_mut(&args[0], env, coord)?,
                lower_in_mut(&args[1], env, coord)?,
            ))
        }
        "CAR" => {
            if args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "1",
                    got: args.len(),
                });
            }
            Ok(Expr::car(lower_in_mut(&args[0], env, coord)?))
        }
        "CDR" => {
            if args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "1",
                    got: args.len(),
                });
            }
            Ok(Expr::cdr(lower_in_mut(&args[0], env, coord)?))
        }
        "-" => match args.len() {
            0 => Err(CompileError::BadArity {
                head: head_name,
                expected: "at least 1",
                got: 0,
            }),
            1 => {
                let x = lower_in_mut(&args[0], env, coord)?;
                Ok(Expr::sub(Expr::Const(0), x))
            }
            _ => {
                let mut acc = lower_in_mut(&args[0], env, coord)?;
                for a in &args[1..] {
                    acc = Expr::sub(acc, lower_in_mut(a, env, coord)?);
                }
                Ok(acc)
            }
        },
        "PROGN" => {
            if args.is_empty() {
                return Ok(Expr::Nil);
            }
            let lowered: Result<Vec<_>, _> =
                args.iter().map(|a| lower_in_mut(a, env, coord)).collect();
            Ok(Expr::progn(lowered?))
        }
        // Numeric comparisons. Binary only for now; CL allows
        // variadic with chained semantics ((< 1 2 3) = (and …))
        // — that lands when `and` does.
        "<" => binary_op(&head_name, args, env, coord, Expr::lt),
        ">" => binary_op(&head_name, args, env, coord, Expr::gt),
        "<=" => binary_op(&head_name, args, env, coord, Expr::le),
        ">=" => binary_op(&head_name, args, env, coord, Expr::ge),
        "=" => binary_op(&head_name, args, env, coord, Expr::num_eq),
        // Type predicates. Each takes one argument.
        "NULL" => unary_op(&head_name, args, env, coord, Expr::is_null),
        "CONSP" => unary_op(&head_name, args, env, coord, Expr::is_cons),
        "ATOM" => unary_op(&head_name, args, env, coord, Expr::is_atom),
        "LISTP" => unary_op(&head_name, args, env, coord, Expr::is_listp),
        // EQL is currently the same as EQ — distinctions come
        // when floats/chars/bignums need value-equality semantics.
        "EQL" => binary_op(&head_name, args, env, coord, Expr::eq),
        "LET" => lower_let(args, env, coord),
        "DEFUN" => Err(CompileError::BadDefun(
            "(defun ...) is only allowed at top level".into(),
        )),
        // Unknown head: it's a function call.
        _ => {
            let sym_word = coord.intern(&head_name);
            let lowered_args: Result<Vec<_>, _> = args
                .iter()
                .map(|a| lower_in_mut(a, env, coord))
                .collect();
            Ok(Expr::call(sym_word.raw(), lowered_args?))
        }
    }
}

/// `(let ((var val) ...) body...)`. Bindings evaluated in OUTER env
/// (parallel let, not let*); body sees them as new Locals. Multiple
/// body forms are wrapped in implicit progn.
fn lower_let(
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Err(CompileError::BadArity {
            head: "LET".into(),
            expected: "at least 1 (bindings)",
            got: 0,
        });
    }
    let bindings_form = &args[0];
    let body_forms = &args[1..];

    // Parse bindings list — each entry is (name value).
    let bindings_list = list_to_vec(bindings_form)?;
    let mut binding_exprs: Vec<Expr> = Vec::new();
    let mut binding_names: Vec<Arc<str>> = Vec::new();

    for binding in &bindings_list {
        let pair = list_to_vec(binding)?;
        if pair.len() != 2 {
            return Err(CompileError::NotImplemented(format!(
                "let binding must be (name value), got {pair:?}"
            )));
        }
        let name = match &pair[0] {
            Value::Symbol(s) => Arc::clone(&s.name),
            other => {
                return Err(CompileError::NotImplemented(format!(
                    "let binding name must be a symbol, got {other:?}"
                )));
            }
        };
        // Evaluate value in OUTER env (parallel binding semantics).
        let val_expr = lower_in_mut(&pair[1], env, coord)?;
        binding_exprs.push(val_expr);
        binding_names.push(name);
    }

    // Extend env with new locals, lower body, restore env.
    let cp = env.checkpoint();
    for name in &binding_names {
        env.push_local(Arc::clone(name));
    }

    let body_expr = if body_forms.is_empty() {
        Expr::Nil
    } else if body_forms.len() == 1 {
        lower_in_mut(&body_forms[0], env, coord)?
    } else {
        let lowered: Result<Vec<_>, _> = body_forms
            .iter()
            .map(|f| lower_in_mut(f, env, coord))
            .collect();
        Expr::progn(lowered?)
    };

    env.restore(cp);
    Ok(Expr::let_(binding_exprs, body_expr))
}

fn binary_op(
    head: &str,
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
    build: fn(Expr, Expr) -> Expr,
) -> Result<Expr, CompileError> {
    if args.len() != 2 {
        return Err(CompileError::BadArity {
            head: head.to_string(),
            expected: "2",
            got: args.len(),
        });
    }
    Ok(build(
        lower_in_mut(&args[0], env, coord)?,
        lower_in_mut(&args[1], env, coord)?,
    ))
}

fn unary_op(
    head: &str,
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
    build: fn(Expr) -> Expr,
) -> Result<Expr, CompileError> {
    if args.len() != 1 {
        return Err(CompileError::BadArity {
            head: head.to_string(),
            expected: "1",
            got: args.len(),
        });
    }
    Ok(build(lower_in_mut(&args[0], env, coord)?))
}

fn fold_arithmetic(
    _head: &str,
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
    identity: i64,
    combine: fn(Expr, Expr) -> Expr,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Ok(Expr::Const(identity));
    }
    let mut acc = lower_in_mut(&args[0], env, coord)?;
    for a in &args[1..] {
        acc = combine(acc, lower_in_mut(a, env, coord)?);
    }
    Ok(acc)
}

fn list_to_vec(v: &Value) -> Result<Vec<Value>, CompileError> {
    let mut out = Vec::new();
    let mut cur = v.clone();
    loop {
        match cur {
            Value::Nil => return Ok(out),
            Value::Cons(c) => {
                out.push(c.car.clone());
                cur = c.cdr.clone();
            }
            other => {
                return Err(CompileError::ImproperList(format!("{other:?}")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ncl_reader::read_one;
    use ncl_runtime::{GcConfig, GcCoordinator};

    fn small_coord() -> Arc<GcCoordinator> {
        GcCoordinator::new(GcConfig {
            young_bytes: 16 * 1024,
            old_bytes: 16 * 1024,
            static_bytes: 8 * 1024,
            tlab_cells: 64,
        })
    }

    fn lower_top(src: &str) -> Result<Expr, CompileError> {
        let v = read_one(src).expect("read");
        lower(&v, &small_coord())
    }

    #[test]
    fn fixnum_literal() {
        assert_eq!(lower_top("42").unwrap(), Expr::Const(42));
    }

    #[test]
    fn binary_add() {
        assert_eq!(
            lower_top("(+ 1 2)").unwrap(),
            Expr::add(Expr::Const(1), Expr::Const(2)),
        );
    }

    #[test]
    fn unknown_head_lowers_to_call() {
        let coord = small_coord();
        let v = read_one("(foo 1 2)").unwrap();
        let e = lower(&v, &coord).unwrap();
        match e {
            Expr::Call { sym_word, args } => {
                // The symbol got interned.
                let interned = coord.find_interned("FOO").expect("interned");
                assert_eq!(sym_word, interned.raw());
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Expr::Const(1));
                assert_eq!(args[1], Expr::Const(2));
            }
            other => panic!("expected Call, got {other:?}"),
        }
    }

    #[test]
    fn param_in_function_body() {
        let coord = small_coord();
        let env = LocalEnv::with_params(&[Arc::from("X")]);
        let v = read_one("(+ x x)").unwrap();
        let e = lower_in(&v, &env, &coord).unwrap();
        // Both `x` references should be Param(0).
        assert_eq!(
            e,
            Expr::add(Expr::Param(0), Expr::Param(0)),
        );
    }

    #[test]
    fn param_priority_over_t() {
        // A parameter named T shadows the truth-value literal.
        let coord = small_coord();
        let env = LocalEnv::with_params(&[Arc::from("T")]);
        let e = lower_in(&read_one("t").unwrap(), &env, &coord).unwrap();
        assert_eq!(e, Expr::Param(0));
    }

    #[test]
    fn defun_at_nested_position_errors() {
        let coord = small_coord();
        let v = read_one("(if t (defun foo () 1) 2)").unwrap();
        let r = lower(&v, &coord);
        assert!(matches!(r, Err(CompileError::BadDefun(_))));
    }

    #[test]
    fn eq_and_if_still_work() {
        assert_eq!(
            lower_top("(eq 1 2)").unwrap(),
            Expr::eq(Expr::Const(1), Expr::Const(2)),
        );
        assert_eq!(
            lower_top("(if t 1 2)").unwrap(),
            Expr::if_(Expr::True, Expr::Const(1), Expr::Const(2)),
        );
    }

    #[test]
    fn nil_lowers_to_nil() {
        assert_eq!(lower_top("nil").unwrap(), Expr::Nil);
        assert_eq!(lower_top("()").unwrap(), Expr::Nil);
    }
}
