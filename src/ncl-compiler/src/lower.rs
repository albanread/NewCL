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

/// Parameter environment for lowering a function body.
#[derive(Debug, Clone, Default)]
pub struct LocalEnv {
    params: Vec<Arc<str>>,
}

impl LocalEnv {
    pub fn empty() -> LocalEnv { LocalEnv { params: Vec::new() } }

    pub fn with_params(names: &[Arc<str>]) -> LocalEnv {
        LocalEnv { params: names.to_vec() }
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| &**p == name)
    }

    pub fn len(&self) -> usize { self.params.len() }
    pub fn is_empty(&self) -> bool { self.params.is_empty() }
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
    match v {
        Value::Fixnum(n) => Ok(Expr::Const(*n)),
        Value::Nil => Ok(Expr::Nil),
        Value::Symbol(s) => {
            // Local parameter first, then T self-evaluating.
            if let Some(idx) = env.find(&s.name) {
                Ok(Expr::Param(idx))
            } else if &*s.name == "T" {
                Ok(Expr::True)
            } else {
                // Global value-cell reference would land here;
                // not yet supported (defparameter / defvar arrive
                // when symbol value cells are wired through).
                Err(CompileError::NotImplemented(format!(
                    "global value reference: {}",
                    s.name
                )))
            }
        }
        Value::Cons(_) => lower_call_in(v, env, coord),
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

fn lower_call_in(
    v: &Value,
    env: &LocalEnv,
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
                lower_in(&args[0], env, coord)?,
                lower_in(&args[1], env, coord)?,
            ))
        }
        "IF" => match args.len() {
            2 => Ok(Expr::if_(
                lower_in(&args[0], env, coord)?,
                lower_in(&args[1], env, coord)?,
                Expr::Nil,
            )),
            3 => Ok(Expr::if_(
                lower_in(&args[0], env, coord)?,
                lower_in(&args[1], env, coord)?,
                lower_in(&args[2], env, coord)?,
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
                lower_in(&args[0], env, coord)?,
                lower_in(&args[1], env, coord)?,
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
            Ok(Expr::car(lower_in(&args[0], env, coord)?))
        }
        "CDR" => {
            if args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "1",
                    got: args.len(),
                });
            }
            Ok(Expr::cdr(lower_in(&args[0], env, coord)?))
        }
        "-" => match args.len() {
            0 => Err(CompileError::BadArity {
                head: head_name,
                expected: "at least 1",
                got: 0,
            }),
            1 => {
                let x = lower_in(&args[0], env, coord)?;
                Ok(Expr::sub(Expr::Const(0), x))
            }
            _ => {
                let mut acc = lower_in(&args[0], env, coord)?;
                for a in &args[1..] {
                    acc = Expr::sub(acc, lower_in(a, env, coord)?);
                }
                Ok(acc)
            }
        },
        // DEFUN is not handled here — it's a top-level meta-form
        // intercepted by the Session driver before lowering. If
        // it appears nested inside an expression, surface a clear
        // error rather than silently miscompiling.
        "DEFUN" => Err(CompileError::BadDefun(
            "(defun ...) is only allowed at top level".into(),
        )),
        // Unknown head: it's a function call. Intern the symbol
        // (allocates a stable Symbol-tagged Word in static if not
        // already there) and emit a Call.
        _ => {
            let sym_word = coord.intern(&head_name);
            let lowered_args: Result<Vec<_>, _> = args
                .iter()
                .map(|a| lower_in(a, env, coord))
                .collect();
            Ok(Expr::call(sym_word.raw(), lowered_args?))
        }
    }
}

fn fold_arithmetic(
    _head: &str,
    args: &[Value],
    env: &LocalEnv,
    coord: &Arc<GcCoordinator>,
    identity: i64,
    combine: fn(Expr, Expr) -> Expr,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Ok(Expr::Const(identity));
    }
    let mut acc = lower_in(&args[0], env, coord)?;
    for a in &args[1..] {
        acc = combine(acc, lower_in(a, env, coord)?);
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
