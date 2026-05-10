//! Lowering: `ncl_runtime::Value` → `ncl_ir::Expr`.
//!
//! Phase 3a recognises fixnum literals and three arithmetic
//! operators (`+`, `-`, `*`) in operator position. Variadic
//! arities fold left:
//!
//! ```text
//! (+ a b c)  →  Add(Add(a, b), c)
//! ```
//!
//! Empty `(+)` is `0`, single-arg `(+ x)` is `x`. Same shape for
//! `*` (identity 1) and `-` (single-arg negates). Anything else
//! returns `CompileError::NotImplemented` — the compiler grows as
//! the language does.

use ncl_ir::Expr;
use ncl_runtime::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum CompileError {
    /// We don't yet know how to compile this value.
    NotImplemented(String),
    /// Function applied with the wrong number of arguments.
    BadArity { head: String, expected: &'static str, got: usize },
    /// A list whose tail wasn't a proper list (had a dotted cdr).
    ImproperList(String),
}

/// Lower a single Value to an Expr.
pub fn lower(v: &Value) -> Result<Expr, CompileError> {
    match v {
        Value::Fixnum(n) => Ok(Expr::Const(*n)),
        Value::Nil => Ok(Expr::Nil),
        Value::Cons(_) => lower_call(v),
        other => Err(CompileError::NotImplemented(format!("{other:?}"))),
    }
}

fn lower_call(v: &Value) -> Result<Expr, CompileError> {
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
        "+" => fold_arithmetic(&head_name, args, 0, Expr::add),
        "*" => fold_arithmetic(&head_name, args, 1, Expr::mul),
        "CONS" => {
            if args.len() != 2 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "2",
                    got: args.len(),
                });
            }
            Ok(Expr::cons(lower(&args[0])?, lower(&args[1])?))
        }
        "CAR" => {
            if args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "1",
                    got: args.len(),
                });
            }
            Ok(Expr::car(lower(&args[0])?))
        }
        "CDR" => {
            if args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "1",
                    got: args.len(),
                });
            }
            Ok(Expr::cdr(lower(&args[0])?))
        }
        "-" => match args.len() {
            0 => Err(CompileError::BadArity {
                head: head_name,
                expected: "at least 1",
                got: 0,
            }),
            1 => {
                // `(- x)` is `0 - x`.
                let x = lower(&args[0])?;
                Ok(Expr::sub(Expr::Const(0), x))
            }
            _ => {
                // `(- a b c)` is `((a - b) - c)` — left-fold from a.
                let mut acc = lower(&args[0])?;
                for a in &args[1..] {
                    acc = Expr::sub(acc, lower(a)?);
                }
                Ok(acc)
            }
        },
        other => Err(CompileError::NotImplemented(format!(
            "unknown function: {other}"
        ))),
    }
}

/// `(op)` → `identity`, `(op x)` → `x`, `(op a b c …)` → left-fold.
/// Used for `+` (identity 0) and `*` (identity 1).
fn fold_arithmetic(
    _head: &str,
    args: &[Value],
    identity: i64,
    combine: fn(Expr, Expr) -> Expr,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Ok(Expr::Const(identity));
    }
    let mut acc = lower(&args[0])?;
    for a in &args[1..] {
        acc = combine(acc, lower(a)?);
    }
    Ok(acc)
}

/// Walk a Lisp cons-list into a Vec of its elements. Errors if the
/// cdr-chain has a non-NIL terminator (improper list).
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

    fn read_lower(src: &str) -> Result<Expr, CompileError> {
        let v = read_one(src).expect("read");
        lower(&v)
    }

    #[test]
    fn fixnum_literal() {
        assert_eq!(read_lower("42").unwrap(), Expr::Const(42));
        assert_eq!(read_lower("-7").unwrap(), Expr::Const(-7));
        assert_eq!(read_lower("0").unwrap(), Expr::Const(0));
    }

    #[test]
    fn binary_add() {
        let e = read_lower("(+ 1 2)").unwrap();
        assert_eq!(e, Expr::add(Expr::Const(1), Expr::Const(2)));
    }

    #[test]
    fn nested_add() {
        // (+ (+ 1 2) 3) → Add(Add(1, 2), 3)
        let e = read_lower("(+ (+ 1 2) 3)").unwrap();
        assert_eq!(
            e,
            Expr::add(Expr::add(Expr::Const(1), Expr::Const(2)), Expr::Const(3)),
        );
    }

    #[test]
    fn variadic_add_left_folds() {
        let e = read_lower("(+ 1 2 3 4)").unwrap();
        // ((1 + 2) + 3) + 4
        assert_eq!(
            e,
            Expr::add(
                Expr::add(
                    Expr::add(Expr::Const(1), Expr::Const(2)),
                    Expr::Const(3),
                ),
                Expr::Const(4),
            ),
        );
    }

    #[test]
    fn nullary_arithmetic_is_identity() {
        assert_eq!(read_lower("(+)").unwrap(), Expr::Const(0));
        assert_eq!(read_lower("(*)").unwrap(), Expr::Const(1));
    }

    #[test]
    fn unary_passthrough_for_plus_star() {
        assert_eq!(read_lower("(+ 7)").unwrap(), Expr::Const(7));
        assert_eq!(read_lower("(* 9)").unwrap(), Expr::Const(9));
    }

    #[test]
    fn unary_minus_negates() {
        // (- 5) is 0 - 5
        let e = read_lower("(- 5)").unwrap();
        assert_eq!(e, Expr::sub(Expr::Const(0), Expr::Const(5)));
    }

    #[test]
    fn binary_sub_and_mul() {
        assert_eq!(
            read_lower("(- 10 3)").unwrap(),
            Expr::sub(Expr::Const(10), Expr::Const(3)),
        );
        assert_eq!(
            read_lower("(* 6 7)").unwrap(),
            Expr::mul(Expr::Const(6), Expr::Const(7)),
        );
    }

    #[test]
    fn variadic_sub_left_folds() {
        // (- 100 1 2 3) = ((100 - 1) - 2) - 3
        let e = read_lower("(- 100 1 2 3)").unwrap();
        assert_eq!(
            e,
            Expr::sub(
                Expr::sub(
                    Expr::sub(Expr::Const(100), Expr::Const(1)),
                    Expr::Const(2),
                ),
                Expr::Const(3),
            ),
        );
    }

    #[test]
    fn mixed_arithmetic() {
        // (* (+ 1 2) (- 10 4))
        let e = read_lower("(* (+ 1 2) (- 10 4))").unwrap();
        assert_eq!(
            e,
            Expr::mul(
                Expr::add(Expr::Const(1), Expr::Const(2)),
                Expr::sub(Expr::Const(10), Expr::Const(4)),
            ),
        );
    }

    #[test]
    fn unknown_function_errors() {
        let r = read_lower("(foo 1 2)");
        assert!(matches!(r, Err(CompileError::NotImplemented(_))));
    }

    #[test]
    fn nullary_minus_errors() {
        let r = read_lower("(-)");
        assert!(matches!(r, Err(CompileError::BadArity { .. })));
    }

    #[test]
    fn non_fixnum_atoms_not_yet_supported() {
        let r = read_lower("3.14"); // float
        assert!(matches!(r, Err(CompileError::NotImplemented(_))));
        let r = read_lower(r#""hi""#); // string
        assert!(matches!(r, Err(CompileError::NotImplemented(_))));
    }
}
