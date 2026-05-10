//! Typed intermediate representation. The reader produces
//! `ncl_runtime::Value`s; the compiler lowers those to `Expr`; the
//! LLVM back-end lowers `Expr` to LLVM IR.
//!
//! Phase 3 starts with the smallest IR that can represent fixnum
//! arithmetic. As the language grows, this enum grows alongside —
//! it is deliberately not factored into a generic AST framework
//! (see MANIFESTO.md, "When in doubt, be Lisp": don't build a
//! pass framework before you need one).

/// A typed expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// An immediate fixnum.
    Const(i64),
    /// Binary addition (overflows wrap silently in Phase 3; the
    /// trap-and-promote-to-bignum path lands when the numeric
    /// tower does).
    Add(Box<Expr>, Box<Expr>),
    /// Binary subtraction.
    Sub(Box<Expr>, Box<Expr>),
    /// Binary multiplication.
    Mul(Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn add(a: Expr, b: Expr) -> Expr { Expr::Add(Box::new(a), Box::new(b)) }
    pub fn sub(a: Expr, b: Expr) -> Expr { Expr::Sub(Box::new(a), Box::new(b)) }
    pub fn mul(a: Expr, b: Expr) -> Expr { Expr::Mul(Box::new(a), Box::new(b)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_round_trip() {
        let e = Expr::Const(42);
        assert_eq!(e, Expr::Const(42));
    }

    #[test]
    fn add_constructs() {
        let e = Expr::add(Expr::Const(1), Expr::Const(2));
        match e {
            Expr::Add(a, b) => {
                assert_eq!(*a, Expr::Const(1));
                assert_eq!(*b, Expr::Const(2));
            }
            _ => panic!(),
        }
    }
}
