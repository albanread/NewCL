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
    /// The literal `nil`.
    Nil,
    /// The truth value `t`.
    True,
    /// Reference to the Nth parameter of the current function.
    /// Only valid inside a function body — top-level expressions
    /// don't have parameters and Param is a compile error there.
    Param(usize),
    /// Reference to the Nth let-bound local. Indexed in the order
    /// the let bindings were entered (per nested let scopes), reset
    /// when the let scope exits.
    Local(usize),
    /// Sequential evaluation: each form runs, the last one's value
    /// is the result. Empty progn yields `nil` per CL convention.
    Progn(Vec<Expr>),
    /// Lexical binding scope. Each `bindings[i]` is evaluated in
    /// the *outer* env (parallel binding, like CL's `let`); their
    /// results are pushed to the locals vec for `body` to read via
    /// `Local(prev_top + i)`. Body is single-expression (the
    /// lowering wraps multiple body forms in a `Progn`).
    Let { bindings: Vec<Expr>, body: Box<Expr> },
    /// Binary addition (overflows wrap silently in Phase 3; the
    /// trap-and-promote-to-bignum path lands when the numeric
    /// tower does).
    Add(Box<Expr>, Box<Expr>),
    /// Binary subtraction.
    Sub(Box<Expr>, Box<Expr>),
    /// Binary multiplication.
    Mul(Box<Expr>, Box<Expr>),
    /// Allocate a cons cell. Calls `ncl_alloc_cons` at runtime.
    Cons(Box<Expr>, Box<Expr>),
    /// Read the car field of a cons.
    Car(Box<Expr>),
    /// Read the cdr field of a cons.
    Cdr(Box<Expr>),
    /// Object identity. Returns `t` if the two operands have the
    /// same Word bits, else `nil`.
    Eq(Box<Expr>, Box<Expr>),
    /// Conditional. If the first sub-expression evaluates to
    /// anything other than `nil`, evaluate the second; else the
    /// third.
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Call a Lisp function via its Symbol's function cell.
    /// `sym_word` is the raw bits of the Symbol-tagged Word
    /// addressing the symbol in static memory; the call dispatches
    /// through `ncl_call` which atomically loads the function cell.
    Call { sym_word: u64, args: Vec<Expr> },
}

impl Expr {
    pub fn add(a: Expr, b: Expr) -> Expr { Expr::Add(Box::new(a), Box::new(b)) }
    pub fn sub(a: Expr, b: Expr) -> Expr { Expr::Sub(Box::new(a), Box::new(b)) }
    pub fn mul(a: Expr, b: Expr) -> Expr { Expr::Mul(Box::new(a), Box::new(b)) }
    pub fn cons(car: Expr, cdr: Expr) -> Expr { Expr::Cons(Box::new(car), Box::new(cdr)) }
    pub fn car(x: Expr) -> Expr { Expr::Car(Box::new(x)) }
    pub fn cdr(x: Expr) -> Expr { Expr::Cdr(Box::new(x)) }
    pub fn eq(a: Expr, b: Expr) -> Expr { Expr::Eq(Box::new(a), Box::new(b)) }
    pub fn if_(c: Expr, t: Expr, e: Expr) -> Expr {
        Expr::If(Box::new(c), Box::new(t), Box::new(e))
    }
    pub fn call(sym_word: u64, args: Vec<Expr>) -> Expr {
        Expr::Call { sym_word, args }
    }
    pub fn param(idx: usize) -> Expr { Expr::Param(idx) }
    pub fn local(idx: usize) -> Expr { Expr::Local(idx) }
    pub fn progn(forms: Vec<Expr>) -> Expr { Expr::Progn(forms) }
    pub fn let_(bindings: Vec<Expr>, body: Expr) -> Expr {
        Expr::Let { bindings, body: Box::new(body) }
    }
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
