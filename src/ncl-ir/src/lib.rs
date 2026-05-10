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
    /// A pre-tagged raw Word constant. Used for symbol references,
    /// statically-allocated quoted lists, and other compile-time-
    /// resolved values whose tagged bit pattern is known. Emitted
    /// as a single i64 constant.
    Word(u64),
    /// The literal `nil`.
    Nil,
    /// The truth value `t`.
    True,
    /// Reference to the Nth parameter of the current function.
    /// Only valid inside a function body — top-level expressions
    /// don't have parameters and Param is a compile error there.
    Param(usize),
    /// `&rest` accessor: build and return a freshly-allocated list
    /// containing args[start..n_args] in order. Used at the entry
    /// of variadic functions to bind the rest parameter. The
    /// allocation lives in the calling thread's young heap.
    BindRest(u32),
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
    /// Integer division, truncating toward zero. Both operands are
    /// tagged fixnums; the lowering untags, runs LLVM `sdiv`, and
    /// re-tags the quotient. Division by zero is currently UB at
    /// the LLVM level (lands as a hardware SIGFPE / STATUS_INTEGER_
    /// DIVIDE_BY_ZERO) — proper condition signalling waits on the
    /// condition system.
    Truncate(Box<Expr>, Box<Expr>),
    /// Remainder. Result has the sign of the dividend (matches
    /// LLVM `srem` and CL's `rem`). Both operands tagged; result is
    /// already tagged because `srem (a<<3) (b<<3) = (a rem b) << 3`.
    Rem(Box<Expr>, Box<Expr>),
    /// Allocate a cons cell. Calls `ncl_alloc_cons` at runtime.
    Cons(Box<Expr>, Box<Expr>),
    /// Read the car field of a cons.
    Car(Box<Expr>),
    /// Read the cdr field of a cons.
    Cdr(Box<Expr>),
    /// Object identity. Returns `t` if the two operands have the
    /// same Word bits, else `nil`.
    Eq(Box<Expr>, Box<Expr>),
    /// Signed integer comparisons on tagged fixnums. `a < b` etc.
    /// Both operands must already be tagged fixnums (low 3 bits 0);
    /// the comparison is on the raw 64-bit values, which preserves
    /// signed ordering of the un-tagged values because shifting both
    /// by 3 doesn't change relative ordering.
    Lt(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    NumEq(Box<Expr>, Box<Expr>),
    /// Type predicates. Each returns `t` (Word::T) or `nil`.
    /// `null` checks `is x == nil`. `consp` / `atom` / `listp`
    /// check the tag bits (atom = !cons; listp = nil or cons).
    IsNull(Box<Expr>),
    IsCons(Box<Expr>),
    IsAtom(Box<Expr>),
    IsListp(Box<Expr>),
    /// Conditional. If the first sub-expression evaluates to
    /// anything other than `nil`, evaluate the second; else the
    /// third.
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Call a Lisp function via its Symbol's function cell.
    /// `sym_word` is the raw bits of the Symbol-tagged Word
    /// addressing the symbol in static memory; the call dispatches
    /// through `ncl_call` which atomically loads the function cell.
    Call { sym_word: u64, args: Vec<Expr> },
    /// Read a Symbol's value cell (acquire). Used for global
    /// variable references like `*counter*` after a `defparameter`
    /// or `setq`.
    LoadGlobal(u64),
    /// Atomically store a value into a Symbol's value cell
    /// (release). The lowered form of `(setq name value)` and
    /// `(defparameter name value)`.
    StoreGlobal { sym_word: u64, value: Box<Expr> },
    /// Closure reference: read `env[i]` from the current call's env
    /// argument. Only valid inside a lambda body that has at least
    /// `i+1` captures.
    ClosureRef(usize),
    /// Load a Symbol's function cell as a first-class Function
    /// value. The lowered form of `#'name` and `(function name)`.
    /// Calling the returned Word with `funcall` invokes the named
    /// function at the time of THIS lookup (atomic load_acquire on
    /// the symbol's function cell).
    LoadFunction(u64),
    /// Polymorphic `(length s)` — works on strings (codepoint count)
    /// and lists (cons-cell count). Calls `ncl_length`.
    Length(Box<Expr>),
    /// Structural equality. Recurses through cons trees and
    /// compares strings codepoint-by-codepoint; falls back to eq
    /// for other atoms. Calls `ncl_equal`.
    Equal(Box<Expr>, Box<Expr>),
    /// `(string= a b)` — both operands must be strings. Returns T
    /// or NIL.
    StringEq(Box<Expr>, Box<Expr>),
    /// `(string-char s i)` or `(aref s i)` for strings — read the
    /// i-th codepoint as a character.
    StringChar(Box<Expr>, Box<Expr>),
    /// Mutate the car of a cons cell. The lowered form of
    /// `(setf (car x) v)`. Evaluates to the new value.
    SetCar(Box<Expr>, Box<Expr>),
    /// Mutate the cdr of a cons cell. The lowered form of
    /// `(setf (cdr x) v)`.
    SetCdr(Box<Expr>, Box<Expr>),
    /// Mutate the i-th codepoint of a string. The lowered form of
    /// `(setf (aref s i) c)` or `(setf (char s i) c)`. Evaluates to
    /// the new character.
    SetChar {
        s: Box<Expr>,
        idx: Box<Expr>,
        ch: Box<Expr>,
    },
    /// Lambda expression. JIT-compiles `body` as a separate function
    /// with the lambda's signature; at the construction site,
    /// evaluates each `captures[i]` in outer scope and packs the
    /// values into the lambda's env vector.
    Lambda {
        arity: u32,
        body: Box<Expr>,
        captures: Vec<Expr>,
    },
    /// Call a first-class function value. `fn_expr` evaluates to a
    /// Function-tagged Word; the call dispatches through `ncl_funcall`.
    Funcall {
        fn_expr: Box<Expr>,
        args: Vec<Expr>,
    },
}

impl Expr {
    pub fn add(a: Expr, b: Expr) -> Expr { Expr::Add(Box::new(a), Box::new(b)) }
    pub fn sub(a: Expr, b: Expr) -> Expr { Expr::Sub(Box::new(a), Box::new(b)) }
    pub fn mul(a: Expr, b: Expr) -> Expr { Expr::Mul(Box::new(a), Box::new(b)) }
    pub fn truncate(a: Expr, b: Expr) -> Expr {
        Expr::Truncate(Box::new(a), Box::new(b))
    }
    pub fn rem(a: Expr, b: Expr) -> Expr { Expr::Rem(Box::new(a), Box::new(b)) }
    pub fn cons(car: Expr, cdr: Expr) -> Expr { Expr::Cons(Box::new(car), Box::new(cdr)) }
    pub fn car(x: Expr) -> Expr { Expr::Car(Box::new(x)) }
    pub fn cdr(x: Expr) -> Expr { Expr::Cdr(Box::new(x)) }
    pub fn eq(a: Expr, b: Expr) -> Expr { Expr::Eq(Box::new(a), Box::new(b)) }
    pub fn lt(a: Expr, b: Expr) -> Expr { Expr::Lt(Box::new(a), Box::new(b)) }
    pub fn gt(a: Expr, b: Expr) -> Expr { Expr::Gt(Box::new(a), Box::new(b)) }
    pub fn le(a: Expr, b: Expr) -> Expr { Expr::Le(Box::new(a), Box::new(b)) }
    pub fn ge(a: Expr, b: Expr) -> Expr { Expr::Ge(Box::new(a), Box::new(b)) }
    pub fn num_eq(a: Expr, b: Expr) -> Expr { Expr::NumEq(Box::new(a), Box::new(b)) }
    pub fn is_null(x: Expr) -> Expr { Expr::IsNull(Box::new(x)) }
    pub fn is_cons(x: Expr) -> Expr { Expr::IsCons(Box::new(x)) }
    pub fn is_atom(x: Expr) -> Expr { Expr::IsAtom(Box::new(x)) }
    pub fn is_listp(x: Expr) -> Expr { Expr::IsListp(Box::new(x)) }
    pub fn if_(c: Expr, t: Expr, e: Expr) -> Expr {
        Expr::If(Box::new(c), Box::new(t), Box::new(e))
    }
    pub fn call(sym_word: u64, args: Vec<Expr>) -> Expr {
        Expr::Call { sym_word, args }
    }
    pub fn load_global(sym_word: u64) -> Expr { Expr::LoadGlobal(sym_word) }
    pub fn store_global(sym_word: u64, value: Expr) -> Expr {
        Expr::StoreGlobal { sym_word, value: Box::new(value) }
    }
    pub fn closure_ref(idx: usize) -> Expr { Expr::ClosureRef(idx) }
    pub fn load_function(sym_word: u64) -> Expr { Expr::LoadFunction(sym_word) }
    pub fn length(x: Expr) -> Expr { Expr::Length(Box::new(x)) }
    pub fn equal(a: Expr, b: Expr) -> Expr {
        Expr::Equal(Box::new(a), Box::new(b))
    }
    pub fn string_eq(a: Expr, b: Expr) -> Expr {
        Expr::StringEq(Box::new(a), Box::new(b))
    }
    pub fn string_char(s: Expr, i: Expr) -> Expr {
        Expr::StringChar(Box::new(s), Box::new(i))
    }
    pub fn set_car(cons: Expr, value: Expr) -> Expr {
        Expr::SetCar(Box::new(cons), Box::new(value))
    }
    pub fn set_cdr(cons: Expr, value: Expr) -> Expr {
        Expr::SetCdr(Box::new(cons), Box::new(value))
    }
    pub fn set_char(s: Expr, idx: Expr, ch: Expr) -> Expr {
        Expr::SetChar { s: Box::new(s), idx: Box::new(idx), ch: Box::new(ch) }
    }
    pub fn lambda(arity: u32, body: Expr, captures: Vec<Expr>) -> Expr {
        Expr::Lambda { arity, body: Box::new(body), captures }
    }
    pub fn funcall(fn_expr: Expr, args: Vec<Expr>) -> Expr {
        Expr::Funcall { fn_expr: Box::new(fn_expr), args }
    }
    pub fn param(idx: usize) -> Expr { Expr::Param(idx) }
    pub fn bind_rest(start: u32) -> Expr { Expr::BindRest(start) }
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
