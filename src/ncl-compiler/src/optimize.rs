//! optimize.rs — Lisp-aware type/representation inference (Slice 1).
//!
//! See docs/compiler_completion.md. LLVM `-O2` does all the generic SSA
//! optimization once we emit IR, but it cannot reason about NCL's 3-bit
//! tag scheme or the heap-float box — those source-language facts are gone
//! by the time it sees `i64` Words. This pass supplies the one fact that
//! pays off most: *which values are provably double-floats*, so codegen
//! can unbox them without the `coerce_to_f64` tag-check diamond.
//!
//! Slice 1 is a forward, flow-insensitive-enough abstract interpretation
//! over the `Expr` tree. It tracks one bit per in-scope `Local` slot —
//! "provably a double-float" — seeded from float literals, already-unboxed
//! float reads, and float arithmetic (CL contagion), and threaded through
//! `Let`/`If`/`Progn`. Reads of a proven-float slot are wrapped in
//! `Expr::TheFloat`, which emit unboxes with a single cell-2 load and no
//! diamond.
//!
//! ## Soundness
//!
//! A wrong wrap would unbox a non-float and read garbage, so the pass is
//! conservative by construction:
//!   * a slot is marked `Float` ONLY from a proven-float init — float
//!     literal, `F64{Param,Local}Read`, or `+`/`-`/`*` where an operand is
//!     proven float (float contagion: if either addend is a float and the
//!     result is returned at all, it is a float);
//!   * only *plain* `Local` reads are wrapped — mutated locals are boxed
//!     cells read via `Car`, so a plain `Local(i)` read is single-assignment
//!     and keeps its binding-init type;
//!   * the env-stack mirrors lowering exactly: every local is introduced by
//!     an `Expr::Let` (the prologue is the outermost `Let`; synthetic temps
//!     are `Let`s too), pushed on entry and truncated on exit, so `Local(i)`
//!     resolves to the right slot even though sibling scopes reuse indices;
//!   * a nested `Lambda` body gets a fresh env (its locals are a separate
//!     scope; outer vars are `ClosureRef`);
//!   * loop bodies suppress wrapping (`in_loop`) — loop-carried slots are
//!     handled by the Slice 2 fixpoint, not yet.
//!
//! Coarsening is always safe; only over-promotion miscompiles. When in
//! doubt this pass returns `Other`.

use ncl_ir::Expr;

/// Slice-1 lattice: just "provably double-float" vs everything else.
/// (The full lattice — Fixnum/Cons/String/⊥/… — arrives in later slices.)
#[derive(Clone, Copy, PartialEq, Eq)]
enum AbsType {
    Float,
    Other,
}

impl AbsType {
    #[inline]
    fn is_float(self) -> bool {
        matches!(self, AbsType::Float)
    }
}

/// Run float-representation inference over a function body, returning the
/// rewritten tree with proven-float `Local` reads wrapped in `TheFloat`.
pub(crate) fn infer_float_unboxing(body: Expr) -> Expr {
    let mut env: Vec<AbsType> = Vec::new();
    walk(body, &mut env, false).1
}

/// Forward walk. `env[i]` is the abstract type of `Local(i)` for in-scope
/// slots. Returns `(inferred type of this expr, rewritten expr)`.
fn walk(e: Expr, env: &mut Vec<AbsType>, in_loop: bool) -> (AbsType, Expr) {
    // Rewrite a boxed child, discarding its type.
    macro_rules! rw {
        ($child:expr) => {
            Box::new(walk(*$child, env, in_loop).1)
        };
    }
    // Rewrite a Vec of children, discarding their types.
    let rw_vec = |v: Vec<Expr>, env: &mut Vec<AbsType>| -> Vec<Expr> {
        v.into_iter().map(|x| walk(x, env, in_loop).1).collect()
    };

    match e {
        // ── proven-float leaves (emit already unboxes these) ──────────────
        Expr::Float { .. } | Expr::F64ParamRead(_) | Expr::F64LocalRead(_) => {
            (AbsType::Float, e)
        }

        // ── local read: the one place a wrap is inserted ──────────────────
        Expr::Local(i) => {
            let t = env.get(i).copied().unwrap_or(AbsType::Other);
            if t.is_float() && !in_loop {
                (AbsType::Float, Expr::TheFloat(Box::new(Expr::Local(i))))
            } else {
                (t, Expr::Local(i))
            }
        }

        // ── let: infer inits (outer env), push, walk body, pop ────────────
        Expr::Let { bindings, body } => {
            let base = env.len();
            let mut new_bindings = Vec::with_capacity(bindings.len());
            let mut init_types = Vec::with_capacity(bindings.len());
            for b in bindings {
                let (t, b2) = walk(b, env, in_loop);
                init_types.push(t);
                new_bindings.push(b2);
            }
            for t in init_types {
                env.push(t);
            }
            let (bt, body2) = walk(*body, env, in_loop);
            env.truncate(base);
            (
                bt,
                Expr::Let {
                    bindings: new_bindings,
                    body: Box::new(body2),
                },
            )
        }

        // ── arithmetic: float contagion ───────────────────────────────────
        Expr::Add(a, b) => arith(*a, *b, env, in_loop, Expr::Add),
        Expr::Sub(a, b) => arith(*a, *b, env, in_loop, Expr::Sub),
        Expr::Mul(a, b) => arith(*a, *b, env, in_loop, Expr::Mul),

        // ── if: both branches in the incoming env; join result types ──────
        Expr::If(c, t, e2) => {
            let c2 = walk(*c, env, in_loop).1;
            let (tt, t2) = walk(*t, env, in_loop);
            let (te, e22) = walk(*e2, env, in_loop);
            let jt = if tt.is_float() && te.is_float() {
                AbsType::Float
            } else {
                AbsType::Other
            };
            (jt, Expr::If(Box::new(c2), Box::new(t2), Box::new(e22)))
        }

        // ── progn: sequence; type is the last form's ──────────────────────
        Expr::Progn(v) => {
            let mut out = Vec::with_capacity(v.len());
            let mut last = AbsType::Other;
            for x in v {
                let (t, x2) = walk(x, env, in_loop);
                last = t;
                out.push(x2);
            }
            (last, Expr::Progn(out))
        }

        // ── lambda: captures in the OUTER env, body in a FRESH scope ──────
        Expr::Lambda {
            arity,
            body,
            captures,
        } => {
            let captures = rw_vec(captures, env);
            let mut inner: Vec<AbsType> = Vec::new();
            let body2 = walk(*body, &mut inner, false).1;
            (
                AbsType::Other,
                Expr::Lambda {
                    arity,
                    body: Box::new(body2),
                    captures,
                },
            )
        }

        // ── loops: walk the body but suppress wraps (Slice 2 does carries) ─
        Expr::FastLoop { test, result, body } => {
            let test = Box::new(walk(*test, env, true).1);
            let result = Box::new(walk(*result, env, true).1);
            let body = Box::new(walk(*body, env, true).1);
            (AbsType::Other, Expr::FastLoop { test, result, body })
        }
        Expr::InlineLoop { body } => {
            let body = Box::new(walk(*body, env, true).1);
            (AbsType::Other, Expr::InlineLoop { body })
        }
        Expr::TailLoop { arity, body } => {
            let body = Box::new(walk(*body, env, true).1);
            (AbsType::Other, Expr::TailLoop { arity, body })
        }
        Expr::LoopBreak { value } => (AbsType::Other, Expr::LoopBreak { value: rw!(value) }),
        Expr::SelfTailNext { args } => {
            (AbsType::Other, Expr::SelfTailNext { args: rw_vec(args, env) })
        }

        // ── dynamic bind: value + body in the current env ─────────────────
        Expr::DynamicBind { sym_word, value, body } => (
            AbsType::Other,
            Expr::DynamicBind {
                sym_word,
                value: rw!(value),
                body: rw!(body),
            },
        ),

        // ── calls / data: recurse to find nested float-locals (type Other) ─
        Expr::Call { sym_word, args } => {
            (AbsType::Other, Expr::Call { sym_word, args: rw_vec(args, env) })
        }
        Expr::Funcall { fn_expr, args } => (
            AbsType::Other,
            Expr::Funcall { fn_expr: rw!(fn_expr), args: rw_vec(args, env) },
        ),
        Expr::Apply { fn_expr, prefix, tail } => (
            AbsType::Other,
            Expr::Apply {
                fn_expr: rw!(fn_expr),
                prefix: rw_vec(prefix, env),
                tail: rw!(tail),
            },
        ),
        Expr::Cons(a, b) => (AbsType::Other, Expr::Cons(rw!(a), rw!(b))),
        Expr::Car(a) => (AbsType::Other, Expr::Car(rw!(a))),
        Expr::Cdr(a) => (AbsType::Other, Expr::Cdr(rw!(a))),
        Expr::SetCar(a, b) => (AbsType::Other, Expr::SetCar(rw!(a), rw!(b))),
        Expr::SetCdr(a, b) => (AbsType::Other, Expr::SetCdr(rw!(a), rw!(b))),
        Expr::Lt(a, b) => (AbsType::Other, Expr::Lt(rw!(a), rw!(b))),
        Expr::Gt(a, b) => (AbsType::Other, Expr::Gt(rw!(a), rw!(b))),
        Expr::Le(a, b) => (AbsType::Other, Expr::Le(rw!(a), rw!(b))),
        Expr::Ge(a, b) => (AbsType::Other, Expr::Ge(rw!(a), rw!(b))),
        Expr::NumEq(a, b) => (AbsType::Other, Expr::NumEq(rw!(a), rw!(b))),
        Expr::Eq(a, b) => (AbsType::Other, Expr::Eq(rw!(a), rw!(b))),
        Expr::Equal(a, b) => (AbsType::Other, Expr::Equal(rw!(a), rw!(b))),
        Expr::Truncate(a, b) => (AbsType::Other, Expr::Truncate(rw!(a), rw!(b))),
        Expr::Rem(a, b) => (AbsType::Other, Expr::Rem(rw!(a), rw!(b))),
        Expr::StoreGlobal { sym_word, value } => (
            AbsType::Other,
            Expr::StoreGlobal { sym_word, value: rw!(value) },
        ),
        Expr::Values(v) => (AbsType::Other, Expr::Values(rw_vec(v, env))),
        Expr::EnsureSingleMv(inner) => {
            (AbsType::Other, Expr::EnsureSingleMv(rw!(inner)))
        }
        Expr::Length(a) => (AbsType::Other, Expr::Length(rw!(a))),
        Expr::IsNull(a) => (AbsType::Other, Expr::IsNull(rw!(a))),
        Expr::IsCons(a) => (AbsType::Other, Expr::IsCons(rw!(a))),
        Expr::IsAtom(a) => (AbsType::Other, Expr::IsAtom(rw!(a))),
        Expr::IsListp(a) => (AbsType::Other, Expr::IsListp(rw!(a))),

        // ── everything else: a sound leaf — no wrap, no recursion ─────────
        // (Misses optimizing nested floats inside exotic nodes, which is
        // safe; later slices can extend coverage.)
        other => (AbsType::Other, other),
    }
}

/// `+`/`-`/`*`: float contagion. If either operand is proven float, the
/// result (when it returns at all) is a float.
fn arith(
    a: Expr,
    b: Expr,
    env: &mut Vec<AbsType>,
    in_loop: bool,
    build: fn(Box<Expr>, Box<Expr>) -> Expr,
) -> (AbsType, Expr) {
    let (ta, a2) = walk(a, env, in_loop);
    let (tb, b2) = walk(b, env, in_loop);
    let t = if ta.is_float() || tb.is_float() {
        AbsType::Float
    } else {
        AbsType::Other
    };
    (t, build(Box::new(a2), Box::new(b2)))
}
