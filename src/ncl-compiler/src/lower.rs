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

use std::collections::HashSet;
use std::sync::Arc;

use ncl_ir::Expr;
use ncl_runtime::{GcCoordinator, Value, Word};

/// A binding kind in the lexical environment.
///
/// The `*Cell` variants represent variables that have been boxed
/// because they are the target of `setq`/`setf`. The slot itself
/// holds a 1-cell cons (init . nil); reads emit `(car slot)` and
/// writes emit `(rplaca slot new)`. Boxing is mandatory for any
/// mutated binding because the same cell may be shared with an
/// inner lambda's closure env — a non-boxed value would diverge
/// across the inner/outer copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    /// Function parameter — read from the call's args[idx].
    Param(usize),
    /// Let-bound local — read from the emitter's locals stack[idx].
    Local(usize),
    /// Closure reference — read from env[idx] in the current
    /// function's closure environment.
    ClosureRef(usize),
    /// Function parameter, boxed for mutation. Reserved — the param
    /// boxing prologue isn't wired yet, so mutating a param still
    /// errors at compile time. Present so the binding-kind enum
    /// covers the full design.
    ParamCell(usize),
    /// Let-bound local, boxed for mutation.
    LocalCell(usize),
    /// Closure-captured cell. The captured Word IS the cons; reads
    /// auto-deref, writes go through SetCar.
    ClosureRefCell(usize),
}

/// Lexical environment for lowering. Tracks names → bindings.
/// Lambda bodies install a `capture_parent` that lookups fall
/// back to: any outer-env name referenced inside the lambda is
/// added to the lambda's captures list and recorded as a new
/// `ClosureRef` binding.
#[derive(Debug, Clone, Default)]
pub struct LocalEnv {
    bindings: Vec<(Arc<str>, Binding)>,
    local_count: usize,
    closure_count: usize,
    /// Set when this env is the inner env of a lambda body. Lookups
    /// that miss in `bindings` fall back to here, capturing on hit.
    capture_parent: Option<Box<LocalEnv>>,
    /// Outer-scope expressions corresponding to ClosureRef indices.
    /// Each is evaluated at lambda-creation time and stored in the
    /// closure's env vector.
    captures: Vec<Expr>,
}

impl LocalEnv {
    pub fn empty() -> LocalEnv { LocalEnv::default() }

    pub fn with_params(names: &[Arc<str>]) -> LocalEnv {
        let bindings = names
            .iter()
            .enumerate()
            .map(|(i, n)| (Arc::clone(n), Binding::Param(i)))
            .collect();
        LocalEnv {
            bindings,
            local_count: 0,
            closure_count: 0,
            capture_parent: None,
            captures: Vec::new(),
        }
    }

    /// Build an env for lowering a lambda body. The lambda's own
    /// params are at Param(0..N); names not bound here fall back
    /// to `parent` and become captures (ClosureRef).
    pub fn for_lambda(params: &[Arc<str>], parent: LocalEnv) -> LocalEnv {
        let bindings = params
            .iter()
            .enumerate()
            .map(|(i, n)| (Arc::clone(n), Binding::Param(i)))
            .collect();
        LocalEnv {
            bindings,
            local_count: 0,
            closure_count: 0,
            capture_parent: Some(Box::new(parent)),
            captures: Vec::new(),
        }
    }

    /// Read-only lookup in this env's bindings only. Does NOT
    /// consult capture_parent.
    pub fn find_local(&self, name: &str) -> Option<Binding> {
        self.bindings
            .iter()
            .rev()
            .find(|(n, _)| &**n == name)
            .map(|(_, b)| *b)
    }

    /// Lookup, with capture from `capture_parent` if needed. May
    /// add to `captures` and `bindings` (returning the new
    /// ClosureRef). Used by `lower_in_mut` for symbol resolution.
    ///
    /// When the parent binding is a Cell variant, the captured slot
    /// holds the cons (the box); the inner binding is a
    /// `ClosureRefCell` so reads and writes auto-deref through it.
    pub fn find_or_capture(&mut self, name: &str) -> Option<Binding> {
        if let Some(b) = self.find_local(name) {
            return Some(b);
        }
        let parent = self.capture_parent.as_ref()?;
        let parent_b = parent.find_local(name)?;
        let idx = self.closure_count;
        let (outer_expr, inner_binding) = match parent_b {
            Binding::Param(i) => (Expr::Param(i), Binding::ClosureRef(idx)),
            Binding::Local(i) => (Expr::Local(i), Binding::ClosureRef(idx)),
            Binding::ClosureRef(i) => (Expr::ClosureRef(i), Binding::ClosureRef(idx)),
            // Cell variants: capture the cons itself; inner sees a Cell.
            Binding::ParamCell(i) => (Expr::Param(i), Binding::ClosureRefCell(idx)),
            Binding::LocalCell(i) => (Expr::Local(i), Binding::ClosureRefCell(idx)),
            Binding::ClosureRefCell(i) => (Expr::ClosureRef(i), Binding::ClosureRefCell(idx)),
        };
        self.captures.push(outer_expr);
        self.bindings.push((Arc::from(name), inner_binding));
        self.closure_count += 1;
        Some(inner_binding)
    }

    /// Compatibility wrapper: callers that don't care about
    /// captures can still use `find`. Same as `find_local`.
    pub fn find(&self, name: &str) -> Option<Binding> {
        self.find_local(name)
    }

    pub fn push_local(&mut self, name: Arc<str>) -> usize {
        let idx = self.local_count;
        self.bindings.push((name, Binding::Local(idx)));
        self.local_count += 1;
        idx
    }

    /// Push a let-binding that needs boxing (because it's a setq /
    /// setf target somewhere in the body, possibly inside a captured
    /// lambda). The slot is a `LocalCell`; the caller is expected to
    /// have already wrapped the binding's init expression in a cons.
    pub fn push_local_cell(&mut self, name: Arc<str>) -> usize {
        let idx = self.local_count;
        self.bindings.push((name, Binding::LocalCell(idx)));
        self.local_count += 1;
        idx
    }

    pub fn checkpoint(&self) -> (usize, usize) {
        (self.bindings.len(), self.local_count)
    }

    pub fn restore(&mut self, cp: (usize, usize)) {
        self.bindings.truncate(cp.0);
        self.local_count = cp.1;
    }

    /// Take ownership of the captures list. Called after lambda-
    /// body lowering completes, to extract the captures for the
    /// resulting `Expr::Lambda`.
    pub fn take_captures(&mut self) -> Vec<Expr> {
        std::mem::take(&mut self.captures)
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
        Value::Char(c) => Ok(Expr::Word(ncl_runtime::Word::char(*c).raw())),
        Value::String(s) => {
            let w = ncl_runtime::gc_string::alloc_string_in_static(
                coord.static_area(),
                s.as_str(),
            )
            .ok_or_else(|| {
                CompileError::NotImplemented(
                    "static area exhausted while allocating string literal".into(),
                )
            })?;
            Ok(Expr::Word(w.raw()))
        }
        Value::Symbol(s) => {
            // Local first (param/let/closure-capture), then T, then
            // global value-cell load. Capture happens lazily — if
            // we're inside a lambda body and the name resolves in
            // the parent env, find_or_capture adds it as a closure
            // capture and returns ClosureRef.
            if let Some(b) = env.find_or_capture(&s.name) {
                Ok(match b {
                    Binding::Param(i) => Expr::Param(i),
                    Binding::Local(i) => Expr::Local(i),
                    Binding::ClosureRef(i) => Expr::ClosureRef(i),
                    // Cell variants auto-deref through (car cell).
                    Binding::ParamCell(i) => Expr::car(Expr::Param(i)),
                    Binding::LocalCell(i) => Expr::car(Expr::Local(i)),
                    Binding::ClosureRefCell(i) => Expr::car(Expr::ClosureRef(i)),
                })
            } else if &*s.name == "T" {
                Ok(Expr::True)
            } else {
                let sym_word = coord.intern(&s.name);
                Ok(Expr::load_global(sym_word.raw()))
            }
        }
        Value::Cons(_) => lower_call_in_mut(v, env, coord),
        other => Err(CompileError::NotImplemented(format!("{other:?}"))),
    }
}

/// Lower a quoted form `(quote x)`. Symbols are interned in the
/// coordinator's static area (returning a stable Symbol-tagged
/// Word). Cons-shaped data is built recursively in static at
/// compile time — each cons cell allocated via
/// `static_area.try_alloc_cons`. This means a `'(1 2 3)` literal
/// lives forever and is shared across all references; no GC
/// pressure, and `(eq '(1 2 3) '(1 2 3))` may even be true if the
/// compiler shares them (it doesn't yet, but could).
fn lower_quoted(v: &Value, coord: &Arc<GcCoordinator>) -> Result<Expr, CompileError> {
    let word = build_quoted_word(v, coord)?;
    // Specialise the IR variant where it's cleaner; otherwise emit
    // a raw Word constant.
    Ok(match word {
        w if w.is_nil() => Expr::Nil,
        w if w.is_t() => Expr::True,
        w if w.is_fixnum() => Expr::Const(w.as_fixnum().unwrap()),
        w => Expr::Word(w.raw()),
    })
}

/// Build a Word for a quoted Value, allocating compound structure
/// in the coordinator's static area as needed. Recurses into cons
/// cells.
fn build_quoted_word(
    v: &Value,
    coord: &Arc<GcCoordinator>,
) -> Result<Word, CompileError> {
    match v {
        Value::Fixnum(n) => Ok(Word::fixnum(*n)),
        Value::Nil => Ok(Word::NIL),
        Value::Symbol(s) if &*s.name == "T" => Ok(Word::T),
        Value::Symbol(s) => Ok(coord.intern(&s.name)),
        Value::Cons(c) => {
            let car_word = build_quoted_word(&c.car, coord)?;
            let cdr_word = build_quoted_word(&c.cdr, coord)?;
            coord
                .static_area()
                .try_alloc_cons(car_word, cdr_word)
                .ok_or_else(|| {
                    CompileError::NotImplemented(
                        "static area exhausted while building quoted list".into(),
                    )
                })
        }
        Value::Char(c) => Ok(Word::char(*c)),
        Value::String(s) => ncl_runtime::gc_string::alloc_string_in_static(
            coord.static_area(),
            s.as_str(),
        )
        .ok_or_else(|| {
            CompileError::NotImplemented(
                "static area exhausted while allocating quoted string".into(),
            )
        }),
        other => Err(CompileError::NotImplemented(format!(
            "quoted sub-value: {other:?}"
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
            lower_quoted(&args[0], coord)
        }
        "LIST" => {
            // (list a b c) ≡ (cons a (cons b (cons c nil)))
            // Pure source-level desugaring.
            let mut result = Expr::Nil;
            for arg in args.iter().rev() {
                let lowered = lower_in_mut(arg, env, coord)?;
                result = Expr::cons(lowered, result);
            }
            Ok(result)
        }
        "LAMBDA" => lower_lambda(args, env, coord),
        "FUNCALL" => lower_funcall(args, env, coord),
        // (function name)  — equivalent to #'name in source. Loads
        // the symbol's function cell as a first-class value.
        "FUNCTION" => {
            if args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "1",
                    got: args.len(),
                });
            }
            let name = match &args[0] {
                Value::Symbol(s) => Arc::clone(&s.name),
                other => {
                    return Err(CompileError::NotImplemented(format!(
                        "(function …) requires a symbol, got {other:?}"
                    )));
                }
            };
            let sym_word = coord.intern(&name);
            Ok(Expr::load_function(sym_word.raw()))
        }
        // (setq name value)  — assign the value cell of `name`. CL
        // also allows (setq a 1 b 2 …) for parallel assignments;
        // v1 supports the binary form only.
        "SETQ" => lower_setq(&head_name, args, env, coord),
        // (setf place value)  — generalised assignment. We dispatch
        // by syntactic shape of `place`: a bare symbol falls through
        // to setq; (car x) / (cdr x) / (aref s i) / (char s i)
        // emit the matching mutation primitive.
        "SETF" => lower_setf(args, env, coord),
        // (defparameter name value [doc])
        // (defvar name value [doc])
        // For v1 both behave like setq — they install/replace the
        // value cell and ignore the optional doc string. The
        // distinction (defvar only sets if unbound) lands later.
        "DEFPARAMETER" | "DEFVAR" => lower_defparameter(&head_name, args, env, coord),
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
        // Boolean operators desugar to existing IR (if + let). No
        // new Expr variants, no LLVM changes — pure lowering.
        "NOT" => {
            if args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: head_name,
                    expected: "1",
                    got: args.len(),
                });
            }
            // (not x) ≡ (if x nil t)
            Ok(Expr::if_(
                lower_in_mut(&args[0], env, coord)?,
                Expr::Nil,
                Expr::True,
            ))
        }
        "AND" => lower_and(args, env, coord),
        "OR" => lower_or(args, env, coord),
        "COND" => lower_cond(args, env, coord),
        // (when test body...) ≡ (if test (progn body...) nil)
        // (unless test body...) ≡ (if test nil (progn body...))
        // Implicit progn over body forms; empty body yields nil.
        "WHEN" => lower_when(&head_name, args, env, coord, /*invert*/ false),
        "UNLESS" => lower_when(&head_name, args, env, coord, /*invert*/ true),
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
        "LENGTH" => unary_op(&head_name, args, env, coord, Expr::length),
        "EQUAL" => binary_op(&head_name, args, env, coord, Expr::equal),
        "STRING=" => binary_op(&head_name, args, env, coord, Expr::string_eq),
        // (char s i) and (aref s i) for strings — until we have
        // generic vectors, both forms route here.
        "CHAR" | "STRING-CHAR" | "AREF" => {
            binary_op(&head_name, args, env, coord, Expr::string_char)
        }
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
/// Walk a Value form and collect every name that appears as the
/// target of a mutating special form. The `shadowed` set records
/// names that have been re-bound by an enclosing scope inside the
/// form being walked — those don't refer to the outer binding so
/// their mutations don't count.
///
/// Recognised mutators:
///   - `(setq sym v)`              — sym is a target
///   - `(setf sym v)`              — sym is a target (place is a symbol)
///
/// Place-form setfs like `(setf (car x) v)` are NOT counted: those
/// mutate the *contents* of x, not the binding x itself.
///
/// Recognised scoping forms (these extend `shadowed`):
///   - `(let ((n1 v1) ...) body...)`
///   - `(lambda (params) body...)`
///   - `(defun name (params) body...)`
///
/// `(quote ...)` is skipped — quoted forms aren't evaluated.
fn collect_mutations(
    form: &Value,
    shadowed: &HashSet<Arc<str>>,
    into: &mut HashSet<Arc<str>>,
) {
    let items = match form {
        Value::Cons(_) => match list_to_vec(form) {
            Ok(v) => v,
            Err(_) => return, // improper list — nothing to analyse
        },
        _ => return, // atoms have no mutating forms inside
    };
    let head_name = match items.first() {
        Some(Value::Symbol(s)) => s.name.to_string(),
        _ => {
            for it in &items {
                collect_mutations(it, shadowed, into);
            }
            return;
        }
    };
    let args = &items[1..];

    match head_name.as_str() {
        "QUOTE" => {}
        "SETQ" if args.len() == 2 => {
            if let Value::Symbol(s) = &args[0] {
                if !shadowed.contains(&s.name) {
                    into.insert(Arc::clone(&s.name));
                }
            }
            collect_mutations(&args[1], shadowed, into);
        }
        "SETF" if args.len() == 2 => {
            if let Value::Symbol(s) = &args[0] {
                if !shadowed.contains(&s.name) {
                    into.insert(Arc::clone(&s.name));
                }
            } else {
                // (setf (place ...) v) — recurse into both place
                // sub-args and the new-value form.
                collect_mutations(&args[0], shadowed, into);
            }
            collect_mutations(&args[1], shadowed, into);
        }
        "LET" if !args.is_empty() => {
            if let Ok(bindings) = list_to_vec(&args[0]) {
                let mut new_shadowed = shadowed.clone();
                for binding in &bindings {
                    if let Ok(pair) = list_to_vec(binding) {
                        if pair.len() == 2 {
                            // Init expr sees OUTER scope.
                            collect_mutations(&pair[1], shadowed, into);
                            if let Value::Symbol(s) = &pair[0] {
                                new_shadowed.insert(Arc::clone(&s.name));
                            }
                        }
                    }
                }
                for body_form in &args[1..] {
                    collect_mutations(body_form, &new_shadowed, into);
                }
            }
        }
        "LAMBDA" if !args.is_empty() => {
            let mut new_shadowed = shadowed.clone();
            if let Ok(params) = list_to_vec(&args[0]) {
                for param in &params {
                    if let Value::Symbol(s) = param {
                        new_shadowed.insert(Arc::clone(&s.name));
                    }
                }
            }
            for body_form in &args[1..] {
                collect_mutations(body_form, &new_shadowed, into);
            }
        }
        "DEFUN" if args.len() >= 2 => {
            // (defun name (params) body...). Params shadow; defun's
            // own name does not (it's a global side effect, not a
            // lexical binding).
            let mut new_shadowed = shadowed.clone();
            if let Ok(params) = list_to_vec(&args[1]) {
                for param in &params {
                    if let Value::Symbol(s) = param {
                        new_shadowed.insert(Arc::clone(&s.name));
                    }
                }
            }
            for body_form in &args[2..] {
                collect_mutations(body_form, &new_shadowed, into);
            }
        }
        _ => {
            for it in args {
                collect_mutations(it, shadowed, into);
            }
        }
    }
}

/// Run `collect_mutations` over a slice of body forms, returning the
/// union. Each form is analysed with the same starting shadowed set
/// (no carry-over since these are siblings, not nested).
fn mutated_in_body(
    body_forms: &[Value],
    shadowed: &HashSet<Arc<str>>,
) -> HashSet<Arc<str>> {
    let mut out = HashSet::new();
    for f in body_forms {
        collect_mutations(f, shadowed, &mut out);
    }
    out
}

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

    // Determine which of OUR bindings are mutated anywhere reachable
    // from the body — including from inside nested lambdas. Those
    // names need boxing; init becomes (cons init nil) and the
    // binding becomes a LocalCell.
    let mutated = mutated_in_body(body_forms, &HashSet::new());
    let needs_box: Vec<bool> = binding_names
        .iter()
        .map(|n| mutated.contains(n))
        .collect();
    for (i, b) in needs_box.iter().enumerate() {
        if *b {
            // Wrap init in (cons init nil) — the cell representation.
            let init = std::mem::replace(&mut binding_exprs[i], Expr::Nil);
            binding_exprs[i] = Expr::cons(init, Expr::Nil);
        }
    }

    // Extend env with new locals (cell or plain), lower body, restore env.
    let cp = env.checkpoint();
    for (i, name) in binding_names.iter().enumerate() {
        if needs_box[i] {
            env.push_local_cell(Arc::clone(name));
        } else {
            env.push_local(Arc::clone(name));
        }
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

/// `(and)` → `t`. `(and x)` → `x`. `(and a b c)` short-circuits:
/// if `a` is nil, return nil; else if `b` is nil, return nil; else
/// return `c` (evaluated). Each form evaluated at most once.
///
/// Desugars to nested `if`. No `let` needed because each form
/// appears exactly once in the lowered tree.
fn lower_and(
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Ok(Expr::True);
    }
    if args.len() == 1 {
        return lower_in_mut(&args[0], env, coord);
    }
    let first = lower_in_mut(&args[0], env, coord)?;
    let rest = lower_and(&args[1..], env, coord)?;
    Ok(Expr::if_(first, rest, Expr::Nil))
}

/// `(or)` → nil. `(or x)` → `x`. `(or a b c)` returns the first
/// non-nil value, or nil if none are. Each form evaluated at most
/// once — that means we can't write `(if a a (or b c))` (would
/// evaluate `a` twice if non-nil); we use `let` to bind the result
/// of evaluating the head.
fn lower_or(
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Ok(Expr::Nil);
    }
    if args.len() == 1 {
        return lower_in_mut(&args[0], env, coord);
    }
    // Lower head in OUTER env first.
    let first = lower_in_mut(&args[0], env, coord)?;
    // Push a synthetic local for the head's value, lower the rest
    // with the local in scope, then restore.
    let cp = env.checkpoint();
    let tmp_idx = env.push_local(std::sync::Arc::from("__or_tmp__"));
    let rest = lower_or(&args[1..], env, coord)?;
    env.restore(cp);
    // (let ((tmp first)) (if tmp tmp rest))
    Ok(Expr::let_(
        vec![first],
        Expr::if_(Expr::Local(tmp_idx), Expr::Local(tmp_idx), rest),
    ))
}

/// `(cond)` → nil. Each clause is `(test form...)`. The first
/// clause whose test evaluates to non-nil has its body forms
/// evaluated as an implicit progn; the value of the last form is
/// the cond's result. If no test matches, the result is nil.
fn lower_cond(
    clauses: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    if clauses.is_empty() {
        return Ok(Expr::Nil);
    }
    let clause = list_to_vec(&clauses[0])?;
    if clause.is_empty() {
        return Err(CompileError::NotImplemented(
            "cond clause must have at least a test".to_string(),
        ));
    }
    let test = lower_in_mut(&clause[0], env, coord)?;
    // Body: implicit progn of forms after the test.
    // CL's `(test)` (clause with only a test) would return test's
    // value if non-nil; defer that case.
    let body = if clause.len() == 1 {
        return Err(CompileError::NotImplemented(
            "cond clause with only a test (no body) not yet supported"
                .to_string(),
        ));
    } else if clause.len() == 2 {
        lower_in_mut(&clause[1], env, coord)?
    } else {
        let lowered: Result<Vec<_>, _> = clause[1..]
            .iter()
            .map(|f| lower_in_mut(f, env, coord))
            .collect();
        Expr::progn(lowered?)
    };
    let rest = lower_cond(&clauses[1..], env, coord)?;
    Ok(Expr::if_(test, body, rest))
}

fn lower_when(
    head: &str,
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
    invert: bool,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Err(CompileError::BadArity {
            head: head.to_string(),
            expected: "at least 1 (test)",
            got: 0,
        });
    }
    let test = lower_in_mut(&args[0], env, coord)?;
    let body_forms = &args[1..];
    let body = if body_forms.is_empty() {
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
    if invert {
        Ok(Expr::if_(test, Expr::Nil, body))
    } else {
        Ok(Expr::if_(test, body, Expr::Nil))
    }
}

fn lower_setq(
    head: &str,
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    if args.len() != 2 {
        return Err(CompileError::BadArity {
            head: head.to_string(),
            expected: "2",
            got: args.len(),
        });
    }
    let name = match &args[0] {
        Value::Symbol(s) => Arc::clone(&s.name),
        other => {
            return Err(CompileError::NotImplemented(format!(
                "setq's first argument must be a symbol, got {other:?}"
            )));
        }
    };
    // Resolve to a local first. If it's already a Cell variant, the
    // mutation analysis pass at the binding's declaration site has
    // already arranged for boxing; we emit a write through the box.
    // A non-cell local means the binding wasn't detected as mutated
    // at its declaration — most often because the declaration is a
    // function parameter (param boxing isn't wired yet). That's a
    // compile error rather than a silent miscompile.
    if let Some(b) = env.find_or_capture(&name) {
        let value = lower_in_mut(&args[1], env, coord)?;
        return match b {
            Binding::LocalCell(i) => Ok(Expr::set_car(Expr::Local(i), value)),
            Binding::ParamCell(i) => Ok(Expr::set_car(Expr::Param(i), value)),
            Binding::ClosureRefCell(i) => {
                Ok(Expr::set_car(Expr::ClosureRef(i), value))
            }
            Binding::Param(_) => Err(CompileError::NotImplemented(format!(
                "setq of function parameter: {name} (mutable parameters not yet supported)"
            ))),
            Binding::Local(_) => Err(CompileError::NotImplemented(format!(
                "internal: non-cell local {name} reached setq path \
                 — mutation analysis missed this binding"
            ))),
            Binding::ClosureRef(_) => Err(CompileError::NotImplemented(format!(
                "setq of captured non-cell variable: {name} (parent scope did not box)"
            ))),
        };
    }
    let sym_word = coord.intern(&name);
    let value = lower_in_mut(&args[1], env, coord)?;
    Ok(Expr::store_global(sym_word.raw(), value))
}

/// `(setf place value)`. We pattern-match on the shape of `place`:
///
/// - bare symbol → `(setq name value)`
/// - `(car x)`   → SetCar
/// - `(cdr x)`   → SetCdr
/// - `(first x)` → SetCar (alias)
/// - `(rest x)`  → SetCdr (alias)
/// - `(aref s i)` / `(char s i)` → SetChar
///
/// CL's full setf is a macro extensible via `defsetf` /
/// `define-setf-expander`. Until macros land, this is a built-in
/// that recognises a fixed set of place forms.
fn lower_setf(
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    if args.len() != 2 {
        return Err(CompileError::BadArity {
            head: "SETF".into(),
            expected: "2",
            got: args.len(),
        });
    }
    let place = &args[0];
    let value_form = &args[1];

    // Bare symbol: same as setq.
    if let Value::Symbol(_) = place {
        return lower_setq("SETF", args, env, coord);
    }

    // (head subargs...) — must be one of the recognised place forms.
    let place_items = list_to_vec(place)?;
    let place_head = match place_items.first() {
        Some(Value::Symbol(s)) => s.name.to_string(),
        _ => {
            return Err(CompileError::NotImplemented(format!(
                "setf place must be a symbol or recognised form, got {place:?}"
            )));
        }
    };
    let place_args = &place_items[1..];

    match place_head.as_str() {
        "CAR" | "FIRST" => {
            if place_args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: format!("setf {place_head}"),
                    expected: "(car place)",
                    got: place_args.len(),
                });
            }
            let cons = lower_in_mut(&place_args[0], env, coord)?;
            let value = lower_in_mut(value_form, env, coord)?;
            Ok(Expr::set_car(cons, value))
        }
        "CDR" | "REST" => {
            if place_args.len() != 1 {
                return Err(CompileError::BadArity {
                    head: format!("setf {place_head}"),
                    expected: "(cdr place)",
                    got: place_args.len(),
                });
            }
            let cons = lower_in_mut(&place_args[0], env, coord)?;
            let value = lower_in_mut(value_form, env, coord)?;
            Ok(Expr::set_cdr(cons, value))
        }
        "AREF" | "CHAR" | "STRING-CHAR" => {
            if place_args.len() != 2 {
                return Err(CompileError::BadArity {
                    head: format!("setf {place_head}"),
                    expected: "(aref s i)",
                    got: place_args.len(),
                });
            }
            let s = lower_in_mut(&place_args[0], env, coord)?;
            let idx = lower_in_mut(&place_args[1], env, coord)?;
            let ch = lower_in_mut(value_form, env, coord)?;
            Ok(Expr::set_char(s, idx, ch))
        }
        other => Err(CompileError::NotImplemented(format!(
            "setf place not yet supported: ({other} ...)"
        ))),
    }
}

fn lower_defparameter(
    head: &str,
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    // (defparameter name value [doc])  — doc string is optional and
    // ignored in v1.
    if args.len() < 2 || args.len() > 3 {
        return Err(CompileError::BadArity {
            head: head.to_string(),
            expected: "2 or 3",
            got: args.len(),
        });
    }
    let name = match &args[0] {
        Value::Symbol(s) => Arc::clone(&s.name),
        other => {
            return Err(CompileError::NotImplemented(format!(
                "{head}'s first argument must be a symbol, got {other:?}"
            )));
        }
    };
    let sym_word = coord.intern(&name);
    let value = lower_in_mut(&args[1], env, coord)?;
    Ok(Expr::store_global(sym_word.raw(), value))
}

/// Lower a `(lambda (params...) body...)` form. Builds an inner
/// env that captures any outer-env names referenced in the body,
/// lowers the body in that env (which records captures lazily),
/// and emits an `Expr::Lambda` with both the body and the
/// outer-scope expressions for each capture.
fn lower_lambda(
    args: &[Value],
    outer_env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Err(CompileError::BadArity {
            head: "LAMBDA".into(),
            expected: "at least 1 (param list)",
            got: 0,
        });
    }
    let params = parse_param_list_lambda(&args[0])?;
    let body_forms = &args[1..];

    // Inner env starts with params at Param(0..N) and a capture
    // parent pointing at the outer env (clone — we only read from
    // it during lookup_or_capture).
    let mut inner_env = LocalEnv::for_lambda(&params, outer_env.clone());

    let body_expr = if body_forms.is_empty() {
        Expr::Nil
    } else if body_forms.len() == 1 {
        lower_in_mut(&body_forms[0], &mut inner_env, coord)?
    } else {
        let lowered: Result<Vec<_>, _> = body_forms
            .iter()
            .map(|f| lower_in_mut(f, &mut inner_env, coord))
            .collect();
        Expr::progn(lowered?)
    };

    let captures = inner_env.take_captures();
    Ok(Expr::lambda(params.len() as u32, body_expr, captures))
}

/// Parse a lambda parameter list into a Vec of names.
fn parse_param_list_lambda(v: &Value) -> Result<Vec<Arc<str>>, CompileError> {
    let mut out = Vec::new();
    let mut cur = v.clone();
    loop {
        match cur {
            Value::Nil => return Ok(out),
            Value::Cons(c) => {
                let Value::Symbol(s) = &c.car else {
                    return Err(CompileError::NotImplemented(format!(
                        "lambda parameter must be a symbol, got {:?}", c.car,
                    )));
                };
                out.push(Arc::clone(&s.name));
                cur = c.cdr.clone();
            }
            other => {
                return Err(CompileError::NotImplemented(format!(
                    "lambda param list must be a proper list, got {other:?}"
                )));
            }
        }
    }
}

/// `(funcall fn arg1 arg2 ...)` — call an arbitrary function value.
fn lower_funcall(
    args: &[Value],
    env: &mut LocalEnv,
    coord: &Arc<GcCoordinator>,
) -> Result<Expr, CompileError> {
    if args.is_empty() {
        return Err(CompileError::BadArity {
            head: "FUNCALL".into(),
            expected: "at least 1 (function)",
            got: 0,
        });
    }
    let fn_expr = lower_in_mut(&args[0], env, coord)?;
    let lowered_args: Result<Vec<_>, _> = args[1..]
        .iter()
        .map(|a| lower_in_mut(a, env, coord))
        .collect();
    Ok(Expr::funcall(fn_expr, lowered_args?))
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
