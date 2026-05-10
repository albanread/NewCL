//! Macroexpansion: walk a `Value` tree, replace any cons form whose
//! head names a registered macro by calling the macro's compiled
//! function and recursing on the result.
//!
//! Macros store JIT-compiled functions in the coordinator's macro
//! registry (separate from symbol function cells). At expansion
//! time we:
//!   1. Convert each unevaluated argument form from `Value` to a
//!      runtime `Word` (cons-cell tree).
//!   2. Call the macro function via the standard JIT calling
//!      convention.
//!   3. Convert the returned `Word` back into a `Value`.
//!   4. Recurse on the result — the expansion may itself contain
//!      macro calls.
//!
//! `(quote ...)` is opaque: we do not recurse inside quoted data.
//! Every other cons form, whether a special form or a function
//! call, has its sub-forms walked.

use std::sync::Arc;

use ncl_runtime::{
    gc_function, gc_string, sym_names, symbol::Symbol, GcCoordinator, MutatorState,
    Tag, Value, Word,
};

use crate::EvalError;

/// Convert a compile-time `Value` into a runtime `Word`. Cons cells
/// and strings allocate on the calling thread's young heap; symbols
/// go through the coordinator's intern table; T and NIL get their
/// canonical immediate words.
pub fn value_to_word(
    v: &Value,
    m: &mut MutatorState,
    coord: &Arc<GcCoordinator>,
) -> Result<Word, EvalError> {
    Ok(match v {
        Value::Nil => Word::NIL,
        Value::Fixnum(n) => Word::fixnum(*n),
        Value::Char(c) => Word::char(*c),
        Value::Symbol(s) if &*s.name == "T" => Word::T,
        Value::Symbol(s) => coord.intern(&s.name),
        Value::String(s) => gc_string::alloc_string_in_young(m, s.as_str()),
        Value::Cons(c) => {
            let car = value_to_word(&c.car, m, coord)?;
            let cdr = value_to_word(&c.cdr, m, coord)?;
            m.alloc_cons(car, cdr)
        }
        other => {
            return Err(EvalError::Compile(crate::CompileError::NotImplemented(
                format!("value_to_word: unsupported value {other:?}"),
            )));
        }
    })
}

/// Convert a runtime `Word` back into a compile-time `Value`. Used
/// to consume a macro's expansion so that the lowering pass can
/// process it as ordinary source.
pub fn word_to_value(w: Word) -> Result<Value, EvalError> {
    if w.is_nil() {
        return Ok(Value::Nil);
    }
    if w.is_t() {
        return Ok(Value::Symbol(Symbol::fresh_uninterned(Arc::from("T"))));
    }
    Ok(match w.tag() {
        Tag::Fixnum => Value::Fixnum(w.as_fixnum().unwrap()),
        Tag::Cons => {
            let p = w.as_ptr::<u64>(Tag::Cons).expect("cons");
            let car = word_to_value(Word::from_raw(unsafe { *p }))?;
            let cdr = word_to_value(Word::from_raw(unsafe { *p.add(1) }))?;
            Value::cons(car, cdr)
        }
        Tag::Symbol => {
            let name = sym_names::lookup(w.raw()).unwrap_or_else(|| {
                Arc::from(format!("<sym {:#x}>", w.raw()).as_str())
            });
            Value::Symbol(Symbol::fresh_uninterned(name))
        }
        Tag::Immediate => match w.as_char() {
            Some(c) => Value::Char(c),
            None => {
                return Err(EvalError::Compile(crate::CompileError::NotImplemented(
                    format!("word_to_value: unknown immediate {:#x}", w.raw()),
                )));
            }
        },
        Tag::String => {
            let s: String = gc_string::chars_of(w).collect();
            Value::String(Arc::new(s))
        }
        other => {
            return Err(EvalError::Compile(crate::CompileError::NotImplemented(
                format!("word_to_value: unsupported tag {other:?}"),
            )));
        }
    })
}

/// Recursively walk `v`, expanding any macro calls. Returns a new
/// `Value` with all reachable expansions performed.
///
/// Special forms whose structure includes non-evaluated positions
/// — `quote`, `defun`, `defmacro`, `lambda`, `let`, `let*` — are
/// recognised so we don't try to "expand" the names in their
/// binding lists. (Without this, `(let ((when 1)) ...)` would walk
/// into the binding pair `(when 1)`, see `when` as the head, and
/// erroneously invoke the `when` macro on the literal `1`.)
///
/// Every other cons form is treated uniformly: if its head names a
/// registered macro, call it and recurse on the result; otherwise
/// walk all subforms.
pub fn macroexpand_all(
    v: &Value,
    coord: &Arc<GcCoordinator>,
    mutator: &mut MutatorState,
) -> Result<Value, EvalError> {
    let Value::Cons(_) = v else {
        return Ok(v.clone());
    };
    if let Some(head_name) = head_symbol_name(v) {
        match &*head_name {
            "QUOTE" => return Ok(v.clone()),
            "DEFUN" | "DEFMACRO" => {
                // (defun/defmacro NAME PARAMS body...) — leave name
                // and params alone, expand body forms.
                let items = list_to_vec(v)?;
                return rebuild_with_body_skip(&items, 3, coord, mutator);
            }
            "LAMBDA" => {
                // (lambda PARAMS body...) — leave params alone,
                // expand body forms.
                let items = list_to_vec(v)?;
                return rebuild_with_body_skip(&items, 2, coord, mutator);
            }
            "LET" | "LET*" => {
                // (let BINDINGS body...). BINDINGS is a list of
                // (name init) pairs; we expand each init but leave
                // the binding names alone.
                let items = list_to_vec(v)?;
                return rebuild_let(&items, coord, mutator);
            }
            _ => {}
        }
        // Macro head? Expand, then recurse on the result.
        if let Some(macro_fn) = coord.macro_for(&head_name) {
            let args_value = cdr_of(v);
            let args_vec = list_to_vec(&args_value)?;
            let mut arg_words: Vec<u64> = Vec::with_capacity(args_vec.len());
            for a in &args_vec {
                arg_words.push(value_to_word(a, mutator, coord)?.raw());
            }
            let env = gc_function::env(macro_fn);
            let code = gc_function::code_ptr(macro_fn);
            let f: gc_function::LispCodeFn =
                unsafe { std::mem::transmute(code) };
            let result_word_raw = unsafe {
                f(
                    mutator as *mut _,
                    env.raw(),
                    arg_words.as_ptr(),
                    arg_words.len() as u64,
                )
            };
            let result_value = word_to_value(Word::from_raw(result_word_raw))?;
            return macroexpand_all(&result_value, coord, mutator);
        }
    }
    // Non-macro cons with no special structure: walk subforms,
    // preserving any dotted tail.
    let (cars, dotted_tail) = list_to_vec_or_dotted(v);
    let mut acc = match dotted_tail {
        None => Value::Nil,
        Some(t) => macroexpand_all(&t, coord, mutator)?,
    };
    for car in cars.iter().rev() {
        let expanded = macroexpand_all(car, coord, mutator)?;
        acc = Value::cons(expanded, acc);
    }
    Ok(acc)
}

/// Rebuild a form like `(head ... a1 a2 a3)` where the first
/// `keep` items pass through unchanged and the rest are
/// macroexpanded body forms.
fn rebuild_with_body_skip(
    items: &[Value],
    keep: usize,
    coord: &Arc<GcCoordinator>,
    mutator: &mut MutatorState,
) -> Result<Value, EvalError> {
    let mut out: Vec<Value> = items.iter().take(keep).cloned().collect();
    for body in &items[keep.min(items.len())..] {
        out.push(macroexpand_all(body, coord, mutator)?);
    }
    Ok(Value::list(out))
}

/// Rebuild a `(let BINDINGS body...)` or `(let* BINDINGS body...)`
/// form: each binding's init expression gets macroexpanded but the
/// binding's name does not; body forms get expanded normally.
fn rebuild_let(
    items: &[Value],
    coord: &Arc<GcCoordinator>,
    mutator: &mut MutatorState,
) -> Result<Value, EvalError> {
    if items.len() < 2 {
        // Malformed; leave it for the lowering pass to error on.
        return Ok(Value::list(items.to_vec()));
    }
    let head = items[0].clone();
    let bindings_form = &items[1];
    let new_bindings = match list_to_vec(bindings_form) {
        Ok(bindings) => {
            let mut out = Vec::with_capacity(bindings.len());
            for b in &bindings {
                match list_to_vec(b) {
                    Ok(pair) if pair.len() == 2 => {
                        let init = macroexpand_all(&pair[1], coord, mutator)?;
                        out.push(Value::list(vec![pair[0].clone(), init]));
                    }
                    _ => out.push(b.clone()),
                }
            }
            Value::list(out)
        }
        Err(_) => bindings_form.clone(),
    };
    let mut new_items = vec![head, new_bindings];
    for body in &items[2..] {
        new_items.push(macroexpand_all(body, coord, mutator)?);
    }
    Ok(Value::list(new_items))
}

// ---- helpers --------------------------------------------------------------

fn head_symbol_name(v: &Value) -> Option<Arc<str>> {
    let Value::Cons(c) = v else { return None; };
    match &c.car {
        Value::Symbol(s) => Some(Arc::clone(&s.name)),
        _ => None,
    }
}

fn cdr_of(v: &Value) -> Value {
    match v {
        Value::Cons(c) => c.cdr.clone(),
        _ => Value::Nil,
    }
}

fn list_to_vec(v: &Value) -> Result<Vec<Value>, EvalError> {
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
                return Err(EvalError::Compile(crate::CompileError::ImproperList(
                    format!("{other:?}"),
                )));
            }
        }
    }
}

/// Walk a (possibly dotted) cons chain, collecting cars and
/// reporting the tail. `None` means the chain ended with NIL
/// (a proper list); `Some(v)` means it ended dotted with `v`.
fn list_to_vec_or_dotted(v: &Value) -> (Vec<Value>, Option<Value>) {
    let mut out = Vec::new();
    let mut cur = v.clone();
    loop {
        match cur {
            Value::Nil => return (out, None),
            Value::Cons(c) => {
                out.push(c.car.clone());
                cur = c.cdr.clone();
            }
            other => return (out, Some(other)),
        }
    }
}
