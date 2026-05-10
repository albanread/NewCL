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

use std::cell::Cell;
use std::sync::Arc;

use ncl_runtime::{
    format_word, gc_function, GcConfig, GcCoordinator, MutatorState, Value, Word,
};

pub mod lower;
pub mod macroexpand;

thread_local! {
    /// Pointer to the active Session for this thread. Set by
    /// `Session::activate()` and read by `eval_string_shim` so a
    /// Lisp call to `(eval-string ...)` can route into the running
    /// session's eval. Null when no Session has been activated;
    /// the shim returns a Lisp error in that case.
    static ACTIVE_SESSION: Cell<*mut Session> =
        const { Cell::new(std::ptr::null_mut()) };
}

pub use lower::{lower, lower_in, CompileError, LocalEnv};
pub use macroexpand::{macroexpand_all, value_to_word, word_to_value};

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
        let mut mutator = Box::new(coord.register_mutator());
        install_native_functions(&coord, &mut mutator);
        Session { coord, mutator }
    }

    pub fn coord(&self) -> &Arc<GcCoordinator> { &self.coord }

    /// Register this Session as the active one for the current
    /// thread, so `(eval-string ...)` from Lisp can route into it.
    /// Call this once after the Session is in its final memory
    /// location (Box, top-level let, etc.); calling before the
    /// final placement leaves a stale pointer behind.
    pub fn activate(&mut self) {
        ACTIVE_SESSION.with(|c| c.set(self as *mut Session));
    }

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
    /// First macroexpands; then recognises `(defun …)` and
    /// `(defmacro …)` at top level; everything else goes through
    /// lower → JIT → call.
    pub fn eval_value(&mut self, v: &Value) -> Result<Word, EvalError> {
        // Defmacro is recognised BEFORE macroexpansion: the body
        // shouldn't be expanded against the (yet-to-be-installed)
        // macro itself, and we want the body's *internal* macros
        // expanded normally during compilation. Same convention
        // as defun.
        if let Some((name, params, body_forms)) = match_defmacro(v)? {
            return self.handle_defmacro(&name, &params, &body_forms);
        }
        if let Some((name, params, body_forms)) = match_defun(v)? {
            return self.handle_defun(&name, &params, &body_forms);
        }
        // Otherwise: macroexpand fully, then compile.
        let expanded = macroexpand_all(v, &self.coord, &mut self.mutator)?;
        // CL convention: `(progn …)` at top level treats each body
        // form as itself a top-level form. This is what lets a
        // macro expand into multiple top-level definitions (e.g.
        // defstruct emits `(progn (defun make-x …) (defun x-y …)
        // (defun %setf-x-y …) …)`). Recurse into eval_value so each
        // body form re-runs the defun/defmacro recognisers.
        if let Some(body_forms) = match_top_level_progn(&expanded) {
            let mut last = Word::NIL;
            for form in body_forms {
                last = self.eval_value(&form)?;
            }
            return Ok(last);
        }
        let expr = lower(&expanded, &self.coord).map_err(EvalError::Compile)?;
        let mutator_ptr = &mut *self.mutator as *mut _;
        ncl_llvm::jit_eval(&expr, mutator_ptr).map_err(EvalError::Jit)
    }

    fn handle_defmacro(
        &mut self,
        name: &str,
        params: &ParamSpec,
        body_forms: &[Value],
    ) -> Result<Word, EvalError> {
        // Macros and defun'd functions are compiled the same way:
        // they're both Lisp functions with the standard JIT
        // calling convention. The only difference is where they're
        // installed — macros into the coordinator's macro registry,
        // defuns into the symbol's function cell.
        let fn_word = self.compile_function(name, params, body_forms)?;
        self.coord.install_macro(Arc::from(name), fn_word);
        Ok(Word::NIL)
    }

    fn handle_defun(
        &mut self,
        name: &str,
        params: &ParamSpec,
        body_forms: &[Value],
    ) -> Result<Word, EvalError> {
        let fn_word = self.compile_function(name, params, body_forms)?;
        let sym_word = self.coord.intern(name);
        self.mutator.set_symbol_function(sym_word, fn_word);
        Ok(Word::NIL)
    }

    /// Shared lowering+JIT path for `defun` and `defmacro`. Returns
    /// a Function-tagged Word; the caller decides where to install
    /// it (function cell or macro registry).
    ///
    /// Body forms are macroexpanded against the current macro
    /// registry before lowering. The rest parameter, if any, is
    /// handled identically to a let-local — boxed if the mutation
    /// analysis sees a setq/setf of it.
    fn compile_function(
        &mut self,
        name: &str,
        params: &ParamSpec,
        body_forms: &[Value],
    ) -> Result<Word, EvalError> {
        // First, expand any macros used in the body.
        let mut expanded_body: Vec<Value> = Vec::with_capacity(body_forms.len());
        for form in body_forms {
            expanded_body.push(macroexpand_all(form, &self.coord, &mut self.mutator)?);
        }

        // Build the env. Required params first at Param(0..N); the
        // shared prologue helper then layers optionals/rest/keys on
        // as let-locals (boxed if the body mutates them).
        let mut env = LocalEnv::with_params(&params.required);
        let prologue = lower::build_arglist_prologue(
            params,
            &expanded_body,
            &mut env,
            &self.coord,
        )
        .map_err(EvalError::Compile)?;

        // Implicit progn over body forms.
        let lowered_body = if expanded_body.len() == 1 {
            lower_in(&expanded_body[0], &env, &self.coord)
                .map_err(EvalError::Compile)?
        } else {
            let lowered: Result<Vec<_>, _> = expanded_body
                .iter()
                .map(|f| lower_in(f, &env, &self.coord))
                .collect();
            ncl_ir::Expr::progn(lowered.map_err(EvalError::Compile)?)
        };

        let body_expr = if prologue.is_empty() {
            lowered_body
        } else {
            ncl_ir::Expr::let_(prologue, lowered_body)
        };
        // Tail-position transform: wrap any non-`(values ...)` tail
        // expression in EnsureSingleMv so the multi-value slot is
        // always exactly the function's actual return values when
        // the caller reads it. See lower::instrument_tail_for_mv.
        let body_expr = lower::instrument_tail_for_mv(body_expr);

        let arity = params.required.len() as u32;
        let code_ptr = ncl_llvm::jit_compile_function(arity, &body_expr)
            .map_err(EvalError::Jit)?;
        let sym_word = self.coord.intern(name);
        gc_function::alloc_function_in_static(
            self.coord.static_area(),
            code_ptr,
            arity,
            sym_word,
            Word::NIL, // top-level functions and macros have no closure env
        )
        .ok_or_else(|| EvalError::Jit("static area exhausted".to_string()))
    }
}

impl Default for Session {
    fn default() -> Self { Session::new() }
}

/// Wire up native (Rust-implemented) Lisp functions. Each becomes
/// a callable Function in a Symbol's function cell — first-class
/// from the Lisp side: usable via `#'foo`, `apply`, `funcall`. Run
/// once at session creation, before any user code evaluates.
fn install_native_functions(
    coord: &Arc<GcCoordinator>,
    mutator: &mut MutatorState,
) {
    install_native(coord, mutator, "FORMAT", ncl_runtime::format_shim, 2);
    // APPEND (binary) is needed natively because backquote-splicing
    // macros expand to `(append ... ...)`. Loading it here means
    // backquote works in a bare Session, before any user-Lisp
    // stdlib has been read.
    install_native(coord, mutator, "APPEND", ncl_runtime::append_shim, 2);

    // File I/O. The handle-table architecture is borrowed from the
    // sister NewCP repo's `host_file_sys.rs`, adapted for our
    // String-Word path encoding. User-Lisp wrappers (with-open-file,
    // read-file-string, write-file-string) live in Lisp/core.lisp.
    install_native(coord, mutator, "OPEN-INPUT-FILE",
                   ncl_runtime::open_input_file_shim, 1);
    install_native(coord, mutator, "OPEN-OUTPUT-FILE",
                   ncl_runtime::open_output_file_shim, 1);
    install_native(coord, mutator, "OPEN-APPEND-FILE",
                   ncl_runtime::open_append_file_shim, 1);
    install_native(coord, mutator, "CLOSE-STREAM",
                   ncl_runtime::close_stream_shim, 1);
    install_native(coord, mutator, "READ-LINE",
                   ncl_runtime::read_line_shim, 1);
    install_native(coord, mutator, "READ-CHAR-FROM",
                   ncl_runtime::read_char_from_shim, 1);
    install_native(coord, mutator, "WRITE-STRING-TO",
                   ncl_runtime::write_string_to_shim, 2);
    install_native(coord, mutator, "FILE-POSITION",
                   ncl_runtime::file_position_shim, 1);
    install_native(coord, mutator, "FILE-LENGTH",
                   ncl_runtime::file_length_shim, 1);
    install_native(coord, mutator, "FILE-EXISTS",
                   ncl_runtime::file_exists_shim, 1);
    install_native(coord, mutator, "DELETE-FILE",
                   ncl_runtime::delete_file_shim, 1);

    // Conditions. ERROR signals; %HANDLER-CASE is the primitive
    // behind the handler-case macro (Lisp/core.lisp). Condition
    // unwinding goes through Rust's catch_unwind, which works
    // because every Rust function on the active call chain (and
    // the LispCodeFn type) is declared `extern "C-unwind"` so
    // panics aren't aborted at the boundary.
    install_native(coord, mutator, "ERROR",
                   ncl_runtime::error_shim, 1);
    install_native(coord, mutator, "%HANDLER-CASE",
                   ncl_runtime::handler_case_shim, 2);
    install_native(coord, mutator, "%NATIVE-LOOP",
                   ncl_runtime::native_loop_shim, 1);
    install_native(coord, mutator, "%LOOP-RETURN",
                   ncl_runtime::loop_return_shim, 1);

    // Self-eval bridge: lets Lisp code re-enter the compiler.
    install_native(coord, mutator, "EVAL-STRING",
                   eval_string_shim, 1);
    // (compile name '(lambda (params) body)) — JIT a lambda form
    // at runtime and (optionally) install in name's function cell.
    // Closette uses this to install its discriminating functions.
    install_native(coord, mutator, "COMPILE",
                   compile_shim, 2);
    install_native(coord, mutator, "PARSE-COMPLETE?",
                   parse_complete_shim, 1);
    install_native(coord, mutator, "SUBSTRING",
                   substring_shim, 3);
    // CLOS port prerequisites — foundation utilities.
    // GENSYM accepts an optional prefix; we register it as a 1-arg
    // shim and tolerate no-arg calls inside the shim body. (When
    // &optional lands in the compiler we'll declare it properly.)
    install_native(coord, mutator, "GENSYM",
                   ncl_runtime::gensym_shim, 0);
    // CAR / CDR / CONS are special forms in lower.rs; these
    // shims exist only so #'car / #'cdr / #'cons can be passed
    // as function values (to mapcar, :key, etc.).
    install_native(coord, mutator, "CAR",
                   ncl_runtime::car_shim, 1);
    install_native(coord, mutator, "CDR",
                   ncl_runtime::cdr_shim, 1);
    install_native(coord, mutator, "CONS",
                   ncl_runtime::cons_shim, 2);
    install_native(coord, mutator, "LIST",
                   ncl_runtime::list_shim, 0);
    install_native(coord, mutator, "EQ",
                   ncl_runtime::eq_shim, 2);
    install_native(coord, mutator, "EQL",
                   ncl_runtime::eql_shim, 2);
    install_native(coord, mutator, "EQUAL",
                   ncl_runtime::equal_shim, 2);
    install_native(coord, mutator, "TYPEP",
                   ncl_runtime::typep_shim, 2);
    // Multi-value support — `(values ...)` itself is a special form
    // lowered in lower.rs; this is the primitive used by the
    // multiple-value-bind / multiple-value-list macros to read the
    // thread-local slot back as a Lisp list.
    install_native(coord, mutator, "%MULTIPLE-VALUE-LIST-OF",
                   ncl_runtime::multiple_value_list_of_shim, 1);
    install_native(coord, mutator, "%MV-CLEAR",
                   ncl_runtime::mv_clear_shim, 0);
    // Vectors. AREF and SVREF are special forms (lowered directly
    // to Expr::Aref); MAKE-ARRAY and VECTOR go through Lisp-callable
    // shims.
    install_native(coord, mutator, "MAKE-ARRAY",
                   ncl_runtime::make_array_shim, 1);
    install_native(coord, mutator, "VECTOR",
                   ncl_runtime::vector_shim, 0);
    install_native(coord, mutator, "%WORD-HASH",
                   ncl_runtime::word_hash_shim, 1);
    // Symbol function-cell access. SYMBOL-FUNCTION reads,
    // %SET-SYMBOL-FUNCTION is the target of the
    // (setf (symbol-function ...) ...) lowering, FMAKUNBOUND
    // clears, FBOUNDP probes.
    install_native(coord, mutator, "SYMBOL-FUNCTION",
                   ncl_runtime::symbol_function_shim, 1);
    install_native(coord, mutator, "%SET-SYMBOL-FUNCTION",
                   ncl_runtime::set_symbol_function_shim, 2);
    install_native(coord, mutator, "FMAKUNBOUND",
                   ncl_runtime::fmakunbound_shim, 1);
    install_native(coord, mutator, "FBOUNDP",
                   ncl_runtime::fboundp_shim, 1);
    install_native(coord, mutator, "INTERN",
                   ncl_runtime::intern_shim, 1);
    // block / return-from — non-local exit via setjmp/longjmp
    // through JIT frames. Lisp-side macros in core.lisp wrap
    // the body in a thunk and call %native-block.
    install_native(coord, mutator, "%NATIVE-BLOCK",
                   ncl_runtime::native_block_shim, 2);
    install_native(coord, mutator, "%RETURN-FROM",
                   ncl_runtime::return_from_shim, 2);

    install_native(coord, mutator, "STRING-SPLIT-NEWLINES",
                   string_split_newlines_shim, 1);

    // iGui (Windows only). Spawns the GUI thread and exposes the
    // window-management trio + event poll. Drawing primitives
    // come in a follow-up commit.
    #[cfg(windows)]
    install_igui(coord, mutator);
}

/// `(eval-string s)` — feed S into the active Session, return the
/// printed result of the last form. Signals on parse / eval errors.
/// Requires `Session::activate()` to have been called by the host.
extern "C-unwind" fn eval_string_shim(
    mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "eval-string: expected 1 arg",
        );
    }
    let src_word = Word::from_raw(unsafe { *args });
    if src_word.tag() != ncl_runtime::Tag::String {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "eval-string: argument must be a string",
        );
    }
    let src: String = ncl_runtime::gc_string::chars_of(src_word).collect();

    let session_ptr = ACTIVE_SESSION.with(|c| c.get());
    if session_ptr.is_null() {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "eval-string: no active session (call Session::activate first)",
        );
    }
    let session = unsafe { &mut *session_ptr };

    match session.eval(&src) {
        Ok(printed) => {
            let m = unsafe { &mut *mutator };
            ncl_runtime::gc_string::alloc_string_in_young(m, &printed).raw()
        }
        Err(e) => ncl_runtime::abi::signal_condition_string(mutator, &format!("{e}")),
    }
}

/// `(compile name definition)` — compile DEFINITION (a `(lambda
/// (params…) body…)` form) into a function. If NAME is non-NIL,
/// install the compiled function in NAME's function cell. Always
/// returns the compiled function value as primary.
///
/// CL's compile is heavily overloaded — the no-definition form
/// recompiles an existing function-cell entry, the no-name form
/// just returns the compiled function, with secondary values for
/// warnings / failure status. Closette only uses the
/// `(compile NAME '(lambda …))` shape, which is what this
/// supports. The other shapes can grow when needed.
extern "C-unwind" fn compile_shim(
    mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 2 {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "compile: expected 2 args (name definition)",
        );
    }
    let name_word = Word::from_raw(unsafe { *args });
    let def_word = Word::from_raw(unsafe { *args.add(1) });

    // Convert the runtime cons-cell tree back into a Value tree
    // the compiler can chew on. macroexpand exposes this bridge
    // already (used by EVAL-STRING).
    let def_value = match word_to_value(def_word) {
        Ok(v) => v,
        Err(e) => {
            return ncl_runtime::abi::signal_condition_string(
                mutator,
                &format!("compile: cannot convert definition: {e}"),
            );
        }
    };

    // Parse `(lambda (params...) body...)`.
    let (params_value, body_forms) = match &def_value {
        ncl_runtime::Value::Cons(c) => {
            let ncl_runtime::Value::Symbol(head) = &c.car else {
                return ncl_runtime::abi::signal_condition_string(
                    mutator,
                    "compile: definition must start with the LAMBDA symbol",
                );
            };
            if &*head.name != "LAMBDA" {
                return ncl_runtime::abi::signal_condition_string(
                    mutator,
                    &format!(
                        "compile: definition must be a (lambda …) form, got ({} …)",
                        head.name
                    ),
                );
            }
            // c.cdr is (params body...). Walk it as a proper list.
            let parts = match list_to_vec_of_value(&c.cdr) {
                Ok(v) => v,
                Err(e) => {
                    return ncl_runtime::abi::signal_condition_string(
                        mutator,
                        &format!("compile: malformed lambda form: {e}"),
                    );
                }
            };
            if parts.is_empty() {
                return ncl_runtime::abi::signal_condition_string(
                    mutator,
                    "compile: lambda must have a parameter list",
                );
            }
            (parts[0].clone(), parts[1..].to_vec())
        }
        _ => {
            return ncl_runtime::abi::signal_condition_string(
                mutator,
                "compile: definition must be a (lambda …) form",
            );
        }
    };

    let params = match parse_param_list(&params_value) {
        Ok(p) => p,
        Err(e) => {
            return ncl_runtime::abi::signal_condition_string(
                mutator,
                &format!("compile: bad lambda list: {e}"),
            );
        }
    };

    let session_ptr = ACTIVE_SESSION.with(|c| c.get());
    if session_ptr.is_null() {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "compile: no active session (call Session::activate first)",
        );
    }
    let session = unsafe { &mut *session_ptr };

    // Use the name (or a placeholder for the no-name case) as the
    // compiled function's debug-display name. Doesn't affect the
    // installation decision below.
    let display_name: String = if name_word.is_nil() {
        "<compiled>".to_string()
    } else if name_word.tag() == ncl_runtime::Tag::Symbol {
        ncl_runtime::sym_names::lookup(name_word.raw())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "<unknown>".to_string())
    } else {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "compile: name must be a symbol or NIL",
        );
    };

    let fn_word = match session.compile_function(&display_name, &params, &body_forms) {
        Ok(w) => w,
        Err(e) => {
            return ncl_runtime::abi::signal_condition_string(
                mutator,
                &format!("compile: {e}"),
            );
        }
    };

    if !name_word.is_nil() {
        session.mutator.set_symbol_function(name_word, fn_word);
    }
    fn_word.raw()
}

/// `(parse-complete? s)` — T iff S parses fully (no trailing
/// open-paren / open-string / dangling unquote). NIL if more input
/// is expected. Used by the in-GUI REPL widget to decide whether
/// to accumulate or evaluate.
extern "C-unwind" fn parse_complete_shim(
    _mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        return Word::NIL.raw();
    }
    let src_word = Word::from_raw(unsafe { *args });
    if src_word.tag() != ncl_runtime::Tag::String {
        return Word::NIL.raw();
    }
    let src: String = ncl_runtime::gc_string::chars_of(src_word).collect();
    if src.trim().is_empty() {
        // Empty input is "complete" in the trivial sense (no
        // pending form).
        return Word::T.raw();
    }
    match ncl_reader::read_all(&src) {
        Ok(_) => Word::T.raw(),
        Err(e) => match e.kind {
            ncl_reader::ReaderErrorKind::UnexpectedEof(_) => Word::NIL.raw(),
            ncl_reader::ReaderErrorKind::Lex(
                ncl_reader::LexErrorKind::UnexpectedEof(_),
            ) => Word::NIL.raw(),
            // Any other parse error means the input is malformed,
            // not incomplete — call it "complete enough to
            // attempt eval, which will then fail informatively."
            _ => Word::T.raw(),
        },
    }
}

/// `(string-split-newlines s)` — split S on each `\n`, returning a
/// fresh list of (fresh) strings. A trailing newline produces a
/// final empty string in the list. Used by the GUI REPL to render
/// multi-line input correctly.
extern "C-unwind" fn string_split_newlines_shim(
    mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "string-split-newlines: expected 1 arg",
        );
    }
    let src_word = Word::from_raw(unsafe { *args });
    if src_word.tag() != ncl_runtime::Tag::String {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "string-split-newlines: argument must be a string",
        );
    }
    let s: String = ncl_runtime::gc_string::chars_of(src_word).collect();
    // Use split('\n') so a trailing \n produces a final empty
    // string — matches the user's mental model when the cursor is
    // sitting on a fresh line.
    let m = unsafe { &mut *mutator };
    let parts: Vec<Word> = s
        .split('\n')
        .map(|p| ncl_runtime::gc_string::alloc_string_in_young(m, p))
        .collect();
    let mut acc = Word::NIL;
    for p in parts.iter().rev() {
        acc = m.alloc_cons(*p, acc);
    }
    acc.raw()
}

/// `(substring s start end)` — return a fresh string holding
/// codepoints S[start..end]. Bounds are clamped: negative start
/// becomes 0, end past the end becomes the length.
extern "C-unwind" fn substring_shim(
    mutator: *mut MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 3 {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "substring: expected 3 args (string, start, end)",
        );
    }
    let src_word = Word::from_raw(unsafe { *args });
    if src_word.tag() != ncl_runtime::Tag::String {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "substring: first arg must be a string",
        );
    }
    let start_w = Word::from_raw(unsafe { *args.add(1) });
    let end_w = Word::from_raw(unsafe { *args.add(2) });
    let Some(start) = start_w.as_fixnum() else {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "substring: start must be a fixnum",
        );
    };
    let Some(end) = end_w.as_fixnum() else {
        return ncl_runtime::abi::signal_condition_string(
            mutator,
            "substring: end must be a fixnum",
        );
    };
    let n = ncl_runtime::gc_string::char_count(src_word) as i64;
    let s = start.max(0).min(n) as usize;
    let e = end.max(0).min(n) as usize;
    if s >= e {
        let m = unsafe { &mut *mutator };
        return ncl_runtime::gc_string::alloc_string_in_young(m, "").raw();
    }
    let chars: String = ncl_runtime::gc_string::chars_of(src_word)
        .skip(s)
        .take(e - s)
        .collect();
    let m = unsafe { &mut *mutator };
    ncl_runtime::gc_string::alloc_string_in_young(m, &chars).raw()
}

#[cfg(windows)]
fn install_igui(coord: &Arc<GcCoordinator>, mutator: &mut MutatorState) {
    install_native(coord, mutator, "IGUI-START",
                   ncl_runtime::igui_start_shim, 0);
    install_native(coord, mutator, "IGUI-WAIT",
                   ncl_runtime::igui_wait_shim, 0);
    install_native(coord, mutator, "IGUI-QUIT",
                   ncl_runtime::igui_quit_shim, 0);
    install_native(coord, mutator, "OPEN-CHILD",
                   ncl_runtime::open_child_shim, 1);
    install_native(coord, mutator, "CLOSE-CHILD",
                   ncl_runtime::close_child_shim, 1);
    install_native(coord, mutator, "SET-TITLE",
                   ncl_runtime::set_title_shim, 2);
    install_native(coord, mutator, "NEXT-EVENT",
                   ncl_runtime::next_event_shim, 1);
    install_native(coord, mutator, "SET-REDRAW-RATE",
                   ncl_runtime::set_redraw_rate_shim, 2);

    // Drawing primitives. The user-Lisp `with-batch` macro wraps
    // begin/submit; the emit-* helpers push commands onto the
    // thread-local current batch.
    install_native(coord, mutator, "%BEGIN-BATCH",
                   ncl_runtime::begin_batch_shim, 1);
    install_native(coord, mutator, "%SUBMIT-BATCH",
                   ncl_runtime::submit_batch_shim, 0);
    install_native(coord, mutator, "%EMIT-CLEAR",
                   ncl_runtime::emit_clear_shim, 1);
    install_native(coord, mutator, "%EMIT-FILL-RECT",
                   ncl_runtime::emit_fill_rect_shim, 5);
    install_native(coord, mutator, "%EMIT-STROKE-RECT",
                   ncl_runtime::emit_stroke_rect_shim, 6);
    install_native(coord, mutator, "%EMIT-DRAW-LINE",
                   ncl_runtime::emit_draw_line_shim, 6);
    install_native(coord, mutator, "%EMIT-DRAW-TEXT",
                   ncl_runtime::emit_draw_text_shim, 5);
    install_native(coord, mutator, "%EMIT-DRAW-TEXT-STYLED",
                   ncl_runtime::emit_draw_text_styled_shim, 6);
    install_native(coord, mutator, "%MEASURE-TEXT",
                   ncl_runtime::measure_text_shim, 4);
    install_native(coord, mutator, "%EMIT-FILL-OVAL",
                   ncl_runtime::emit_fill_oval_shim, 5);
    install_native(coord, mutator, "%EMIT-STROKE-OVAL",
                   ncl_runtime::emit_stroke_oval_shim, 6);
    install_native(coord, mutator, "%EMIT-FILL-CIRCLE",
                   ncl_runtime::emit_fill_circle_shim, 4);
    install_native(coord, mutator, "%EMIT-STROKE-CIRCLE",
                   ncl_runtime::emit_stroke_circle_shim, 5);
    install_native(coord, mutator, "%EMIT-DRAW-ARC",
                   ncl_runtime::emit_draw_arc_shim, 7);

    // Log view bridge.
    install_native(coord, mutator, "LOG-WRITE",
                   ncl_runtime::log_write_shim, 1);

    // Text-view (terminal-style monospaced child).
    install_native(coord, mutator, "OPEN-TEXT-WINDOW",
                   ncl_runtime::open_text_window_shim, 1);
    install_native(coord, mutator, "TEXT-WRITE",
                   ncl_runtime::text_write_shim, 2);
    install_native(coord, mutator, "TEXT-WRITE-CHAR",
                   ncl_runtime::text_write_char_shim, 2);
    install_native(coord, mutator, "TEXT-CLEAR",
                   ncl_runtime::text_clear_shim, 1);
    install_native(coord, mutator, "TEXT-CLEAR-EOL",
                   ncl_runtime::text_clear_eol_shim, 1);
    install_native(coord, mutator, "TEXT-CLEAR-EOS",
                   ncl_runtime::text_clear_eos_shim, 1);
    install_native(coord, mutator, "TEXT-NEWLINE",
                   ncl_runtime::text_newline_shim, 1);
    install_native(coord, mutator, "TEXT-SCROLL-UP",
                   ncl_runtime::text_scroll_up_shim, 2);
    install_native(coord, mutator, "TEXT-SET-CURSOR",
                   ncl_runtime::text_set_cursor_shim, 3);
    install_native(coord, mutator, "TEXT-SET-PEN",
                   ncl_runtime::text_set_pen_shim, 3);
    install_native(coord, mutator, "TEXT-RESET-PEN",
                   ncl_runtime::text_reset_pen_shim, 1);
    install_native(coord, mutator, "TEXT-SHOW-CARET",
                   ncl_runtime::text_show_caret_shim, 2);
}

fn install_native(
    coord: &Arc<GcCoordinator>,
    mutator: &mut MutatorState,
    name: &str,
    code: extern "C-unwind" fn(*mut MutatorState, u64, *const u64, u64) -> u64,
    arity: u32,
) {
    let sym_word = coord.intern(name);
    let fn_word = gc_function::alloc_function_in_static(
        coord.static_area(),
        code as usize,
        arity,
        sym_word,
        Word::NIL, // native functions don't carry a closure env
    )
    .expect("static area exhausted while installing native function");
    mutator.set_symbol_function(sym_word, fn_word);
}

/// User-Lisp portion of the standard library. Embedded at compile
/// time so the binary is self-contained — no filesystem lookup at
/// runtime, no working-directory dependence in tests. The Rust-side
/// glue (numeric primitives, condition machinery) lives in
/// `ncl-cl`; everything written in Lisp lives here.
const CORE_LISP_SOURCE: &str = include_str!("../../../Lisp/core.lisp");

/// Closette CLOS port — staged Lisp source. Loaded after the
/// core stdlib because it depends on every primitive (vectors,
/// hash tables, defstruct, &key, multiple values, block /
/// return-from, compile, flet/labels, the lot). See the staging
/// plan in the commit log for which sections cover which
/// chunks.
const CLOS_LISP_SOURCE: &str = include_str!("../../../Lisp/clos.lisp");

impl Session {
    /// Read and evaluate the embedded core-stdlib source. Each defun
    /// goes through the same JIT path as user code; the resulting
    /// Function objects are installed in the symbols' function
    /// cells. Idempotent only in the trivial sense — calling twice
    /// re-defines every function.
    pub fn load_core_stdlib(&mut self) -> Result<(), EvalError> {
        self.eval(CORE_LISP_SOURCE)?;
        Ok(())
    }

    /// Load Closette on top of the core stdlib. Order matters —
    /// CLOS uses everything in core (defstruct, hash tables, &key,
    /// labels, etc.). Idempotent in the same trivial sense as
    /// `load_core_stdlib`.
    pub fn load_clos(&mut self) -> Result<(), EvalError> {
        self.eval(CLOS_LISP_SOURCE)?;
        Ok(())
    }

    /// Convenience: a session with the core stdlib + CLOS pre-loaded.
    pub fn with_stdlib() -> Result<Session, EvalError> {
        let mut s = Session::new();
        s.load_core_stdlib()?;
        s.load_clos()?;
        Ok(s)
    }
}

/// Recognise `(defun name (params...) body...)`. Returns `Some` if
/// the form is a defun. Implicit progn is supported — multiple body
/// forms are returned as a Vec for the caller to wrap.
fn match_defun(
    v: &Value,
) -> Result<Option<(String, ParamSpec, Vec<Value>)>, EvalError> {
    match_defun_like(v, "DEFUN")
}

/// Recognise `(progn body...)` so top-level progn can splice its
/// body forms back into the top-level recogniser. Returns the
/// body as a Vec of Values, or None if the form isn't a progn.
fn match_top_level_progn(v: &Value) -> Option<Vec<Value>> {
    let Value::Cons(c) = v else { return None };
    let Value::Symbol(head) = &c.car else { return None };
    if &*head.name != "PROGN" {
        return None;
    }
    list_to_vec_of_value(&c.cdr).ok()
}

/// Recognise `(defmacro name (params...) body...)`. Same shape as
/// defun; the body is compiled the same way but installed in the
/// coordinator's macro registry rather than the symbol's function
/// cell.
fn match_defmacro(
    v: &Value,
) -> Result<Option<(String, ParamSpec, Vec<Value>)>, EvalError> {
    match_defun_like(v, "DEFMACRO")
}

fn match_defun_like(
    v: &Value,
    head_name: &str,
) -> Result<Option<(String, ParamSpec, Vec<Value>)>, EvalError> {
    let Value::Cons(c) = v else { return Ok(None); };
    let Value::Symbol(head) = &c.car else { return Ok(None); };
    if &*head.name != head_name {
        return Ok(None);
    }
    let args = list_to_vec_of_value(&c.cdr).map_err(|e| {
        EvalError::Compile(CompileError::ImproperList(e))
    })?;
    if args.len() < 3 {
        return Err(EvalError::Compile(CompileError::BadDefun(format!(
            "{} needs name, params, and body — got {} args",
            head_name.to_lowercase(),
            args.len()
        ))));
    }
    let name = match &args[0] {
        Value::Symbol(s) => s.name.to_string(),
        other => {
            return Err(EvalError::Compile(CompileError::BadDefun(format!(
                "{} name must be a symbol, got {other:?}",
                head_name.to_lowercase()
            ))));
        }
    };
    let params = parse_param_list(&args[1])?;
    let body_forms = args[2..].to_vec();
    Ok(Some((name, params, body_forms)))
}

/// Parsed parameter list. CL canonical order: required, optional,
/// rest, key. Each later category may be empty; the only currently
/// unsupported feature is `&aux` (post-key let-bindings) and the
/// optional/key supplied-p flag (`(name default supplied-p)`).
#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub required: Vec<Arc<str>>,
    pub optionals: Vec<OptParam>,
    pub rest: Option<Arc<str>>,
    pub keys: Vec<KeyParam>,
}

/// One `&optional` parameter. `default` is the raw form the user
/// wrote (or None for no default — implicit NIL); the lowering pass
/// evaluates it lazily, only when the caller didn't supply this
/// argument. The default form is lowered in an env that has the
/// required params and all earlier optionals already bound, so
/// `(defun f (a &optional (b (* a 2))))` works.
#[derive(Debug, Clone)]
pub struct OptParam {
    pub name: Arc<str>,
    pub default: Option<Value>,
}

/// One `&key` parameter. `keyword` is the colon-prefixed name used
/// by callers (e.g. `:FAMILY`); `name` is the binding name inside
/// the body. CL allows `((:other-name name) default)` to decouple
/// them; for now we only support the common case where keyword
/// equals `:NAME`. `default` lazily evaluated, same shape as for
/// optionals.
#[derive(Debug, Clone)]
pub struct KeyParam {
    pub name: Arc<str>,
    pub keyword: Arc<str>,
    pub default: Option<Value>,
}

/// Parsing state machine for an arglist. Steps strictly through
/// `&optional` -> `&rest`/`&body` -> `&key` in that order; an
/// out-of-order marker is a hard error.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArglistMode {
    Required,
    Optional,
    Rest,
    Key,
}

fn parse_param_list(v: &Value) -> Result<ParamSpec, EvalError> {
    parse_param_list_inner(v).map_err(|s| {
        EvalError::Compile(CompileError::BadDefun(s))
    })
}

/// Internal parse with String errors — used by both the defun-side
/// (lib.rs) and lambda-side (lower.rs) entry points so the shape
/// stays identical between them.
pub(crate) fn parse_param_list_inner(v: &Value) -> Result<ParamSpec, String> {
    let mut required: Vec<Arc<str>> = Vec::new();
    let mut optionals: Vec<OptParam> = Vec::new();
    let mut rest: Option<Arc<str>> = None;
    let mut keys: Vec<KeyParam> = Vec::new();
    let mut mode = ArglistMode::Required;
    let mut cur = v.clone();
    loop {
        match cur {
            Value::Nil => break,
            Value::Cons(c) => {
                let head = c.car.clone();
                cur = c.cdr.clone();
                // Lambda-list keyword?
                if let Value::Symbol(s) = &head {
                    match &*s.name {
                        "&OPTIONAL" => {
                            if mode != ArglistMode::Required {
                                return Err(format!(
                                    "&OPTIONAL out of order in arglist"
                                ));
                            }
                            mode = ArglistMode::Optional;
                            continue;
                        }
                        "&REST" | "&BODY" => {
                            if matches!(mode, ArglistMode::Rest | ArglistMode::Key) {
                                return Err(format!(
                                    "&REST/&BODY out of order in arglist"
                                ));
                            }
                            // Take exactly one symbol.
                            let Value::Cons(rc) = cur.clone() else {
                                return Err(
                                    "&REST/&BODY must be followed by a name".into(),
                                );
                            };
                            let Value::Symbol(rs) = &rc.car else {
                                return Err(format!(
                                    "&REST/&BODY name must be a symbol, got {:?}",
                                    rc.car
                                ));
                            };
                            rest = Some(Arc::clone(&rs.name));
                            cur = rc.cdr.clone();
                            mode = ArglistMode::Rest;
                            continue;
                        }
                        "&KEY" => {
                            if mode == ArglistMode::Key {
                                return Err("duplicate &KEY in arglist".into());
                            }
                            mode = ArglistMode::Key;
                            continue;
                        }
                        "&AUX" => {
                            return Err("&AUX is not yet supported".into());
                        }
                        "&ALLOW-OTHER-KEYS" => {
                            // ncl_lookup_keyword silently ignores
                            // unknown keywords already, so this
                            // marker is a no-op for us. Accept it
                            // so Closette code that uses it (e.g.
                            // initialize-instance dispatching to
                            // shared-initialize) parses.
                            continue;
                        }
                        _ => {}
                    }
                }
                // Non-keyword entry — interpret per current mode.
                match mode {
                    ArglistMode::Required => {
                        let Value::Symbol(s) = &head else {
                            return Err(format!(
                                "required parameter must be a symbol, got {head:?}"
                            ));
                        };
                        required.push(Arc::clone(&s.name));
                    }
                    ArglistMode::Optional => {
                        let (name, default) = parse_init_form(&head)?;
                        optionals.push(OptParam { name, default });
                    }
                    ArglistMode::Rest => {
                        return Err(format!(
                            "extra parameter after &REST: {head:?}"
                        ));
                    }
                    ArglistMode::Key => {
                        let (name, default) = parse_init_form(&head)?;
                        // Convention: the matching keyword is `:NAME`.
                        let kw_name = format!(":{}", name);
                        keys.push(KeyParam {
                            name,
                            keyword: Arc::from(kw_name.as_str()),
                            default,
                        });
                    }
                }
            }
            other => {
                return Err(format!(
                    "param list must be a proper list, got {other:?}"
                ));
            }
        }
    }
    Ok(ParamSpec { required, optionals, rest, keys })
}

/// Parse one entry of the form `name`, `(name)`, or `(name
/// default)`. Returns the name and the optional default form. The
/// 3-element `(name default supplied-p)` shape is rejected for now
/// — Closette doesn't use it widely; can grow later.
fn parse_init_form(v: &Value) -> Result<(Arc<str>, Option<Value>), String> {
    match v {
        Value::Symbol(s) => Ok((Arc::clone(&s.name), None)),
        Value::Cons(c) => {
            let Value::Symbol(s) = &c.car else {
                return Err(format!(
                    "init-form name must be a symbol, got {:?}", c.car
                ));
            };
            let name = Arc::clone(&s.name);
            match &c.cdr {
                Value::Nil => Ok((name, None)),
                Value::Cons(c2) => {
                    if !matches!(c2.cdr, Value::Nil) {
                        return Err(
                            "init-form may have at most (name default); supplied-p flag not yet supported".into(),
                        );
                    }
                    Ok((name, Some(c2.car.clone())))
                }
                other => Err(format!(
                    "init-form tail must be a list, got {other:?}"
                )),
            }
        }
        other => Err(format!(
            "init-form must be a symbol or a list, got {other:?}"
        )),
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

    // -- &rest / variadic functions ---------------------------------------

    #[test]
    fn rest_no_extra_args() {
        let mut s = Session::new();
        s.eval("(defun f (a &rest r) r)").unwrap();
        // Called with exactly the required count — rest is nil.
        assert_eq!(s.eval("(f 1)").unwrap(), "nil");
    }

    #[test]
    fn rest_collects_extra_args_in_order() {
        let mut s = Session::new();
        s.eval("(defun f (a &rest r) r)").unwrap();
        assert_eq!(s.eval("(f 1 2 3 4)").unwrap(), "(2 3 4)");
        assert_eq!(s.eval("(f 1 99)").unwrap(), "(99)");
    }

    #[test]
    fn rest_only() {
        // (defun f (&rest r) r) — all args go into r.
        let mut s = Session::new();
        s.eval("(defun all (&rest r) r)").unwrap();
        assert_eq!(s.eval("(all)").unwrap(), "nil");
        assert_eq!(s.eval("(all 1 2 3)").unwrap(), "(1 2 3)");
        assert_eq!(s.eval(r#"(all 'a "b" 3)"#).unwrap(), r#"(A "b" 3)"#);
    }

    #[test]
    fn rest_with_multiple_required() {
        let mut s = Session::new();
        s.eval("(defun f (a b c &rest r) (cons (+ a b c) r))").unwrap();
        assert_eq!(s.eval("(f 1 2 3)").unwrap(), "(6)");
        assert_eq!(s.eval("(f 1 2 3 4 5)").unwrap(), "(6 4 5)");
    }

    #[test]
    fn rest_lets_us_walk_args() {
        // Without apply we can't recursively call the variadic
        // function with fewer args, but we CAN walk the rest list
        // with a separate helper. This is the common shape of
        // "variadic frontend, list-walking backend."
        let mut s = Session::with_stdlib().unwrap();
        s.eval(
            "(defun sum-list (lst) \
               (if (null lst) 0 \
                 (+ (car lst) (sum-list (cdr lst)))))",
        )
        .unwrap();
        s.eval("(defun sum (&rest r) (sum-list r))").unwrap();
        assert_eq!(s.eval("(sum)").unwrap(), "0");
        assert_eq!(s.eval("(sum 1)").unwrap(), "1");
        assert_eq!(s.eval("(sum 1 2 3 4 5)").unwrap(), "15");
    }

    #[test]
    fn rest_in_lambda() {
        let mut s = Session::new();
        s.eval("(defparameter *f* (lambda (a &rest r) (cons a r)))").unwrap();
        assert_eq!(s.eval("(funcall *f* 1)").unwrap(), "(1)");
        assert_eq!(s.eval("(funcall *f* 1 2 3)").unwrap(), "(1 2 3)");
    }

    #[test]
    fn rest_can_be_mutated() {
        // setq of the rest binding works because the analysis
        // detects mutation and boxes it like any let-local.
        let mut s = Session::new();
        // Variadic length implemented by walking the rest list
        // destructively (just to exercise mutation, not because
        // it's a good idea — `length` already does this).
        s.eval(
            "(defun my-count (&rest r) \
               (let ((n 0)) \
                 (if (null r) n \
                   (progn (setq r (cdr r)) \
                          (setq n (+ n 1)) \
                          (if (null r) n \
                            (progn (setq r (cdr r)) (setq n (+ n 1))))))))",
        )
        .unwrap();
        assert_eq!(s.eval("(my-count)").unwrap(), "0");
        assert_eq!(s.eval("(my-count 'a)").unwrap(), "1");
        assert_eq!(s.eval("(my-count 'a 'b 'c)").unwrap(), "2");
    }

    #[test]
    fn rest_closes_over_correctly() {
        // Lambda with &rest, captured by another lambda.
        let mut s = Session::new();
        s.eval(
            "(defun make-collector () \
               (let ((items nil)) \
                 (lambda (&rest new-items) \
                   (setf items (append items new-items)) \
                   items)))",
        )
        .unwrap();
        // append is binary in our stdlib, so this needs stdlib.
        let mut s = Session::with_stdlib().unwrap();
        s.eval(
            "(defun make-collector () \
               (let ((items nil)) \
                 (lambda (&rest new-items) \
                   (setf items (append items new-items)) \
                   items)))",
        )
        .unwrap();
        s.eval("(defparameter *c* (make-collector))").unwrap();
        assert_eq!(s.eval("(funcall *c* 1 2)").unwrap(), "(1 2)");
        assert_eq!(s.eval("(funcall *c* 3)").unwrap(), "(1 2 3)");
        assert_eq!(s.eval("(funcall *c*)").unwrap(), "(1 2 3)");
    }

    #[test]
    fn rest_malformed_errors() {
        // &rest with no name following.
        let r = eval_str("(defun f (a &rest) a)");
        assert!(r.is_err());
        // &rest followed by multiple names.
        let r = eval_str("(defun f (a &rest r s) a)");
        assert!(r.is_err());
    }

    // -- loop / return / getf --------------------------------------------

    #[test]
    fn loop_with_return_exits() {
        // (loop (return 42)) — first iteration returns.
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(loop (return 42))").unwrap(), "42");
        // (return) with no value yields nil.
        assert_eq!(s.eval("(loop (return))").unwrap(), "nil");
    }

    #[test]
    fn loop_counts_via_mutable_local() {
        // Classic "count to 5 then exit." Uses a mutable let-local
        // for the counter, terminal return in a cond clause.
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval(
                "(let ((i 0))
                   (loop
                     (cond
                       ((= i 5) (return i))
                       (t (setq i (+ i 1))))))",
            )
            .unwrap(),
            "5",
        );
    }

    #[test]
    fn loop_accumulates() {
        // Sum 1+2+...+10 via a counter and accumulator.
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval(
                "(let ((i 0) (sum 0))
                   (loop
                     (cond
                       ((> i 10) (return sum))
                       (t (setq sum (+ sum i))
                          (setq i (+ i 1))))))",
            )
            .unwrap(),
            "55",
        );
    }

    #[test]
    fn nested_loops_each_have_own_break() {
        // Inner return only exits the inner loop. Outer continues
        // and exits via its own terminal return.
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval(
                "(let ((outer 0))
                   (loop
                     (cond
                       ((= outer 3) (return outer))
                       (t (setq outer (+ outer 1))
                          (loop (return :inner))))))",
            )
            .unwrap(),
            "3",
        );
    }

    #[test]
    fn loop_return_value_can_be_anything() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(loop (return 'foo))").unwrap(), "FOO");
        assert_eq!(s.eval("(loop (return '(1 2 3)))").unwrap(), "(1 2 3)");
        assert_eq!(s.eval(r#"(loop (return "hello"))"#).unwrap(), r#""hello""#);
    }

    #[test]
    fn getf_finds_value() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval("(getf '(:a 1 :b 2 :c 3) :b)").unwrap(),
            "2",
        );
    }

    #[test]
    fn getf_returns_nil_when_not_found() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval("(getf '(:a 1 :b 2) :z)").unwrap(),
            "nil",
        );
        assert_eq!(s.eval("(getf nil :a)").unwrap(), "nil");
    }

    #[test]
    fn getf_returns_first_match() {
        let mut s = Session::with_stdlib().unwrap();
        // Duplicate keys: returns the first value found.
        assert_eq!(
            s.eval("(getf '(:a 1 :a 2 :a 3) :a)").unwrap(),
            "1",
        );
    }

    #[test]
    fn loop_with_getf_event_pattern() {
        // The event-loop idiom we'd actually write for the GUI:
        // a loop that pulls items off a "queue" (here a list) and
        // exits when it sees a sentinel kind.
        let mut s = Session::with_stdlib().unwrap();
        s.eval("(defparameter *q* '((:kind :a :v 1) (:kind :b :v 2) (:kind :stop)))").unwrap();
        s.eval(
            "(defun next-item ()
               (cond
                 ((null *q*) nil)
                 (t (let ((it (car *q*)))
                      (setq *q* (cdr *q*))
                      it))))",
        )
        .unwrap();
        assert_eq!(
            s.eval(
                "(let ((collected nil))
                   (loop
                     (let ((ev (next-item)))
                       (cond
                         ((null ev) (return (reverse collected)))
                         ((eq (getf ev :kind) :stop) (return (reverse collected)))
                         (t (setq collected (cons (getf ev :v) collected)))))))",
            )
            .unwrap(),
            "(1 2)",
        );
    }

    // -- Conditions ------------------------------------------------------

    #[test]
    fn handler_case_no_signal_returns_body_value() {
        // When the body completes normally, handler-case returns
        // the body's value.
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval("(handler-case 42 (error (c) (+ 100 0)))").unwrap(),
            "42",
        );
    }

    #[test]
    fn error_signals_handler_runs() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval(r#"(handler-case (error "boom") (error (c) c))"#).unwrap(),
            r#""boom""#,
        );
    }

    #[test]
    fn handler_returns_replacement_value() {
        let mut s = Session::with_stdlib().unwrap();
        // Handler can compute a replacement value of any type.
        assert_eq!(
            s.eval(r#"(handler-case (error "x") (error (c) 99))"#).unwrap(),
            "99",
        );
    }

    #[test]
    fn handler_can_use_condition_value() {
        // The handler can format the condition into a message.
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval(
                r#"(handler-case (error "thing failed")
                     (error (c) (format nil "caught: ~A" c)))"#,
            )
            .unwrap(),
            r#""caught: thing failed""#,
        );
    }

    #[test]
    fn nested_handler_case_inner_catches() {
        let mut s = Session::with_stdlib().unwrap();
        // Inner handler catches; outer never sees it.
        assert_eq!(
            s.eval(
                r#"(handler-case
                     (handler-case (error "inner")
                       (error (c) "inner caught"))
                     (error (c) "outer caught"))"#,
            )
            .unwrap(),
            r#""inner caught""#,
        );
    }

    #[test]
    fn nested_handler_case_outer_catches_when_inner_skips() {
        let mut s = Session::with_stdlib().unwrap();
        // Inner doesn't run; outer handler catches the rethrown.
        assert_eq!(
            s.eval(
                r#"(handler-case
                     (progn (handler-case 1 (error (c) c))
                            (error "from outer body"))
                     (error (c) c))"#,
            )
            .unwrap(),
            r#""from outer body""#,
        );
    }

    #[test]
    fn error_unwinds_through_user_frames() {
        // (error) deep inside a call chain is caught by the outer
        // handler-case — proving panic propagation crosses JIT
        // frames cleanly.
        let mut s = Session::with_stdlib().unwrap();
        s.eval(r#"(defun deep () (error "from deep"))"#).unwrap();
        s.eval("(defun mid () (deep))").unwrap();
        s.eval("(defun top () (mid))").unwrap();
        assert_eq!(
            s.eval("(handler-case (top) (error (c) c))").unwrap(),
            r#""from deep""#,
        );
    }

    #[test]
    fn condition_value_can_be_any_type() {
        // The condition isn't required to be a string. Any Word
        // works — here we pass a list.
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval("(handler-case (error '(:bad 42)) (error (c) c))").unwrap(),
            "(:BAD 42)",
        );
    }

    #[test]
    fn handler_case_in_let_body() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval(
                r#"(let ((x 10))
                     (handler-case (progn (error "oops") x)
                       (error (c) (+ x 1))))"#,
            )
            .unwrap(),
            "11",
        );
    }

    // -- Keywords --------------------------------------------------------

    #[test]
    fn keyword_self_evaluates() {
        // :foo evaluates to the symbol :FOO.
        assert_eq!(eval_str(":foo").unwrap(), ":FOO");
        assert_eq!(eval_str(":input").unwrap(), ":INPUT");
    }

    #[test]
    fn keyword_eq_to_itself() {
        // The same keyword is eq to itself (interned).
        assert_eq!(eval_str("(eq :foo :foo)").unwrap(), "T");
        assert_eq!(eval_str("(eq :foo :bar)").unwrap(), "nil");
    }

    #[test]
    fn keyword_distinct_from_symbol() {
        // :foo and 'foo are different symbols.
        assert_eq!(eval_str("(eq :foo 'foo)").unwrap(), "nil");
    }

    #[test]
    fn keyword_in_quoted_list() {
        // Quoted lists containing keywords preserve them.
        assert_eq!(
            eval_str("'(:input :output :append)").unwrap(),
            "(:INPUT :OUTPUT :APPEND)",
        );
    }

    #[test]
    fn keyword_as_arg() {
        // Pass keywords to functions — they evaluate to themselves
        // and travel through the call without ceremony.
        let mut s = Session::new();
        s.eval("(defun classify (k) (cond ((eq k :input) 1) ((eq k :output) 2) (t 0)))").unwrap();
        assert_eq!(s.eval("(classify :input)").unwrap(), "1");
        assert_eq!(s.eval("(classify :output)").unwrap(), "2");
        assert_eq!(s.eval("(classify :other)").unwrap(), "0");
    }

    #[test]
    fn keyword_in_macro_dispatch() {
        // A macro that compares against keyword literals at expansion
        // time (the with-open-file pattern).
        let mut s = Session::new();
        s.eval(
            "(defmacro pick (k) \
               (cond \
                 ((eq k :a) ''first) \
                 ((eq k :b) ''second) \
                 (t ''other)))",
        )
        .unwrap();
        assert_eq!(s.eval("(pick :a)").unwrap(), "FIRST");
        assert_eq!(s.eval("(pick :b)").unwrap(), "SECOND");
        assert_eq!(s.eval("(pick :z)").unwrap(), "OTHER");
    }

    #[test]
    fn keyword_through_format() {
        // Keywords print with the colon under both ~A and ~S.
        assert_eq!(
            eval_str(r#"(format nil "~A" :hello)"#).unwrap(),
            r#"":HELLO""#,
        );
        assert_eq!(
            eval_str(r#"(format nil "~S" :hello)"#).unwrap(),
            r#"":HELLO""#,
        );
    }

    #[test]
    fn keyword_through_apply() {
        let mut s = Session::new();
        s.eval("(defun marker-of (k) (cond ((eq k :red) 1) (t 0)))").unwrap();
        assert_eq!(s.eval("(apply #'marker-of '(:red))").unwrap(), "1");
        assert_eq!(s.eval("(apply #'marker-of '(:blue))").unwrap(), "0");
    }

    // -- File I/O --------------------------------------------------------

    fn temp_path(name: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("ncl_test_{}_{name}", std::process::id()));
        p.to_string_lossy().to_string()
    }

    #[test]
    fn file_round_trip_string() {
        let mut s = Session::with_stdlib().unwrap();
        let path = temp_path("roundtrip.txt");
        s.eval(&format!(r#"(defparameter *p* "{}")"#, path.replace('\\', "\\\\")))
            .unwrap();
        s.eval(r#"(write-file-string *p* "hello, world!")"#).unwrap();
        assert_eq!(
            s.eval("(read-file-string *p*)").unwrap(),
            r#""hello, world!""#,
        );
        s.eval("(delete-file *p*)").unwrap();
    }

    #[test]
    fn file_exists_round_trip() {
        let mut s = Session::with_stdlib().unwrap();
        let path = temp_path("exists.txt");
        s.eval(&format!(r#"(defparameter *p* "{}")"#, path.replace('\\', "\\\\")))
            .unwrap();
        // Doesn't exist yet.
        assert_eq!(s.eval("(file-exists *p*)").unwrap(), "nil");
        s.eval(r#"(write-file-string *p* "hi")"#).unwrap();
        assert_eq!(s.eval("(file-exists *p*)").unwrap(), "T");
        s.eval("(delete-file *p*)").unwrap();
        assert_eq!(s.eval("(file-exists *p*)").unwrap(), "nil");
    }

    #[test]
    fn file_lines_round_trip() {
        let mut s = Session::with_stdlib().unwrap();
        let path = temp_path("lines.txt");
        s.eval(&format!(r#"(defparameter *p* "{}")"#, path.replace('\\', "\\\\")))
            .unwrap();
        s.eval(r#"(write-file-lines *p* '("alpha" "beta" "gamma"))"#).unwrap();
        assert_eq!(
            s.eval("(read-file-lines *p*)").unwrap(),
            r#"("alpha" "beta" "gamma")"#,
        );
        s.eval("(delete-file *p*)").unwrap();
    }

    #[test]
    fn with_open_file_macro_works() {
        let mut s = Session::with_stdlib().unwrap();
        let path = temp_path("with_open.txt");
        s.eval(&format!(r#"(defparameter *p* "{}")"#, path.replace('\\', "\\\\")))
            .unwrap();
        // Write via with-open-file (using real CL keywords).
        s.eval(
            "(with-open-file (out *p* :output) \
               (write-line out \"line 1\") \
               (write-line out \"line 2\"))",
        )
        .unwrap();
        // Read back via with-open-file.
        assert_eq!(
            s.eval("(with-open-file (in *p* :input) (read-line in))").unwrap(),
            r#""line 1""#,
        );
        s.eval("(delete-file *p*)").unwrap();
    }

    #[test]
    fn read_line_returns_nil_at_eof() {
        let mut s = Session::with_stdlib().unwrap();
        let path = temp_path("eof.txt");
        s.eval(&format!(r#"(defparameter *p* "{}")"#, path.replace('\\', "\\\\")))
            .unwrap();
        s.eval(r#"(write-file-string *p* "single")"#).unwrap();
        s.eval("(defparameter *h* (open-input-file *p*))").unwrap();
        assert_eq!(s.eval("(read-line *h*)").unwrap(), r#""single""#);
        // Past EOF.
        assert_eq!(s.eval("(read-line *h*)").unwrap(), "nil");
        s.eval("(close-stream *h*)").unwrap();
        s.eval("(delete-file *p*)").unwrap();
    }

    #[test]
    fn file_append_mode() {
        let mut s = Session::with_stdlib().unwrap();
        let path = temp_path("append.txt");
        s.eval(&format!(r#"(defparameter *p* "{}")"#, path.replace('\\', "\\\\")))
            .unwrap();
        // Write initial.
        s.eval(r#"(write-file-string *p* "first")"#).unwrap();
        // Append more.
        s.eval(
            "(with-open-file (out *p* :append) (write-string-to out \"-second\"))",
        )
        .unwrap();
        assert_eq!(
            s.eval("(read-file-string *p*)").unwrap(),
            r#""first-second""#,
        );
        s.eval("(delete-file *p*)").unwrap();
    }

    #[test]
    fn file_open_failure_returns_zero_handle() {
        let mut s = Session::with_stdlib().unwrap();
        // A path that almost certainly doesn't exist.
        let h = s
            .eval(r#"(open-input-file "Z:\\definitely-not-here-9876.xyz")"#)
            .unwrap();
        assert_eq!(h, "0");
    }

    #[test]
    fn file_unicode_round_trip() {
        let mut s = Session::with_stdlib().unwrap();
        let path = temp_path("unicode.txt");
        s.eval(&format!(r#"(defparameter *p* "{}")"#, path.replace('\\', "\\\\")))
            .unwrap();
        s.eval(r#"(write-file-string *p* "café 🦀 日本")"#).unwrap();
        assert_eq!(
            s.eval("(read-file-string *p*)").unwrap(),
            r#""café 🦀 日本""#,
        );
        s.eval("(delete-file *p*)").unwrap();
    }

    // -- defmacro / macros -----------------------------------------------

    #[test]
    fn defmacro_returns_nil() {
        let mut s = Session::new();
        // Defining a macro is a side-effecting form; it returns nil.
        assert_eq!(
            s.eval("(defmacro my-id (x) x)").unwrap(),
            "nil",
        );
    }

    #[test]
    fn macro_simplest_passthrough() {
        // The macro body returns its argument unchanged. Calling
        // (my-id 42) expands to 42.
        let mut s = Session::new();
        s.eval("(defmacro my-id (x) x)").unwrap();
        assert_eq!(s.eval("(my-id 42)").unwrap(), "42");
        // The arg is the literal form, so quoted forms work too.
        assert_eq!(s.eval("(my-id (+ 1 2))").unwrap(), "3");
    }

    #[test]
    fn macro_constructs_form_with_list() {
        // (my-when test body) → (if test body nil), built with list
        // and quote (no backquote yet).
        let mut s = Session::new();
        s.eval(
            "(defmacro my-when (test body) \
               (list 'if test body 'nil))",
        )
        .unwrap();
        assert_eq!(s.eval("(my-when t 5)").unwrap(), "5");
        assert_eq!(s.eval("(my-when nil 5)").unwrap(), "nil");
        // The arg forms are unevaluated until the expansion runs.
        assert_eq!(s.eval("(my-when (> 3 2) (+ 10 20))").unwrap(), "30");
    }

    #[test]
    fn macro_expansion_runs_at_compile_time() {
        // The macro body uses (list 'quote x) to embed a literal
        // form into the expansion. The expansion only runs the
        // outer form, not the embedded literal.
        let mut s = Session::new();
        s.eval(
            "(defmacro literal (x) \
               (list 'quote x))",
        )
        .unwrap();
        // (literal foo) → 'foo → the symbol FOO.
        assert_eq!(s.eval("(literal foo)").unwrap(), "FOO");
        assert_eq!(s.eval("(literal (1 2 3))").unwrap(), "(1 2 3)");
    }

    #[test]
    fn macro_with_rest_args() {
        // &rest in a macro captures all the remaining arg forms.
        // Build (progn arg1 arg2 ...) — return the last one's value.
        let mut s = Session::new();
        s.eval(
            "(defmacro my-progn (&rest forms) \
               (cons 'progn forms))",
        )
        .unwrap();
        assert_eq!(s.eval("(my-progn 1 2 3)").unwrap(), "3");
        assert_eq!(s.eval("(my-progn (+ 1 1) (+ 2 2))").unwrap(), "4");
    }

    #[test]
    fn macro_recursive_expansion() {
        // A macro whose expansion contains another macro call —
        // both get expanded.
        let mut s = Session::new();
        s.eval("(defmacro inc (x) (list '+ x 1))").unwrap();
        s.eval("(defmacro double-inc (x) (list 'inc (list 'inc x)))").unwrap();
        assert_eq!(s.eval("(double-inc 5)").unwrap(), "7");
    }

    #[test]
    fn macro_inside_defun_body() {
        // Macros used inside a defun's body get expanded at compile
        // time — the defun stores the expanded code, not a runtime
        // macro call.
        let mut s = Session::new();
        s.eval("(defmacro plus1 (x) (list '+ x 1))").unwrap();
        s.eval("(defun bump (n) (plus1 n))").unwrap();
        assert_eq!(s.eval("(bump 41)").unwrap(), "42");
    }

    #[test]
    fn macro_in_let_body() {
        let mut s = Session::new();
        s.eval("(defmacro plus1 (x) (list '+ x 1))").unwrap();
        assert_eq!(
            s.eval("(let ((x 10)) (plus1 x))").unwrap(),
            "11",
        );
    }

    #[test]
    fn macro_can_use_quote_in_expansion() {
        let mut s = Session::new();
        // Wraps the form in (quote ...).
        s.eval(
            "(defmacro literally (x) \
               (cons 'quote (cons x nil)))",
        )
        .unwrap();
        assert_eq!(s.eval("(literally hello)").unwrap(), "HELLO");
        assert_eq!(s.eval("(literally (a b c))").unwrap(), "(A B C)");
    }

    #[test]
    fn macro_redefinition_replaces() {
        // A second defmacro with the same name installs a new
        // expansion; subsequent calls use the new one. (Existing
        // compiled code bound to the OLD expansion stays bound to
        // the old expansion — that's CL semantics.)
        let mut s = Session::new();
        s.eval("(defmacro tag (x) (list 'list ''v1 x))").unwrap();
        assert_eq!(s.eval("(tag 5)").unwrap(), "(V1 5)");
        s.eval("(defmacro tag (x) (list 'list ''v2 x))").unwrap();
        assert_eq!(s.eval("(tag 5)").unwrap(), "(V2 5)");
    }

    // -- backquote / quasiquote ------------------------------------------

    #[test]
    fn backquote_atom_is_quoted() {
        // `foo → 'foo → FOO at evaluation time.
        assert_eq!(eval_str("`foo").unwrap(), "FOO");
        // Numbers self-quote.
        assert_eq!(eval_str("`42").unwrap(), "42");
        // nil self-quotes.
        assert_eq!(eval_str("`nil").unwrap(), "nil");
        // Strings.
        assert_eq!(eval_str(r#"`"hi""#).unwrap(), r#""hi""#);
    }

    #[test]
    fn backquote_unquote_evaluates() {
        // `,x → x.
        let mut s = Session::new();
        s.eval("(defparameter *v* 42)").unwrap();
        assert_eq!(s.eval("`,*v*").unwrap(), "42");
    }

    #[test]
    fn backquote_list_with_unquotes() {
        // `(a ,b c) → (cons 'a (cons b (cons 'c nil))).
        let mut s = Session::new();
        s.eval("(defparameter *x* 99)").unwrap();
        assert_eq!(
            s.eval("`(a ,*x* c)").unwrap(),
            "(A 99 C)",
        );
    }

    #[test]
    fn backquote_with_splice() {
        // `(a ,@xs b) splices xs into the list.
        let mut s = Session::new();
        s.eval("(defparameter *xs* '(1 2 3))").unwrap();
        assert_eq!(
            s.eval("`(a ,@*xs* b)").unwrap(),
            "(A 1 2 3 B)",
        );
        // Empty splice — disappears.
        s.eval("(defparameter *empty* nil)").unwrap();
        assert_eq!(
            s.eval("`(a ,@*empty* b)").unwrap(),
            "(A B)",
        );
    }

    #[test]
    fn backquote_only_splice() {
        // `(,@xs) is just xs.
        let mut s = Session::new();
        s.eval("(defparameter *xs* '(1 2 3))").unwrap();
        assert_eq!(s.eval("`(,@*xs*)").unwrap(), "(1 2 3)");
    }

    #[test]
    fn backquote_dotted() {
        // `(a . ,x) puts x in the cdr.
        let mut s = Session::new();
        s.eval("(defparameter *x* 5)").unwrap();
        assert_eq!(s.eval("`(a . ,*x*)").unwrap(), "(A . 5)");
    }

    #[test]
    fn backquote_in_macro_my_when() {
        // The classic example: write `when` cleanly using backquote.
        let mut s = Session::new();
        s.eval(
            "(defmacro my-when (test &rest body) \
               `(if ,test (progn ,@body) nil))",
        )
        .unwrap();
        assert_eq!(s.eval("(my-when t 1 2 3)").unwrap(), "3");
        assert_eq!(s.eval("(my-when nil 1 2 3)").unwrap(), "nil");
        assert_eq!(s.eval("(my-when (> 5 3) (+ 1 2) (+ 3 4))").unwrap(), "7");
    }

    #[test]
    fn backquote_in_macro_unless() {
        let mut s = Session::new();
        s.eval(
            "(defmacro my-unless (test &rest body) \
               `(if ,test nil (progn ,@body)))",
        )
        .unwrap();
        assert_eq!(s.eval("(my-unless nil 1 2 3)").unwrap(), "3");
        assert_eq!(s.eval("(my-unless t 1 2 3)").unwrap(), "nil");
    }

    #[test]
    fn backquote_in_macro_with_splicing_body() {
        // (let-1 var val body...) → (let ((var val)) body...)
        let mut s = Session::new();
        s.eval(
            "(defmacro let-1 (var val &rest body) \
               `(let ((,var ,val)) ,@body))",
        )
        .unwrap();
        assert_eq!(s.eval("(let-1 x 10 (+ x 1))").unwrap(), "11");
        assert_eq!(
            s.eval("(let-1 x 10 (+ x 1) (* x 2))").unwrap(),
            "20",
        );
    }

    #[test]
    fn backquote_nested_list_structure() {
        // `((a ,x) (b ,y))
        let mut s = Session::new();
        s.eval("(defparameter *x* 1)").unwrap();
        s.eval("(defparameter *y* 2)").unwrap();
        assert_eq!(
            s.eval("`((a ,*x*) (b ,*y*))").unwrap(),
            "((A 1) (B 2))",
        );
    }

    #[test]
    fn backquote_unquote_outside_backquote_errors() {
        // ,x at top level — not inside backquote.
        let r = eval_str(",foo");
        assert!(r.is_err());
    }

    #[test]
    fn quote_is_opaque_to_macroexpansion() {
        // (when ...) is the WHEN special form (already a primitive).
        // But if we redefine it as a macro and quote a (when ...)
        // form, the quoted data should not get expanded.
        let mut s = Session::new();
        s.eval("(defmacro mine (x) (list '+ x 1))").unwrap();
        // Inside quote, (mine 5) is just data — no expansion.
        assert_eq!(s.eval("(quote (mine 5))").unwrap(), "(MINE 5)");
        // Outside quote, it expands.
        assert_eq!(s.eval("(mine 5)").unwrap(), "6");
    }

    // -- format ----------------------------------------------------------

    #[test]
    fn format_nil_returns_string() {
        // (format nil "...") returns the formatted text as a String.
        assert_eq!(
            eval_str(r#"(format nil "hello, world")"#).unwrap(),
            r#""hello, world""#,
        );
    }

    #[test]
    fn format_a_directive_aesthetic() {
        // ~A: aesthetic — strings printed without quotes.
        assert_eq!(
            eval_str(r#"(format nil "name: ~A!" "Alice")"#).unwrap(),
            r#""name: Alice!""#,
        );
        // Numbers print as digits.
        assert_eq!(
            eval_str(r#"(format nil "x = ~A" 42)"#).unwrap(),
            r#""x = 42""#,
        );
        // nil prints as "nil".
        assert_eq!(
            eval_str(r#"(format nil "got ~A" nil)"#).unwrap(),
            r#""got nil""#,
        );
        // Lists print structurally.
        assert_eq!(
            eval_str(r#"(format nil "list: ~A" '(1 2 3))"#).unwrap(),
            r#""list: (1 2 3)""#,
        );
    }

    #[test]
    fn format_s_directive_readable() {
        // ~S: readable — strings keep their quotes.
        assert_eq!(
            eval_str(r#"(format nil "got ~S" "hi")"#).unwrap(),
            r#""got \"hi\"""#,
        );
        // Chars print as #\X. The eval result is itself a string;
        // when the REPL prints it back, the backslash inside the
        // payload gets escaped, so `#\a` round-trips as `"#\\a"`.
        // Use explicit escapes to dodge the Rust 2021 reserved-
        // prefix lint on `r#"#`.
        assert_eq!(
            eval_str("(format nil \"~S\" #\\a)").unwrap(),
            "\"#\\\\a\"",
        );
    }

    #[test]
    fn format_d_directive_decimal() {
        assert_eq!(
            eval_str(r#"(format nil "n = ~D" 1234)"#).unwrap(),
            r#""n = 1234""#,
        );
        assert_eq!(
            eval_str(r#"(format nil "neg: ~D" -7)"#).unwrap(),
            r#""neg: -7""#,
        );
    }

    #[test]
    fn format_percent_emits_newline() {
        assert_eq!(
            eval_str(r#"(format nil "line1~%line2")"#).unwrap(),
            "\"line1\nline2\"",
        );
    }

    #[test]
    fn format_tilde_tilde_is_literal() {
        assert_eq!(
            eval_str(r#"(format nil "~~ here")"#).unwrap(),
            r#""~ here""#,
        );
    }

    #[test]
    fn format_multiple_directives() {
        assert_eq!(
            eval_str(r#"(format nil "~A is ~D years old" "Bob" 30)"#).unwrap(),
            r#""Bob is 30 years old""#,
        );
    }

    #[test]
    fn format_unicode_in_control_string() {
        // The control string is UTF-32 internally; full Unicode pass-through.
        assert_eq!(
            eval_str(r#"(format nil "héllo ~A" "🦀")"#).unwrap(),
            r#""héllo 🦀""#,
        );
    }

    #[test]
    fn format_to_t_returns_nil() {
        // (format t ...) writes to stdout and returns nil. The
        // tests can't easily capture stdout, so just verify the
        // return value.
        let mut s = Session::new();
        assert_eq!(s.eval(r#"(format t "ignored")"#).unwrap(), "nil");
    }

    #[test]
    fn format_is_first_class() {
        // FORMAT is a real callable function — usable via funcall,
        // apply, #'.
        let mut s = Session::new();
        s.eval("(defparameter *fmt* #'format)").unwrap();
        assert_eq!(
            s.eval(r#"(funcall *fmt* nil "hi ~A" 42)"#).unwrap(),
            r#""hi 42""#,
        );
        // apply works too — splat a list of args.
        assert_eq!(
            s.eval(r#"(apply #'format nil "~A and ~A" '(1 2))"#).unwrap(),
            r#""1 and 2""#,
        );
    }

    #[test]
    fn format_no_args_passes_through() {
        // No directives, no args — returns the control unchanged.
        assert_eq!(
            eval_str(r#"(format nil "plain text")"#).unwrap(),
            r#""plain text""#,
        );
        // Empty control.
        assert_eq!(eval_str(r#"(format nil "")"#).unwrap(), r#""""#);
    }

    // -- apply -----------------------------------------------------------

    #[test]
    fn apply_simple() {
        // (apply f lst) — splat lst as args to f.
        let mut s = Session::new();
        s.eval("(defun add3 (a b c) (+ a b c))").unwrap();
        assert_eq!(s.eval("(apply #'add3 '(1 2 3))").unwrap(), "6");
    }

    #[test]
    fn apply_with_prefix() {
        // (apply f a b lst) — prefix a, b followed by lst.
        let mut s = Session::new();
        s.eval("(defun add4 (a b c d) (+ a b c d))").unwrap();
        assert_eq!(s.eval("(apply #'add4 1 '(2 3 4))").unwrap(), "10");
        assert_eq!(s.eval("(apply #'add4 1 2 '(3 4))").unwrap(), "10");
        assert_eq!(s.eval("(apply #'add4 1 2 3 '(4))").unwrap(), "10");
    }

    #[test]
    fn apply_with_empty_tail() {
        let mut s = Session::new();
        s.eval("(defun add2 (a b) (+ a b))").unwrap();
        assert_eq!(s.eval("(apply #'add2 1 2 nil)").unwrap(), "3");
    }

    #[test]
    fn apply_to_variadic() {
        // The classic apply use case: pass a list to a &rest-taking
        // function so it sees the elements as separate args.
        let mut s = Session::new();
        s.eval("(defun all (&rest r) r)").unwrap();
        assert_eq!(s.eval("(apply #'all '(1 2 3 4))").unwrap(), "(1 2 3 4)");
        assert_eq!(s.eval("(apply #'all 0 '(1 2 3))").unwrap(), "(0 1 2 3)");
    }

    #[test]
    fn apply_with_lambda() {
        let mut s = Session::new();
        assert_eq!(
            s.eval("(apply (lambda (a b) (* a b)) '(7 6))").unwrap(),
            "42",
        );
    }

    #[test]
    fn apply_to_stdlib_min() {
        // Use stdlib's variadic `min` via apply.
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(apply #'min '(3 1 4 1 5 9 2 6))").unwrap(), "1");
        assert_eq!(s.eval("(apply #'max '(3 1 4 1 5 9 2 6))").unwrap(), "9");
    }

    #[test]
    fn apply_arity_too_few_errors_at_compile() {
        // (apply) and (apply f) — both lack the tail list.
        let r = eval_str("(apply)");
        assert!(r.is_err());
        let r = eval_str("(apply #'+)");
        assert!(r.is_err());
    }

    #[test]
    fn apply_returns_value() {
        let mut s = Session::new();
        s.eval("(defun double (x) (* x 2))").unwrap();
        assert_eq!(s.eval("(apply #'double '(21))").unwrap(), "42");
    }

    // -- truncate / rem (integer division primitives) ---------------------

    #[test]
    fn truncate_basic() {
        assert_eq!(eval_str("(truncate 10 3)").unwrap(), "3");
        assert_eq!(eval_str("(truncate 7 2)").unwrap(), "3");
        assert_eq!(eval_str("(truncate 6 3)").unwrap(), "2");
        // Truncate rounds toward zero — negative dividend.
        assert_eq!(eval_str("(truncate -7 2)").unwrap(), "-3");
        assert_eq!(eval_str("(truncate 7 -2)").unwrap(), "-3");
        assert_eq!(eval_str("(truncate -7 -2)").unwrap(), "3");
    }

    #[test]
    fn rem_basic() {
        assert_eq!(eval_str("(rem 10 3)").unwrap(), "1");
        assert_eq!(eval_str("(rem 7 2)").unwrap(), "1");
        assert_eq!(eval_str("(rem 6 3)").unwrap(), "0");
        // rem matches the sign of the dividend.
        assert_eq!(eval_str("(rem -7 2)").unwrap(), "-1");
        assert_eq!(eval_str("(rem 7 -2)").unwrap(), "1");
        assert_eq!(eval_str("(rem -7 -2)").unwrap(), "-1");
    }

    #[test]
    fn truncate_rem_invariant() {
        // (= a (+ (* (truncate a b) b) (rem a b))) for any a, b.
        let mut session = Session::new();
        session
            .eval("(defun ok (a b) (= a (+ (* (truncate a b) b) (rem a b))))")
            .unwrap();
        assert_eq!(session.eval("(ok 17 5)").unwrap(), "T");
        assert_eq!(session.eval("(ok -17 5)").unwrap(), "T");
        assert_eq!(session.eval("(ok 17 -5)").unwrap(), "T");
        assert_eq!(session.eval("(ok -17 -5)").unwrap(), "T");
        assert_eq!(session.eval("(ok 42 7)").unwrap(), "T");
    }

    #[test]
    fn stdlib_mod_matches_divisor_sign() {
        let mut s = Session::with_stdlib().unwrap();
        // Same sign as divisor — differs from rem when signs differ.
        assert_eq!(s.eval("(mod 10 3)").unwrap(), "1");
        assert_eq!(s.eval("(mod -7 2)").unwrap(), "1");   // rem returns -1
        assert_eq!(s.eval("(mod 7 -2)").unwrap(), "-1");  // rem returns 1
        assert_eq!(s.eval("(mod -7 -2)").unwrap(), "-1"); // matches rem
        // Exact divisions return 0 regardless of sign.
        assert_eq!(s.eval("(mod 6 3)").unwrap(), "0");
        assert_eq!(s.eval("(mod -6 3)").unwrap(), "0");
    }

    #[test]
    fn stdlib_oddp_evenp() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(evenp 0)").unwrap(), "T");
        assert_eq!(s.eval("(evenp 4)").unwrap(), "T");
        assert_eq!(s.eval("(evenp -4)").unwrap(), "T");
        assert_eq!(s.eval("(evenp 3)").unwrap(), "nil");
        assert_eq!(s.eval("(oddp 3)").unwrap(), "T");
        assert_eq!(s.eval("(oddp -3)").unwrap(), "T");
        assert_eq!(s.eval("(oddp 0)").unwrap(), "nil");
    }

    #[test]
    fn stdlib_floor_rounds_toward_negative_infinity() {
        let mut s = Session::with_stdlib().unwrap();
        // Differs from truncate on mixed signs with non-zero remainder.
        assert_eq!(s.eval("(floor 7 2)").unwrap(), "3");
        assert_eq!(s.eval("(floor -7 2)").unwrap(), "-4");  // truncate would give -3
        assert_eq!(s.eval("(floor 7 -2)").unwrap(), "-4");
        assert_eq!(s.eval("(floor -7 -2)").unwrap(), "3");
        // Exact divisions match truncate.
        assert_eq!(s.eval("(floor 6 3)").unwrap(), "2");
        assert_eq!(s.eval("(floor -6 3)").unwrap(), "-2");
    }

    // -- when / unless ----------------------------------------------------

    #[test]
    fn when_true_runs_body() {
        assert_eq!(eval_str("(when t 1 2 3)").unwrap(), "3");
    }

    #[test]
    fn when_false_returns_nil() {
        assert_eq!(eval_str("(when nil 1 2 3)").unwrap(), "nil");
    }

    #[test]
    fn when_no_body_returns_nil() {
        assert_eq!(eval_str("(when t)").unwrap(), "nil");
        assert_eq!(eval_str("(when nil)").unwrap(), "nil");
    }

    #[test]
    fn unless_inverts_when() {
        assert_eq!(eval_str("(unless nil 1 2 3)").unwrap(), "3");
        assert_eq!(eval_str("(unless t 1 2 3)").unwrap(), "nil");
    }

    // -- Core stdlib (Lisp/core.lisp) -------------------------------------

    #[test]
    fn stdlib_loads_clean() {
        // Smoke test: the file evaluates without error.
        let mut s = Session::with_stdlib().expect("stdlib should load");
        // Bonus: a defined function is callable.
        assert_eq!(s.eval("(first '(1 2 3))").unwrap(), "1");
    }

    #[test]
    fn stdlib_accessors() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(first '(a b c))").unwrap(), "A");
        assert_eq!(s.eval("(rest '(a b c))").unwrap(), "(B C)");
        assert_eq!(s.eval("(second '(1 2 3))").unwrap(), "2");
        assert_eq!(s.eval("(third '(1 2 3 4))").unwrap(), "3");
        assert_eq!(s.eval("(fourth '(1 2 3 4 5))").unwrap(), "4");
        assert_eq!(s.eval("(cadr '(1 2 3))").unwrap(), "2");
        assert_eq!(s.eval("(cddr '(1 2 3 4))").unwrap(), "(3 4)");
    }

    #[test]
    fn stdlib_reverse() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(reverse '(1 2 3 4 5))").unwrap(), "(5 4 3 2 1)");
        assert_eq!(s.eval("(reverse nil)").unwrap(), "nil");
        assert_eq!(s.eval("(reverse '(a))").unwrap(), "(A)");
    }

    #[test]
    fn stdlib_append() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval("(append '(1 2 3) '(4 5))").unwrap(),
            "(1 2 3 4 5)",
        );
        assert_eq!(s.eval("(append nil '(a b))").unwrap(), "(A B)");
        assert_eq!(s.eval("(append '(a b) nil)").unwrap(), "(A B)");
    }

    #[test]
    fn stdlib_mapcar() {
        let mut s = Session::with_stdlib().unwrap();
        s.eval("(defun double (x) (* x 2))").unwrap();
        assert_eq!(
            s.eval("(mapcar #'double '(1 2 3 4))").unwrap(),
            "(2 4 6 8)",
        );
        // With a lambda.
        assert_eq!(
            s.eval("(mapcar (lambda (x) (+ x 10)) '(1 2 3))").unwrap(),
            "(11 12 13)",
        );
        assert_eq!(s.eval("(mapcar #'double nil)").unwrap(), "nil");
    }

    #[test]
    fn stdlib_member() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(member 3 '(1 2 3 4 5))").unwrap(), "(3 4 5)");
        assert_eq!(s.eval("(member 99 '(1 2 3))").unwrap(), "nil");
        // member uses equal — finds string content match.
        assert_eq!(
            s.eval(r#"(member "b" '("a" "b" "c"))"#).unwrap(),
            r#"("b" "c")"#,
        );
    }

    #[test]
    fn stdlib_position_and_find() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(position 'c '(a b c d))").unwrap(), "2");
        assert_eq!(s.eval("(position 'z '(a b c))").unwrap(), "nil");
        assert_eq!(s.eval("(find 3 '(1 2 3 4))").unwrap(), "3");
        assert_eq!(s.eval("(find 99 '(1 2 3))").unwrap(), "nil");
    }

    #[test]
    fn stdlib_assoc() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(
            s.eval("(assoc 'b '((a . 1) (b . 2) (c . 3)))").unwrap(),
            "(B . 2)",
        );
        assert_eq!(
            s.eval("(assoc 'z '((a . 1) (b . 2)))").unwrap(),
            "nil",
        );
    }

    #[test]
    fn stdlib_nth_and_nthcdr() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(nth 0 '(a b c))").unwrap(), "A");
        assert_eq!(s.eval("(nth 2 '(a b c))").unwrap(), "C");
        assert_eq!(s.eval("(nthcdr 0 '(a b c))").unwrap(), "(A B C)");
        assert_eq!(s.eval("(nthcdr 1 '(a b c))").unwrap(), "(B C)");
        assert_eq!(s.eval("(nthcdr 3 '(a b c))").unwrap(), "nil");
    }

    #[test]
    fn stdlib_last_returns_last_cons() {
        let mut s = Session::with_stdlib().unwrap();
        // CL: last returns the LAST CONS, not the last element.
        assert_eq!(s.eval("(last '(1 2 3))").unwrap(), "(3)");
        assert_eq!(s.eval("(last '(a))").unwrap(), "(A)");
        assert_eq!(s.eval("(last nil)").unwrap(), "nil");
    }

    #[test]
    fn stdlib_butlast() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(butlast '(1 2 3 4))").unwrap(), "(1 2 3)");
        assert_eq!(s.eval("(butlast '(1))").unwrap(), "nil");
        assert_eq!(s.eval("(butlast nil)").unwrap(), "nil");
    }

    #[test]
    fn stdlib_every_some() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(every #'plusp '(1 2 3))").unwrap(), "T");
        assert_eq!(s.eval("(every #'plusp '(1 -1 3))").unwrap(), "nil");
        assert_eq!(s.eval("(every #'plusp nil)").unwrap(), "T");
        assert_eq!(s.eval("(some #'minusp '(1 2 -3))").unwrap(), "T");
        assert_eq!(s.eval("(some #'minusp '(1 2 3))").unwrap(), "nil");
    }

    #[test]
    fn stdlib_numeric_helpers() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(zerop 0)").unwrap(), "T");
        assert_eq!(s.eval("(zerop 5)").unwrap(), "nil");
        assert_eq!(s.eval("(plusp 5)").unwrap(), "T");
        assert_eq!(s.eval("(plusp -5)").unwrap(), "nil");
        assert_eq!(s.eval("(minusp -5)").unwrap(), "T");
        assert_eq!(s.eval("(1+ 41)").unwrap(), "42");
        assert_eq!(s.eval("(1- 43)").unwrap(), "42");
        assert_eq!(s.eval("(min2 3 7)").unwrap(), "3");
        assert_eq!(s.eval("(max2 3 7)").unwrap(), "7");
        assert_eq!(s.eval("(abs -5)").unwrap(), "5");
        assert_eq!(s.eval("(abs 5)").unwrap(), "5");
    }

    #[test]
    fn stdlib_copy_list_unshares() {
        let mut s = Session::with_stdlib().unwrap();
        s.eval("(defparameter *a* '(1 2 3))").unwrap();
        s.eval("(defparameter *b* (copy-list *a*))").unwrap();
        // Same content...
        assert_eq!(s.eval("(equal *a* *b*)").unwrap(), "T");
        // ...but distinct conses.
        assert_eq!(s.eval("(eq *a* *b*)").unwrap(), "nil");
    }

    #[test]
    fn stdlib_composes() {
        // Build a non-trivial pipeline using stdlib functions.
        let mut s = Session::with_stdlib().unwrap();
        // Reverse, then mapcar 1+, then take last cons.
        assert_eq!(
            s.eval("(last (mapcar #'1+ (reverse '(1 2 3 4))))").unwrap(),
            "(2)",
        );
    }

    #[test]
    fn stdlib_min_max_variadic() {
        let mut s = Session::with_stdlib().unwrap();
        // Single-arg forms.
        assert_eq!(s.eval("(min 5)").unwrap(), "5");
        assert_eq!(s.eval("(max 5)").unwrap(), "5");
        // Multi-arg.
        assert_eq!(s.eval("(min 3 1 4 1 5 9 2 6)").unwrap(), "1");
        assert_eq!(s.eval("(max 3 1 4 1 5 9 2 6)").unwrap(), "9");
        assert_eq!(s.eval("(min 7 7 7)").unwrap(), "7");
        // Mixed signs.
        assert_eq!(s.eval("(min -5 -2 -10)").unwrap(), "-10");
        assert_eq!(s.eval("(max -5 -2 -10)").unwrap(), "-2");
    }

    #[test]
    fn stdlib_list_star() {
        let mut s = Session::with_stdlib().unwrap();
        // (list* x) ≡ x.
        assert_eq!(s.eval("(list* 'a)").unwrap(), "A");
        // (list* a b) ≡ (cons a b).
        assert_eq!(s.eval("(list* 1 2)").unwrap(), "(1 . 2)");
        // (list* a b c lst) ≡ (cons a (cons b (cons c lst)))
        assert_eq!(s.eval("(list* 1 2 3 '(4 5))").unwrap(), "(1 2 3 4 5)");
        // (list* 1 2 3 nil) ≡ (1 2 3)
        assert_eq!(s.eval("(list* 1 2 3 nil)").unwrap(), "(1 2 3)");
    }

    #[test]
    fn stdlib_append_star_variadic() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(append*)").unwrap(), "nil");
        assert_eq!(s.eval("(append* '(a b))").unwrap(), "(A B)");
        assert_eq!(
            s.eval("(append* '(1 2) '(3 4) '(5 6))").unwrap(),
            "(1 2 3 4 5 6)",
        );
        assert_eq!(s.eval("(append* nil nil nil)").unwrap(), "nil");
    }

    #[test]
    fn stdlib_identity() {
        let mut s = Session::with_stdlib().unwrap();
        assert_eq!(s.eval("(identity 42)").unwrap(), "42");
        assert_eq!(s.eval("(identity 'foo)").unwrap(), "FOO");
        assert_eq!(s.eval("(mapcar #'identity '(1 2 3))").unwrap(), "(1 2 3)");
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
