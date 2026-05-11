use std::cell::Cell;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::mem::MaybeUninit;
use std::process::ExitCode;
use std::sync::Mutex;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() {
    eprintln!("usage: ncl [--lean] [--repl | (--eval <src> | --load <file>)...] [--repl]");
    eprintln!("       ncl --version | --help");
    eprintln!("  --eval, -e <src>     evaluate a source string");
    eprintln!("  --load, -l <file>    read and evaluate the file");
    eprintln!("  --repl, -r           enter the interactive REPL (default if no flags given)");
    eprintln!("  --lean, -L           start with core only (no CLOS, no Library/init.lisp)");
    eprintln!("  --version, -V        print version and exit");
    eprintln!("  --help, -h           print this message and exit");
    eprintln!("  multiple --eval / --load can be chained; --repl runs after them");
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = env::args().skip(1).collect();

    // Early-exit flags. Scan ALL of argv so position doesn't matter —
    // `ncl --version`, `ncl --lean --version`, `ncl -e foo -V` all
    // print version-and-exit before any session work.
    if raw_args.iter().any(|a| a == "--version" || a == "-V") {
        println!("NewCormanLisp {VERSION}");
        return ExitCode::SUCCESS;
    }
    if raw_args.iter().any(|a| a == "--help" || a == "-h") {
        usage();
        return ExitCode::SUCCESS;
    }

    // Bare `ncl` invocation drops into the REPL with the stdlib loaded.
    let want_repl = raw_args.is_empty()
        || raw_args.iter().any(|a| a == "--repl" || a == "-r");

    // --lean: skip CLOS, skip Library/init.lisp. User explicitly opted
    // out of the standard auto-loaded surface. Useful for scripts or
    // sandboxing.
    let lean = raw_args.iter().any(|a| a == "--lean" || a == "-L");

    let session_result = if lean {
        ncl_compiler::Session::with_minimal_stdlib()
    } else {
        ncl_compiler::Session::with_stdlib()
    };
    let session = match session_result {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ncl: stdlib load failed: {e}");
            return ExitCode::from(1);
        }
    };
    // Park the Session at a stable address so `(eval-string ...)` can
    // route into it from inside Lisp.
    let mut session = Box::new(session);
    session.activate();

    // ─── User library bootstrap ──────────────────────────────────────────
    //
    // Look for `Library/` next to the executable. If it exists, push
    // it onto *load-path* and (if Library/init.lisp is present) load
    // that init file. Failures here are warnings, not fatal — the
    // user can still drop into the REPL and work with just the baked-
    // in stdlib.
    //
    // Skipped entirely when --lean is set. In lean mode there's no
    // load / require / *load-path* in the session at all (those live
    // in the bottom of core.lisp — still loaded, since they don't
    // depend on CLOS — so library bootstrap is suppressed by choice,
    // not by absence).
    if !lean {
        if let Some(library_dir) = find_library_dir() {
            let setup = format!(
                "(setq *load-path* (cons \"{}\" *load-path*))",
                library_dir.replace('\\', "/")
            );
            if let Err(e) = session.eval(&setup) {
                eprintln!("ncl: warning: could not extend *load-path*: {e}");
            }
            let init_path = format!("{library_dir}/init.lisp");
            if std::path::Path::new(&init_path).exists() {
                let load = format!("(load \"{}\")", init_path.replace('\\', "/"));
                if let Err(e) = session.eval(&load) {
                    eprintln!("ncl: warning: Library/init.lisp failed: {e}");
                }
            }
        }
    }

    let mut last_output: Option<String> = None;
    let mut args = raw_args.into_iter().peekable();

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--eval" | "-e" => {
                let Some(src) = args.next() else {
                    eprintln!("ncl: {flag} requires a source string");
                    usage();
                    return ExitCode::from(2);
                };
                match session.eval(&src) {
                    Ok(s) => last_output = Some(s),
                    Err(e) => {
                        eprintln!("ncl: {e}");
                        return ExitCode::from(1);
                    }
                }
            }
            "--load" | "-l" => {
                let Some(path) = args.next() else {
                    eprintln!("ncl: {flag} requires a file path");
                    usage();
                    return ExitCode::from(2);
                };
                let src = match fs::read_to_string(&path) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("ncl: cannot read {path}: {e}");
                        return ExitCode::from(1);
                    }
                };
                match session.eval(&src) {
                    Ok(s) => last_output = Some(s),
                    Err(e) => {
                        eprintln!("ncl: {path}: {e}");
                        return ExitCode::from(1);
                    }
                }
            }
            "--repl" | "-r" => {
                // Handled below; just accept and continue scanning.
            }
            "--lean" | "-L" => {
                // Handled at session-construction time above; accept here.
            }
            other => {
                eprintln!("ncl: unknown argument '{other}'");
                usage();
                return ExitCode::from(2);
            }
        }
    }

    if let Some(s) = &last_output {
        println!("{s}");
    }

    if want_repl {
        return run_repl(&mut session);
    }

    ExitCode::SUCCESS
}

/// Resolve the path to `Library/` next to the executable. Returns
/// the absolute path string if the directory exists, else None.
///
/// Search order:
///   1. NCL_LIBRARY env var (override for dev / install bundles)
///   2. <exe_dir>/Library  (the shipping shape)
///   3. <exe_dir>/../../Lisp/Library  (developer running cargo run)
///
/// Anything not found falls through to None; the loader is optional.
fn find_library_dir() -> Option<String> {
    if let Ok(p) = std::env::var("NCL_LIBRARY") {
        if std::path::Path::new(&p).is_dir() {
            return Some(p);
        }
    }
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    let beside = exe_dir.join("Library");
    if beside.is_dir() {
        return Some(beside.to_string_lossy().into_owned());
    }
    // Dev fallback: target/release/ncl.exe → repo-root/Lisp/Library
    let dev = exe_dir
        .ancestors()
        .nth(2)
        .map(|p| p.join("Lisp").join("Library"));
    if let Some(d) = dev {
        if d.is_dir() {
            return Some(d.to_string_lossy().into_owned());
        }
    }
    None
}

// ─── setjmp/longjmp bindings ────────────────────────────────────────────
//
// libc doesn't expose setjmp on Windows because the MSVC ABI for
// setjmp / longjmp is compiler-specific (it's a builtin, technically).
// We declare the C runtime entry points by hand. The jmp_buf size
// is platform-dependent — on x86_64 MSVC it's 16 × 8 = 128 bytes,
// plus 16-byte alignment slack — 256 bytes with 16-byte alignment is
// comfortably oversized for every target we care about.

#[repr(C, align(16))]
struct JmpBuf([u8; 256]);

unsafe extern "C" {
    /// Save calling-thread register state into env. Returns 0 on
    /// the initial call, returns the `val` passed to longjmp on
    /// the longjmp resume.
    #[link_name = "_setjmp"]
    fn setjmp_raw(env: *mut JmpBuf) -> i32;

    /// Restore registers from env and resume at the setjmp call
    /// site as if it returned `val`.
    fn longjmp(env: *mut JmpBuf, val: i32) -> !;
}

// ─── REPL panic-recovery via setjmp/longjmp ────────────────────────────
//
// Most user-level errors (undefined function, unbound variable) are
// converted to catchable Lisp conditions inside the runtime — the
// REPL wraps each input in a top-level `handler-case` and prints
// the result. But some Rust panics (ncl_car of non-cons, length on
// improper list, etc.) still escape: panicking out of a Rust runtime
// helper fails to unwind cleanly through MCJIT-emitted JIT frames on
// Windows because the unwinder needs SEH .pdata tables that MCJIT
// doesn't reliably register.
//
// The standard workaround is the same one Lisp REPLs have used since
// the 1970s: a setjmp at the prompt and a longjmp from a global
// panic hook. setjmp captures registers, longjmp restores them; no
// frame unwinding involved. The OS is happy, the JIT frames are
// happy, and the user gets back to a prompt instead of a crashed
// process.

thread_local! {
    /// Per-thread pointer to the active jmp_buf, set by `run_repl`
    /// before each input form. The panic hook reads this; if non-
    /// null, longjmps to it. Cleared after each successful eval so
    /// panics outside the REPL fall through to the default handler.
    static REPL_JMP_BUF: Cell<*mut JmpBuf> = const { Cell::new(std::ptr::null_mut()) };

    /// One-line description of what was running when a panic fired,
    /// captured by the hook before the longjmp. The REPL reads it
    /// after recovering and prints it as the recovery message.
    static REPL_PANIC_MSG: Mutex<Option<String>> = const { Mutex::new(None) };
}

fn install_repl_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .map(String::from)
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<unknown panic>".to_string());
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()));
        let full = match location {
            Some(loc) => format!("panic at {loc}: {msg}"),
            None => format!("panic: {msg}"),
        };
        REPL_PANIC_MSG.with(|cell| {
            if let Ok(mut guard) = cell.lock() {
                *guard = Some(full);
            }
        });

        let buf_ptr = REPL_JMP_BUF.with(|c| c.get());
        if !buf_ptr.is_null() {
            // Clear before the longjmp — we won't return here.
            REPL_JMP_BUF.with(|c| c.set(std::ptr::null_mut()));
            unsafe { longjmp(buf_ptr, 1) };
        }
        // No REPL active: let the default handler print and abort.
        eprintln!("{}", REPL_PANIC_MSG.with(|cell| {
            cell.lock().ok().and_then(|g| g.clone()).unwrap_or_default()
        }));
    }));
}

/// Wrap the user's source in a top-level handler-case so the
/// runtime's converted-to-condition panics (undefined function,
/// unbound variable) print as messages instead of crashing the
/// session. Returns NIL or the formatted error string for the
/// REPL to display.
fn wrap_for_repl(src: &str) -> String {
    format!(
        "(handler-case (progn {src}) (error (c) (format nil \"** ~A\" c)))"
    )
}

/// Interactive read-eval-print loop. Reads from stdin, accumulates
/// input until the form is parseable (handles multi-line entry by
/// detecting an UnexpectedEof from the reader and prompting again),
/// hands it to the session, prints the result.
///
/// Exit on Ctrl+D / EOF, or by typing `(exit)` or `(quit)` at the
/// top-level prompt. Panics inside the eval are caught via a
/// setjmp/longjmp pair and the prompt is restored.
fn run_repl(session: &mut ncl_compiler::Session) -> ExitCode {
    install_repl_panic_hook();

    println!("NewCormanLisp {VERSION} REPL");
    println!("  (exit) or Ctrl+D / Ctrl+Z to leave");
    println!();

    let stdin = io::stdin();
    let mut buf = String::new();
    let mut handle = stdin.lock();

    loop {
        let prompt = if buf.trim().is_empty() { "ncl> " } else { "...> " };
        print!("{prompt}");
        let _ = io::stdout().flush();

        let mut line = String::new();
        match handle.read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("ncl: stdin: {e}");
                return ExitCode::from(1);
            }
        }
        buf.push_str(&line);

        let trimmed = buf.trim();
        if trimmed.is_empty() {
            buf.clear();
            continue;
        }
        if trimmed == "(exit)" || trimmed == "(quit)" {
            break;
        }

        // Probe the reader for completeness.
        match ncl_reader::read_all(&buf) {
            Ok(_) => {
                let src = wrap_for_repl(&buf);
                eval_with_recovery(session, &src);
                buf.clear();
            }
            Err(e) => {
                if is_incomplete(&e) {
                    continue;
                }
                eprintln!("ncl: read error: {:?}", e.kind);
                buf.clear();
            }
        }
    }

    let _ = std::panic::take_hook();
    ExitCode::SUCCESS
}

/// Run one eval inside a setjmp shield. If the eval panics, the
/// installed hook longjmps back here; we print the captured panic
/// message and return without crashing the REPL.
fn eval_with_recovery(session: &mut ncl_compiler::Session, src: &str) {
    let mut jmpbuf: MaybeUninit<JmpBuf> = MaybeUninit::uninit();
    REPL_JMP_BUF.with(|c| c.set(jmpbuf.as_mut_ptr()));

    let r = unsafe { setjmp_raw(jmpbuf.as_mut_ptr()) };
    if r == 0 {
        // First entry — try the eval.
        match session.eval(src) {
            Ok(result) => println!("{result}"),
            Err(e) => eprintln!("ncl: {e}"),
        }
    } else {
        // We just got longjmp'd back. Read whatever the panic hook
        // captured and print it.
        let msg = REPL_PANIC_MSG.with(|cell| {
            cell.lock().ok().and_then(|mut g| g.take()).unwrap_or_default()
        });
        eprintln!("ncl: ** recovered from {msg} **");
    }

    // Disarm the buf so panics OUTSIDE this eval can't longjmp into
    // a stale stack frame.
    REPL_JMP_BUF.with(|c| c.set(std::ptr::null_mut()));
}

/// Is this reader error "input is unfinished, please type more"?
/// Matches end-of-input both from the lexer (mid-string, mid-#\, etc.)
/// and from the parser (unclosed list, dangling `'`/`,`, etc.).
fn is_incomplete(e: &ncl_reader::ReaderError) -> bool {
    matches!(e.kind, ncl_reader::ReaderErrorKind::UnexpectedEof(_))
        || matches!(
            &e.kind,
            ncl_reader::ReaderErrorKind::Lex(ncl_reader::LexErrorKind::UnexpectedEof(_))
        )
}
