use std::cell::Cell;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::mem::MaybeUninit;
use std::process::ExitCode;
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Mutex;
use std::thread;
#[cfg(not(windows))]
use std::time::Duration;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() {
    eprintln!("usage: ncl [--lean] [--windows] [--repl | (--eval <src> | --load <file> | --check <file>)...] [--repl]");
    eprintln!("       ncl --version | --help");
    eprintln!("  --eval,  -e <src>    evaluate a source string");
    eprintln!("  --load,  -l <file>   read and evaluate the file");
    eprintln!("  --check, -c <file>   dry-run: parse + macroexpand + lower each top-level form,");
    eprintln!("                       executing only definitions (defun, defmacro, defparameter,");
    eprintln!("                       defconstant, require, …). Non-definition forms get lowered");
    eprintln!("                       through the JIT pipeline but never run, so the file's main");
    eprintln!("                       side-effects (FFI calls, network I/O, window creation) are");
    eprintln!("                       suppressed. Surfaces reader, macroexpand, and compile errors.");
    eprintln!("  --repl,  -r          enter the interactive REPL (default if no flags given)");
    eprintln!("  --lean,  -L          start with core only (no CLOS, no Library/init.lisp)");
    eprintln!("  --windows, -W        enable the Windows surface: thread 0 runs a Win32");
    eprintln!("                       message pump, Lisp runs on a worker thread, and");
    eprintln!("                       (windows-enabled-p) is T. Without this flag the");
    eprintln!("                       process is byte-for-byte unchanged from today.");
    eprintln!("  --version, -V        print version and exit");
    eprintln!("  --help,    -h        print this message and exit");
    eprintln!("  multiple --eval / --load / --check can be chained; --repl runs after them");
    eprintln!();
    eprintln!("Environment variables:");
    eprintln!("  NCL_HEAP_BACKEND     pick the GC implementation:");
    eprintln!("                         semispace  (default, production)");
    eprintln!("                         page-heap  (under construction — see docs/GC_DESIGN.md)");
    eprintln!("  NCL_LIBRARY          override the Library/ directory location");
    eprintln!("  NCL_PACK_DIR         override the packs/ directory (Win32 metadata pack)");
    eprintln!("  NCL_YOUNG_MB         young-heap reservation in MB (default 256)");
    eprintln!("  NCL_OLD_MB           old-heap reservation in MB (default 2048)");
    eprintln!("  NCL_STATIC_MB        static-area reservation in MB (default 1024,");
    eprintln!("                       elastic on Windows — only committed as used)");
    eprintln!("  NCL_TLAB_KB          per-mutator TLAB size in KB (default 2048)");
    eprintln!("                       smaller values force GC pressure for testing");
}

fn main() -> ExitCode {
    // Install the Windows last-resort SEH filter before anything that
    // could fault. On non-Windows this is a no-op. Idempotent.
    ncl_runtime::brk::install_crash_handler();

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

    // --windows: thread 0 becomes the Win32 UI thread (message pump),
    // Lisp eval moves to a worker thread. See docs/WINDOWS_FFI.md.
    let want_windows = raw_args.iter().any(|a| a == "--windows" || a == "-W");

    if want_windows {
        run_with_windows_surface(raw_args)
    } else {
        run_without_windows_surface(raw_args)
    }
}

/// Today's startup path. Lisp runs on thread 0; no message pump; no
/// Windows surface. Pulled out of main() unchanged so we can compose
/// it as either thread-0 work (no `--windows`) or worker work
/// (`--windows`).
fn run_without_windows_surface(raw_args: Vec<String>) -> ExitCode {
    lisp_main(raw_args)
}

/// `--windows` path. Thread 0 registers itself as the UI thread,
/// flips the `windows-enabled` flag, spawns the Lisp worker, then
/// runs the Win32 message pump. When the worker finishes, it posts
/// `WM_QUIT` to thread 0; the pump unblocks and we return the
/// worker's exit code.
fn run_with_windows_surface(raw_args: Vec<String>) -> ExitCode {
    use std::sync::mpsc;

    // Flip the flag BEFORE spawning the worker so init.lisp sees it.
    ncl_runtime::win_surface::set_windows_enabled(true);
    ncl_runtime::win_surface::register_ui_thread();
    // Create the hidden HWND_MESSAGE dispatch window before the
    // worker can possibly send the first WM_NCL_EXECUTE message.
    ncl_runtime::win_surface::init_ui_dispatch();
    // Load the Win32 metadata pack (Phase 4). Looks under
    // <exe-dir>/../packs/windows_api.pack first (dev), then
    // <exe-dir>/packs/windows_api.pack (install). Failure is
    // non-fatal: %ffi-call still works directly; only (win32 …) /
    // (defwin32 …) need the pack and they report a clean error
    // when they can't find it.
    if let Some(pack_path) = find_pack_path() {
        ncl_runtime::win_metadata::try_load_pack(&pack_path);
    }

    let (tx, rx) = mpsc::sync_channel::<ExitCode>(1);

    let _worker = match std::thread::Builder::new()
        .name("ncl-lisp-worker".into())
        .spawn(move || {
            let code = lisp_main(raw_args);
            let _ = tx.send(code);
            ncl_runtime::win_surface::post_quit_to_ui_thread();
        }) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("ncl: cannot spawn worker thread: {e}");
            return ExitCode::from(1);
        }
    };

    // Thread 0 takes over the pump. Returns when the worker posts WM_QUIT.
    ncl_runtime::win_surface::run_message_pump();

    // Read back the worker's exit code (sent before the WM_QUIT post,
    // so this never blocks).
    rx.try_recv().unwrap_or(ExitCode::SUCCESS)
}

/// The Lisp side of startup — runs on thread 0 without `--windows`,
/// on the worker thread with `--windows`. Builds the session, loads
/// stdlib + Library/init.lisp, processes `--eval` / `--load` flags,
/// optionally runs the REPL.
fn lisp_main(raw_args: Vec<String>) -> ExitCode {
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

    // Publish the coordinator so thread 0 can register itself as a
    // mutator on the same heap when WM_NCL_EXECUTE arrives. No-op
    // when --windows is off (the OnceLock is set but no one reads
    // it); harmless to do unconditionally.
    ncl_runtime::win_surface::publish_coordinator(session.coord().clone());

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
            "--check" | "-c" => {
                // Dry-run: parse + macroexpand + lower each form,
                // executing only definitions. Non-definition forms
                // pass through the JIT pipeline (so syntax / macro /
                // lowering errors surface) but never run.
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
                match session.check(&src) {
                    Ok(n) => {
                        println!("[CHECK] {path}: OK ({n} forms)");
                        last_output = None;
                    }
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
            "--windows" | "-W" => {
                // Handled in main() before session creation; accept here.
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
        // `--windows` was already consumed by main() to set up the UI
        // thread, but lisp_main still needs to know about it so the
        // REPL can interleave iGui-mailbox draining with stdin reads.
        // Also poll iGui when igui-start was called (sets windows_enabled
        // without --windows on the command line).
        let with_windows = std::env::args().any(|a| a == "--windows" || a == "-W")
            || ncl_runtime::win_surface::windows_enabled();
        return run_repl(&mut session, with_windows);
    }

    ExitCode::SUCCESS
}

/// Resolve the path to `packs/windows_api.pack` for the Windows
/// surface metadata. Search order matches `find_library_dir`:
///   1. NCL_PACK_DIR env override
///   2. <exe_dir>/packs/windows_api.pack  (installed shape)
///   3. <exe_dir>/../../packs/windows_api.pack  (dev build)
fn find_pack_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("NCL_PACK_DIR") {
        let cand = std::path::Path::new(&p).join("windows_api.pack");
        if cand.is_file() { return Some(cand); }
    }
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    let beside = exe_dir.join("packs").join("windows_api.pack");
    if beside.is_file() { return Some(beside); }
    let dev = exe_dir.ancestors().nth(2)
        .map(|p| p.join("packs").join("windows_api.pack"));
    if let Some(d) = dev {
        if d.is_file() { return Some(d); }
    }
    None
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
///
/// When `with_windows` is true, the loop also drains the iGui event
/// mailbox between stdin reads. The motivating case is `:eval-buffer`
/// events fired by ledit's F5/Ctrl+R: those land in the mailbox and
/// would otherwise be ignored unless a Lisp-level `(event-loop-for ...)`
/// happens to be running. With this interleaving, F5 in the editor
/// always evaluates the buffer through the active session, even when
/// the user is just sitting at the REPL prompt — matching the NewFB
/// and NewBCPL "run from the IDE" model.
///
/// The mailbox poll is a 50ms blocking recv; stdin is a `try_recv`
/// against a channel fed by a helper thread. Stdin has priority but
/// each iteration always touches both sources, so the worst-case
/// keystroke latency is ~50ms.
fn run_repl(session: &mut ncl_compiler::Session, with_windows: bool) -> ExitCode {
    install_repl_panic_hook();

    println!("NewCormanLisp {VERSION} REPL");
    println!("  (exit) or Ctrl+D / Ctrl+Z to leave");
    println!();

    let stdin_rx = spawn_stdin_reader();
    let mut buf = String::new();

    'repl: loop {
        // Between prompts, drain any hot-reload pending queue. This
        // is a Lisp-level call; if hot-reload was never enabled,
        // (check-reloads) is a NIL-returning no-op. We swallow any
        // Err so a broken reload doesn't take the REPL down — the
        // Lisp handler-case inside check-reloads handles per-file
        // errors; this is the safety net for the wrapper itself.
        if buf.trim().is_empty() {
            let _ = session.eval("(check-reloads)");
        }
        print_prompt(buf.trim().is_empty());

        // Wait for the next line of input. With `--windows`, between
        // try_recv polls of stdin we also drain the iGui mailbox so
        // F5 in ledit reaches the active session.
        let line_result = loop {
            match stdin_rx.try_recv() {
                Ok(r) => break r,
                Err(TryRecvError::Disconnected) => {
                    // Reader thread died — treat like EOF.
                    println!();
                    break 'repl;
                }
                Err(TryRecvError::Empty) => {
                    if with_windows {
                        #[cfg(windows)]
                        {
                            if let Some(ev) =
                                ncl_runtime::igui::channels::next_event(50)
                            {
                                handle_repl_event(session, ev);
                            }
                        }
                        #[cfg(not(windows))]
                        {
                            thread::sleep(Duration::from_millis(50));
                        }
                    } else {
                        // No iGui mailbox to drain — block on stdin so
                        // we don't busy-wait. recv_timeout(forever) is
                        // recv().
                        match stdin_rx.recv() {
                            Ok(r) => break r,
                            Err(_) => {
                                println!();
                                break 'repl;
                            }
                        }
                    }
                }
            }
        };

        let line = match line_result {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ncl: stdin: {e}");
                return ExitCode::from(1);
            }
        };
        if line.is_empty() {
            // EOF (Ctrl+D / Ctrl+Z).
            println!();
            break;
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
                    // Multi-line continuation: keep buf, prompt with
                    // "...> " next iteration.
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

/// Print "ncl> " for a fresh input or "...> " when continuing a
/// multi-line form. Flush stdout so the prompt actually appears
/// before we block on stdin.
fn print_prompt(fresh: bool) {
    let prompt = if fresh { "ncl> " } else { "...> " };
    print!("{prompt}");
    let _ = io::stdout().flush();
}

/// Spawn a thread that drains stdin line-by-line into a channel.
/// Each item is either `Ok(line)` (with the trailing `\n`) or
/// `Ok("")` to signal EOF; on read error we send `Err(e)` once and
/// exit. Decoupling stdin from the main thread lets the main loop
/// also poll the iGui mailbox.
fn spawn_stdin_reader() -> mpsc::Receiver<io::Result<String>> {
    let (tx, rx) = mpsc::channel::<io::Result<String>>();
    thread::spawn(move || {
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        loop {
            let mut line = String::new();
            match handle.read_line(&mut line) {
                Ok(0) => {
                    // EOF — send empty line as sentinel and exit.
                    let _ = tx.send(Ok(String::new()));
                    break;
                }
                Ok(_) => {
                    if tx.send(Ok(line)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }
    });
    rx
}

/// Handle one iGui event during a REPL idle tick. Today we care
/// about `EvalBuffer` (the editor's F5 / Ctrl+R run-buffer event);
/// every other event kind is dropped, because no Lisp-level
/// `(event-loop-for ...)` is consuming them and there's no other
/// listener to forward to.
///
/// `EvalBuffer` runs the buffer source through `session.eval`. The
/// result lands in the iGui log overlay (Ctrl+Shift+L) so the user
/// gets immediate feedback even though they're focused on the
/// editor pane, not the REPL pane.
#[cfg(windows)]
fn handle_repl_event(
    session: &mut ncl_compiler::Session,
    ev: ncl_runtime::igui::channels::IGuiEvent,
) {
    use ncl_runtime::igui::channels::IGuiEvent;
    use ncl_runtime::igui::log_view;
    match ev {
        IGuiEvent::EvalBuffer { source } => {
            log_view::append(&format!(
                "[F5] evaluating {} chars from editor", source.len()
            ));
            match session.eval(&source) {
                Ok(result) => {
                    let summary = summarize_eval_result(&result);
                    log_view::append(&format!("[F5] => {summary}"));
                }
                Err(e) => {
                    log_view::append(&format!("[F5] error: {e}"));
                }
            }
        }
        _ => {
            // No active listener — drop. Demos that want events
            // start their own (event-loop-for ...) inside the form
            // they run; while that form is on the stack, this REPL
            // poll loop is paused, so the demo gets first crack at
            // every event from the mailbox.
        }
    }
}

/// Condense an eval result to one short line for the log overlay.
/// Multi-line results just get a "(N lines)" tag — the user can
/// re-run from a window with their own clause to see the full text.
#[cfg(windows)]
fn summarize_eval_result(result: &str) -> String {
    let trimmed = result.trim_end();
    if trimmed.is_empty() {
        return "nil".to_string();
    }
    if let Some(idx) = trimmed.find('\n') {
        let head: String = trimmed[..idx].chars().take(60).collect();
        let lines = 1 + trimmed.bytes().filter(|b| *b == b'\n').count();
        return format!("({lines} lines) {head}...");
    }
    if trimmed.chars().count() <= 80 {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(77).collect();
    format!("{head}...")
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
