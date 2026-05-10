use std::env;
use std::fs;
use std::process::ExitCode;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() {
    eprintln!("usage: ncl [--version | (--eval <src> | --load <file>)... ]");
    eprintln!("  --eval, -e <src>   evaluate a source string");
    eprintln!("  --load, -l <file>  read and evaluate the file");
    eprintln!("  multiple --eval / --load can be chained, in order");
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1).peekable();

    // Bare invocations: print version banner.
    if args.peek().is_none() {
        println!("NewCormanLisp {VERSION}");
        println!("Try: ncl --eval \"(+ 1 2)\"");
        return ExitCode::SUCCESS;
    }
    if matches!(args.peek().map(String::as_str), Some("--version") | Some("-V")) {
        println!("NewCormanLisp {VERSION}");
        return ExitCode::SUCCESS;
    }

    // Single shared session — stdlib loads once, --eval / --load can
    // build on each other across the command line.
    let mut session = match ncl_compiler::Session::with_stdlib() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ncl: stdlib load failed: {e}");
            return ExitCode::from(1);
        }
    };

    let mut last_output: Option<String> = None;

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
            other => {
                eprintln!("ncl: unknown argument '{other}'");
                usage();
                return ExitCode::from(2);
            }
        }
    }

    if let Some(s) = last_output {
        println!("{s}");
    }
    ExitCode::SUCCESS
}
