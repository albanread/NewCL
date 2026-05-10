use std::env;
use std::process::ExitCode;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("--version") | Some("-V") => {
            println!("NewCormanLisp {VERSION}");
            ExitCode::SUCCESS
        }
        Some("--eval") | Some("-e") => {
            let Some(src) = args.next() else {
                eprintln!("ncl: --eval requires a source string");
                eprintln!("usage: ncl --eval \"(+ 1 2)\"");
                return ExitCode::from(2);
            };
            match ncl_compiler::eval_str(&src) {
                Ok(n) => {
                    println!("{n}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("ncl: {e}");
                    ExitCode::from(1)
                }
            }
        }
        Some(cmd) => {
            eprintln!("ncl: unknown command '{cmd}'");
            eprintln!("usage: ncl [--version | --eval <source>]");
            ExitCode::from(2)
        }
        None => {
            println!("NewCormanLisp {VERSION}");
            println!("Try: ncl --eval \"(+ 1 2)\"");
            ExitCode::SUCCESS
        }
    }
}
