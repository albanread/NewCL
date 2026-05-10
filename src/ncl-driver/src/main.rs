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
        Some(cmd) => {
            eprintln!("ncl: unknown command '{cmd}'");
            eprintln!("(no commands are wired yet — see MANIFESTO.md)");
            ExitCode::from(2)
        }
        None => {
            println!("NewCormanLisp {VERSION} — pre-bootstrap.");
            println!("See MANIFESTO.md for the plan.");
            ExitCode::SUCCESS
        }
    }
}
