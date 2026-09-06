//! Offline Core-IR-to-Rust codegen driver: generation is a deliberate offline step whose output is
//! checked in, not a `build.rs`/proc-macro step -- reviewable and debuggable the same way every
//! other crate in this workspace is source-controlled. Parses a `style-c/sexpr/*.gdls` file through
//! `gdl::style_c::parse_game`, lowers the resulting `Program` through `gdl::codegen::generate`,
//! and prints the generated Rust source to stdout -- redirect it into a checked-in
//! `games/*-gen/src/lib.rs`, the same way `games/ttt-gen/src/lib.rs` was produced:
//!
//! ```text
//! cargo run -p gdl --bin codegen -- style-c/sexpr/tic-tac-toe.gdls TicTacToe "Tic-Tac-Toe" \
//!   > games/ttt-gen/src/lib.rs
//! ```

use std::io::Write;
use std::process::{Command, ExitCode, Stdio};

/// Pipes `source` through `rustfmt` -- the template in `gdl::codegen::rect` builds its method
/// bodies as single long `format!`-joined lines (`region_expr`/`bool_expr` don't themselves know
/// column widths), so without this the checked-in output would fail this repo's own `cargo fmt
/// --check`. Falls back to the unformatted source (rather than failing the whole run) if
/// `rustfmt` isn't on `PATH`, so this stays usable in an environment that only has `cargo`.
fn rustfmt(source: &str) -> String {
    let Ok(mut child) = Command::new("rustfmt")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
    else {
        eprintln!("warning: rustfmt not found on PATH -- emitting unformatted source");
        return source.to_string();
    };
    // `rustfmt` reading a large stdin while also being read from can deadlock if done
    // sequentially on one thread -- write from a second thread, same as `std::process::Command`'s
    // own docs recommend for a piped-both-ways child.
    let mut stdin = child.stdin.take().expect("piped stdin");
    let source_owned = source.to_string();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(source_owned.as_bytes());
    });
    let output = child.wait_with_output().expect("rustfmt should run");
    writer.join().expect("stdin-writer thread should not panic");
    if output.status.success() {
        String::from_utf8(output.stdout).expect("rustfmt output should be UTF-8")
    } else {
        eprintln!("warning: rustfmt failed -- emitting unformatted source");
        source.to_string()
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let [_, sexpr_path, struct_name, game_name] = args.as_slice() else {
        eprintln!(
            "usage: codegen <path/to/game.gdls> <StructName> <\"Game Name\">\n\n\
             Writes the generated Rust source to stdout."
        );
        return ExitCode::FAILURE;
    };

    let source = match std::fs::read_to_string(sexpr_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading {sexpr_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let program = match gdl::style_c::parse_game(&source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error parsing {sexpr_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    match gdl::codegen::generate(game_name, struct_name, sexpr_path, &program) {
        Ok(rust_source) => {
            print!("{}", rustfmt(&rust_source));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error generating code for {sexpr_path}: {e}");
            ExitCode::FAILURE
        }
    }
}
