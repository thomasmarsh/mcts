//! `game-connect4` binary. Currently only the `dump` subcommand, which
//! writes self-play records for an offline value-net training loop; see
//! `game_connect4::dump`. Connect Four is not yet wired into `game-host`'s
//! JSON-line protocol, so there is no other subcommand.

use std::env;

fn main() {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("dump") => game_connect4::dump::run(args),
        other => {
            eprintln!(
                "game-connect4: only the `dump` subcommand is supported (got {other:?})"
            );
            std::process::exit(2);
        }
    }
}
