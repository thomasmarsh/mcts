//! Standalone 2-player Focus binary that speaks the JSON-line subprocess
//! protocol on stdin/stdout. Built by `cargo build -p game-focus --bin
//! game-focus-2p`; see `game_focus::adapter` for the shared implementation.

fn main() {
    game_focus::adapter::main::<2>();
}
