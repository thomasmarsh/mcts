//! Core IR -> Rust source codegen (`DESIGN.md`'s "backend primops -> Rust source" arrow).
//! [`crate::core::interp`] is deliberately the slow tree-walking evaluator this crate checks new
//! `Region`/`BoolExpr` combinators against; this module is the second, real backend -- it lowers a
//! [`crate::core::Program`] into the text of a Rust source file that stands on its own as an
//! ordinary `games/*` crate (a `Game` impl, zobrist hashing, `Display`), the same shape every
//! hand-written game crate already has (see `games/ttt`).
//!
//! One pass per [`crate::core::Topology`] variant, matching `core::interp`'s own per-topology
//! split (`core::rect`/`core::hex`) and `DESIGN.md`'s "Topology is a type parameter" principle --
//! [`rect`] is the only one implemented so far, since Tic-Tac-Toe is the only corpus game
//! currently routed through codegen rather than the interpreter.
//!
//! Scoped narrowly, per `DESIGN.md`'s "grow from real lowerings" principle: [`rect::generate`]
//! only lowers the `Region`/`BoolExpr` variants Tic-Tac-Toe's own `Program` actually uses
//! (`Occupied`/`Union`/`Complement`/`Sites`, `Contains`/`Any`) and returns [`Error`] on anything
//! else (`Intersect`/`Shift`/`Adjacent`/`Flood`, `Connects`) rather than guessing at a lowering no
//! corpus game has forced yet. [`hex::generate`] is the second backend, forced in by Hex's
//! `Connects` edge-to-edge end rule (`ROADMAP.md` phase 6) -- it lowers the same `Region` shapes
//! as `rect`, plus `BoolExpr::Connects` itself (specialized to `Connectivity::Six`, the only one
//! any corpus Hex-topology game uses, via a generated `hex_connects` helper that calls
//! `bitboard::Board::flood6` directly rather than reproducing
//! `core::interp::bounded_fixpoint`'s general fixpoint loop). Only `HexShape::Rhombus` boards are
//! supported so far -- `HexShape::Triangle` (Y) additionally needs `Region::Intersect` to mask
//! `(sites Empty)` down to the triangular half of the grid, a real next step, not attempted here.

pub mod hex;
pub mod rect;

/// FNV-1a over `s`'s bytes -- just needs to be a stable, well-distributed seed for a generated
/// game's `LazyZobristTable`, not cryptographic; deterministic on the input `struct_name` so
/// regenerating the same game twice produces byte-identical output. Shared by every backend
/// module (not `Region`/`BoolExpr`-shaped, so it doesn't belong to any one topology's own pass).
pub(crate) fn fnv1a(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// A codegen failure: either `program`'s topology has no backend yet, or it uses a `Region`/
/// `BoolExpr` shape this backend doesn't lower yet. Not a panic, matching `style_c::Error`'s own
/// discipline of growing one accepted shape at a time.
#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

/// Generates a standalone Rust source file implementing `program` as a `mcts::game::Game`, named
/// `struct_name` (e.g. `"TicTacToe"`). Dispatches on `program.topology`; `source_path` is only
/// used for the generated file's own provenance comment.
pub fn generate(
    game_name: &str,
    struct_name: &str,
    source_path: &str,
    program: &crate::core::Program,
) -> Result<String, Error> {
    match &program.topology {
        crate::core::Topology::Rect(r) => {
            rect::generate(game_name, struct_name, source_path, *r, program)
        }
        crate::core::Topology::Hex(h) => {
            hex::generate(game_name, struct_name, source_path, *h, program)
        }
    }
}
