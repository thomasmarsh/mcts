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
//! corpus game has forced yet -- Hex's `Connects` end rule is the next real forcing case, once a
//! second game gets routed through codegen instead of the interpreter.

pub mod rect;

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
        crate::core::Topology::Hex(_) => Err(Error(
            "codegen: Hex topology has no backend yet -- routing a Hex-topology game (Shift/\
             Adjacent/Flood/Connects end rules) through codegen is the forcing case, not \
             attempted yet"
                .into(),
        )),
    }
}
