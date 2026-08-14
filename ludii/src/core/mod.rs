//! Core IR: a small, regular language that sits between the Ludii [`crate::ast`] and a concrete
//! board backend (see `DESIGN.md`). [`crate::elaborate`] turns `.lud` source into `ast::*`;
//! [`lower`] turns a self-contained `ast::game::Game` into a [`Program`] here; [`interp`]
//! evaluates a `Program` directly against a `Rect`-backed board rather than compiling it, per
//! `DESIGN.md`'s "Interpret Core, don't codegen yet" bootstrap order.
//!
//! Scoped deliberately narrowly, to exactly what Tic-Tac-Toe's `.lud` needs: a single `Rect`
//! topology, a region algebra of `union`/`complement`/`member` (the last realized by the
//! backend's bitboard indexing rather than a dedicated IR node), a placement move generator, and
//! a static line-win terminal check. Growing this into `DESIGN.md`'s full Core IR (raster ops,
//! `flood`/`connects`, control combinators, other topologies) is future work, one real game at a
//! time.

pub mod interp;
pub mod lower;
pub mod rect;

pub use lower::lower_game;

pub use rect::Rect;

/// A player, identified by their position in the game's equipment list -- the first
/// `(piece ...)` in a game's equipment is `Player(0)`, the second is `Player(1)`, and so on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player(pub usize);

/// A region-valued Core IR expression: the sites currently occupied by a player, or a
/// combination of those via the region algebra's `union`/`complement`.
#[derive(Debug, Clone, PartialEq)]
pub enum Region {
    Occupied(Player),
    Union(Box<Region>, Box<Region>),
    Complement(Box<Region>),
}

/// A move generator: for now, only "add a piece to every site in a region" -- Tic-Tac-Toe's
/// `(move Add (to (sites Empty)))`.
#[derive(Debug, Clone, PartialEq)]
pub struct MoveGen {
    pub to: Region,
}

/// A terminal/end-condition check: the mover wins if their own occupied region contains a line
/// of at least `length` sites, in any of the `Rect` topology's four line directions -- the
/// general form of `(is Line <length>)` `(result Mover Win)` that a `Rect` topology can decide
/// with static line masks (no flood/search needed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndRule {
    pub line_length: usize,
}

/// A complete Core IR program for a single game: its board topology, player count, move
/// generator, and end conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub topology: Rect,
    pub num_players: usize,
    pub move_gen: MoveGen,
    pub end: Vec<EndRule>,
}
