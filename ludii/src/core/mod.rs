//! Core IR: a small, regular language that sits between the Ludii [`crate::ast`] and a concrete
//! board backend (see `DESIGN.md`). [`crate::elaborate`] turns `.lud` source into `ast::*`;
//! [`lower`] turns a self-contained `ast::game::Game` into a [`Program`] here; [`interp`]
//! evaluates a `Program` directly against a `Rect`-backed board rather than compiling it, per
//! `DESIGN.md`'s "Interpret Core, don't codegen yet" bootstrap order.
//!
//! Scoped to exactly what Tic-Tac-Toe's and Hex's `.lud` files need: two topologies (`Rect`,
//! `Hex`), a region algebra of `union`/`complement`/`member`/a static `Sites` list (the `member`
//! test itself realized by the backend's bitboard indexing rather than a dedicated IR node), a
//! placement move generator, and two dedicated terminal checks (a static line-win, and an
//! edge-to-edge connectivity check). Growing this into `DESIGN.md`'s full Core IR (raster ops,
//! composable `flood`/`connects` as region-algebra combinators, control combinators, other
//! topologies) is future work, one real game at a time.

pub mod hex;
pub mod interp;
pub mod lower;
pub mod rect;

pub use lower::lower_game;

pub use hex::Hex;
pub use rect::Rect;

/// A player, identified by their position in the game's equipment list -- the first
/// `(piece ...)` in a game's equipment is `Player(0)`, the second is `Player(1)`, and so on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player(pub usize);

/// A board topology. See `DESIGN.md`'s "Topology model" for the full sketch -- only the two
/// variants an actual corpus game needs so far are implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    Rect(Rect),
    Hex(Hex),
}

/// A region-valued Core IR expression: the sites currently occupied by a player, a combination
/// of those via the region algebra's `union`/`complement`, or a fixed, static list of sites (a
/// board edge -- Hex's `(regions P1 {(sites Side NE) ...})`).
#[derive(Debug, Clone, PartialEq)]
pub enum Region {
    Occupied(Player),
    Union(Box<Region>, Box<Region>),
    Complement(Box<Region>),
    Sites(Vec<usize>),
}

/// A move generator: for now, only "add a piece to every site in a region" -- both Tic-Tac-Toe's
/// and Hex's `(move Add (to (sites Empty)))`.
#[derive(Debug, Clone, PartialEq)]
pub struct MoveGen {
    pub to: Region,
}

/// A terminal/end-condition check.
#[derive(Debug, Clone, PartialEq)]
pub enum EndRule {
    /// The mover wins if their own occupied region contains a line of at least `length` sites,
    /// in any of the `Rect` topology's four line directions -- the general form of `(is Line
    /// <length>)` `(result Mover Win)` that a `Rect` topology can decide with static line masks
    /// (no flood/search needed).
    Line { length: usize },
    /// The mover wins if their own occupied region contains a hex-connected (six-way adjacency)
    /// group touching both of the sites in `program.player_regions[mover]` -- the general form of
    /// Hex's `(is Connected Mover)` `(result Mover Win)`, where "the mover's regions" are looked
    /// up from that player's `(regions ...)` equipment.
    Connected,
}

/// A complete Core IR program for a single game: its board topology, player count, move
/// generator, end conditions, and (for games with a `Connected` end rule) each player's pair of
/// named board regions.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub topology: Topology,
    pub num_players: usize,
    pub move_gen: MoveGen,
    pub end: Vec<EndRule>,
    /// Indexed by player. Empty when no `EndRule::Connected` end rule references it (e.g.
    /// Tic-Tac-Toe).
    pub player_regions: Vec<(Region, Region)>,
}
