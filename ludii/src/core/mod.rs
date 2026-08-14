//! Core IR: a small, regular language that sits between the Ludii [`crate::ast`] and a concrete
//! board backend (see `DESIGN.md`). [`crate::elaborate`] turns `.lud` source into `ast::*`;
//! [`lower`] turns a self-contained `ast::game::Game` into a [`Program`] here; [`interp`]
//! evaluates a `Program` directly against a `Rect`-backed board rather than compiling it, per
//! `DESIGN.md`'s "Interpret Core, don't codegen yet" bootstrap order.
//!
//! Scoped to exactly what Tic-Tac-Toe's and Hex's `.lud` files need: two topologies (`Rect`,
//! `Hex`), a region algebra of `union`/`complement`/`member`/a static `Sites` list plus real
//! `shift`/`adjacent`/`flood` combinators (the `member` test itself realized by the backend's
//! bitboard indexing rather than a dedicated IR node), a placement move generator, and a single
//! composable end-rule predicate (`BoolExpr`) built from those combinators -- Tic-Tac-Toe's
//! line-win and Hex's edge-to-edge connectivity-win are now two particular `BoolExpr` values, not
//! two dedicated Rust variants (see `DESIGN.md`'s "promote to a composable primitive" corollary).
//! Growing this into `DESIGN.md`'s full Core IR (raster ops, `has_cycle`, control combinators,
//! other topologies) is future work, one real game at a time.

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

/// One of a topology's eight queen-move shift directions -- names
/// `game_core::bitboard::BitBoard::shift_*` directly, so lowering a `Region::Shift`/`adjacent`
/// term to a backend call is the identity mapping. `Hex`'s six-way adjacency uses `Northeast`/
/// `Southwest` for its one diagonal pair (`Northwest`/`Southeast` is the *other* square diagonal,
/// deliberately excluded -- see `game_core::bitboard::BitBoard::flood6`'s doc comment); `Rect`'s
/// `Eight` connectivity is the only user of the remaining two variants so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    East,
    South,
    West,
    Northeast,
    Northwest,
    Southeast,
    Southwest,
}

/// One of a topology's adjacency direction sets that a `flood`/`adjacent`/`shift` combinator can
/// use -- DESIGN.md's Region algebra table's `conn: Connectivity` parameter. `Rect` supports
/// `Four` (orthogonal) or `Eight` (queen-move) adjacency; `Hex` has one six-way notion (`Six`).
/// Each variant's direction set exactly mirrors an existing, proven `BitBoard` method
/// (`flood4`/`flood6`/`flood8`) -- see [`interp::adjacent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    Four,
    Six,
    Eight,
}

/// A region-valued Core IR expression: the sites currently occupied by a player, a combination of
/// those via the region algebra's `union`/`complement`, a fixed static list of sites (a board edge
/// -- Hex's `(regions P1 {(sites Side NE) ...})`), or one of the three real Region-algebra
/// combinators DESIGN.md's table lists (`shift`/`adjacent`/`flood`).
#[derive(Debug, Clone, PartialEq)]
pub enum Region {
    Occupied(Player),
    Union(Box<Region>, Box<Region>),
    Complement(Box<Region>),
    Sites(Vec<usize>),
    /// `region` shifted one step in `dir` -- DESIGN.md's `shift(dir): Region -> Region`.
    Shift {
        region: Box<Region>,
        dir: Direction,
    },
    /// The cells adjacent to (but not inside) `region`, under `conn`-adjacency -- DESIGN.md's
    /// `adjacent(conn): Region -> Region`.
    Adjacent {
        region: Box<Region>,
        conn: Connectivity,
    },
    /// The `conn`-connected component(s) of `region` reachable from `seed` -- DESIGN.md's
    /// `flood(seed, conn): Region -> Region`. This is [`interp::bounded_fixpoint`]'s `Aux = ()`
    /// instantiation (a bounded trace over bare `Region` state, per DESIGN.md's "Categorical
    /// structure" section) -- see that function's doc comment for how the same node shape was
    /// checked against `has_cycle`'s richer simultaneous state.
    Flood {
        region: Box<Region>,
        seed: Box<Region>,
        conn: Connectivity,
    },
}

/// A move generator: for now, only "add a piece to every site in a region" -- both Tic-Tac-Toe's
/// and Hex's `(move Add (to (sites Empty)))`.
#[derive(Debug, Clone, PartialEq)]
pub struct MoveGen {
    pub to: Region,
}

/// A Boolean-valued Core IR expression over the region under test -- DESIGN.md's "Design
/// principles" corollary: "an end rule is really 'some Boolean/Region predicate over the board is
/// true'". [`EndRule`] evaluates this against a specific player's occupied region; `mover` itself
/// is a runtime binding the interpreter supplies (see `interp::State::winner`), not a value
/// inside this AST -- the same restriction DESIGN.md's "First-order, not full lambda calculus"
/// non-goal already places on function values applies here to a first-class `Player`/`Region`
/// variable.
#[derive(Debug, Clone, PartialEq)]
pub enum BoolExpr {
    /// `sites` is entirely contained in the region under test -- the general form of `(is Line
    /// <length>)`. Tic-Tac-Toe's end rule lowers to `Any` of one `Contains` per candidate line
    /// (see `core::rect::Rect::lines`).
    Contains(Region),
    /// The region under test contains a `conn`-connected component reachable from the mover's
    /// first named region (`Program.player_regions[mover].0`) that also touches their second
    /// (`.1`) -- DESIGN.md's `connects(edge_a, edge_b): Region -> Bool`, built from
    /// [`Region::Flood`] plus an intersection test (see `interp::eval_bool`). `edge_a`/`edge_b`
    /// aren't embedded here the way `Contains`'s static `sites` is, because `Program.end` is
    /// shared across every player while `player_regions` varies per player -- the interpreter
    /// looks the pair up per mover, the same way it looks up which player's occupied region is
    /// "the board under test" in the first place.
    Connects { conn: Connectivity },
    /// True if any of the given expressions is true.
    Any(Vec<BoolExpr>),
}

/// A terminal/end-condition check: the mover wins if `condition` holds against their own occupied
/// region. Per DESIGN.md's "Design principles" corollary, this is one composable predicate built
/// from `Region`/`BoolExpr` combinators rather than a dedicated Rust variant per end-rule shape --
/// the previous `EndRule::Line`/`EndRule::Connected` enum is now just two particular `BoolExpr`
/// values a `.lud` file can lower to (see `core::lower::lower_end`), not two special cases
/// `interp::State::winner` has to know about by name.
#[derive(Debug, Clone, PartialEq)]
pub struct EndRule {
    pub condition: BoolExpr,
}

/// A complete Core IR program for a single game: its board topology, player count, move
/// generator, end conditions, and (for games with a `BoolExpr::Connects` end rule) each player's
/// pair of named board regions.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub topology: Topology,
    pub num_players: usize,
    pub move_gen: MoveGen,
    pub end: Vec<EndRule>,
    /// Indexed by player. Empty when no end rule's `BoolExpr::Connects` references it (e.g.
    /// Tic-Tac-Toe).
    pub player_regions: Vec<(Region, Region)>,
}
