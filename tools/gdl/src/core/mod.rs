//! Core IR: a small, regular language that a concrete board backend compiles down to (see
//! `DESIGN.md`). [`crate::style_c`] parses a direct s-expression rendering of this module's own
//! types into a [`Program`]; [`interp`] evaluates a `Program` directly against a `Rect`-backed
//! board rather than compiling it, per `DESIGN.md`'s "Interpret Core, don't codegen yet" bootstrap
//! order. There used to be a second frontend here lowering Ludii's own `.lud`/ludeme AST
//! (`crate::ast`/`crate::elaborate`) into a `Program` -- retired per `ROADMAP.md`'s decision to
//! stop loading `.lud` source in code at all; a `.lud` file is read by a person, not this crate.
//!
//! Scoped to exactly what Tic-Tac-Toe/Hex/Y need: two topologies (`Rect`, `Hex`), a region algebra
//! of `union`/`intersect`/`complement`/`member`/a static `Sites` list plus real `shift`/`adjacent`/
//! `flood` combinators (the `member` test itself realized by the backend's bitboard indexing
//! rather than a dedicated IR node), a placement move generator, and a single composable end-rule
//! predicate (`BoolExpr`) built from those combinators -- Tic-Tac-Toe's line-win, Hex's edge-to-edge
//! connectivity-win, and Y's three-edge connectivity-win are three particular `BoolExpr`/`Region`
//! values, not three dedicated Rust variants (see `DESIGN.md`'s "promote to a composable
//! primitive" corollary). Growing this into `DESIGN.md`'s full Core IR (raster ops, `has_cycle`,
//! control combinators, other topologies) is future work, one real game at a time.

pub mod hex;
pub mod interp;
pub mod rect;

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
/// `bitboard::Board::shift_*` directly, so lowering a `Region::Shift`/`adjacent`
/// term to a backend call is the identity mapping. `Hex`'s six-way adjacency uses `Northeast`/
/// `Southwest` for its one diagonal pair (`Northwest`/`Southeast` is the *other* square diagonal,
/// deliberately excluded -- see `bitboard::Board::flood6`'s doc comment); `Rect`'s
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
    /// The sites in both `a` and `b` -- DESIGN.md's Region algebra table's `intersect`. Forced in
    /// by Y's triangular board: legal moves need to be "empty AND inside the triangle," not just
    /// "empty," since a triangle's valid sites are a proper subset of the `side x side` grid its
    /// [`crate::core::hex::Hex::valid_sites`] is carved out of.
    Intersect(Box<Region>, Box<Region>),
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

/// The union of every player's occupied region -- "any site with a piece on it." `pub(crate)`
/// since [`crate::style_c`]'s sexpr frontend reuses it for its own `(sites Empty)` sugar rather
/// than duplicating the fold.
pub(crate) fn all_occupied(num_players: usize) -> Region {
    (1..num_players).fold(Region::Occupied(Player(0)), |acc, i| {
        Region::Union(Box::new(acc), Box::new(Region::Occupied(Player(i))))
    })
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
    /// The region under test contains a single `conn`-connected component that touches every one
    /// of the mover's named regions (`Program.player_regions[mover]`) -- DESIGN.md's
    /// `connects(edge_a, edge_b): Region -> Bool`, generalized from a fixed pair to an arbitrary
    /// list: flood from the first named region and check the result intersects every other one
    /// (see `interp::eval_bool`). Two named regions (Hex's edge-to-edge win) and three (Y's
    /// three-side win, the game that forced this generalization -- a fixed `(Region, Region)`
    /// pair has no third slot) are both just different lengths of the same list, not two
    /// different `BoolExpr` shapes. The regions themselves aren't embedded here the way
    /// `Contains`'s static `sites` is, because `Program.end` is shared across every player while
    /// `player_regions` varies per player -- the interpreter looks the list up per mover, the
    /// same way it looks up which player's occupied region is "the board under test" in the
    /// first place.
    Connects { conn: Connectivity },
    /// True if any of the given expressions is true.
    Any(Vec<BoolExpr>),
}

/// A terminal/end-condition check: the mover wins if `condition` holds against their own occupied
/// region. Per DESIGN.md's "Design principles" corollary, this is one composable predicate built
/// from `Region`/`BoolExpr` combinators rather than a dedicated Rust variant per end-rule shape --
/// the previous `EndRule::Line`/`EndRule::Connected` enum is now just particular `BoolExpr` values
/// (see `crate::style_c`), not special cases `interp::State::winner` has to know about by name.
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
    /// Indexed by player; each player's entry is the list of named regions their
    /// `BoolExpr::Connects` end rule must all be touched by one connected component (two for
    /// Hex's edge-to-edge win, three for Y's three-side win -- see `BoolExpr::Connects`'s doc
    /// comment). Empty when no end rule's `BoolExpr::Connects` references it (e.g. Tic-Tac-Toe).
    pub player_regions: Vec<Vec<Region>>,
}
