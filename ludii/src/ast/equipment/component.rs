//! Component equipment ludemes (Language Reference 3.2-3.3): physical pieces of equipment
//! other than boards -- cards, dice, pieces, dominoes and tiles.

use crate::ast::located::LBox;
use crate::ast::moves::Moves;
use crate::ast::numeric::int::IntFunction;
use crate::ast::types::{CardType, CompassDirection, RoleType, Walk};

/// `(flips <int> <int>)` (15.6.2): the pair of local-state values a piece flips between, e.g.
/// an Othello disc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flips {
    pub from: i64,
    pub to: i64,
}

/// `(card ...)` (3.2.1): a standalone card component with its own moves. Distinct from
/// [`crate::ast::common::DeckCard`], the lighter per-suit data nested inside `(deck {...})`.
#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    pub name: String,
    pub owner: RoleType,
    pub card_type: CardType,
    pub rank: LBox<IntFunction>,
    pub value: LBox<IntFunction>,
    pub trump_rank: LBox<IntFunction>,
    pub trump_value: LBox<IntFunction>,
    pub suit: LBox<IntFunction>,
    pub moves: Option<LBox<Moves>>,
    pub max_state: Option<LBox<IntFunction>>,
    pub max_count: Option<LBox<IntFunction>>,
    pub max_value: Option<LBox<IntFunction>>,
}

/// `(die ...)` (3.2.3): a single non-stochastic die used as a piece (turned to show faces,
/// rather than rolled -- see [`crate::ast::equipment::container::Dice`] for rollable dice).
#[derive(Debug, Clone, PartialEq)]
pub struct Die {
    pub name: String,
    pub owner: RoleType,
    pub num_faces: LBox<IntFunction>,
    pub facing: Option<CompassDirection>,
    pub moves: Option<LBox<Moves>>,
}

/// `(piece ...)` (3.2.4): a game piece.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub name: String,
    pub owner: Option<RoleType>,
    pub facing: Option<CompassDirection>,
    pub flips: Option<Flips>,
    pub moves: Option<LBox<Moves>>,
    pub max_state: Option<LBox<IntFunction>>,
    pub max_count: Option<LBox<IntFunction>>,
    pub max_value: Option<LBox<IntFunction>>,
}

/// `(domino ...)` (3.3.1): a single domino component, not automatically part of a
/// [`crate::ast::equipment::other::Dominoes`] set.
#[derive(Debug, Clone, PartialEq)]
pub struct Domino {
    pub name: String,
    pub owner: RoleType,
    pub value: LBox<IntFunction>,
    pub value2: LBox<IntFunction>,
    pub moves: Option<LBox<Moves>>,
}

/// `(path ...)` (3.3.2): one internal connection of a [`Tile`], between two of its sides.
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub from: LBox<IntFunction>,
    pub slots_from: Option<LBox<IntFunction>>,
    pub to: LBox<IntFunction>,
    pub slots_to: Option<LBox<IntFunction>>,
    pub colour: LBox<IntFunction>,
}

/// The outline shape of a [`Tile`]: a single turtle-graphics walk, or several (for tiles made
/// of multiple sub-shapes).
#[derive(Debug, Clone, PartialEq)]
pub enum TileShape {
    Single(Walk),
    Many(Vec<Walk>),
}

/// How many connection slots each side of a [`Tile`] has.
#[derive(Debug, Clone, PartialEq)]
pub enum TileSlots {
    PerSide(LBox<IntFunction>),
    Explicit(Vec<LBox<IntFunction>>),
}

/// `(tile ...)` (3.3.3): a tile component that follows the board's tiling, with internal
/// path connections (e.g. for Trax-like games).
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    pub name: String,
    pub owner: Option<RoleType>,
    pub shape: Option<TileShape>,
    pub num_sides: Option<LBox<IntFunction>>,
    pub slots: Option<TileSlots>,
    pub paths: Vec<Path>,
    pub flips: Option<Flips>,
    pub moves: Option<LBox<Moves>>,
    pub max_state: Option<LBox<IntFunction>>,
    pub max_count: Option<LBox<IntFunction>>,
    pub max_value: Option<LBox<IntFunction>>,
}
