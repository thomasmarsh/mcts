//! Move-generator operator ludemes (Language Reference 8.11-8.12): iterate over sites/pieces/
//! players to generate moves, or combine/filter existing move lists.

use crate::ast::boolean::BooleanFunction;
use crate::ast::common::{Between, From, PlayerOrRole, To};
use crate::ast::direction::DirectionFunction;
use crate::ast::located::LBox;
use crate::ast::moves::{Moves, Then};
use crate::ast::numeric::int::IntFunction;
use crate::ast::numeric::int_array::IntArrayFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{SiteType, StackDirection};

/// The name(s) of piece(s) that `(forEach Piece ...)` (8.11.1) applies to.
#[derive(Debug, Clone, PartialEq)]
pub enum PieceNameSpec {
    One(String),
    Many(Vec<String>),
}

/// The target of `(forEach Direction ...)` (8.11.1): either a `to` location descriptor, or a
/// nested move list to apply per direction.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectionTarget {
    To(To),
    Moves(LBox<Moves>),
}

/// The many forms of the `(forEach ...)` (8.11.1) move generator.
#[derive(Debug, Clone, PartialEq)]
pub enum ForEach {
    Level {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        direction: Option<StackDirection>,
        moves: LBox<Moves>,
        then: Option<Then>,
    },
    Team {
        moves: LBox<Moves>,
        then: Option<Then>,
    },
    Group {
        site_type: Option<SiteType>,
        direction: Option<LBox<DirectionFunction>>,
        condition: Option<LBox<BooleanFunction>>,
        moves: LBox<Moves>,
        then: Option<Then>,
    },
    Die {
        index: Option<LBox<IntFunction>>,
        combined: Option<bool>,
        replay_double: Option<bool>,
        condition: Option<LBox<BooleanFunction>>,
        moves: LBox<Moves>,
        then: Option<Then>,
    },
    Direction {
        from: Option<From>,
        direction: Option<LBox<DirectionFunction>>,
        between: Option<Between>,
        target: DirectionTarget,
        then: Option<Then>,
    },
    Site {
        region: LBox<RegionFunction>,
        moves: LBox<Moves>,
        no_move_yet: Option<LBox<Moves>>,
        then: Option<Then>,
    },
    ValueArray {
        array: LBox<IntArrayFunction>,
        moves: LBox<Moves>,
        then: Option<Then>,
    },
    ValueRange {
        min: LBox<IntFunction>,
        max: LBox<IntFunction>,
        moves: LBox<Moves>,
        then: Option<Then>,
    },
    Piece {
        on: Option<SiteType>,
        name: Option<PieceNameSpec>,
        container_index: Option<LBox<IntFunction>>,
        container_name: Option<String>,
        moves: Option<LBox<Moves>>,
        owner: Option<PlayerOrRole>,
        top: Option<bool>,
        then: Option<Then>,
    },
    Player {
        moves: LBox<Moves>,
        then: Option<Then>,
    },
    Players {
        array: LBox<IntArrayFunction>,
        moves: LBox<Moves>,
        then: Option<Then>,
    },
}

/// `(allCombinations ...)` (8.12.1): the cross product of two move lists.
#[derive(Debug, Clone, PartialEq)]
pub struct AllCombinations {
    pub first: LBox<Moves>,
    pub second: LBox<Moves>,
    pub then: Option<Then>,
}

/// `(and ...)` (8.12.2): all moves in the list, if used as a consequence; else a choice among
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct And {
    pub moves: Vec<LBox<Moves>>,
    pub then: Option<Then>,
}

/// `(append ...)` (8.12.3): appends a move list to each move in another list.
#[derive(Debug, Clone, PartialEq)]
pub struct Append {
    pub moves: LBox<Moves>,
    pub then: Option<Then>,
}

/// `(if ...)` (8.12.4): one move list or another, depending on a condition.
#[derive(Debug, Clone, PartialEq)]
pub struct If {
    pub condition: LBox<BooleanFunction>,
    pub if_true: LBox<Moves>,
    pub if_false: Option<LBox<Moves>>,
    pub then: Option<Then>,
}

/// `(or ...)` (8.12.5): a choice among the given move lists.
#[derive(Debug, Clone, PartialEq)]
pub struct Or {
    pub moves: Vec<LBox<Moves>>,
    pub then: Option<Then>,
}

/// Any move-generator operator ludeme.
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    ForEach(Box<ForEach>),
    AllCombinations(AllCombinations),
    And(And),
    Append(Append),
    If(Box<If>),
    Or(Or),
}
