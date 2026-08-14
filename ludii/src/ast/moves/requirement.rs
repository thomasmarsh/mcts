//! Move requirement ludemes (Language Reference 8.3-8.4): filter a list of generated moves
//! down to those satisfying some criterion. Can be expensive to evaluate, per the reference's
//! own warning.

use crate::ast::boolean::BooleanFunction;
use crate::ast::located::LBox;
use crate::ast::moves::{Moves, Then};
use crate::ast::types::RoleType;

/// `(avoidStoredState ...)` (8.3.1): filters out moves that would reach a previously-seen
/// state.
#[derive(Debug, Clone, PartialEq)]
pub struct AvoidStoredState {
    pub moves: LBox<Moves>,
    pub then: Option<Then>,
}

/// `(do ...)` (8.3.2): applies moves, then optional follow-up moves, filtered by a post-hoc
/// condition.
#[derive(Debug, Clone, PartialEq)]
pub struct Do {
    pub moves: LBox<Moves>,
    pub next: Option<LBox<Moves>>,
    pub if_afterwards: Option<LBox<BooleanFunction>>,
    pub then: Option<Then>,
}

/// `(firstMoveOnTrack ...)` (8.3.3): the first legal move on a track (e.g. Backgammon).
#[derive(Debug, Clone, PartialEq)]
pub struct FirstMoveOnTrack {
    pub track: Option<String>,
    pub owner: Option<RoleType>,
    pub moves: LBox<Moves>,
    pub then: Option<Then>,
}

/// `(priority {...})` (8.3.4): prefers the first move list with any legal moves over the rest
/// (e.g. mandatory captures).
#[derive(Debug, Clone, PartialEq)]
pub struct Priority {
    pub lists: Vec<LBox<Moves>>,
    pub then: Option<Then>,
}

/// `(while ...)` (8.3.5): repeats a move while a condition holds.
#[derive(Debug, Clone, PartialEq)]
pub struct While {
    pub condition: LBox<BooleanFunction>,
    pub moves: LBox<Moves>,
    pub then: Option<Then>,
}

/// `maxMovesType` (8.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxMovesType {
    Moves,
    Captures,
}

/// `(max ...)` (8.4.1): filters a move list down to those maximising some property (number of
/// moves/captures in a turn, or distance travelled).
#[derive(Debug, Clone, PartialEq)]
pub enum Max {
    Property {
        kind: MaxMovesType,
        with_value: Option<bool>,
        moves: LBox<Moves>,
        then: Option<Then>,
    },
    Distance {
        track: Option<String>,
        owner: Option<RoleType>,
        moves: LBox<Moves>,
        then: Option<Then>,
    },
}

/// Any move-requirement ludeme.
#[derive(Debug, Clone, PartialEq)]
pub enum Requirement {
    AvoidStoredState(AvoidStoredState),
    Do(Box<Do>),
    FirstMoveOnTrack(FirstMoveOnTrack),
    Priority(Priority),
    While(While),
    Max(Max),
}
