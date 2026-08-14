//! Move ludemes (Language Reference chapter 8): the largest single chapter in the language,
//! covering the decision a player makes ([`decision`]), the effects that follow from it
//! ([`effect`], [`set`], [`state`]), requirements that filter legal moves ([`requirement`]),
//! and the operators that generate and combine move lists ([`operator`]).
//!
//! We distinguish, as the reference does, between decision moves (an actual player choice),
//! effect moves (applied as a consequence of a decision), and move generators (operators that
//! iterate over playable sites) -- but all three ultimately produce a [`Moves`] value, since
//! they nest into each other freely (e.g. a decision's `then` clause is itself a [`Moves`]).

pub mod decision;
pub mod effect;
pub mod operator;
pub mod requirement;
pub mod set;
pub mod state;

use crate::ast::located::LBox;

/// `(then <nonDecision> [applyAfterAllMoves:<boolean>])` (8.2.31): the moves applied after a
/// move is made, chained onto almost every other move ludeme.
#[derive(Debug, Clone, PartialEq)]
pub struct Then {
    pub moves: LBox<Moves>,
    pub apply_after_all_moves: Option<bool>,
}

/// Any ludeme that produces a move or list of moves: a player decision, an effect, a
/// requirement filter, a state-recording effect, or a generator/combinator operator.
#[derive(Debug, Clone, PartialEq)]
pub enum Moves {
    Decision(Box<decision::Decision>),
    Effect(Box<effect::Effect>),
    Requirement(Box<requirement::Requirement>),
    Set(Box<set::Set>),
    State(Box<state::State>),
    Operator(Box<operator::Operator>),
}
