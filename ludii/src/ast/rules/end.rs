//! End rules (Language Reference 7.2): terminating conditions and results.

use crate::ast::boolean::BooleanFunction;
use crate::ast::common::{Payoff, ScoreEntry};
use crate::ast::located::LBox;
use crate::ast::types::{ResultType, RoleType};

/// `(result <roleType> <resultType>)` (7.2.6): the outcome for a player/team when an end rule
/// is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Result {
    pub role: RoleType,
    pub result: ResultType,
}

/// Who `(forEach ...)` (7.2.3) applies an end condition to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForEachEndRole {
    Role(RoleType),
    Track,
}

/// `(if ...)` (7.2.4): the condition(s) for ending the game and deciding its result.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct If {
    pub condition: Option<LBox<BooleanFunction>>,
    pub subconditions: Vec<If>,
    pub result: Option<Result>,
}

/// Any single ending rule.
#[derive(Debug, Clone, PartialEq)]
pub enum EndRule {
    /// `(byScore [{<score>}])` (7.2.1): ends the game by comparing player scores.
    ByScore(Vec<ScoreEntry>),
    ForEach {
        role: ForEachEndRole,
        condition: LBox<BooleanFunction>,
        result: Result,
    },
    If(Box<If>),
    /// `(payoffs {<payoff>})` (7.2.5): ends the game with a floating-point payoff per player.
    Payoffs(Vec<Payoff>),
    Result(Result),
}

/// `(end (<endRule> | {<endRule>}))` (7.2.2): the ending rules of a game or phase.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct End {
    pub rules: Vec<EndRule>,
}
