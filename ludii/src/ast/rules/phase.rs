//! Phases (Language Reference 7.5): named sub-divisions of a game, each with its own
//! sub-rules, which a game can move between under specified conditions.

use crate::ast::boolean::BooleanFunction;
use crate::ast::common::PlayerOrRole;
use crate::ast::game::Mode;
use crate::ast::located::LBox;
use crate::ast::rules::end::End;
use crate::ast::rules::play::Play;
use crate::ast::types::RoleType;

/// `(nextPhase ...)` (7.5.1): a condition under which control passes to another (or the next)
/// phase.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NextPhase {
    pub who: Option<PlayerOrRole>,
    pub condition: Option<LBox<BooleanFunction>>,
    pub name: Option<String>,
}

/// `(phase ...)` (7.5.2): a named phase of the game, with its own play/end rules.
#[derive(Debug, Clone, PartialEq)]
pub struct Phase {
    pub name: String,
    pub owner: Option<RoleType>,
    pub mode: Option<Mode>,
    pub play: Play,
    pub end: Option<End>,
    pub next_phases: Vec<NextPhase>,
}
