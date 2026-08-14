//! The play rule (Language Reference 7.6): the moves available at each point in the game.

use crate::ast::located::LBox;
use crate::ast::moves::Moves;

/// `(play <moves>)` (7.6.1): the legal-move generator for a game or phase.
#[derive(Debug, Clone, PartialEq)]
pub struct Play(pub LBox<Moves>);
