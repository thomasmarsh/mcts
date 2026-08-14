//! Rule ludemes (Language Reference chapter 7): how a game is played, from initial setup
//! ([`start`]) through the moves of play ([`play`]) to the conditions that end it ([`end`]),
//! with [`meta`]rules and [`phase`]s layered on top.

pub mod end;
pub mod meta;
pub mod phase;
pub mod play;
pub mod start;

use end::End;
use meta::Meta;
use phase::Phase;
use play::Play;
use start::StartRule;

/// `(rules ...)` (7.1.1): the complete rules of a game, either as a single flat `play`/`end`
/// pair, or subdivided into named [`Phase`]s.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Rules {
    pub meta: Option<Meta>,
    pub start: Vec<StartRule>,
    pub play: Option<Play>,
    pub phases: Vec<Phase>,
    pub end: Option<End>,
}
