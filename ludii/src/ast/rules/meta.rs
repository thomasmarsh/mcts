//! Meta rules (Language Reference 7.3-7.4): higher-level rules applied across an entire game,
//! defined before play and superceding all other rules.

use crate::ast::types::{PassEndType, RepetitionType};

/// `(gravity [PyramidalDrop])` (7.3.2): applies gravity after each move. `PyramidalDrop` is
/// currently the only documented gravity type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GravityType {
    PyramidalDrop,
}

/// `(pin SupportMultiple)` (7.3.5): filters remove-moves for pieces still supporting others.
/// `SupportMultiple` is currently the only documented pin type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinType {
    SupportMultiple,
}

/// `(no ...)` (7.4.1): forbids certain moves across the whole game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoRule {
    Repeat(Option<RepetitionType>),
    Suicide,
}

/// Any single meta rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaRule {
    /// `(automove)` (7.3.1): auto-applies any legal move that is the only one available at its
    /// site.
    Automove,
    Gravity(Option<GravityType>),
    PassEnd(PassEndType),
    Pin(PinType),
    /// `(swap)` (7.3.6): activates the swap (pie) rule.
    Swap,
    No(NoRule),
}

/// `(meta ({<metaRule>} | <metaRule>))` (7.3.3): the metarules of a game.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Meta {
    pub rules: Vec<MetaRule>,
}
