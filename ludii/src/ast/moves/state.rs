//! State-effect move ludemes (Language Reference 8.6-8.10): moves that record information for
//! later use, rather than changing what is on the board.

use crate::ast::common::IntOrRole;
use crate::ast::located::LBox;
use crate::ast::moves::Then;
use crate::ast::numeric::int::IntFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::SiteType;

/// `(addScore ...)` (8.6.1): adds to one or several players' scores.
#[derive(Debug, Clone, PartialEq)]
pub enum AddScore {
    One {
        who: IntOrRole,
        amount: LBox<IntFunction>,
        then: Option<Then>,
    },
    Many {
        who: Vec<IntOrRole>,
        amounts: Vec<LBox<IntFunction>>,
        then: Option<Then>,
    },
}

/// `(forget ...)` (8.7.1): forgets a previously-remembered value (or all of them).
#[derive(Debug, Clone, PartialEq)]
pub enum Forget {
    All {
        name: Option<String>,
        then: Option<Then>,
    },
    Value {
        name: Option<String>,
        value: LBox<IntFunction>,
        then: Option<Then>,
    },
}

/// `(remember ...)` (8.8.1): remembers a value, or the current state, for later use.
#[derive(Debug, Clone, PartialEq)]
pub enum Remember {
    Value {
        name: Option<String>,
        value: LBox<IntFunction>,
        unique: Option<bool>,
        then: Option<Then>,
    },
    State(Option<Then>),
}

/// `(swap ...)` (8.9.1): swaps two pieces, or two players.
#[derive(Debug, Clone, PartialEq)]
pub enum Swap {
    Pieces {
        first: Option<LBox<IntFunction>>,
        second: Option<LBox<IntFunction>>,
        then: Option<Then>,
    },
    Players {
        first: IntOrRole,
        second: IntOrRole,
        then: Option<Then>,
    },
}

/// `(take ...)` (8.10.1): takes a domino, or takes control of another player's pieces.
#[derive(Debug, Clone, PartialEq)]
pub enum Take {
    Domino(Option<Then>),
    Control {
        of: IntOrRole,
        by: IntOrRole,
        at: Option<LBox<IntFunction>>,
        to: Option<LBox<RegionFunction>>,
        site_type: Option<SiteType>,
        then: Option<Then>,
    },
}

/// Any state-recording move effect.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    AddScore(AddScore),
    /// `(moveAgain [<then>])` (8.6.2): the mover takes another move this turn.
    MoveAgain(Option<Then>),
    Forget(Forget),
    Remember(Remember),
    Swap(Swap),
    Take(Take),
}
