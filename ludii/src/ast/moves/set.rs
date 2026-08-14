//! The effect-level `(set ...)` "super ludeme" (Language Reference 8.5): sets some aspect of
//! the game state in response to a move -- a counter, the next player, a site's local state,
//! and so on. Ten distinct forms in the reference grammar; [`Set`] gives each its own variant.

use crate::ast::common::{IntOrRole, PlayerOrRole, SiteOrRegion};
use crate::ast::located::LBox;
use crate::ast::moves::decision::{NextPlayerChoice, RotationChoice, TrumpSuitChoice};
use crate::ast::moves::Then;
use crate::ast::numeric::int::IntFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{HiddenData, RoleType, SiteType};

/// `setPlayerType` (8.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetPlayerType {
    Value,
    Score,
}

/// `setSiteType` (8.5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetSiteType {
    Count,
    State,
    Value,
}

/// `setValueType` (8.5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetValueType {
    Counter,
    Pot,
}

/// Either a single site or a whole region, for `(set Pending ...)`.
#[derive(Debug, Clone, PartialEq)]
pub enum PendingTarget {
    Site(LBox<IntFunction>),
    Region(LBox<RegionFunction>),
}

/// Any `(set ...)` (8.5.1) effect.
#[derive(Debug, Clone, PartialEq)]
pub enum Set {
    Team {
        index: LBox<IntFunction>,
        members: Vec<RoleType>,
        then: Option<Then>,
    },
    Hidden {
        data: Vec<HiddenData>,
        site_type: Option<SiteType>,
        location: SiteOrRegion,
        level: Option<LBox<IntFunction>>,
        value: Option<bool>,
        to: PlayerOrRole,
        then: Option<Then>,
    },
    TrumpSuit {
        choice: TrumpSuitChoice,
        then: Option<Then>,
    },
    NextPlayer {
        next: NextPlayerChoice,
        then: Option<Then>,
    },
    Rotation {
        to: Option<crate::ast::common::To>,
        directions: Option<RotationChoice>,
        previous: Option<bool>,
        next: Option<bool>,
        then: Option<Then>,
    },
    PlayerProperty {
        kind: SetPlayerType,
        who: IntOrRole,
        value: LBox<IntFunction>,
        then: Option<Then>,
    },
    Pending {
        target: Option<PendingTarget>,
        then: Option<Then>,
    },
    Var {
        name: Option<String>,
        value: Option<LBox<IntFunction>>,
        then: Option<Then>,
    },
    ValueType {
        kind: SetValueType,
        value: Option<LBox<IntFunction>>,
        then: Option<Then>,
    },
    SiteProperty {
        kind: SetSiteType,
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        level: Option<LBox<IntFunction>>,
        value: LBox<IntFunction>,
        then: Option<Then>,
    },
}
