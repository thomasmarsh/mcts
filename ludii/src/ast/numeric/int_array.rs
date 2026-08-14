//! Integer array functions (Language Reference chapter 11): ludemes returning an array of
//! integers, e.g. for remembered value sets or lists of player indices.

use crate::ast::boolean::BooleanFunction;
use crate::ast::common::SiteOrRegion;
use crate::ast::direction::DirectionFunction;
use crate::ast::located::LBox;
use crate::ast::numeric::int::IntFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{AbsoluteDirection, RoleType, SiteType};

/// How an `(array ...)` (11.1.1) is populated.
#[derive(Debug, Clone, PartialEq)]
pub enum ArraySource {
    Region(LBox<RegionFunction>),
    Ints(Vec<LBox<IntFunction>>),
}

/// The value subtracted by `(difference ...)` (11.3.1): another array, or a single integer.
#[derive(Debug, Clone, PartialEq)]
pub enum DifferenceOperand {
    Array(LBox<IntArrayFunction>),
    Int(LBox<IntFunction>),
}

/// `playersManyType` (11.4.2): sets of players relative to the mover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayersManyType {
    All,
    NonMover,
    Enemy,
}

/// `playersTeamType` (11.4.3): a specific team.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayersTeamType {
    Team1,
    Team2,
    Team3,
    Team4,
    Team5,
    Team6,
    Team7,
    Team8,
    Team9,
    Team10,
    Team11,
    Team12,
    Team13,
    Team14,
    Team15,
    Team16,
}

/// The `(players ...)` (11.4.1) ludeme's two forms.
#[derive(Debug, Clone, PartialEq)]
pub enum PlayersSpec {
    Team {
        team: PlayersTeamType,
        condition: Option<LBox<BooleanFunction>>,
    },
    Many {
        kind: PlayersManyType,
        of: Option<LBox<IntFunction>>,
        condition: Option<LBox<BooleanFunction>>,
    },
}

/// `(sizes Group ...)` (11.5.1): the sizes of connected component groups on the board.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SizesGroup {
    pub site_type: Option<SiteType>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub owner: Option<RoleType>,
    pub of: Option<LBox<IntFunction>>,
    pub condition: Option<LBox<BooleanFunction>>,
    pub min: Option<LBox<IntFunction>>,
}

/// Any ludeme that computes an array of integers.
#[derive(Debug, Clone, PartialEq)]
pub enum IntArrayFunction {
    /// `(array ...)` (11.1.1).
    Array(ArraySource),
    /// `(team)` (11.2.1): the team iterator value.
    Team,
    /// `(difference ...)` (11.3.1).
    Difference {
        array: LBox<IntArrayFunction>,
        subtract: DifferenceOperand,
    },
    /// `(if ...)` (11.3.2).
    If {
        condition: LBox<BooleanFunction>,
        then: LBox<IntArrayFunction>,
        otherwise: Option<LBox<IntArrayFunction>>,
    },
    /// `(intersection ...)` (11.3.3).
    Intersection(Vec<LBox<IntArrayFunction>>),
    /// `(results from:... to:... ...)` (11.3.4): the function value from each "from" site to
    /// each "to" site.
    Results {
        from: SiteOrRegion,
        to: SiteOrRegion,
        function: LBox<IntFunction>,
    },
    /// `(union ...)` (11.3.5).
    Union(Vec<LBox<IntArrayFunction>>),
    /// `(players ...)` (11.4.1).
    Players(PlayersSpec),
    /// `(sizes Group ...)` (11.5.1).
    Sizes(SizesGroup),
    /// `(rotations ...)` (11.6.1).
    Rotations(Vec<AbsoluteDirection>),
    /// `(values Remembered [<string>])` (11.7.1).
    ValuesRemembered(Option<String>),
}
