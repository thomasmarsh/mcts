//! Equipment ludemes that are neither components nor containers (Language Reference 3.7):
//! dominoes sets, deduction-puzzle hints, integer maps and static board regions.

use crate::ast::common::{Hint, Pair};
use crate::ast::located::LBox;
use crate::ast::numeric::int::IntFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{RegionTypeStatic, RoleType, SiteType};

/// `(dominoes upTo:<int>)` (3.7.1): a full dominoes set up to a given highest value.
#[derive(Debug, Clone, PartialEq)]
pub struct Dominoes {
    pub up_to: Option<LBox<IntFunction>>,
}

/// `(hints ...)` (3.7.2): the named collection of hint values for a deduction puzzle.
#[derive(Debug, Clone, PartialEq)]
pub struct Hints {
    pub name: Option<String>,
    pub hints: Vec<Hint>,
    pub site_type: Option<SiteType>,
}

/// `(map ...)` (3.7.3): a named mapping, either between site/role pairs or between plain
/// integers.
#[derive(Debug, Clone, PartialEq)]
pub enum Map {
    Pairs {
        name: Option<String>,
        pairs: Vec<Pair>,
    },
    IntMap {
        name: Option<String>,
        keys: Vec<LBox<IntFunction>>,
        values: Vec<LBox<IntFunction>>,
    },
}

/// How a [`Regions`] equipment item's sites are specified.
#[derive(Debug, Clone, PartialEq)]
pub enum RegionsSpec {
    Sites(Vec<LBox<IntFunction>>),
    Region(LBox<RegionFunction>),
    Regions(Vec<LBox<RegionFunction>>),
    Static(RegionTypeStatic),
    StaticMany(Vec<RegionTypeStatic>),
}

/// `(regions ...)` (3.7.4): a named, static region of the board, e.g. a player's home region
/// or a deduction-puzzle hint region.
#[derive(Debug, Clone, PartialEq)]
pub struct Regions {
    pub name: Option<String>,
    pub owner: Option<RoleType>,
    pub spec: RegionsSpec,
    pub hint_name: Option<String>,
}
