//! Start rules (Language Reference 7.7-7.12): the initial setup of equipment and state before
//! play begins. `(place ...)` (7.10.1) and `(set ...)` (7.11.1) are each large "super ludemes"
//! with several distinct forms, given their own [`Place`]/[`SetStart`] variants below.

use crate::ast::boolean::BooleanFunction;
use crate::ast::common::SiteOrRegion;
use crate::ast::located::LBox;
use crate::ast::numeric::int::IntFunction;
use crate::ast::numeric::int_array::IntArrayFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{DealableType, HiddenData, RoleType, SiteType};

/// `(deal <dealableType> [<int>])` (7.7.1): deals components between players.
#[derive(Debug, Clone, PartialEq)]
pub struct Deal {
    pub dealable: DealableType,
    pub count: Option<LBox<IntFunction>>,
}

/// `(set [<siteType>] {{<int>}})` (7.8.1): sets deduction-puzzle variables to known values.
#[derive(Debug, Clone, PartialEq)]
pub struct PuzzleSet {
    pub site_type: Option<SiteType>,
    pub entries: Vec<(LBox<IntFunction>, LBox<IntFunction>)>,
}

/// The many forms of the start-rule `(forEach ...)` (7.9.1), for running a starting rule
/// several times while varying a parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ForEachStart {
    Team(StartRule),
    Site {
        region: LBox<RegionFunction>,
        condition: Option<LBox<BooleanFunction>>,
        rule: StartRule,
    },
    Value {
        min: LBox<IntFunction>,
        max: LBox<IntFunction>,
        rule: StartRule,
    },
    Player(StartRule),
    Array {
        array: LBox<IntArrayFunction>,
        rule: StartRule,
    },
}

/// The component(s) placed by `(place Stack ...)` (7.10.1).
#[derive(Debug, Clone, PartialEq)]
pub enum StackItems {
    Single(String),
    Many(Vec<String>),
}

/// The destination(s) of a `(place Stack ...)` (7.10.1).
#[derive(Debug, Clone, PartialEq)]
pub enum StackLocation {
    Site(LBox<IntFunction>),
    Sites(Vec<LBox<IntFunction>>),
    Region(LBox<RegionFunction>),
    Coord(String),
    Coords(Vec<String>),
}

/// `(place ...)` (7.10.1): places item(s) at the start of the game. Placing a single item at
/// one site, several items at several sites, a stack, or randomly, are each distinct forms in
/// the reference grammar.
#[derive(Debug, Clone, PartialEq)]
pub enum Place {
    Site {
        item: String,
        container: Option<String>,
        site_type: Option<SiteType>,
        at: Option<LBox<IntFunction>>,
        coord: Option<String>,
        count: Option<LBox<IntFunction>>,
        state: Option<LBox<IntFunction>>,
        rotation: Option<LBox<IntFunction>>,
        value: Option<LBox<IntFunction>>,
    },
    Sites {
        item: String,
        site_type: Option<SiteType>,
        at: Vec<LBox<IntFunction>>,
        region: Option<LBox<RegionFunction>>,
        coords: Vec<String>,
        counts: Vec<LBox<IntFunction>>,
        state: Option<LBox<IntFunction>>,
        rotation: Option<LBox<IntFunction>>,
        value: Option<LBox<IntFunction>>,
    },
    Stack {
        items: StackItems,
        container: Option<String>,
        site_type: Option<SiteType>,
        location: Option<StackLocation>,
        count: Option<LBox<IntFunction>>,
        counts: Vec<LBox<IntFunction>>,
        state: Option<LBox<IntFunction>>,
        rotation: Option<LBox<IntFunction>>,
        value: Option<LBox<IntFunction>>,
    },
    Random {
        region: Option<LBox<RegionFunction>>,
        items: Vec<String>,
        count: Option<LBox<IntFunction>>,
        state: Option<LBox<IntFunction>>,
        value: Option<LBox<IntFunction>>,
        site_type: Option<SiteType>,
    },
    RandomStack {
        items: Vec<String>,
        counts: Vec<LBox<IntFunction>>,
        state: Option<LBox<IntFunction>>,
        value: Option<LBox<IntFunction>>,
        at: LBox<IntFunction>,
        site_type: Option<SiteType>,
    },
    RandomStackCounts {
        counts: Vec<crate::ast::common::ItemCount>,
        at: LBox<IntFunction>,
        site_type: Option<SiteType>,
    },
}

/// The value remembered by `(set RememberValue ...)` (7.11.1).
#[derive(Debug, Clone, PartialEq)]
pub enum RememberOperand {
    Int(LBox<IntFunction>),
    Region(LBox<RegionFunction>),
}

/// The candidate suit(s) of `(set TrumpSuit ...)` (7.11.1).
#[derive(Debug, Clone, PartialEq)]
pub enum TrumpSuitOperand {
    Int(LBox<IntFunction>),
    Choices(LBox<IntArrayFunction>),
}

/// The location(s) affected by `(set <roleType> ...)` (7.11.1).
#[derive(Debug, Clone, PartialEq)]
pub enum OwnedSites {
    Site {
        at: Option<LBox<IntFunction>>,
        coord: Option<String>,
    },
    Sites {
        at: Vec<LBox<IntFunction>>,
        region: Option<LBox<RegionFunction>>,
        coords: Vec<String>,
    },
}

/// `setStartSitesType` (7.11.3): board-site properties settable in the starting rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetStartSitesType {
    Count,
    Cost,
    Phase,
}

/// The site(s) affected by `(set <setStartSitesType> ...)` (7.11.1).
#[derive(Debug, Clone, PartialEq)]
pub enum SitesTarget {
    At(LBox<IntFunction>),
    To(LBox<RegionFunction>),
}

/// `setStartPlayerType` (7.11.2): player properties settable in the starting rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetStartPlayerType {
    Amount,
    Score,
}

/// The many forms of the start-rule `(set ...)` "super ludeme" (7.11.1).
#[derive(Debug, Clone, PartialEq)]
pub enum SetStart {
    RememberValue {
        name: Option<String>,
        value: RememberOperand,
        unique: Option<bool>,
    },
    Hidden {
        data: Vec<HiddenData>,
        site_type: Option<SiteType>,
        location: SiteOrRegion,
        level: Option<LBox<IntFunction>>,
        value: Option<bool>,
        to: RoleType,
    },
    TrumpSuit(TrumpSuitOperand),
    Owned {
        owner: RoleType,
        site_type: Option<SiteType>,
        sites: OwnedSites,
    },
    SitesProperty {
        kind: SetStartSitesType,
        value: LBox<IntFunction>,
        site_type: Option<SiteType>,
        target: SitesTarget,
    },
    PlayerProperty {
        kind: SetStartPlayerType,
        owner: Option<RoleType>,
        value: LBox<IntFunction>,
    },
    Team {
        index: LBox<IntFunction>,
        members: Vec<RoleType>,
    },
}

/// `(split Deck)` (7.12.1): splits a deck of cards between players. `Deck` is currently the
/// only documented split target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split;

/// Any ludeme describing part of the game's initial setup.
#[derive(Debug, Clone, PartialEq)]
pub enum StartRule {
    Deal(Deal),
    PuzzleSet(PuzzleSet),
    ForEach(Box<ForEachStart>),
    Place(Box<Place>),
    Set(Box<SetStart>),
    Split(Split),
}
