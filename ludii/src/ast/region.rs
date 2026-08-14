//! Region functions (Language Reference chapter 12): ludemes returning a region -- a
//! collection of sites, static (e.g. player homes) or dynamic (e.g. currently-empty sites).
//! The `(sites ...)` ludeme (12.4.2) is the largest single ludeme in the language, with around
//! two dozen distinct forms; [`Sites`] gives each its own variant.

use crate::ast::boolean::BooleanFunction;
use crate::ast::common::{Piece, PlayerOrRole, SiteOrRegion};
use crate::ast::located::LBox;
use crate::ast::moves::Moves;
use crate::ast::numeric::int::IntFunction;
use crate::ast::numeric::int_array::IntArrayFunction;
use crate::ast::numeric::range::RangeFunction;
use crate::ast::types::{
    AbsoluteDirection, CompassDirection, HiddenData, RegionTypeDynamic, RelationType, RoleType,
    SiteType, StackDirection, Walk,
};

/// `(last Between)` (12.2.1): sites between the "from" and "to" of the last move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LastBetween;

/// The value subtracted by `(difference <region> ...)` (12.3.1): another region, or a site.
#[derive(Debug, Clone, PartialEq)]
pub enum RegionDifferenceOperand {
    Region(LBox<RegionFunction>),
    Site(LBox<IntFunction>),
}

/// A container referenced by index or by name.
#[derive(Debug, Clone, PartialEq)]
pub enum ContainerRef {
    Index(LBox<IntFunction>),
    Name(String),
}

/// A component referenced by index, single name, or list of names.
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentRef {
    Index(LBox<IntFunction>),
    Name(String),
    Names(Vec<String>),
}

/// `(expand ...)` (12.3.2): expands a region/site outward by a number of steps.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Expand {
    pub container: Option<ContainerRef>,
    pub region: Option<LBox<RegionFunction>>,
    pub origin: Option<LBox<IntFunction>>,
    pub steps: Option<LBox<IntFunction>>,
    pub direction: Option<AbsoluteDirection>,
    pub site_type: Option<SiteType>,
}

/// `lineOfSightType` (12.4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineOfSightType {
    Empty,
    Farthest,
    Piece,
}

/// `sitesEdgeType` (12.4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SitesEdgeType {
    Axial,
    Horizontal,
    Vertical,
    Angled,
    Slash,
    Slosh,
}

/// `sitesIndexType` (12.4.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SitesIndexType {
    Row,
    Column,
    Phase,
    Cell,
    Edge,
    State,
    Empty,
    Layer,
}

/// `sitesMoveType` (12.4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SitesMoveType {
    From,
    Between,
    To,
}

/// `sitesPlayerType` (12.4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SitesPlayerType {
    Hand,
    Winning,
}

/// `sitesSimpleType` (12.4.7): sites requiring no parameters beyond the graph element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SitesSimpleType {
    Board,
    Top,
    Bottom,
    Left,
    Right,
    Inner,
    Outer,
    Perimeter,
    Corners,
    ConcaveCorners,
    ConvexCorners,
    Major,
    Minor,
    Centre,
    Hint,
    ToClear,
    LineOfPlay,
    Pending,
    Playable,
    LastTo,
    LastFrom,
}

/// Who is considered "inside" a `(sites Loop ...)`.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopSurround {
    Role(RoleType),
    Roles(Vec<RoleType>),
}

/// The piece(s) a `(sites Pattern ...)` must match.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternWhat {
    Single(LBox<IntFunction>),
    Many(Vec<LBox<IntFunction>>),
}

/// The origin of a `(sites Group ...)`.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupFrom {
    At(LBox<IntFunction>),
    Region(LBox<RegionFunction>),
}

/// Explicit site lists, as either raw indices or an [`IntArrayFunction`].
#[derive(Debug, Clone, PartialEq)]
pub enum SiteList {
    Sites(Vec<LBox<IntFunction>>),
    Array(LBox<IntArrayFunction>),
}

/// The target of a `(sites Side ...)` query.
#[derive(Debug, Clone, PartialEq)]
pub enum SideTarget {
    Player(LBox<IntFunction>),
    Role(RoleType),
    Compass(CompassDirection),
}

/// The many forms of the `(sites ...)` "super ludeme" (12.4.2), by far the largest single
/// ludeme in the language.
#[derive(Debug, Clone, PartialEq)]
pub enum Sites {
    /// `(sites)`: the sites iterated by an enclosing move generator.
    Current,
    Loop {
        inside: Option<bool>,
        site_type: Option<SiteType>,
        surround: Option<LoopSurround>,
        direction: Option<AbsoluteDirection>,
        owner: Option<LBox<IntFunction>>,
        from: Option<SiteOrRegion>,
    },
    Pattern {
        walk: Walk,
        site_type: Option<SiteType>,
        from: Option<LBox<IntFunction>>,
        what: Option<PatternWhat>,
    },
    Hidden {
        data: Option<HiddenData>,
        site_type: Option<SiteType>,
        to: PlayerOrRole,
    },
    Between {
        direction: Option<AbsoluteDirection>,
        site_type: Option<SiteType>,
        from: LBox<IntFunction>,
        from_included: Option<bool>,
        to: LBox<IntFunction>,
        to_included: Option<bool>,
        condition: Option<LBox<BooleanFunction>>,
    },
    LargePiece {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
    },
    Random {
        region: Option<LBox<RegionFunction>>,
        num: Option<LBox<IntFunction>>,
    },
    Crossing {
        at: LBox<IntFunction>,
        owner: Option<PlayerOrRole>,
    },
    Group {
        site_type: Option<SiteType>,
        from: GroupFrom,
        direction: Option<AbsoluteDirection>,
        condition: Option<LBox<BooleanFunction>>,
    },
    Edge(SitesEdgeType),
    Simple {
        kind: SitesSimpleType,
        site_type: Option<SiteType>,
    },
    Coordinates {
        site_type: Option<SiteType>,
        coords: Vec<String>,
    },
    FromMoves {
        kind: SitesMoveType,
        moves: LBox<Moves>,
    },
    Ints(SiteList),
    Walk {
        site_type: Option<SiteType>,
        from: Option<LBox<IntFunction>>,
        walks: Vec<Walk>,
        rotations: Option<bool>,
    },
    Index {
        kind: SitesIndexType,
        site_type: Option<SiteType>,
        index: Option<LBox<IntFunction>>,
    },
    Side {
        site_type: Option<SiteType>,
        target: Option<SideTarget>,
    },
    Distance {
        site_type: Option<SiteType>,
        relation: Option<RelationType>,
        step: Option<Walk>,
        new_rotation: Option<LBox<IntFunction>>,
        from: LBox<IntFunction>,
        range: LBox<RangeFunction>,
    },
    OfPlayer {
        owner: Option<PlayerOrRole>,
        site_type: Option<SiteType>,
        name: Option<String>,
    },
    Track {
        owner: Option<PlayerOrRole>,
        name: Option<String>,
        from: Option<LBox<IntFunction>>,
        to: Option<LBox<IntFunction>>,
    },
    PlayerRelated {
        kind: SitesPlayerType,
        site_type: Option<SiteType>,
        owner: Option<PlayerOrRole>,
        rules: Option<LBox<Moves>>,
        name: Option<String>,
    },
    Start(Piece),
    Occupied {
        by: PlayerOrRole,
        container: Option<ContainerRef>,
        component: Option<ComponentRef>,
        top: Option<bool>,
        on: Option<SiteType>,
    },
    Incident {
        result_type: SiteType,
        of_type: SiteType,
        at: LBox<IntFunction>,
        owner: Option<PlayerOrRole>,
    },
    Around {
        site_type: Option<SiteType>,
        from: SiteOrRegion,
        dynamic: Option<RegionTypeDynamic>,
        distance: Option<LBox<IntFunction>>,
        direction: Option<AbsoluteDirection>,
        condition: Option<LBox<BooleanFunction>>,
        include_self: Option<bool>,
    },
    Direction {
        from: SiteOrRegion,
        direction: Option<AbsoluteDirection>,
        included: Option<bool>,
        stop: Option<LBox<BooleanFunction>>,
        stop_included: Option<bool>,
        distance: Option<LBox<IntFunction>>,
        site_type: Option<SiteType>,
    },
    LineOfSight {
        kind: Option<LineOfSightType>,
        site_type: Option<SiteType>,
        at: Option<LBox<IntFunction>>,
        direction: Option<AbsoluteDirection>,
    },
}

/// Any ludeme that computes a region (a collection of sites).
#[derive(Debug, Clone, PartialEq)]
pub enum RegionFunction {
    ForEachLevel {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        direction: Option<StackDirection>,
        condition: Option<LBox<BooleanFunction>>,
        start_at: Option<LBox<IntFunction>>,
    },
    ForEachTeam(LBox<RegionFunction>),
    ForEachFilter {
        region: LBox<RegionFunction>,
        condition: LBox<BooleanFunction>,
    },
    ForEachOf {
        of: LBox<RegionFunction>,
        region: LBox<RegionFunction>,
    },
    ForEachPlayers {
        players: LBox<IntArrayFunction>,
        region: LBox<RegionFunction>,
    },
    LastBetween(LastBetween),
    Difference {
        region: LBox<RegionFunction>,
        subtract: RegionDifferenceOperand,
    },
    Expand(Box<Expand>),
    If {
        condition: LBox<BooleanFunction>,
        then: LBox<RegionFunction>,
        otherwise: Option<LBox<RegionFunction>>,
    },
    Intersection(Vec<LBox<RegionFunction>>),
    Union(Vec<LBox<RegionFunction>>),
    Sites(Box<Sites>),
}
