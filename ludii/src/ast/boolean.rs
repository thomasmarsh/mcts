//! Boolean functions (Language Reference chapter 9): ludemes returning true/false about the
//! current game state. `(is ...)` (9.7.1) is the largest single ludeme in this chapter, with
//! around twenty distinct query forms; [`Is`] gives each its own variant.

use crate::ast::common::{IntOrRole, PlayerOrRole, SiteOrRegion};
use crate::ast::located::LBox;
use crate::ast::moves::Moves;
use crate::ast::numeric::float::FloatFunction;
use crate::ast::numeric::int::IntFunction;
use crate::ast::numeric::int_array::IntArrayFunction;
use crate::ast::numeric::range::RangeFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{
    AbsoluteDirection, HiddenData, PuzzleElementType, RegionTypeStatic, RelationType, RoleType,
    SiteType, Walk,
};

/// The source value converted by `(toBool ...)` (9.1.1).
#[derive(Debug, Clone, PartialEq)]
pub enum ToBoolSource {
    Int(LBox<IntFunction>),
    Float(LBox<FloatFunction>),
}

/// `allSimpleType` (9.2.2): `(all ...)` queries with no parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllSimpleType {
    DiceUsed,
    DiceEqual,
    Passed,
}

/// `allSitesType` (9.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllSitesType {
    Sites,
    Different,
}

/// The many forms of the `(all ...)` "super ludeme" (9.2.1), for querying whether a condition
/// holds across every element of a collection.
#[derive(Debug, Clone, PartialEq)]
pub enum All {
    Groups {
        site_type: Option<SiteType>,
        direction: Option<crate::ast::direction::DirectionFunction>,
        of: Option<LBox<BooleanFunction>>,
        condition: LBox<BooleanFunction>,
    },
    Values {
        array: LBox<IntArrayFunction>,
        condition: LBox<BooleanFunction>,
    },
    Sites {
        kind: AllSitesType,
        region: LBox<RegionFunction>,
        condition: LBox<BooleanFunction>,
    },
    Simple(AllSimpleType),
}

/// One or more site exceptions for `(all Different ...)` (9.5.1).
#[derive(Debug, Clone, PartialEq)]
pub enum DifferentExceptions {
    One(LBox<IntFunction>),
    Many(Vec<LBox<IntFunction>>),
}

/// `(all Different ...)` (9.5.1): a deduction-puzzle constraint that all values in a region
/// are distinct.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PuzzleAllDifferent {
    pub site_type: Option<SiteType>,
    pub region: Option<LBox<RegionFunction>>,
    pub exceptions: Option<DifferentExceptions>,
}

/// `isPuzzleRegionResultType` (9.6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsPuzzleRegionResultType {
    Count,
    Sum,
}

/// The many forms of the deduction-puzzle `(is ...)` ludeme (9.6.1). Distinct from the
/// general-purpose [`Is`] (9.7.1).
#[derive(Debug, Clone, PartialEq)]
pub enum PuzzleIs {
    Solved,
    Unique {
        site_type: Option<SiteType>,
    },
    Region {
        kind: IsPuzzleRegionResultType,
        site_type: Option<SiteType>,
        region: Option<LBox<RegionFunction>>,
        of: Option<LBox<IntFunction>>,
        name: Option<String>,
        result: LBox<IntFunction>,
    },
}

/// `isTreeType` (9.7.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsTreeType {
    Tree,
    SpanningTree,
    CaterpillarTree,
    TreeCentre,
}

/// `isPlayerType` (9.7.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsPlayerType {
    Mover,
    Next,
    Prev,
    Friend,
    Enemy,
    Active,
}

/// `isSimpleType` (9.7.7): `(is ...)` queries with no parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsSimpleType {
    Cycle,
    Pending,
    Full,
}

/// `isStringType` (9.7.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsStringType {
    Proposed,
    Decided,
}

/// `isGraphType` (9.7.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsGraphType {
    LastFrom,
    LastTo,
}

/// `isIntegerType` (9.7.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsIntegerType {
    Odd,
    Even,
    Visited,
    SidesMatch,
    PipsMatch,
    Flat,
    AnyDie,
}

/// `isComponentType` (9.7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsComponentType {
    Threatened,
    Within,
}

/// `isConnectType` (9.7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsConnectType {
    Connected,
    Blocked,
}

/// `isSiteType` (9.7.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsSiteType {
    Empty,
    Occupied,
}

/// The location(s) checked by `(is <isComponentType> ...)` (9.7.1).
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentLocation {
    At(LBox<IntFunction>),
    In(LBox<RegionFunction>),
}

/// The piece(s) making up a line for `(is Line ...)` (9.7.1).
#[derive(Debug, Clone, PartialEq)]
pub enum LineWhat {
    Single(LBox<IntFunction>),
    Many(Vec<LBox<IntFunction>>),
}

/// `(is Line ...)` (9.7.1): whether a line of pieces of a minimum length exists.
#[derive(Debug, Clone, PartialEq)]
pub struct IsLine {
    pub site_type: Option<SiteType>,
    pub min_length: LBox<IntFunction>,
    pub direction: Option<AbsoluteDirection>,
    pub through: Option<LineThrough>,
    pub owner: Option<RoleType>,
    pub what: Option<LineWhat>,
    pub exact: Option<bool>,
    pub contiguous: Option<bool>,
    pub condition: Option<LBox<BooleanFunction>>,
    pub by_level: Option<bool>,
}

/// The `through:`/`throughAny:` constraint of [`IsLine`].
#[derive(Debug, Clone, PartialEq)]
pub enum LineThrough {
    Site(LBox<IntFunction>),
    AnyOf(LBox<RegionFunction>),
}

/// The set of items considered "inside" for `(is Loop ...)` (9.7.1).
#[derive(Debug, Clone, PartialEq)]
pub enum LoopSurround {
    Role(RoleType),
    Roles(Vec<RoleType>),
}

/// `(is Loop ...)` (9.7.1): whether pieces form a closed loop.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IsLoop {
    pub site_type: Option<SiteType>,
    pub surround: Option<LoopSurround>,
    pub direction: Option<crate::ast::direction::DirectionFunction>,
    pub owner: Option<LBox<IntFunction>>,
    pub from: Option<SiteOrRegion>,
    pub path: Option<bool>,
}

/// The configuration checked by `(is Target ...)` (9.7.1).
#[derive(Debug, Clone, PartialEq)]
pub enum TargetContainer {
    Index(LBox<IntFunction>),
    Name(String),
}

/// The disjoint regions connected by `(is <isConnectType> ...)` (9.7.1).
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectRegions {
    Regions(Vec<LBox<RegionFunction>>),
    Role(RoleType),
    Static(RegionTypeStatic),
}

/// The set(s) of sites `(is In ...)` (9.7.1) tests membership against.
#[derive(Debug, Clone, PartialEq)]
pub enum InSet {
    Region(LBox<RegionFunction>),
    Array(LBox<IntArrayFunction>),
}

/// The many forms of the general-purpose `(is ...)` "super ludeme" (9.7.1).
#[derive(Debug, Clone, PartialEq)]
pub enum Is {
    Hidden {
        data: Option<HiddenData>,
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        level: Option<LBox<IntFunction>>,
        to: PlayerOrRole,
    },
    Repeat(Option<crate::ast::types::RepetitionType>),
    Pattern {
        walk: Walk,
        site_type: Option<SiteType>,
        from: Option<LBox<IntFunction>>,
        what: Option<LineWhat>,
    },
    Tree {
        kind: IsTreeType,
        owner: PlayerOrRole,
    },
    RegularGraph {
        owner: PlayerOrRole,
        k: Option<LBox<IntFunction>>,
        odd: Option<bool>,
        even: Option<bool>,
    },
    Player {
        kind: IsPlayerType,
        who: IntOrRole,
    },
    Triggered {
        event: String,
        who: IntOrRole,
    },
    Simple(IsSimpleType),
    Crossing(LBox<IntFunction>, LBox<IntFunction>),
    Str {
        kind: IsStringType,
        value: String,
    },
    Graph {
        kind: IsGraphType,
        site_type: SiteType,
    },
    Integer {
        kind: IsIntegerType,
        value: Option<LBox<IntFunction>>,
    },
    Component {
        kind: IsComponentType,
        piece: Option<LBox<IntFunction>>,
        site_type: Option<SiteType>,
        location: Option<ComponentLocation>,
        moves: Option<LBox<Moves>>,
    },
    Related {
        relation: RelationType,
        site_type: Option<SiteType>,
        first: LBox<IntFunction>,
        second: SiteOrRegion,
    },
    Target {
        container: Option<TargetContainer>,
        configuration: Vec<LBox<IntFunction>>,
        sites: SiteList,
    },
    Connect {
        kind: IsConnectType,
        min_regions: Option<LBox<IntFunction>>,
        site_type: Option<SiteType>,
        at: Option<LBox<IntFunction>>,
        direction: Option<crate::ast::direction::DirectionFunction>,
        regions: ConnectRegions,
    },
    Line(Box<IsLine>),
    Loop(Box<IsLoop>),
    Path {
        site_type: SiteType,
        from: Option<LBox<IntFunction>>,
        owner: Option<PlayerOrRole>,
        length: LBox<RangeFunction>,
        closed: Option<bool>,
    },
    Site {
        kind: IsSiteType,
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
    },
    In {
        sites: SiteList,
        set: InSet,
    },
}

/// A single site, or a list of them.
#[derive(Debug, Clone, PartialEq)]
pub enum SiteList {
    One(LBox<IntFunction>),
    Many(Vec<LBox<IntFunction>>),
}

/// `(no Pieces ...)` (9.9.1): whether a piece type (or all types) have no pieces placed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NoPieces {
    pub site_type: Option<SiteType>,
    pub owner: Option<RoleType>,
    pub of: Option<LBox<IntFunction>>,
    pub name: Option<String>,
    pub in_region: Option<LBox<RegionFunction>>,
}

/// The many forms of the `(no ...)` "super ludeme" (9.9.1).
#[derive(Debug, Clone, PartialEq)]
pub enum No {
    Pieces(NoPieces),
    Moves(RoleType),
}

/// Any ludeme that computes a boolean value.
#[derive(Debug, Clone, PartialEq)]
pub enum BooleanFunction {
    Bool(bool),

    ToBool(ToBoolSource),
    All(Box<All>),
    CanMove(LBox<Moves>),
    ForAll {
        element: PuzzleElementType,
        condition: LBox<BooleanFunction>,
    },
    PuzzleAllDifferent(PuzzleAllDifferent),
    PuzzleIs(Box<PuzzleIs>),
    Is(Box<Is>),

    And(Vec<LBox<BooleanFunction>>),
    Equals(EqualsOperand),
    Ge(LBox<IntFunction>, LBox<IntFunction>),
    Gt(LBox<IntFunction>, LBox<IntFunction>),
    If {
        condition: LBox<BooleanFunction>,
        then: LBox<BooleanFunction>,
        otherwise: Option<LBox<BooleanFunction>>,
    },
    Le(LBox<IntFunction>, LBox<IntFunction>),
    Lt(LBox<IntFunction>, LBox<IntFunction>),
    Not(LBox<BooleanFunction>),
    NotEquals(EqualsOperand),
    Or(Vec<LBox<BooleanFunction>>),
    Xor(LBox<BooleanFunction>, LBox<BooleanFunction>),

    No(Box<No>),
    WasPass,
}

/// The operands of `(= ...)` / `(!= ...)` (9.8.2, 9.8.9): either two int-like values (an
/// `<int>`, or a `<roleType>` compared by index), or two regions.
#[derive(Debug, Clone, PartialEq)]
pub enum EqualsOperand {
    Int(LBox<IntFunction>, IntOrRole),
    Region(LBox<RegionFunction>, LBox<RegionFunction>),
}
