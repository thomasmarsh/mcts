//! Integer functions (Language Reference chapter 10): by far the most common expression type
//! in Ludii rules, returning a single (possibly negative, possibly "undefined") integer value.

use crate::ast::boolean::BooleanFunction;
use crate::ast::common::{IntOrRole, PlayerOrRole, SiteOrRegion};
use crate::ast::direction::DirectionFunction;
use crate::ast::equipment::container::TrackOwner;
use crate::ast::located::LBox;
use crate::ast::numeric::float::FloatFunction;
use crate::ast::numeric::int_array::IntArrayFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{RelationType, RoleType, SiteType, StackDirection, Walk};

/// `(coord ...)` (10.2.5): a site identified by coordinate string, or by row/column indices.
#[derive(Debug, Clone, PartialEq)]
pub enum Coord {
    String(String),
    RowColumn {
        row: LBox<IntFunction>,
        column: LBox<IntFunction>,
    },
}

/// `(id ...)` (10.2.8): identifies a component and/or role by name.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Id {
    pub name: Option<String>,
    pub owner: Option<RoleType>,
}

/// The key of a `(mapEntry ...)` (10.2.10) lookup.
#[derive(Debug, Clone, PartialEq)]
pub enum MapKey {
    Int(LBox<IntFunction>),
    Role(RoleType),
}

/// How a piece is identified for a `(where ...)` (10.3.1) query.
#[derive(Debug, Clone, PartialEq)]
pub enum PieceIdent {
    Name {
        name: String,
        owner: IntOrRole,
        state: Option<LBox<IntFunction>>,
    },
    Index(LBox<IntFunction>),
}

/// `(where ...)` (10.3.1): the site, or stack level, of a piece.
#[derive(Debug, Clone, PartialEq)]
pub enum Where {
    Site {
        piece: PieceIdent,
        site_type: Option<SiteType>,
    },
    Level {
        piece: PieceIdent,
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        from_top: Option<bool>,
    },
}

/// `cardSiteType` (10.4.2): the property of a card that `(card ...)` (10.4.1) returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardSiteType {
    Rank,
    Suit,
    TrumpRank,
    TrumpValue,
}

/// `(card ...)` (10.4.1): a query about the current state of card components.
#[derive(Debug, Clone, PartialEq)]
pub enum Card {
    TrumpSuit,
    Property {
        kind: CardSiteType,
        at: LBox<IntFunction>,
        level: Option<LBox<IntFunction>>,
    },
}

/// `countComponentType` (10.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountComponentType {
    Pieces,
    Pips,
}

/// `countSimpleType` (10.5.3): properties countable with no parameters (beyond site type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountSimpleType {
    Rows,
    Columns,
    Turns,
    Moves,
    Trials,
    MovesThisTurn,
    Phases,
    Vertices,
    Edges,
    Cells,
    Players,
    Active,
    LegalMoves,
}

/// `countSiteType` (10.5.4): properties countable at/around a site or region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountSiteType {
    Sites,
    Adjacent,
    Neighbours,
    Orthogonal,
    Diagonal,
    Off,
}

/// A site or region for a [`Count`] site-based query.
#[derive(Debug, Clone, PartialEq)]
pub enum CountLocation {
    In(LBox<RegionFunction>),
    At(LBox<IntFunction>),
    Container(String),
}

/// `(count ...)` (10.5.1): the many forms of the `Count` "super ludeme".
#[derive(Debug, Clone, PartialEq)]
pub enum Count {
    ValueIn {
        value: LBox<IntFunction>,
        array: LBox<IntArrayFunction>,
    },
    Stack {
        direction: Option<StackDirection>,
        site_type: Option<SiteType>,
        location: SiteOrRegion,
        condition: Option<LBox<BooleanFunction>>,
        stop: Option<LBox<BooleanFunction>>,
    },
    Simple {
        kind: CountSimpleType,
        site_type: Option<SiteType>,
    },
    Site {
        kind: Option<CountSiteType>,
        site_type: Option<SiteType>,
        location: Option<CountLocation>,
    },
    Component {
        kind: CountComponentType,
        site_type: Option<SiteType>,
        owner: Option<RoleType>,
        of: Option<LBox<IntFunction>>,
        name: Option<String>,
        region: Option<LBox<RegionFunction>>,
        condition: Option<LBox<BooleanFunction>>,
    },
    Groups {
        site_type: Option<SiteType>,
        direction: Option<LBox<DirectionFunction>>,
        condition: Option<LBox<BooleanFunction>>,
        min: Option<LBox<IntFunction>>,
    },
    Liberties {
        site_type: Option<SiteType>,
        at: Option<LBox<IntFunction>>,
        direction: Option<LBox<DirectionFunction>>,
        condition: Option<LBox<BooleanFunction>>,
    },
    Steps {
        site_type: Option<SiteType>,
        relation: Option<RelationType>,
        step: Option<Walk>,
        new_rotation: Option<LBox<IntFunction>>,
        from: LBox<IntFunction>,
        to: SiteOrRegion,
    },
    StepsOnTrack {
        owner: Option<TrackOwner>,
        from: Option<LBox<IntFunction>>,
        to: Option<LBox<IntFunction>>,
    },
}

/// `lastType` (10.8.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastType {
    From,
    LevelFrom,
    To,
    LevelTo,
}

/// `(trackSite ...)` (10.15.1): a site reached by moving along a track.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackSite {
    FirstSite {
        owner: Option<PlayerOrRole>,
        track: Option<String>,
        from: Option<LBox<IntFunction>>,
        condition: Option<LBox<BooleanFunction>>,
    },
    EndSite {
        owner: Option<PlayerOrRole>,
        track: Option<String>,
    },
    Move {
        from: Option<LBox<IntFunction>>,
        owner: Option<TrackOwner>,
        steps: LBox<IntFunction>,
    },
}

/// `valueSimpleType` (10.16.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSimpleType {
    Pending,
    MoveLimit,
    TurnLimit,
}

/// `(value ...)` (10.16.1): the many forms of the `Value` "super ludeme".
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Random(LBox<crate::ast::numeric::range::RangeFunction>),
    Simple(ValueSimpleType),
    Player(IntOrRole),
    Piece {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        level: Option<LBox<IntFunction>>,
    },
    /// The value currently being iterated by an enclosing `(forEach Value ...)`.
    Iterated,
}

/// Any ludeme that computes an integer value.
#[derive(Debug, Clone, PartialEq)]
pub enum IntFunction {
    Int(i64),

    /// `(toInt ...)` (10.1.1).
    ToInt(ToIntSource),

    // -- Board (10.2) --
    Ahead {
        site_type: Option<SiteType>,
        from: LBox<IntFunction>,
        steps: Option<LBox<IntFunction>>,
        direction: Option<LBox<DirectionFunction>>,
    },
    ArrayValue {
        array: LBox<IntArrayFunction>,
        index: LBox<IntFunction>,
    },
    CentrePoint {
        site_type: Option<SiteType>,
    },
    Column {
        site_type: Option<SiteType>,
        of: LBox<IntFunction>,
    },
    Coord(Coord),
    Cost {
        site_type: Option<SiteType>,
        location: SiteOrRegion,
    },
    HandSite {
        owner: IntOrRole,
        site: Option<LBox<IntFunction>>,
    },
    Id(Id),
    Layer {
        of: LBox<IntFunction>,
        site_type: Option<SiteType>,
    },
    MapEntry {
        name: Option<String>,
        key: MapKey,
    },
    Phase {
        site_type: Option<SiteType>,
        of: LBox<IntFunction>,
    },
    RegionSite {
        region: LBox<RegionFunction>,
        index: LBox<IntFunction>,
    },
    Row {
        site_type: Option<SiteType>,
        of: LBox<IntFunction>,
    },

    // -- Board - Where (10.3) --
    Where(Where),

    // -- Card (10.4) --
    Card(Card),

    // -- Count (10.5) --
    Count(Box<Count>),

    // -- Dice (10.6) --
    Face(LBox<IntFunction>),

    // -- Iterator (10.7): context values used while generating/applying moves --
    Between,
    /// `(edge [<int> <int>])` (10.7.2): an edge index (explicit), or the context's current
    /// edge value.
    Edge(Option<(LBox<IntFunction>, LBox<IntFunction>)>),
    From {
        at: Option<crate::ast::types::WhenType>,
    },
    Hint {
        site_type: Option<SiteType>,
        at: Option<LBox<IntFunction>>,
    },
    Level,
    Pips,
    PlayerContext,
    Site,
    To,
    Track,

    // -- Last (10.8) --
    Last {
        kind: LastType,
        after_consequence: Option<bool>,
    },

    // -- Match (10.9) --
    MatchScore(RoleType),

    // -- Math (10.10) --
    Abs(LBox<IntFunction>),
    Add(Vec<LBox<IntFunction>>),
    Div(LBox<IntFunction>, LBox<IntFunction>),
    If {
        condition: LBox<BooleanFunction>,
        then: LBox<IntFunction>,
        otherwise: LBox<IntFunction>,
    },
    Max(MaxMinOperand),
    Min(MaxMinOperand),
    Mod(LBox<IntFunction>, LBox<IntFunction>),
    Mul(Vec<LBox<IntFunction>>),
    Pow(LBox<IntFunction>, LBox<IntFunction>),
    Sub(Option<LBox<IntFunction>>, LBox<IntFunction>),

    // -- Size (10.11) --
    SizeArray(LBox<IntArrayFunction>),
    SizeStack {
        site_type: Option<SiteType>,
        location: Option<SiteOrRegion>,
    },
    SizeLargePiece {
        site_type: Option<SiteType>,
        location: Option<SiteOrRegion>,
    },
    SizeGroup {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        direction: Option<LBox<DirectionFunction>>,
        condition: Option<LBox<BooleanFunction>>,
    },
    SizeTerritory {
        site_type: Option<SiteType>,
        owner: IntOrRole,
        direction: Option<LBox<DirectionFunction>>,
    },

    // -- Stacking (10.12) --
    TopLevel {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
    },

    // -- State (10.13) --
    Amount(IntOrRole),
    Counter,
    Mover,
    Next,
    Pot,
    Prev(Option<crate::ast::types::PrevType>),
    Rotation {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        level: Option<LBox<IntFunction>>,
    },
    Score(IntOrRole),
    State {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        level: Option<LBox<IntFunction>>,
    },
    Var(Option<String>),
    What {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        level: Option<LBox<IntFunction>>,
    },
    Who {
        site_type: Option<SiteType>,
        at: LBox<IntFunction>,
        level: Option<LBox<IntFunction>>,
    },

    // -- Tile (10.14) --
    PathExtent {
        colour: Option<LBox<IntFunction>>,
        from: Option<SiteOrRegion>,
    },

    // -- TrackSite (10.15) --
    TrackSite(Box<TrackSite>),

    // -- Value (10.16) --
    Value(Box<Value>),
}

/// The source value converted by `(toInt ...)` (10.1.1).
#[derive(Debug, Clone, PartialEq)]
pub enum ToIntSource {
    Bool(LBox<BooleanFunction>),
    Float(LBox<FloatFunction>),
}

/// The operand of `(max ...)` / `(min ...)` (10.10.5-6): either two values, or an array.
#[derive(Debug, Clone, PartialEq)]
pub enum MaxMinOperand {
    Pair(LBox<IntFunction>, LBox<IntFunction>),
    Array(LBox<IntArrayFunction>),
}
