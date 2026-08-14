//! Small utility ludemes shared across many other chapters (Language Reference chapter 15,
//! minus the direction/turtle-step vocabulary which lives in [`crate::ast::types`]).
//!
//! [`From`], [`To`], [`Between`] and [`Piece`] in particular are the location-descriptor
//! ludemes threaded through almost every ludeme in [`crate::ast::moves`].

use crate::ast::located::LBox;
use crate::ast::moves::effect::Apply;
use crate::ast::numeric::float::FloatFunction;
use crate::ast::numeric::int::IntFunction;
use crate::ast::numeric::int_array::IntArrayFunction;
use crate::ast::numeric::range::RangeFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{LandmarkType, RoleType, SiteType};

/// A single site index, or a region of them; many location descriptors accept either.
#[derive(Debug, Clone, PartialEq)]
pub enum SiteOrRegion {
    Site(LBox<IntFunction>),
    Region(LBox<RegionFunction>),
}

/// `(from ...)` (15.6.3): describes the origin location of a move.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct From {
    pub site_type: Option<SiteType>,
    pub location: Option<SiteOrRegion>,
    pub level: Option<LBox<IntFunction>>,
    pub condition: Option<LBox<crate::ast::boolean::BooleanFunction>>,
}

/// `(to ...)` (15.6.6): describes the destination location of a move, and the effect to apply
/// there.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct To {
    pub site_type: Option<SiteType>,
    pub location: Option<SiteOrRegion>,
    pub level: Option<LBox<IntFunction>>,
    pub rotations: Option<LBox<IntArrayFunction>>,
    pub condition: Option<LBox<crate::ast::boolean::BooleanFunction>>,
    pub apply: Option<LBox<Apply>>,
}

/// `(between ...)` (15.6.1): describes the location(s) between a move's "from" and "to" sites.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Between {
    pub before: Option<LBox<IntFunction>>,
    pub range: Option<LBox<RangeFunction>>,
    pub after: Option<LBox<IntFunction>>,
    pub condition: Option<LBox<crate::ast::boolean::BooleanFunction>>,
    pub trail: Option<LBox<IntFunction>>,
    pub apply: Option<LBox<Apply>>,
}

/// How a [`Piece`] identifies the component(s) it refers to.
#[derive(Debug, Clone, PartialEq)]
pub enum PieceRef {
    Name(String),
    Index(LBox<IntFunction>),
    Names(Vec<String>),
    Indices(Vec<LBox<IntFunction>>),
}

/// `(piece ...)` (15.6.4): describes a component ("what" data), and optionally the local state
/// to place it with.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub reference: PieceRef,
    pub state: Option<LBox<IntFunction>>,
}

/// `(player ...)` (15.6.5): describes a player index ("who" data). Named `PlayerRef` here to
/// avoid clashing with [`crate::ast::game::Player`], the equipment-definition ludeme.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlayerRef {
    pub index: Option<LBox<IntFunction>>,
}

/// The key half of a `(pair ...)` (15.5.2).
#[derive(Debug, Clone, PartialEq)]
pub enum PairKey {
    Int(i64),
    Str(String),
    Role(RoleType),
}

/// The value half of a `(pair ...)` (15.5.2).
#[derive(Debug, Clone, PartialEq)]
pub enum PairValue {
    Int(i64),
    Str(String),
    Role(RoleType),
    Landmark(LandmarkType),
}

/// `(pair ...)` (15.5.2): a single key/value entry of a `(map ...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    pub key: PairKey,
    pub value: PairValue,
}

/// `(count <string> <int>)` (15.5.1): associates a named item with a count, e.g. within
/// `(place Random {(count "Pawn1" 8) ...})`. Named `ItemCount` here to avoid clashing with
/// [`IntFunction::Count`].
#[derive(Debug, Clone, PartialEq)]
pub struct ItemCount {
    pub item: String,
    pub count: LBox<IntFunction>,
}

/// `(graph vertices:{...} edges:{...})` (15.4.1): an explicit custom board graph.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphLiteral {
    pub vertices: Vec<Vec<f64>>,
    pub edges: Vec<(u32, u32)>,
}

/// A single point of a [`Poly`], expressed either as float coordinates or as dimension
/// expressions (so it can depend on board size parameters).
#[derive(Debug, Clone, PartialEq)]
pub enum PolyPoint {
    Float(f64, f64),
    Dim(
        LBox<crate::ast::numeric::dim::DimFunction>,
        LBox<crate::ast::numeric::dim::DimFunction>,
    ),
}

/// `(poly {...})` (15.4.2): a polygon (possibly concave) used to clip or shape board graphs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Poly {
    pub points: Vec<PolyPoint>,
}

/// `(card ...)` (15.3.1): per-card data used inside a `(deck {...})` definition. Distinct from
/// [`crate::ast::equipment::component::Card`], the standalone equipment-component ludeme (3.2.1).
#[derive(Debug, Clone, PartialEq)]
pub struct DeckCard {
    pub rank: crate::ast::types::CardType,
    pub rank_value: LBox<IntFunction>,
    pub value: LBox<IntFunction>,
    pub trump_rank: Option<LBox<IntFunction>>,
    pub trump_value: Option<LBox<IntFunction>>,
    pub biased: Option<LBox<IntFunction>>,
}

/// The sites a [`Hint`] applies to: either a single site or a set of them.
#[derive(Debug, Clone, PartialEq)]
pub enum HintSites {
    Site(LBox<IntFunction>),
    Region(Vec<LBox<IntFunction>>),
}

/// `(hint ...)` (15.3.2): a single deduction-puzzle hint value attached to a site or region.
#[derive(Debug, Clone, PartialEq)]
pub struct Hint {
    pub sites: HintSites,
    pub value: Option<LBox<IntFunction>>,
}

/// `(values <siteType> <range>)` (15.3.4): the set of legal values of a graph variable in a
/// deduction puzzle.
#[derive(Debug, Clone, PartialEq)]
pub struct ValuesRange {
    pub site_type: SiteType,
    pub range: LBox<RangeFunction>,
}

/// A player, identified either by a raw index expression or by [`RoleType`] -- the very common
/// `([<int>] | [<roleType>])` parameter shape (e.g. `(handSite Mover)` vs. `(handSite 0)`).
#[derive(Debug, Clone, PartialEq)]
pub enum IntOrRole {
    Int(LBox<IntFunction>),
    Role(RoleType),
}

/// A player, identified either by a [`PlayerRef`] (`(player ...)`) or by [`RoleType`] -- the
/// `([<player>] | [<roleType>])` parameter shape.
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerOrRole {
    Player(PlayerRef),
    Role(RoleType),
}

/// `(payoff <roleType> <floatFunction>)` (15.2.1): one player's payoff, within `(payoffs {...})`.
#[derive(Debug, Clone, PartialEq)]
pub struct Payoff {
    pub role: RoleType,
    pub value: LBox<FloatFunction>,
}

/// `(score <roleType> <int>)` (15.2.2): one player's final score, within `(byScore {...})`.
/// Distinct from [`IntFunction::Score`], the in-play score query.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreEntry {
    pub role: RoleType,
    pub value: LBox<IntFunction>,
}
