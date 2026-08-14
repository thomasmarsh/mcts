//! Effect move ludemes (Language Reference 8.2): moves applied as the direct result of a
//! decision, or chained after one via `(then ...)`.

use crate::ast::boolean::BooleanFunction;
use crate::ast::common::{Between, From, IntOrRole, Piece, PlayerOrRole, SiteOrRegion, To};
use crate::ast::direction::DirectionFunction;
use crate::ast::graph::GraphFunction;
use crate::ast::located::LBox;
use crate::ast::moves::decision::Messages;
use crate::ast::moves::{Moves, Then};
use crate::ast::numeric::float::FloatFunction;
use crate::ast::numeric::int::IntFunction;
use crate::ast::numeric::int_array::IntArrayFunction;
use crate::ast::numeric::range::RangeFunction;
use crate::ast::region::RegionFunction;
use crate::ast::types::{DealableType, RelationType, RoleType, SiteType, WhenType};

/// `(apply ...)` (8.2.2): applies an effect, optionally gated by a condition.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Apply {
    pub condition: Option<LBox<BooleanFunction>>,
    pub effect: Option<LBox<Moves>>,
}

/// `(add ...)` (8.2.1): places one or more components at a site or collection of sites.
#[derive(Debug, Clone, PartialEq)]
pub struct Add {
    pub piece: Option<Piece>,
    pub to: To,
    pub count: Option<LBox<IntFunction>>,
    pub stack: Option<bool>,
    pub then: Option<Then>,
}

/// `(attract ...)` (8.2.3): attracts pieces as close as possible to a site.
#[derive(Debug, Clone, PartialEq)]
pub struct Attract {
    pub from: Option<From>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub then: Option<Then>,
}

/// `(bet ...)` (8.2.4): bets an amount within a range.
#[derive(Debug, Clone, PartialEq)]
pub struct Bet {
    pub who: PlayerOrRole,
    pub range: LBox<RangeFunction>,
    pub then: Option<Then>,
}

/// `(claim ...)` (8.2.5): claims a site by adding a piece there.
#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    pub piece: Option<Piece>,
    pub to: To,
    pub then: Option<Then>,
}

/// `(custodial ...)` (8.2.6): applies an effect to sites flanked between two others.
#[derive(Debug, Clone, PartialEq)]
pub struct Custodial {
    pub from: Option<From>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub between: Option<Between>,
    pub to: Option<To>,
    pub then: Option<Then>,
}

/// `(deal ...)` (8.2.7): deals cards or dominoes during play. Distinct from
/// [`crate::ast::rules::start::Deal`], the starting-rules version.
#[derive(Debug, Clone, PartialEq)]
pub struct Deal {
    pub dealable: DealableType,
    pub count: Option<LBox<IntFunction>>,
    pub begin_with: Option<LBox<IntFunction>>,
    pub then: Option<Then>,
}

/// `(directional ...)` (8.2.8): applies an effect to all pieces in a direction from a site.
#[derive(Debug, Clone, PartialEq)]
pub struct Directional {
    pub from: Option<From>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub to: Option<To>,
    pub then: Option<Then>,
}

/// `(enclose ...)` (8.2.9): applies an effect to an enclosed group of pieces (e.g. Go capture).
#[derive(Debug, Clone, PartialEq)]
pub struct Enclose {
    pub site_type: Option<SiteType>,
    pub from: Option<From>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub between: Option<Between>,
    pub num_exception: Option<LBox<IntFunction>>,
    pub then: Option<Then>,
}

/// `(flip ...)` (8.2.10): flips a piece between its recorded states.
#[derive(Debug, Clone, PartialEq)]
pub struct Flip {
    pub site_type: Option<SiteType>,
    pub at: Option<LBox<IntFunction>>,
    pub then: Option<Then>,
}

/// `(fromTo ...)` (8.2.11): moves a piece from one site to another with no adjacency
/// requirement between them.
#[derive(Debug, Clone, PartialEq)]
pub struct FromTo {
    pub from: From,
    pub to: To,
    pub count: Option<LBox<IntFunction>>,
    pub copy: Option<bool>,
    pub stack: Option<bool>,
    pub mover: Option<RoleType>,
    pub then: Option<Then>,
}

/// `(hop ...)` (8.2.12): a piece hops over a hurdle in a direction.
#[derive(Debug, Clone, PartialEq)]
pub struct Hop {
    pub from: Option<From>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub between: Option<Between>,
    pub to: To,
    pub stack: Option<bool>,
    pub then: Option<Then>,
}

/// `(intervene ...)` (8.2.13): applies an effect to all sites flanking a site.
#[derive(Debug, Clone, PartialEq)]
pub struct Intervene {
    pub from: Option<From>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub between: Option<Between>,
    pub to: Option<To>,
    pub then: Option<Then>,
}

/// `(leap ...)` (8.2.14): leaps to sites defined by a turtle-graphics walk (e.g. Chess knight).
#[derive(Debug, Clone, PartialEq)]
pub struct Leap {
    pub from: Option<From>,
    pub walks: Vec<crate::ast::types::Walk>,
    pub forward: Option<bool>,
    pub rotations: Option<bool>,
    pub to: To,
    pub then: Option<Then>,
}

/// The payload of a `(note ...)` (8.2.15) message.
#[derive(Debug, Clone, PartialEq)]
pub enum NoteMessage {
    Str(String),
    Int(LBox<IntFunction>),
    IntArray(LBox<IntArrayFunction>),
    Float(LBox<FloatFunction>),
    Bool(LBox<BooleanFunction>),
    Region(LBox<RegionFunction>),
    Range(LBox<RangeFunction>),
    Direction(LBox<DirectionFunction>),
    Graph(LBox<GraphFunction>),
}

/// `(note ...)` (8.2.15): sends a note/message to a player or all players.
#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub player: Option<IntOrRole>,
    pub message: NoteMessage,
    pub to: Option<IntOrRole>,
    pub then: Option<Then>,
}

/// `(promote ...)` (8.2.18): promotes a piece into another type.
#[derive(Debug, Clone, PartialEq)]
pub struct Promote {
    pub site_type: Option<SiteType>,
    pub at: Option<LBox<IntFunction>>,
    pub piece: Piece,
    pub owner: Option<PlayerOrRole>,
    pub then: Option<Then>,
}

/// `(push ...)` (8.2.20): pushes all pieces from a site in one direction.
#[derive(Debug, Clone, PartialEq)]
pub struct Push {
    pub from: Option<From>,
    pub direction: LBox<DirectionFunction>,
    pub then: Option<Then>,
}

/// `(random ...)` (8.2.21): chooses randomly, either among weighted alternative move lists, or
/// a fixed number of moves sampled from a single list.
#[derive(Debug, Clone, PartialEq)]
pub enum Random {
    Weighted {
        weights: Vec<LBox<FloatFunction>>,
        choices: Vec<LBox<Moves>>,
    },
    Sample {
        moves: LBox<Moves>,
        num: LBox<IntFunction>,
    },
}

/// `(remove ...)` (8.2.22): removes an item from a site.
#[derive(Debug, Clone, PartialEq)]
pub struct Remove {
    pub site_type: Option<SiteType>,
    pub location: SiteOrRegion,
    pub level: Option<LBox<IntFunction>>,
    pub at: Option<WhenType>,
    pub count: Option<LBox<IntFunction>>,
    pub then: Option<Then>,
}

/// `(satisfy ...)` (8.2.24): deduction-puzzle constraints used to filter legal moves.
#[derive(Debug, Clone, PartialEq)]
pub struct Satisfy {
    pub constraints: Vec<LBox<BooleanFunction>>,
}

/// `(select ...)` (8.2.25): selects a site, or a from/to pair.
#[derive(Debug, Clone, PartialEq)]
pub struct Select {
    pub from: From,
    pub to: Option<To>,
    pub mover: Option<RoleType>,
    pub then: Option<Then>,
}

/// `(shoot ...)` (8.2.26): shoots an item from one site to another (e.g. Amazons).
#[derive(Debug, Clone, PartialEq)]
pub struct Shoot {
    pub piece: Piece,
    pub from: Option<From>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub between: Option<Between>,
    pub to: Option<To>,
    pub then: Option<Then>,
}

/// `(slide ...)` (8.2.27): slides a piece through a number of sites in a direction.
#[derive(Debug, Clone, PartialEq)]
pub struct Slide {
    pub from: Option<From>,
    pub track: Option<String>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub between: Option<Between>,
    pub to: Option<To>,
    pub stack: Option<bool>,
    pub then: Option<Then>,
}

/// `(sow ...)` (8.2.28): removes counters from a site and places them one-by-one along a
/// track (Mancala games).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sow {
    pub site_type: Option<SiteType>,
    pub origin: Option<LBox<IntFunction>>,
    pub count: Option<LBox<IntFunction>>,
    pub num_per_hole: Option<LBox<IntFunction>>,
    pub track: Option<String>,
    pub owner: Option<LBox<IntFunction>>,
    pub condition: Option<LBox<BooleanFunction>>,
    pub sow_effect: Option<LBox<Moves>>,
    pub apply: Option<LBox<Moves>>,
    pub include_self: Option<bool>,
    /// `origin:<boolean>` (place a counter in the origin hole at the start of sowing) --
    /// distinct from the positional `origin` site above.
    pub seed_origin: Option<bool>,
    pub skip_if: Option<LBox<BooleanFunction>>,
    pub backtracking: Option<bool>,
    pub forward: Option<bool>,
    pub then: Option<Then>,
}

/// `(step ...)` (8.2.29): moves to a connected site.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub from: Option<From>,
    pub direction: Option<LBox<DirectionFunction>>,
    pub to: To,
    pub stack: Option<bool>,
    pub then: Option<Then>,
}

/// `(surround ...)` (8.2.30): applies an effect to sites surrounded in a specific direction.
#[derive(Debug, Clone, PartialEq)]
pub struct Surround {
    pub from: Option<From>,
    pub relation: Option<RelationType>,
    pub between: Option<Between>,
    pub to: Option<To>,
    pub except: Option<LBox<IntFunction>>,
    pub with: Option<Piece>,
    pub then: Option<Then>,
}

/// `(trigger ...)` (8.2.32): sets a "triggered" flag for a player and event name.
#[derive(Debug, Clone, PartialEq)]
pub struct Trigger {
    pub event: String,
    pub who: IntOrRole,
    pub then: Option<Then>,
}

/// Any effect move: applied directly as the result of a decision, or chained via `then`.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    Add(Add),
    Apply(Apply),
    Attract(Attract),
    Bet(Bet),
    Claim(Claim),
    Custodial(Custodial),
    Deal(Deal),
    Directional(Directional),
    Enclose(Enclose),
    Flip(Flip),
    FromTo(FromTo),
    Hop(Hop),
    Intervene(Intervene),
    Leap(Leap),
    Note(Note),
    /// `(pass [<then>])` (8.2.16).
    Pass(Option<Then>),
    /// `(playCard [<then>])` (8.2.17).
    PlayCard(Option<Then>),
    Promote(Promote),
    /// `(propose ...)` (8.2.19).
    Propose(Messages, Option<Then>),
    Push(Push),
    Random(Random),
    Remove(Remove),
    /// `(roll [<then>])` (8.2.23).
    Roll(Option<Then>),
    Satisfy(Satisfy),
    Select(Select),
    Shoot(Shoot),
    Slide(Slide),
    Sow(Box<Sow>),
    Step(Step),
    Surround(Surround),
    Trigger(Trigger),
    /// `(vote ...)` (8.2.33).
    Vote(Messages, Option<Then>),
}
