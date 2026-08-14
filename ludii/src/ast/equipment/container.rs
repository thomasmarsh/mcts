//! Container equipment ludemes (Language Reference 3.4-3.6): boards and other holders of
//! components (decks, dice cups, hands).

use crate::ast::common::{DeckCard, ValuesRange};
use crate::ast::graph::GraphFunction;
use crate::ast::located::LBox;
use crate::ast::numeric::dim::DimFunction;
use crate::ast::numeric::int::IntFunction;
use crate::ast::types::{RoleType, SiteType, StoreType, TilingBoardlessType};

/// One step of a [`Track`]'s route, when described as a brace-delimited list rather than a
/// plain coordinate string.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackStep {
    Site(i64),
    Range(i64, i64),
    End,
}

/// The route of a [`Track`]: either the compact string notation (e.g. `"1,E,N,W"`) or an
/// explicit list of steps/site indices.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackRoute {
    Description(String),
    Steps(Vec<TrackStep>),
}

/// The owner of a [`Track`]: a plain integer index, or a [`RoleType`].
#[derive(Debug, Clone, PartialEq)]
pub enum TrackOwner {
    Index(i64),
    Role(RoleType),
}

/// `(track ...)` (3.4.3): a named path around a container, typically the board, used by race
/// games and by pieces that move along predefined routes.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub name: String,
    pub route: TrackRoute,
    pub is_loop: Option<bool>,
    pub owner: Option<TrackOwner>,
    pub directed: Option<bool>,
}

/// `(board ...)` (3.4.1): a board defined by its graph, plus any tracks and deduction-puzzle
/// value ranges laid over it.
#[derive(Debug, Clone, PartialEq)]
pub struct Board {
    pub graph: LBox<GraphFunction>,
    pub tracks: Vec<Track>,
    pub values: Vec<ValuesRange>,
    pub use_site_type: Option<SiteType>,
    pub large_stack: Option<bool>,
}

/// `(boardless ...)` (3.4.2): a container that grows as pieces are played onto it, rather than
/// having a fixed graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Boardless {
    pub tiling: TilingBoardlessType,
    pub fake_size: Option<LBox<DimFunction>>,
    pub large_stack: Option<bool>,
}

/// `(mancalaBoard ...)` (3.5.1): a Mancala-style board of rows/columns of holes plus optional
/// stores.
#[derive(Debug, Clone, PartialEq)]
pub struct MancalaBoard {
    pub rows: LBox<IntFunction>,
    pub columns: LBox<IntFunction>,
    pub store: Option<StoreType>,
    pub num_stores: Option<LBox<IntFunction>>,
    pub large_stack: Option<bool>,
    pub tracks: Vec<Track>,
}

/// `(surakartaBoard ...)` (3.5.2): a board with capture loops that pieces travel around, as in
/// Surakarta.
#[derive(Debug, Clone, PartialEq)]
pub struct SurakartaBoard {
    pub graph: LBox<GraphFunction>,
    pub loops: Option<LBox<IntFunction>>,
    pub from: Option<LBox<IntFunction>>,
    pub large_stack: Option<bool>,
}

/// `(deck ...)` (3.6.1): a deck of cards.
#[derive(Debug, Clone, PartialEq)]
pub struct Deck {
    pub owner: Option<RoleType>,
    pub cards_by_suit: Option<LBox<IntFunction>>,
    pub suits: Option<LBox<IntFunction>>,
    pub cards: Vec<DeckCard>,
}

/// How the faces of a [`Dice`] set are determined.
#[derive(Debug, Clone, PartialEq)]
pub enum DiceFaces {
    Faces(Vec<LBox<IntFunction>>),
    FacesByDie(Vec<Vec<LBox<IntFunction>>>),
    From(LBox<IntFunction>),
}

/// `(dice ...)` (3.6.2): a set of rollable dice.
#[derive(Debug, Clone, PartialEq)]
pub struct Dice {
    pub num_faces: Option<LBox<IntFunction>>,
    pub faces: Option<DiceFaces>,
    pub owner: Option<RoleType>,
    pub num: LBox<IntFunction>,
    pub biased: Option<Vec<LBox<IntFunction>>>,
}

/// `(hand ...)` (3.6.3): a player's hand, for components held off the board.
#[derive(Debug, Clone, PartialEq)]
pub struct Hand {
    pub owner: RoleType,
    pub size: Option<LBox<IntFunction>>,
}
