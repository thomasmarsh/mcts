//! The root of a Ludii ludemeplex (Language Reference chapter 2): `(game ...)`, and the
//! `(match ...)` / `(games ...)` / `(subgame ...)` ludemes used to combine several games into a
//! super-game.

use crate::ast::equipment::Equipment;
use crate::ast::located::LBox;
use crate::ast::numeric::int::IntFunction;
use crate::ast::rules::end::End;
use crate::ast::rules::Rules;
use crate::ast::types::{CompassDirection, ModeType};

/// The root ludeme of a single game description (2.1.1): `(game <string> <players> [<mode>]
/// <equipment> <rules>)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Game {
    pub name: String,
    pub players: Players,
    pub mode: Option<Mode>,
    pub equipment: Equipment,
    pub rules: Rules,
}

/// Either a single `(game ...)` description, or a `(match ...)` of several component games --
/// the two possible roots of a `.lud` file.
#[derive(Debug, Clone, PartialEq)]
pub enum Description {
    Game(Game),
    Match(Match),
}

/// `(match <string> [<players>] <games> <end>)` (2.2.2): a super-game composed of a series of
/// component games, with its own end conditions based on their results.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub name: String,
    pub players: Option<Players>,
    pub games: Games,
    pub end: End,
}

/// `(games (<subgame> | {<subgame>}))` (2.2.1): the component games of a [`Match`].
#[derive(Debug, Clone, PartialEq)]
pub struct Games {
    pub subgames: Vec<Subgame>,
}

/// `(subgame <string> [<string>] [next:<int>] [result:<int>])` (2.2.3): one instance game
/// within a [`Match`].
#[derive(Debug, Clone, PartialEq)]
pub struct Subgame {
    pub name: String,
    pub option: Option<String>,
    pub next: Option<LBox<IntFunction>>,
    pub result: Option<LBox<IntFunction>>,
}

/// `(mode <modeType>)` (2.3.1): the mode of play for a game or phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mode(pub ModeType);

/// `(player <directionFacing>)` (2.4.1): a single player, identified by the compass direction
/// their pieces face (e.g. the side of the board they start on).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player {
    pub facing: CompassDirection,
}

/// `(players ...)` (2.4.2): the players of a game, either as an explicit list with per-player
/// data, or as a plain count.
#[derive(Debug, Clone, PartialEq)]
pub enum Players {
    List(Vec<Player>),
    Count(LBox<IntFunction>),
}
