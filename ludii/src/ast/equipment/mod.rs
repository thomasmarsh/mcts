//! Equipment ludemes (Language Reference chapter 3): everything a game is played with --
//! components, containers, and other supporting equipment such as maps and hint regions.

pub mod component;
pub mod container;
pub mod other;

use component::{Card, Die, Domino, Piece, Tile};
use container::{Board, Boardless, Dice, Hand, MancalaBoard, SurakartaBoard, Track};
use other::{Dominoes, Hints, Map, Regions};

/// A single entry of an [`Equipment`] list: any component, container, or other equipment
/// ludeme.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Card(Card),
    Die(Die),
    Piece(Piece),
    Domino(Domino),
    Tile(Tile),
    Board(Board),
    Boardless(Boardless),
    Track(Track),
    MancalaBoard(MancalaBoard),
    SurakartaBoard(SurakartaBoard),
    Deck(container::Deck),
    Dice(Dice),
    Hand(Hand),
    Dominoes(Dominoes),
    Hints(Hints),
    Map(Map),
    Regions(Regions),
}

/// `(equipment {<item>})` (3.1.1): the equipment list of a game.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Equipment {
    pub items: Vec<Item>,
}
