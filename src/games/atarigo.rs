#![allow(unused)]

// TODO: this is just a placeholder
use crate::game::Game;
use crate::game::PlayerIndex;

#[derive(Clone, Copy, Serialize, Debug)]
struct BitBoard(u64);

use serde::Serialize;

#[derive(Copy, Clone)]
pub enum Player {
    Black,
    White,
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Copy, Serialize, Debug, Hash, PartialEq, Eq)]
pub struct Move(u8);

#[derive(Clone, Copy, Serialize, Debug)]
pub struct State(BitBoard);

impl State {}

#[derive(Clone)]
struct AtariGo;

impl Game for AtariGo {
    type S = State;
    type A = Move;
    type P = Player;

    fn apply(state: State, action: &Move) -> State {
        // 1. Place piece
        // 2. Scan for groups with no liberties and remove
        todo!();
    }

    fn generate_actions(state: &State, actions: &mut Vec<Move>) {
        // 1. Most open points are playable
        // 2. ...unless they would result in self capture
        todo!();
    }

    fn is_terminal(state: &State) -> bool {
        todo!();
    }

    fn player_to_move(state: &State) -> Player {
        todo!();
    }

    fn winner(state: &State) -> Option<Player> {
        todo!();
    }

    fn num_players() -> usize {
        2
    }
}
