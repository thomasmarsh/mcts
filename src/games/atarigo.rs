#![allow(unused)]

// TODO: this is just a placeholder
use super::bitboard::BitBoard;
use crate::game::Game;
use crate::game::PlayerIndex;

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
pub struct State<const N: usize> {
    black: BitBoard<N, N>,
    white: BitBoard<N, N>,
}

impl<const N: usize> State<N> {
    fn occupied(&self) -> BitBoard<N, N> {
        self.black | self.white
    }
}

#[derive(Clone)]
struct AtariGo<const N: usize>;

impl<const N: usize> Game for AtariGo<N> {
    type S = State<N>;
    type A = Move;
    type P = Player;

    fn apply(state: State<N>, action: &Move) -> State<N> {
        // 1. Place piece
        // 2. Scan for groups with no liberties and remove
        todo!();
    }

    fn generate_actions(state: &State<N>, actions: &mut Vec<Move>) {
        // 1. Most open points are playable
        // 2. ...unless they would result in self capture
        todo!();
    }

    fn is_terminal(state: &State<N>) -> bool {
        todo!();
    }

    fn player_to_move(state: &State<N>) -> Player {
        todo!();
    }

    fn winner(state: &State<N>) -> Option<Player> {
        todo!();
    }

    fn num_players() -> usize {
        2
    }
}
