#![allow(unused)]

use super::bitboard;
use super::bitboard::BitBoard;
use crate::display::RectangularBoard;
use crate::display::RectangularBoardDisplay;
use crate::game::Game;
use crate::game::PlayerIndex;

use log::Record;
use serde::Serialize;
use std::fmt;

#[derive(Copy, Clone, Serialize, Debug, Default)]
pub enum Player {
    #[default]
    Black,
    White,
}

impl Player {
    fn next(self) -> Player {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Copy, Serialize, Debug, Hash, PartialEq, Eq)]
pub struct Move(u8, u8);

impl Move {
    #[inline(always)]
    fn src(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    fn dst(self) -> usize {
        self.1 as usize
    }
}

#[derive(Clone, Copy, Serialize, Debug)]
pub struct State<const N: usize, const M: usize> {
    black: BitBoard<N, M>,
    white: BitBoard<N, M>,
    turn: Player,
    winner: bool,
}

impl<const N: usize, const M: usize> Default for State<N, M> {
    fn default() -> Self {
        debug_assert!(N > 5);
        debug_assert!(M > 0);

        let n = BitBoard::wall(bitboard::Direction::North);
        let s = BitBoard::wall(bitboard::Direction::South);

        let black = n | n.shift_south();
        let white = s | s.shift_north();

        Self {
            black,
            white,
            turn: Player::Black,
            winner: false,
        }
    }
}

impl<const N: usize, const M: usize> State<N, M> {
    #[inline(always)]
    fn occupied(&self) -> BitBoard<N, M> {
        self.black | self.white
    }

    #[inline(always)]
    fn player(&self, player: Player) -> (BitBoard<N, M>, BitBoard<N, M>) {
        match player {
            Player::Black => (self.black, BitBoard::wall(bitboard::Direction::South)),
            Player::White => (self.white, BitBoard::wall(bitboard::Direction::North)),
        }
    }

    #[inline(always)]
    fn color(&self, index: usize) -> Player {
        debug_assert!(self.occupied().get(index));
        if self.black.get(index) {
            Player::Black
        } else {
            debug_assert!(self.white.get(index));
            Player::White
        }
    }

    fn moves(&self, actions: &mut Vec<Move>) {
        if self.winner {
            return;
        }

        let (player, _) = self.player(self.turn);
        let (opponent, _) = self.player(self.turn.next());
        let occupied = player | opponent;

        for src in player {
            let mut knight_moves = BitBoard::EMPTY;
            let (row, col) = BitBoard::<N, M>::to_coord(src);
            for (dx, dy) in [
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
                (1, 2),
                (1, -2),
                (-1, 2),
                (-1, -2),
            ] {
                let (r, c) = (row as isize + dx, col as isize + dy);
                if r >= 0 && r < N as isize && c >= 0 && c < M as isize {
                    knight_moves |= BitBoard::from_coord(r as usize, c as usize);
                }
            }

            let available = knight_moves & !player;
            for dst in available {
                actions.push(Move(src as u8, dst as u8));
            }
        }
    }

    #[inline]
    fn apply(&mut self, action: &Move) -> Self {
        debug_assert!(self.occupied().get(action.0 as usize));
        let src = BitBoard::from_index(action.0 as usize);
        let dst = BitBoard::from_index(action.1 as usize);
        let (mut player, goal) = self.player(self.turn);
        player |= dst;
        player &= !src;
        let opponent = self.player(self.turn.next()).0 & !dst;

        match self.turn {
            Player::Black => {
                self.black = player;
                self.white = opponent;
            }
            Player::White => {
                self.white = player;
                self.black = opponent;
            }
        }

        if player.intersects(goal) {
            self.winner = true;
        } else {
            self.turn = self.turn.next();
        }

        *self
    }
}

#[derive(Clone)]

pub struct Knightthrough<const N: usize, const M: usize>;

impl<const N: usize, const M: usize> Game for Knightthrough<N, M> {
    type S = State<N, M>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<N, M>, action: &Move) -> State<N, M> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N, M>, actions: &mut Vec<Move>) {
        state.moves(actions);
    }

    fn is_terminal(state: &State<N, M>) -> bool {
        state.winner
    }

    fn player_to_move(state: &State<N, M>) -> Player {
        state.turn
    }

    fn winner(state: &State<N, M>) -> Option<Player> {
        if state.winner {
            Some(state.turn)
        } else {
            None
        }
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        const COL_NAMES: &[u8] = b"ABCDEFGH";
        let (row, col) = BitBoard::<N, N>::to_coord(action.0 as usize);
        format!("{}{}", COL_NAMES[col] as char, row + 1)
    }

    fn num_players() -> usize {
        2
    }
}

impl<const N: usize, const M: usize> RectangularBoard for State<N, M> {
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = M;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        if self.black.get_at(row, col) {
            'X'
        } else if self.white.get_at(row, col) {
            'O'
        } else {
            '.'
        }
    }
}

impl<const N: usize, const M: usize> fmt::Display for State<N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use crate::util::random_play;

    use super::*;

    #[test]
    fn test_knightthrough() {
        random_play::<Knightthrough<8, 8>>();
    }
}
