#![allow(unused)]

use super::bitboard;
use super::bitboard::BitBoard;
use crate::game::Game;
use crate::game::PlayerIndex;

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
pub struct Move(u8, u64);

#[derive(Clone, Copy, Serialize, Debug)]
pub struct State<const N: usize> {
    black: BitBoard<N, N>,
    white: BitBoard<N, N>,
    ko_black: BitBoard<N, N>,
    ko_white: BitBoard<N, N>,
    turn: Player,
    winner: bool,
}

impl<const N: usize> Default for State<N> {
    fn default() -> Self {
        Self {
            black: BitBoard::default(),
            white: BitBoard::default(),
            ko_black: BitBoard::ones(),
            ko_white: BitBoard::ones(),
            turn: Player::default(),
            winner: false,
        }
    }
}

impl<const N: usize> State<N> {
    #[inline(always)]
    fn occupied(&self) -> BitBoard<N, N> {
        self.black | self.white
    }

    #[inline(always)]
    fn player(&self, player: Player) -> BitBoard<N, N> {
        match player {
            Player::Black => self.black,
            Player::White => self.white,
        }
    }

    #[inline(always)]
    fn player_ko(&self, player: Player) -> BitBoard<N, N> {
        match player {
            Player::Black => self.ko_black,
            Player::White => self.ko_white,
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

    #[inline]
    fn is_ko(&self, index: usize, will_capture: BitBoard<N, N>) -> bool {
        let player = self.player(self.turn) | BitBoard::from_index(index);
        let opponent = self.player(self.turn.next()) & !will_capture;
        let player_ko = self.player_ko(self.turn);
        let opponent_ko = self.player_ko(self.turn.next());
        player_ko == player && opponent_ko == opponent
    }

    #[inline]
    fn valid(&self, index: usize) -> (bool, BitBoard<N, N>) {
        bitboard::check_go_move::<N, N>(
            self.player(self.turn),
            self.player(self.turn.next()),
            index,
        )
    }

    #[inline]
    fn apply(&mut self, action: &Move) -> Self {
        debug_assert!(!self.occupied().get(action.0 as usize));
        let index = action.0 as usize;
        let player = self.player(self.turn) | BitBoard::from_index(index);
        let opponent = self.player(self.turn.next());
        self.ko_black = self.black;
        self.ko_white = self.white;
        match self.turn {
            Player::Black => {
                self.black = player;
                self.white = opponent & !BitBoard::new(action.1);
            }
            Player::White => {
                self.white = player;
                self.black = opponent & !BitBoard::new(action.1);
            }
        }
        if player.has_opposite_connection4(index) {
            self.winner = true;
        } else {
            self.turn = self.turn.next();
        }

        *self
    }
}

#[derive(Clone)]
pub struct Gonnect<const N: usize>;

impl<const N: usize> Game for Gonnect<N> {
    type S = State<N>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<N>, action: &Move) -> State<N> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N>, actions: &mut Vec<Move>) {
        for index in !state.occupied() {
            let (valid, will_capture) = state.valid(index);
            if valid && !state.is_ko(index, will_capture) {
                actions.push(Move(index as u8, will_capture.get_raw()))
            }
        }
    }

    fn is_terminal(state: &State<N>) -> bool {
        state.winner
    }

    fn player_to_move(state: &State<N>) -> Player {
        state.turn
    }

    fn winner(state: &State<N>) -> Option<Player> {
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

impl<const N: usize> fmt::Display for State<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in (0..N).rev() {
            for col in 0..N {
                if self.black.get_at(row, col) {
                    write!(f, " X")?;
                } else if self.white.get_at(row, col) {
                    write!(f, " O")?;
                } else {
                    write!(f, " .")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

struct MovesDisplay<const N: usize>(State<N>);

impl<const N: usize> fmt::Display for MovesDisplay<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut actions = Vec::new();
        Gonnect::generate_actions(&self.0, &mut actions);
        for row in (0..N).rev() {
            for col in 0..N {
                let mut found = false;
                for action in &actions {
                    let (r, c) = BitBoard::<N, N>::to_coord(action.0 as usize);
                    if r == row && c == col {
                        found = true;
                    }
                }

                if self.0.black.get_at(row, col) {
                    write!(f, " X")?;
                } else if self.0.white.get_at(row, col) {
                    write!(f, " O")?;
                } else if found {
                    write!(f, " +")?;
                } else {
                    write!(f, " .")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gonnect() {
        let mut state = State::<3>::default();
        while !Gonnect::is_terminal(&state) {
            println!("state: ({:?} to play)\n{state}", state.turn);
            println!("moves:\n{}", MovesDisplay(state));
            let mut actions = Vec::new();
            Gonnect::generate_actions(&state, &mut actions);
            use rand::Rng;
            let mut rng = rand::thread_rng();
            assert!(!actions.is_empty());
            let idx = rng.gen_range(0..actions.len());
            state = Gonnect::apply(state, &actions[idx]);
        }
    }
}