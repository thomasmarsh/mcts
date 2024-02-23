#![allow(unused)]

// TODO: this is just a placeholder
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

#[derive(Clone, Copy, Serialize, Debug, Default)]
pub struct State<const N: usize> {
    black: BitBoard<N, N>,
    white: BitBoard<N, N>,
    turn: Player,
    pub winner: bool,
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
    fn color(&self, index: usize) -> Player {
        debug_assert!(self.occupied().get(index));
        if self.black.get(index) {
            Player::Black
        } else {
            debug_assert!(self.white.get(index));

            Player::White
        }
    }

    fn valid(&self, index: usize) -> (bool, BitBoard<N, N>) {
        assert!(!self.occupied().get(index));
        let player = self.player(self.turn) | BitBoard::from_index(index);
        let opponent = self.player(self.turn.next());
        let occupied = player | opponent;
        let group = player.flood4(index);
        let adjacent = group.adjacency_mask();
        let occupied_adjacent = (occupied & adjacent);
        let empty_adjacent = !occupied_adjacent;

        // If we have adjacent empty positions we still have liberties.
        let safe = !(empty_adjacent.is_empty());

        let mut seen = BitBoard::empty();
        let mut will_capture = BitBoard::empty();
        for point in occupied_adjacent {
            // By definition, adjacent non-empty points must be the opponent
            assert!(occupied.get(point));
            assert!(opponent.get(point));
            if !seen.get(point) {
                let group = opponent.flood4(point);
                let adjacent = group.adjacency_mask();
                let empty_adjacent = !occupied & adjacent;
                if empty_adjacent.is_empty() {
                    will_capture |= group;
                }
                seen |= group;
            }
        }

        (safe || !(will_capture.is_empty()), will_capture)
    }

    #[inline]
    fn apply(&mut self, action: &Move) -> Self {
        debug_assert!(!self.occupied().get(action.0 as usize));
        let player = self.player(self.turn) | BitBoard::from_index(action.0 as usize);
        let opponent = self.player(self.turn.next());
        match self.turn {
            Player::Black => {
                self.black = player;
                self.white = opponent & (!BitBoard::new(action.1));
            }
            Player::White => {
                self.white = player;
                self.black = opponent & (!BitBoard::new(action.1));
            }
        }
        if action.1 > 0 {
            self.winner = true;
        } else {
            self.turn = self.turn.next();
        }

        *self
    }
}

#[derive(Clone)]
pub struct AtariGo<const N: usize>;

impl<const N: usize> Game for AtariGo<N> {
    type S = State<N>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<N>, action: &Move) -> State<N> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N>, actions: &mut Vec<Move>) {
        for index in !state.occupied() {
            let (valid, will_capture) = state.valid(index);
            if valid {
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
        let (col, row) = BitBoard::<N, N>::to_coord(action.0 as usize);
        format!("{}{}", COL_NAMES[col] as char, row + 1)
    }

    fn num_players() -> usize {
        2
    }
}

impl<const N: usize> fmt::Display for State<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..N {
            for col in 0..N {
                if self.black.get_at(N - row - 1, col) {
                    write!(f, "X")?;
                } else if self.white.get_at(N - row - 1, col) {
                    write!(f, "O")?;
                } else {
                    write!(f, ".")?;
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
    fn test_atarigo() {
        let mut state = State::<7>::default();
        while !AtariGo::is_terminal(&state) {
            println!("state:\n{state}");
            let mut actions = Vec::new();
            AtariGo::generate_actions(&state, &mut actions);
            use rand::Rng;
            let mut rng = rand::thread_rng();
            assert!(!actions.is_empty());
            let idx = rng.gen_range(0..actions.len());
            state = AtariGo::apply(state, &actions[idx]);
        }
    }
}
