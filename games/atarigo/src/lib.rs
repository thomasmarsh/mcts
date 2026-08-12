#![allow(unused)]

use game_core::bigbitboard;
use game_core::bigbitboard::BigBitBoard;
use game_core::display::RectangularBoard;
use game_core::display::RectangularBoardDisplay;
use mcts::game::Game;
use mcts::game::PlayerIndex;

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Copy, Clone, Serialize, Debug, Default, Hash, PartialEq, Eq)]
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

/// A placement at cell `.0`, capturing every stone set in `.1` (computed up
/// front by [`State::valid`] so `apply` never has to recompute it).
#[derive(Clone, Copy, Serialize, Deserialize, Debug, Hash, PartialEq, Eq)]
pub struct Move<const N: usize, const WORDS: usize>(pub u16, pub BigBitBoard<N, N, WORDS>);

impl<const N: usize, const WORDS: usize> Move<N, WORDS> {
    /// Sentinel for "the player to move has no legal (non-suicide)
    /// placement" -- see [`State::apply`].
    pub const NO_MOVE: Move<N, WORDS> = Move(u16::MAX, BigBitBoard::EMPTY);
}

#[derive(Clone, Copy, Serialize, Debug, Default, Hash, PartialEq, Eq)]
pub struct State<const N: usize, const WORDS: usize> {
    pub black: BigBitBoard<N, N, WORDS>,
    pub white: BigBitBoard<N, N, WORDS>,
    pub turn: Player,
    pub winner: bool,
}

impl<const N: usize, const WORDS: usize> State<N, WORDS> {
    #[inline(always)]
    pub fn black(&self) -> BigBitBoard<N, N, WORDS> {
        self.black
    }

    #[inline(always)]
    pub fn white(&self) -> BigBitBoard<N, N, WORDS> {
        self.white
    }

    #[inline(always)]
    pub fn turn(&self) -> Player {
        self.turn
    }

    #[inline(always)]
    pub fn has_winner(&self) -> bool {
        self.winner
    }

    #[inline(always)]
    fn occupied(&self) -> BigBitBoard<N, N, WORDS> {
        self.black | self.white
    }

    #[inline(always)]
    fn player(&self, player: Player) -> BigBitBoard<N, N, WORDS> {
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

    #[inline]
    fn valid(&self, index: usize) -> (bool, BigBitBoard<N, N, WORDS>) {
        bigbitboard::check_go_move::<N, WORDS>(
            self.player(self.turn),
            self.player(self.turn.next()),
            index,
        )
    }

    #[inline]
    fn apply(&mut self, action: &Move<N, WORDS>) -> Self {
        if *action == Move::NO_MOVE {
            // The player to move has no legal (non-suicide) placement and
            // loses; the opponent wins.
            self.winner = true;
            self.turn = self.turn.next();
        } else {
            let index = action.0 as usize;
            debug_assert!(!self.occupied().get(index));
            let player = self.player(self.turn) | BigBitBoard::from_index(index);
            let opponent = self.player(self.turn.next());
            match self.turn {
                Player::Black => {
                    self.black = player;
                    self.white = opponent & !action.1;
                }
                Player::White => {
                    self.white = player;
                    self.black = opponent & !action.1;
                }
            }
            if !action.1.is_empty() {
                self.winner = true;
            } else {
                self.turn = self.turn.next();
            }
        }

        *self
    }
}

#[derive(Clone)]
pub struct AtariGo<const N: usize, const WORDS: usize>;

impl<const N: usize, const WORDS: usize> Game for AtariGo<N, WORDS> {
    type S = State<N, WORDS>;
    type A = Move<N, WORDS>;
    type P = Player;

    fn apply(mut state: State<N, WORDS>, action: &Move<N, WORDS>) -> State<N, WORDS> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N, WORDS>, actions: &mut Vec<Move<N, WORDS>>) {
        for index in !state.occupied() {
            let (valid, will_capture) = state.valid(index);
            if valid {
                actions.push(Move(index as u16, will_capture));
            }
        }
        if actions.is_empty() {
            actions.push(Move::NO_MOVE);
        }
    }

    fn is_terminal(state: &State<N, WORDS>) -> bool {
        state.winner
    }

    fn player_to_move(state: &State<N, WORDS>) -> Player {
        state.turn
    }

    fn winner(state: &State<N, WORDS>) -> Option<Player> {
        if state.winner {
            Some(state.turn)
        } else {
            None
        }
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        if *action == Move::NO_MOVE {
            return "no-move".into();
        }
        const COL_NAMES: &[u8] = b"ABCDEFGHIJKLMNOPQRST";
        let (row, col) = BigBitBoard::<N, N, WORDS>::to_coord(action.0 as usize);
        format!("{}{}", COL_NAMES[col] as char, row + 1)
    }

    fn num_players() -> usize {
        2
    }
}

impl<const N: usize, const WORDS: usize> RectangularBoard for State<N, WORDS> {
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

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

impl<const N: usize, const WORDS: usize> fmt::Display for State<N, WORDS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::{HashSet, VecDeque};

    use super::*;

    /// Deterministic regression coverage for the "no legal move" and capture
    /// logic: play a fixed-seed random game to completion and check the
    /// invariants that a missing `valid` filter (allowing suicide moves) or
    /// a wrong winner assignment would violate.
    ///
    /// AtariGo is guaranteed to terminate within `N*N + 1` plies: every
    /// non-capturing placement strictly reduces the empty-cell count by one,
    /// and any capturing placement ends the game immediately (the first
    /// capture wins), so play never resumes after a capture to refill the
    /// board.
    fn seeded_random_play<const N: usize, const WORDS: usize>(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::<N, WORDS>::default();
        let max_plies = N * N + 2;

        for _ in 0..max_plies {
            if AtariGo::<N, WORDS>::is_terminal(&state) {
                assert!(
                    AtariGo::<N, WORDS>::winner(&state).is_some(),
                    "a terminal AtariGo state must have a winner (draws are not possible)"
                );
                return;
            }
            let mut actions = Vec::new();
            AtariGo::<N, WORDS>::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = AtariGo::<N, WORDS>::apply(state, &action);
        }
        panic!("AtariGo<{N}> (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_atarigo_seeded_playouts_terminate() {
        for seed in 0..200 {
            seeded_random_play::<6, 1>(seed);
        }
    }

    /// Same seeded-playout regression, but on a board size that spans
    /// multiple `BigBitBoard` words (9x9 = 81 bits = 2 words), to prove the
    /// port from `BitBoard` didn't only work on the single-word case.
    #[test]
    fn test_atarigo_9x9_seeded_playouts_terminate() {
        for seed in 0..50 {
            seeded_random_play::<9, 2>(seed);
        }
    }

    /// Exhaustively explore every reachable position from the empty 3x3
    /// board (small enough to enumerate fully) and check that every
    /// terminal position has a winner, every non-terminal position has a
    /// legal move, and the whole reachable state graph is finite -- i.e.
    /// there is no line of play that fails to terminate.
    #[test]
    fn test_atarigo_3x3_all_lines_terminate_with_a_winner() {
        const N: usize = 3;
        const WORDS: usize = 1;
        let start = State::<N, WORDS>::default();
        let mut seen: HashSet<State<N, WORDS>> = HashSet::new();
        let mut queue: VecDeque<State<N, WORDS>> = VecDeque::new();
        seen.insert(start);
        queue.push_back(start);

        let mut explored = 0usize;
        while let Some(state) = queue.pop_front() {
            explored += 1;
            assert!(
                explored <= 200_000,
                "reachable-state graph is unexpectedly large -- possible non-termination"
            );

            if AtariGo::<N, WORDS>::is_terminal(&state) {
                assert!(
                    AtariGo::<N, WORDS>::winner(&state).is_some(),
                    "a terminal AtariGo state must have a winner (draws are not possible)"
                );
                continue;
            }

            let mut actions = Vec::new();
            AtariGo::<N, WORDS>::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );

            for action in actions {
                let next = AtariGo::<N, WORDS>::apply(state, &action);
                if seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
    }
}
