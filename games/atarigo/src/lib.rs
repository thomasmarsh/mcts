#![allow(unused)]

use game_core::bigbitboard::BigBitBoard;
use game_core::display::RectangularBoard;
use game_core::display::RectangularBoardDisplay;
use game_core::go_engine::GoEngine;
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Move<const N: usize, const WORDS: usize>(pub u16, pub BigBitBoard<N, N, WORDS>);

/// Hand-written wire format: `.1`'s words as hex strings, not raw `u64`s.
/// A captured group can span most of a 64-cell word, and a `u64` with
/// several scattered bits set can exceed JS's 2^53 safe-integer range --
/// `serde`'s derived numeric encoding would silently lose precision through
/// `JSON.parse` on the client, corrupting the capture set the server later
/// validates a client-submitted move against. Mirrors the hex-string
/// convention `games/breakthrough`/`games/knightthrough` use for their own
/// 64-bit bitboard wire fields.
impl<const N: usize, const WORDS: usize> Serialize for Move<N, WORDS> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&self.0)?;
        let hex: Vec<String> = self.1.words().iter().map(|w| format!("{w:016x}")).collect();
        tup.serialize_element(&hex)?;
        tup.end()
    }
}

impl<'de, const N: usize, const WORDS: usize> Deserialize<'de> for Move<N, WORDS> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (cell, hex): (u16, Vec<String>) = Deserialize::deserialize(deserializer)?;
        let mut words = [0u64; WORDS];
        for (i, w) in words.iter_mut().enumerate() {
            let s = hex
                .get(i)
                .ok_or_else(|| serde::de::Error::invalid_length(hex.len(), &"WORDS hex words"))?;
            *w = u64::from_str_radix(s, 16).map_err(serde::de::Error::custom)?;
        }
        Ok(Move(cell, BigBitBoard::new(words)))
    }
}

impl<const N: usize, const WORDS: usize> Move<N, WORDS> {
    /// Sentinel for "the player to move has no legal (non-suicide)
    /// placement" -- see [`State::apply`].
    pub const NO_MOVE: Move<N, WORDS> = Move(u16::MAX, BigBitBoard::EMPTY);
}

/// Board occupancy plus captures are tracked by the incremental
/// [`GoEngine`] (union-find groups + cached liberty counts) rather than a
/// bare `black`/`white` `BigBitBoard` pair, so `valid`/`apply` answer
/// legality and capture questions in O(neighbors)/O(group size) instead of
/// re-flooding the board on every candidate cell.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct State<const N: usize, const WORDS: usize, const CELLS: usize> {
    engine: GoEngine<N, WORDS, CELLS>,
    pub turn: Player,
    pub winner: bool,
}

impl<const N: usize, const WORDS: usize, const CELLS: usize> State<N, WORDS, CELLS> {
    /// Rebuilds a state from a plain occupancy pair (e.g. deserialized from
    /// the wire format), flood-filling the engine's group/liberty
    /// bookkeeping from scratch once. Not used on the hot `apply` path,
    /// which advances an already-built engine incrementally instead.
    pub fn from_boards(
        black: BigBitBoard<N, N, WORDS>,
        white: BigBitBoard<N, N, WORDS>,
        turn: Player,
        winner: bool,
    ) -> Self {
        Self {
            engine: GoEngine::from_boards(black, white),
            turn,
            winner,
        }
    }

    #[inline(always)]
    pub fn black(&self) -> BigBitBoard<N, N, WORDS> {
        self.engine.black()
    }

    #[inline(always)]
    pub fn white(&self) -> BigBitBoard<N, N, WORDS> {
        self.engine.white()
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
        self.black() | self.white()
    }

    #[inline(always)]
    fn color(&self, index: usize) -> Player {
        debug_assert!(self.occupied().get(index));
        if self.black().get(index) {
            Player::Black
        } else {
            debug_assert!(self.white().get(index));
            Player::White
        }
    }

    #[inline]
    fn valid(&self, index: usize) -> (bool, BigBitBoard<N, N, WORDS>) {
        self.engine.check(self.turn == Player::Black, index)
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
            let captured = self
                .engine
                .play(self.turn == Player::Black, index)
                .expect("apply called with a move already validated by generate_actions");
            if !captured.is_empty() {
                self.winner = true;
            } else {
                self.turn = self.turn.next();
            }
        }

        *self
    }
}

#[derive(Clone)]
pub struct AtariGo<const N: usize, const WORDS: usize, const CELLS: usize>;

impl<const N: usize, const WORDS: usize, const CELLS: usize> Game for AtariGo<N, WORDS, CELLS> {
    type S = State<N, WORDS, CELLS>;
    type A = Move<N, WORDS>;
    type P = Player;

    fn apply(mut state: State<N, WORDS, CELLS>, action: &Move<N, WORDS>) -> State<N, WORDS, CELLS> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N, WORDS, CELLS>, actions: &mut Vec<Move<N, WORDS>>) {
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

    /// Rejection-sampling fast path for `SimulateStrategy::playout`'s
    /// uniform rollouts: draw a random cell and run `State::valid` (the
    /// `GoEngine`-backed O(neighbors) check) on just that one cell instead of
    /// probing every empty cell via `generate_actions`. Falls back to the
    /// full enumeration once `max_attempts` candidates in a row miss --
    /// bounds the cost on boards where legal placements are sparse (heavy
    /// suicide restriction near the end of the game) instead of looping
    /// indefinitely, and is also what correctly proves "no legal move" when
    /// that's actually true, since a bounded run of misses alone can't tell
    /// "unlucky" apart from "no move exists".
    fn random_action(
        state: &State<N, WORDS, CELLS>,
        rng: &mut rand::rngs::SmallRng,
    ) -> Option<Move<N, WORDS>> {
        use rand::Rng;
        let occupied = state.occupied();
        if occupied.count_ones() as usize == CELLS {
            return Some(Move::NO_MOVE);
        }
        let max_attempts = 64;
        for _ in 0..max_attempts {
            let index = rng.gen_range(0..CELLS);
            if occupied.get(index) {
                continue;
            }
            let (valid, will_capture) = state.valid(index);
            if valid {
                return Some(Move(index as u16, will_capture));
            }
        }
        let mut actions = Vec::new();
        Self::generate_actions(state, &mut actions);
        Some(actions[rng.gen_range(0..actions.len())])
    }

    fn is_terminal(state: &State<N, WORDS, CELLS>) -> bool {
        state.winner
    }

    fn player_to_move(state: &State<N, WORDS, CELLS>) -> Player {
        state.turn
    }

    fn winner(state: &State<N, WORDS, CELLS>) -> Option<Player> {
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

impl<const N: usize, const WORDS: usize, const CELLS: usize> RectangularBoard
    for State<N, WORDS, CELLS>
{
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        if self.black().get_at(row, col) {
            'X'
        } else if self.white().get_at(row, col) {
            'O'
        } else {
            '.'
        }
    }
}

impl<const N: usize, const WORDS: usize, const CELLS: usize> fmt::Display
    for State<N, WORDS, CELLS>
{
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
    fn seeded_random_play<const N: usize, const WORDS: usize, const CELLS: usize>(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::<N, WORDS, CELLS>::default();
        let max_plies = N * N + 2;

        for _ in 0..max_plies {
            if AtariGo::<N, WORDS, CELLS>::is_terminal(&state) {
                assert!(
                    AtariGo::<N, WORDS, CELLS>::winner(&state).is_some(),
                    "a terminal AtariGo state must have a winner (draws are not possible)"
                );
                return;
            }
            let mut actions = Vec::new();
            AtariGo::<N, WORDS, CELLS>::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = AtariGo::<N, WORDS, CELLS>::apply(state, &action);
        }
        panic!("AtariGo<{N}> (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_atarigo_seeded_playouts_terminate() {
        for seed in 0..200 {
            seeded_random_play::<6, 1, 36>(seed);
        }
    }

    /// Same seeded-playout regression, but on a board size that spans
    /// multiple `BigBitBoard` words (9x9 = 81 bits = 2 words), to prove the
    /// port from `BitBoard` didn't only work on the single-word case.
    #[test]
    fn test_atarigo_9x9_seeded_playouts_terminate() {
        for seed in 0..50 {
            seeded_random_play::<9, 2, 81>(seed);
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
        const CELLS: usize = 9;
        let start = State::<N, WORDS, CELLS>::default();
        let mut seen: HashSet<State<N, WORDS, CELLS>> = HashSet::new();
        let mut queue: VecDeque<State<N, WORDS, CELLS>> = VecDeque::new();
        seen.insert(start);
        queue.push_back(start);

        let mut explored = 0usize;
        while let Some(state) = queue.pop_front() {
            explored += 1;
            assert!(
                explored <= 200_000,
                "reachable-state graph is unexpectedly large -- possible non-termination"
            );

            if AtariGo::<N, WORDS, CELLS>::is_terminal(&state) {
                assert!(
                    AtariGo::<N, WORDS, CELLS>::winner(&state).is_some(),
                    "a terminal AtariGo state must have a winner (draws are not possible)"
                );
                continue;
            }

            let mut actions = Vec::new();
            AtariGo::<N, WORDS, CELLS>::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );

            for action in actions {
                let next = AtariGo::<N, WORDS, CELLS>::apply(state, &action);
                if seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
    }

    use game_core::bigbitboard;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Equivalence check: before retiring `check_go_move` as AtariGo's own
    // legality/capture path (now `GoEngine::check`/`play`, see `State::valid`/
    // `apply`), replay the same seeded-random-playout regression games above
    // and assert the engine-backed action set matches a `check_go_move`
    // oracle computed independently from the same board/turn at every ply.

    /// Old-path oracle: legal actions computed directly from `check_go_move`
    /// against a plain `black`/`white` pair, mirroring exactly what
    /// `State::valid`/`generate_actions` did before the `GoEngine` port.
    fn old_path_actions<const N: usize, const WORDS: usize>(
        black: BigBitBoard<N, N, WORDS>,
        white: BigBitBoard<N, N, WORDS>,
        turn: Player,
    ) -> Vec<Move<N, WORDS>> {
        let occupied = black | white;
        let (player, opponent) = match turn {
            Player::Black => (black, white),
            Player::White => (white, black),
        };
        let mut actions = Vec::new();
        for index in !occupied {
            let (valid, will_capture) =
                bigbitboard::check_go_move::<N, WORDS>(player, opponent, index);
            if valid {
                actions.push(Move(index as u16, will_capture));
            }
        }
        if actions.is_empty() {
            actions.push(Move::NO_MOVE);
        }
        actions
    }

    fn seeded_random_play_matches_old_path<
        const N: usize,
        const WORDS: usize,
        const CELLS: usize,
    >(
        seed: u64,
    ) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::<N, WORDS, CELLS>::default();
        let max_plies = N * N + 2;

        for _ in 0..max_plies {
            if AtariGo::<N, WORDS, CELLS>::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            AtariGo::<N, WORDS, CELLS>::generate_actions(&state, &mut actions);
            let old_actions =
                old_path_actions::<N, WORDS>(state.black(), state.white(), state.turn());
            assert_eq!(
                actions, old_actions,
                "engine-backed action set diverged from the check_go_move oracle at seed {seed}"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = AtariGo::<N, WORDS, CELLS>::apply(state, &action);
        }
        panic!("AtariGo<{N}> (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_engine_backed_atarigo_matches_check_go_move_oracle() {
        for seed in 0..200 {
            seeded_random_play_matches_old_path::<6, 1, 36>(seed);
        }
        for seed in 0..50 {
            seeded_random_play_matches_old_path::<9, 2, 81>(seed);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // `random_action`'s rejection-sampling fast path must always agree with `generate_actions`'s
    // full enumeration: every draw is either `Move::NO_MOVE` when that's the only legal action, or
    // an action also present in `generate_actions`'s output.

    fn random_action_matches_generate_actions<
        const N: usize,
        const WORDS: usize,
        const CELLS: usize,
    >(
        seed: u64,
    ) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::<N, WORDS, CELLS>::default();
        let max_plies = N * N + 2;

        for _ in 0..max_plies {
            if AtariGo::<N, WORDS, CELLS>::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            AtariGo::<N, WORDS, CELLS>::generate_actions(&state, &mut actions);
            // Draw several times from the same state to exercise both the
            // rejection-sampling success path and (near the end of the
            // game, when legal placements are sparse) its full-enumeration
            // fallback.
            for _ in 0..8 {
                let drawn = AtariGo::<N, WORDS, CELLS>::random_action(&state, &mut rng).expect(
                    "random_action must return Some whenever generate_actions is non-empty",
                );
                assert!(
                    actions.contains(&drawn),
                    "random_action drew {drawn:?}, not present in generate_actions {actions:?}"
                );
            }
            let action = actions[rng.gen_range(0..actions.len())];
            state = AtariGo::<N, WORDS, CELLS>::apply(state, &action);
        }
        panic!("AtariGo<{N}> (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_atarigo_random_action_matches_generate_actions() {
        for seed in 0..200 {
            random_action_matches_generate_actions::<6, 1, 36>(seed);
        }
        for seed in 0..50 {
            random_action_matches_generate_actions::<9, 2, 81>(seed);
        }
    }
}
