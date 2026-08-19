#![allow(unused)]

use bitboard::{Board, Dyn, GoEngine};
use mcts::game::Game;
use mcts::game::PlayerIndex;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Board sizes 3x3 through 19x19 (`main.rs`'s supported range) all fit in 6
/// words (19*19 = 361 bits), so one `Board<[u64; 6], Dyn, Dyn>`
/// monomorphization serves every size, with the actual size carried as a
/// runtime field rather than a distinct compiled type per size.
pub type Bits = Board<[u64; 6], Dyn, Dyn>;
type Engine = GoEngine<[u64; 6], Dyn, Dyn>;

/// The board size `State::default()` builds, matching `main.rs`'s
/// `DEFAULT_SIZE`.
pub const DEFAULT_SIZE: usize = 9;

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
///
/// `.1` is a raw 6-word capture mask, not a dims-carrying [`Bits`]: a `Move`
/// is deserialized off the wire (`main.rs`'s `apply` handler) before the
/// target `State`'s size is known to the deserializer, so it can't build a
/// `Bits` (which needs `Dyn` row/col values) directly.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Move(pub u16, [u64; 6]);

/// Hand-written wire format: `.1`'s words as hex strings, not raw `u64`s.
/// A captured group can span most of a 64-cell word, and a `u64` with
/// several scattered bits set can exceed JS's 2^53 safe-integer range --
/// `serde`'s derived numeric encoding would silently lose precision through
/// `JSON.parse` on the client, corrupting the capture set the server later
/// validates a client-submitted move against. Mirrors the hex-string
/// convention `games/breakthrough`/`games/knightthrough` use for their own
/// 64-bit bitboard wire fields.
impl Serialize for Move {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&self.0)?;
        let hex: Vec<String> = self.1.iter().map(|w| format!("{w:016x}")).collect();
        tup.serialize_element(&hex)?;
        tup.end()
    }
}

impl<'de> Deserialize<'de> for Move {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (cell, hex): (u16, Vec<String>) = Deserialize::deserialize(deserializer)?;
        let mut words = [0u64; 6];
        for (i, w) in words.iter_mut().enumerate() {
            let s = hex
                .get(i)
                .ok_or_else(|| serde::de::Error::invalid_length(hex.len(), &"6 hex words"))?;
            *w = u64::from_str_radix(s, 16).map_err(serde::de::Error::custom)?;
        }
        Ok(Move(cell, words))
    }
}

impl Move {
    /// Sentinel for "the player to move has no legal (non-suicide)
    /// placement" -- see [`State::apply`].
    pub const NO_MOVE: Move = Move(u16::MAX, [0; 6]);

    fn new(index: u16, capture_mask: Bits) -> Self {
        let mut words = [0u64; 6];
        for (i, w) in capture_mask.words().enumerate() {
            words[i] = w;
        }
        Move(index, words)
    }
}

/// Not `Copy`: `Engine`'s group/liberty bookkeeping is `Vec`-backed (a
/// `Dyn`-dimensioned board has no compile-time cell count to size a fixed
/// array with -- see `bitboard::go::GoEngine`'s doc comment), so `apply`
/// clones rather than implicitly copies.
///
/// Board occupancy plus captures are tracked by the incremental
/// [`GoEngine`] (union-find groups + cached liberty counts) rather than a
/// bare `black`/`white` [`Bits`] pair, so `valid`/`apply` answer legality and
/// capture questions in O(neighbors)/O(group size) instead of re-flooding
/// the board on every candidate cell.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct State {
    engine: Engine,
    pub turn: Player,
    pub winner: bool,
}

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_SIZE)
    }
}

impl State {
    /// A fresh empty `size x size` board.
    pub fn new(size: usize) -> Self {
        Self {
            engine: Engine::new(Dyn(size), Dyn(size)),
            turn: Player::default(),
            winner: false,
        }
    }

    /// Rebuilds a state from a plain occupancy pair (e.g. deserialized from
    /// the wire format), flood-filling the engine's group/liberty
    /// bookkeeping from scratch once. Not used on the hot `apply` path,
    /// which advances an already-built engine incrementally instead.
    pub fn from_boards(black: Bits, white: Bits, turn: Player, winner: bool) -> Self {
        Self {
            engine: Engine::from_boards(black, white),
            turn,
            winner,
        }
    }

    #[inline(always)]
    pub fn black(&self) -> Bits {
        self.engine.black()
    }

    #[inline(always)]
    pub fn white(&self) -> Bits {
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
    fn occupied(&self) -> Bits {
        self.black() | self.white()
    }

    #[inline(always)]
    fn color(&self, index: usize) -> Player {
        debug_assert!(self.occupied().get_index(index));
        if self.black().get_index(index) {
            Player::Black
        } else {
            debug_assert!(self.white().get_index(index));
            Player::White
        }
    }

    #[inline]
    fn valid(&self, index: usize) -> (bool, Bits) {
        self.engine.check(self.turn == Player::Black, index)
    }

    #[inline]
    fn apply(&mut self, action: &Move) -> Self {
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
            if !captured.none_set() {
                self.winner = true;
            } else {
                self.turn = self.turn.next();
            }
        }

        self.clone()
    }
}

#[derive(Clone)]
pub struct AtariGo;

impl Game for AtariGo {
    type S = State;
    type A = Move;
    type P = Player;

    fn apply(mut state: State, action: &Move) -> State {
        state.apply(action)
    }

    fn generate_actions(state: &State, actions: &mut Vec<Move>) {
        for index in !state.occupied() {
            let (valid, will_capture) = state.valid(index);
            if valid {
                actions.push(Move::new(index as u16, will_capture));
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
    fn random_action(state: &State, rng: &mut rand::rngs::SmallRng) -> Option<Move> {
        use rand::Rng;
        let occupied = state.occupied();
        let cells = state.black().len();
        if occupied.count_ones() as usize == cells {
            return Some(Move::NO_MOVE);
        }
        let max_attempts = 64;
        for _ in 0..max_attempts {
            let index = rng.gen_range(0..cells);
            if occupied.get_index(index) {
                continue;
            }
            let (valid, will_capture) = state.valid(index);
            if valid {
                return Some(Move::new(index as u16, will_capture));
            }
        }
        let mut actions = Vec::new();
        Self::generate_actions(state, &mut actions);
        Some(actions[rng.gen_range(0..actions.len())])
    }

    fn is_terminal(state: &State) -> bool {
        state.winner
    }

    fn player_to_move(state: &State) -> Player {
        state.turn
    }

    fn winner(state: &State) -> Option<Player> {
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
        let n = state.black().cols();
        let (row, col) = (action.0 as usize / n, action.0 as usize % n);
        format!("{}{}", COL_NAMES[col] as char, row + 1)
    }

    fn num_players() -> usize {
        2
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const FILES: &[u8] = b"ABCDEFGHIJKLMNOPQRST";
        let n = self.black().rows();
        let char_at = |row: usize, col: usize| {
            if self.black().get(row, col) {
                'X'
            } else if self.white().get(row, col) {
                'O'
            } else {
                '.'
            }
        };
        write!(f, " ")?;
        for c in FILES.iter().take(n) {
            write!(f, " {}", *c as char)?;
        }
        writeln!(f)?;
        for row in (0..n).rev() {
            write!(f, "{}", row + 1)?;
            for col in 0..n {
                write!(f, " {}", char_at(row, col))?;
            }
            write!(f, " {}", row + 1)?;
            writeln!(f)?;
        }
        write!(f, " ")?;
        for c in FILES.iter().take(n) {
            write!(f, " {}", *c as char)?;
        }
        writeln!(f)?;
        Ok(())
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
    fn seeded_random_play(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = n * n + 2;

        for _ in 0..max_plies {
            if AtariGo::is_terminal(&state) {
                assert!(
                    AtariGo::winner(&state).is_some(),
                    "a terminal AtariGo state must have a winner (draws are not possible)"
                );
                return;
            }
            let mut actions = Vec::new();
            AtariGo::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = AtariGo::apply(state, &action);
        }
        panic!("AtariGo(n={n}) (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_atarigo_seeded_playouts_terminate() {
        for seed in 0..200 {
            seeded_random_play(6, seed);
        }
    }

    /// Same seeded-playout regression, but on a board size that spans
    /// multiple words (9x9 = 81 bits = 2 words), to prove the port to
    /// `bitboard::Board` didn't only work on the single-word case.
    #[test]
    fn test_atarigo_9x9_seeded_playouts_terminate() {
        for seed in 0..50 {
            seeded_random_play(9, seed);
        }
    }

    /// Exhaustively explore every reachable position from the empty 3x3
    /// board (small enough to enumerate fully) and check that every
    /// terminal position has a winner, every non-terminal position has a
    /// legal move, and the whole reachable state graph is finite -- i.e.
    /// there is no line of play that fails to terminate.
    #[test]
    fn test_atarigo_3x3_all_lines_terminate_with_a_winner() {
        let start = State::new(3);
        let mut seen: HashSet<State> = HashSet::new();
        let mut queue: VecDeque<State> = VecDeque::new();
        seen.insert(start.clone());
        queue.push_back(start);

        let mut explored = 0usize;
        while let Some(state) = queue.pop_front() {
            explored += 1;
            assert!(
                explored <= 200_000,
                "reachable-state graph is unexpectedly large -- possible non-termination"
            );

            if AtariGo::is_terminal(&state) {
                assert!(
                    AtariGo::winner(&state).is_some(),
                    "a terminal AtariGo state must have a winner (draws are not possible)"
                );
                continue;
            }

            let mut actions = Vec::new();
            AtariGo::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );

            for action in actions {
                let next = AtariGo::apply(state.clone(), &action);
                if seen.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Equivalence check: before retiring `check_go_move` as AtariGo's own
    // legality/capture path (now `GoEngine::check`/`play`, see `State::valid`/
    // `apply`), replay the same seeded-random-playout regression games above
    // and assert the engine-backed action set matches a `check_go_move`
    // oracle computed independently from the same board/turn at every ply.

    /// Old-path oracle: legal actions computed directly from `check_go_move`
    /// against a plain `black`/`white` pair, mirroring exactly what
    /// `State::valid`/`generate_actions` did before the `GoEngine` port.
    fn old_path_actions(black: Bits, white: Bits, turn: Player) -> Vec<Move> {
        let occupied = black | white;
        let (player, opponent) = match turn {
            Player::Black => (black, white),
            Player::White => (white, black),
        };
        let mut actions = Vec::new();
        for index in !occupied {
            let (valid, will_capture) = bitboard::check_go_move(player, opponent, index);
            if valid {
                actions.push(Move::new(index as u16, will_capture));
            }
        }
        if actions.is_empty() {
            actions.push(Move::NO_MOVE);
        }
        actions
    }

    fn seeded_random_play_matches_old_path(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = n * n + 2;

        for _ in 0..max_plies {
            if AtariGo::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            AtariGo::generate_actions(&state, &mut actions);
            let old_actions = old_path_actions(state.black(), state.white(), state.turn());
            assert_eq!(
                actions, old_actions,
                "engine-backed action set diverged from the check_go_move oracle at seed {seed}"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = AtariGo::apply(state, &action);
        }
        panic!("AtariGo(n={n}) (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_engine_backed_atarigo_matches_check_go_move_oracle() {
        for seed in 0..200 {
            seeded_random_play_matches_old_path(6, seed);
        }
        for seed in 0..50 {
            seeded_random_play_matches_old_path(9, seed);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // `random_action`'s rejection-sampling fast path must always agree with `generate_actions`'s
    // full enumeration: every draw is either `Move::NO_MOVE` when that's the only legal action, or
    // an action also present in `generate_actions`'s output.

    fn random_action_matches_generate_actions(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = n * n + 2;

        for _ in 0..max_plies {
            if AtariGo::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            AtariGo::generate_actions(&state, &mut actions);
            // Draw several times from the same state to exercise both the
            // rejection-sampling success path and (near the end of the
            // game, when legal placements are sparse) its full-enumeration
            // fallback.
            for _ in 0..8 {
                let drawn = AtariGo::random_action(&state, &mut rng).expect(
                    "random_action must return Some whenever generate_actions is non-empty",
                );
                assert!(
                    actions.contains(&drawn),
                    "random_action drew {drawn:?}, not present in generate_actions {actions:?}"
                );
            }
            let action = actions[rng.gen_range(0..actions.len())];
            state = AtariGo::apply(state, &action);
        }
        panic!("AtariGo(n={n}) (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_atarigo_random_action_matches_generate_actions() {
        for seed in 0..200 {
            random_action_matches_generate_actions(6, seed);
        }
        for seed in 0..50 {
            random_action_matches_generate_actions(9, seed);
        }
    }
}
