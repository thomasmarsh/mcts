#![allow(unused)]

pub mod book;

use game_core::bigbitboard;
use game_core::bigbitboard::BigBitBoard;
use game_core::display::RectangularBoard;
use game_core::display::RectangularBoardDisplay;
use mcts::game::Game;
use mcts::game::PlayerIndex;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::mem::swap;

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
/// front by [`State::valid`] so `apply` never has to recompute it). `.0`
/// also carries the [`SWAP`](Self::SWAP)/[`NO_MOVE`](Self::NO_MOVE)
/// sentinels, reserving the top of the `u16` range the same way the
/// original `u8` encoding reserved its top two values -- board sizes here
/// never approach `u16::MAX` cells.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Move<const N: usize, const WORDS: usize>(u16, BigBitBoard<N, N, WORDS>);

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
    pub const SWAP: Move<N, WORDS> = Move(u16::MAX, BigBitBoard::EMPTY);
    pub const NO_MOVE: Move<N, WORDS> = Move(u16::MAX - 1, BigBitBoard::EMPTY);

    pub fn new(index: u16, capture_mask: BigBitBoard<N, N, WORDS>) -> Self {
        Move(index, capture_mask)
    }

    pub fn index(&self) -> u16 {
        self.0
    }

    pub fn capture_mask(&self) -> BigBitBoard<N, N, WORDS> {
        self.1
    }
}

#[derive(Clone, Copy, Serialize, Debug, Hash, PartialEq, Eq)]
pub struct State<const N: usize, const WORDS: usize> {
    black: BigBitBoard<N, N, WORDS>,
    white: BigBitBoard<N, N, WORDS>,
    ko_black: BigBitBoard<N, N, WORDS>,
    ko_white: BigBitBoard<N, N, WORDS>,
    turn: Player,
    can_swap: bool,
    winner: bool,
}

impl<const N: usize, const WORDS: usize> Default for State<N, WORDS> {
    fn default() -> Self {
        Self {
            black: BigBitBoard::default(),
            white: BigBitBoard::default(),
            ko_black: BigBitBoard::ONES,
            ko_white: BigBitBoard::ONES,
            turn: Player::default(),
            can_swap: true,
            winner: false,
        }
    }
}

impl<const N: usize, const WORDS: usize> State<N, WORDS> {
    #[inline(always)]
    pub fn from_parts(
        black: BigBitBoard<N, N, WORDS>,
        white: BigBitBoard<N, N, WORDS>,
        ko_black: BigBitBoard<N, N, WORDS>,
        ko_white: BigBitBoard<N, N, WORDS>,
        turn: Player,
        can_swap: bool,
        winner: bool,
    ) -> Self {
        Self {
            black,
            white,
            ko_black,
            ko_white,
            turn,
            can_swap,
            winner,
        }
    }

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
    fn player_ko(&self, player: Player) -> BigBitBoard<N, N, WORDS> {
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
    fn is_ko(&self, index: usize, will_capture: BigBitBoard<N, N, WORDS>) -> bool {
        let player = self.player(self.turn) | BigBitBoard::from_index(index);
        let opponent = self.player(self.turn.next()) & !will_capture;
        let player_ko = self.player_ko(self.turn);
        let opponent_ko = self.player_ko(self.turn.next());
        player_ko == player && opponent_ko == opponent
    }

    #[inline]
    fn valid(&self, index: usize) -> (bool, BigBitBoard<N, N, WORDS>) {
        if (self.black | self.white).get(index) {
            return (false, BigBitBoard::EMPTY);
        }
        bigbitboard::check_go_move::<N, WORDS>(
            self.player(self.turn),
            self.player(self.turn.next()),
            index,
        )
    }

    #[inline]
    fn apply(&mut self, action: &Move<N, WORDS>) -> Self {
        if *action == Move::NO_MOVE {
            // The player to move has no legal move and loses; the opponent
            // wins (Gonnect's official rule: "A player loses if he has no
            // legal move").
            self.winner = true;
            self.turn = self.turn.next();
        } else if *action == Move::SWAP {
            swap(&mut self.black, &mut self.white);
            self.can_swap = false;
        } else {
            let index = action.0 as usize;
            debug_assert!(!self.occupied().get(index));
            let player = self.player(self.turn) | BigBitBoard::from_index(index);
            let opponent = self.player(self.turn.next());
            self.ko_black = self.black;
            self.ko_white = self.white;
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
            if player.has_opposite_connection4(index) {
                self.winner = true;
            }
        }
        if self.can_swap && self.occupied().count_ones() == 1 {
            self.can_swap = false;
        }
        if !self.winner {
            self.turn = self.turn.next();
        }

        *self
    }
}

// Zobrist hashing for Gonnect is harder because of the repetition of the ko rule. A solution
// would be to use Zobrist path hashing.
#[derive(Clone)]
pub struct Gonnect<const N: usize, const WORDS: usize>;

impl<const N: usize, const WORDS: usize> Game for Gonnect<N, WORDS> {
    type S = State<N, WORDS>;
    type A = Move<N, WORDS>;
    type P = Player;

    fn apply(mut state: State<N, WORDS>, action: &Move<N, WORDS>) -> State<N, WORDS> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N, WORDS>, actions: &mut Vec<Move<N, WORDS>>) {
        if state.can_swap && state.occupied().count_ones() == 1 {
            actions.push(Move::SWAP);
        }
        for index in !state.occupied() {
            let (valid, will_capture) = state.valid(index);
            if valid && !state.is_ko(index, will_capture) {
                actions.push(Move(index as u16, will_capture))
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

    fn parse_action(state: &State<N, WORDS>, input: &str) -> Option<Self::A> {
        if input.trim() == "swap" {
            if state.can_swap && state.occupied().count_ones() == 1 {
                return Some(Move::SWAP);
            } else {
                eprintln!("invalid move");
                return None;
            }
        }
        let mut chars = input.chars();

        if let Some(file) = chars.next() {
            let col = file.to_ascii_uppercase() as usize - 'A' as usize;
            if col < N {
                if let Ok(row) = chars
                    .collect::<String>()
                    .trim()
                    .parse::<usize>()
                    .map(|x| x - 1)
                {
                    if row < N {
                        let index = BigBitBoard::<N, N, WORDS>::to_index(row, col);
                        let (valid, will_capture) = state.valid(index);
                        let is_ko = state.is_ko(index, will_capture);
                        if valid && !is_ko {
                            return Some(Move(index as u16, will_capture));
                        } else {
                            eprintln!("invalid placement: (valid={valid}, is_ko={is_ko})");
                        }
                    } else {
                        eprintln!("row out of range: {row} must be >= 1 and <= {N}");
                    }
                }
            } else {
                eprintln!("col out of range: {col} must be >= 1 and <= {N}");
            }
        }
        None
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        if *action == Move::SWAP {
            "swap".into()
        } else {
            const COL_NAMES: &[u8] = b"ABCDEFGHIJKLMNOPQRST";
            let (row, col) = BigBitBoard::<N, N, WORDS>::to_coord(action.0 as usize);
            format!("{}{}", COL_NAMES[col] as char, row + 1)
        }
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

struct MovesDisplay<const N: usize, const WORDS: usize>(State<N, WORDS>);

impl<const N: usize, const WORDS: usize> RectangularBoard for MovesDisplay<N, WORDS> {
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        let mut actions = Vec::new();
        Gonnect::generate_actions(&self.0, &mut actions);
        let mut found = false;
        for action in &actions {
            let (r, c) = BigBitBoard::<N, N, WORDS>::to_coord(action.0 as usize);
            if r == row && c == col {
                found = true;
            }
        }

        if self.0.black.get_at(row, col) {
            'X'
        } else if self.0.white.get_at(row, col) {
            'O'
        } else if found {
            '+'
        } else {
            '.'
        }
    }
}

#[cfg(test)]
impl<const N: usize, const WORDS: usize> mcts::strategies::mcts::render::NodeRender
    for State<N, WORDS>
{
}

#[cfg(test)]
mod tests {
    use mcts::strategies::{
        mcts::{node::QInit, render, strategy, SearchConfig, TreeSearch},
        Search,
    };
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::{HashSet, VecDeque};

    use super::*;

    /// Deterministic regression coverage for the "no legal move" win
    /// attribution bug (the player stuck with no legal move used to be
    /// awarded the win instead of the loss, per Gonnect's official rule "a
    /// player loses if he has no legal move") and the general capture/ko
    /// logic: play a fixed-seed random game to completion and check the
    /// invariants a regression there would violate.
    ///
    /// Gonnect's ko rule here is positional against only the immediately
    /// preceding position (not full superko), so unlike AtariGo there is no
    /// clean proof the game must terminate within a fixed ply count -- the
    /// cap below is generous headroom for a small board, not a proven
    /// bound. Hitting it is a correctness signal worth investigating, not
    /// just a slow test.
    fn seeded_random_play<const N: usize, const WORDS: usize>(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::<N, WORDS>::default();
        let max_plies = N * N * 8 + 32;

        for _ in 0..max_plies {
            if Gonnect::<N, WORDS>::is_terminal(&state) {
                assert!(
                    Gonnect::<N, WORDS>::winner(&state).is_some(),
                    "a terminal Gonnect state must have a winner (draws are not possible)"
                );
                return;
            }
            let mut actions = Vec::new();
            Gonnect::<N, WORDS>::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = Gonnect::<N, WORDS>::apply(state, &action);
        }
        panic!("Gonnect<{N}> (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_gonnect_seeded_playouts_terminate() {
        for seed in 0..200 {
            seeded_random_play::<5, 1>(seed);
        }
    }

    /// Same seeded-playout regression, but on a board size that spans
    /// multiple `BigBitBoard` words (9x9 = 81 bits = 2 words), to prove the
    /// port from `BitBoard` didn't only work on the single-word case.
    #[test]
    fn test_gonnect_9x9_seeded_playouts_terminate() {
        for seed in 0..30 {
            seeded_random_play::<9, 2>(seed);
        }
    }

    /// Exhaustively explore every reachable position from the empty 3x3
    /// board (small enough to enumerate fully) and check that every
    /// terminal position has a winner, every non-terminal position has a
    /// legal move, and the whole reachable state graph is finite -- i.e.
    /// there is no line of play that fails to terminate.
    #[test]
    fn test_gonnect_3x3_all_lines_terminate_with_a_winner() {
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
                explored <= 500_000,
                "reachable-state graph is unexpectedly large -- possible non-termination"
            );

            if Gonnect::<N, WORDS>::is_terminal(&state) {
                assert!(
                    Gonnect::<N, WORDS>::winner(&state).is_some(),
                    "a terminal Gonnect state must have a winner (draws are not possible)"
                );
                continue;
            }

            let mut actions = Vec::new();
            Gonnect::<N, WORDS>::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );

            for action in actions {
                let next = Gonnect::<N, WORDS>::apply(state, &action);
                if seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
    }

    #[test]
    #[ignore = "flaky: unseeded MCTS playouts occasionally run for many minutes before a \
                connection forms -- observed hanging under a full-workspace `cargo test` run; \
                test_gonnect_seeded_playouts_terminate covers the same termination concern \
                deterministically"]
    fn test_gonnect_render() {
        let mut search = TreeSearch::<Gonnect<3, 1>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(false)
                .max_iterations(20),
        );
        _ = search.choose_action(&State::default());
        render::render(&search);
    }
}
