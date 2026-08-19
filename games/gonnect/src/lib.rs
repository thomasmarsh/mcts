#![allow(unused)]

pub mod book;

use bitboard::{Board, Dyn, GoEngine};
use mcts::game::Game;
use mcts::game::PlayerIndex;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Board sizes 3x3 through 19x19 (`main.rs`'s `SUPPORTED_SIZES`) all fit in
/// 6 words (19*19 = 361 bits), so one `Board<[u64; 6], Dyn, Dyn>`
/// monomorphization serves every size, with the actual size carried as a
/// runtime field rather than a distinct compiled type per size.
pub type Bits = Board<[u64; 6], Dyn, Dyn>;
type Engine = GoEngine<[u64; 6], Dyn, Dyn>;

/// The board size `State::default()` builds -- Gonnect's traditional size,
/// matching `main.rs`'s `DEFAULT_SIZE`.
pub const DEFAULT_SIZE: usize = 13;

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
///
/// `.1` is a raw 6-word capture mask, not a dims-carrying [`Bits`]: a `Move`
/// is deserialized off the wire (`main.rs`'s `apply` handler) before the
/// target `State`'s size is known to the deserializer, so it can't build a
/// `Bits` (which needs `Dyn` row/col values) directly. Every caller that
/// needs the capture set as a real board already has a `State` in scope to
/// supply the size (see [`Move::capture_mask`]).
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Move(u16, [u64; 6]);

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
    pub const SWAP: Move = Move(u16::MAX, [0; 6]);
    pub const NO_MOVE: Move = Move(u16::MAX - 1, [0; 6]);

    fn new(index: u16, capture_mask: Bits) -> Self {
        let mut words = [0u64; 6];
        for (i, w) in capture_mask.words().enumerate() {
            words[i] = w;
        }
        Move(index, words)
    }

    pub fn index(&self) -> u16 {
        self.0
    }

    /// Rebuilds the capture mask as a real [`Bits`] board, sized to match
    /// `state` -- see this type's own doc comment for why the raw words
    /// can't be turned back into a `Bits` without an existing state to take
    /// dims from.
    pub fn capture_mask(&self, state: &State) -> Bits {
        let n = state.black().rows();
        let mut mask = Bits::new(Dyn(n), Dyn(n));
        for (w, &word) in self.1.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                mask.set_index(w * 64 + bit);
            }
        }
        mask
    }
}

/// Not `Copy`: `Engine`'s group/liberty bookkeeping is `Vec`-backed (a
/// `Dyn`-dimensioned board has no compile-time cell count to size a fixed
/// array with -- see `bitboard::go::GoEngine`'s doc comment), so `apply`
/// clones rather than implicitly copies.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct State {
    engine: Engine,
    ko_black: Bits,
    ko_white: Bits,
    turn: Player,
    can_swap: bool,
    winner: bool,
}

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_SIZE)
    }
}

impl State {
    /// A fresh empty `size x size` board.
    pub fn new(size: usize) -> Self {
        let ones = !Bits::new(Dyn(size), Dyn(size));
        Self {
            engine: Engine::new(Dyn(size), Dyn(size)),
            ko_black: ones,
            ko_white: ones,
            turn: Player::default(),
            can_swap: true,
            winner: false,
        }
    }

    #[inline(always)]
    pub fn from_parts(
        black: Bits,
        white: Bits,
        ko_black: Bits,
        ko_white: Bits,
        turn: Player,
        can_swap: bool,
        winner: bool,
    ) -> Self {
        Self {
            engine: Engine::from_boards(black, white),
            ko_black,
            ko_white,
            turn,
            can_swap,
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

    /// The `black()`/`white()` snapshot from just before the most recent
    /// placement -- what [`is_ko`](Self::is_ko) compares the position after
    /// a candidate move against to enforce the simple-ko rule. Exposed for
    /// callers (`main.rs`'s wire format) that need to round-trip full state,
    /// not just the board a renderer displays.
    #[inline(always)]
    pub fn ko_black(&self) -> Bits {
        self.ko_black
    }

    #[inline(always)]
    pub fn ko_white(&self) -> Bits {
        self.ko_white
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
    pub fn can_swap(&self) -> bool {
        self.can_swap
    }

    #[inline(always)]
    fn occupied(&self) -> Bits {
        self.black() | self.white()
    }

    #[inline(always)]
    fn player(&self, player: Player) -> Bits {
        match player {
            Player::Black => self.black(),
            Player::White => self.white(),
        }
    }

    #[inline(always)]
    fn player_ko(&self, player: Player) -> Bits {
        match player {
            Player::Black => self.ko_black,
            Player::White => self.ko_white,
        }
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

    /// A board shaped like this state's, with only `index` set -- the
    /// `Dyn`-dims counterpart of a `from_index` static constructor, which
    /// only exists for `Const` dims (no instance to take `rows`/`cols`
    /// from).
    #[inline]
    fn seed(&self, index: usize) -> Bits {
        let mut b = self.black().empty_like();
        b.set_index(index);
        b
    }

    #[inline]
    fn is_ko(&self, index: usize, will_capture: Bits) -> bool {
        let player = self.player(self.turn) | self.seed(index);
        let opponent = self.player(self.turn.next()) & !will_capture;
        let player_ko = self.player_ko(self.turn);
        let opponent_ko = self.player_ko(self.turn.next());
        player_ko == player && opponent_ko == opponent
    }

    #[inline]
    fn valid(&self, index: usize) -> (bool, Bits) {
        self.engine.check(self.turn == Player::Black, index)
    }

    #[inline]
    fn apply(&mut self, action: &Move) -> Self {
        if *action == Move::NO_MOVE {
            // The player to move has no legal move and loses; the opponent
            // wins (Gonnect's official rule: "A player loses if he has no
            // legal move").
            self.winner = true;
            self.turn = self.turn.next();
        } else if *action == Move::SWAP {
            let engine = Engine::from_boards(self.white(), self.black());
            self.engine = engine;
            self.can_swap = false;
        } else {
            let index = action.0 as usize;
            debug_assert!(!self.occupied().get_index(index));
            self.ko_black = self.black();
            self.ko_white = self.white();
            let player = self.player(self.turn) | self.seed(index);
            self.engine
                .play(self.turn == Player::Black, index)
                .expect("apply called with a move already validated by generate_actions");
            if player.has_opposite_connection4(index) {
                self.winner = true;
            }
        }
        // The swap window is open for exactly one reply: the move that
        // brings `occupied` to 1 (Black's opening placement) leaves it open
        // for White; any move after that -- an ordinary placement (occupied
        // now != 1) or an explicit `SWAP` (already cleared above) -- closes
        // it for good, so a later capture that happens to drop `occupied`
        // back to 1 can't reopen it.
        if self.can_swap && self.occupied().count_ones() != 1 {
            self.can_swap = false;
        }
        if !self.winner {
            self.turn = self.turn.next();
        }

        self.clone()
    }
}

// Zobrist hashing for Gonnect is harder because of the repetition of the ko rule. A solution
// would be to use Zobrist path hashing.
#[derive(Clone)]
pub struct Gonnect;

impl Game for Gonnect {
    type S = State;
    type A = Move;
    type P = Player;

    fn apply(mut state: State, action: &Move) -> State {
        state.apply(action)
    }

    fn generate_actions(state: &State, actions: &mut Vec<Move>) {
        if state.can_swap && state.occupied().count_ones() == 1 {
            actions.push(Move::SWAP);
        }
        for index in !state.occupied() {
            let (valid, will_capture) = state.valid(index);
            if valid && !state.is_ko(index, will_capture) {
                actions.push(Move::new(index as u16, will_capture))
            }
        }
        if actions.is_empty() {
            actions.push(Move::NO_MOVE);
        }
    }

    /// Rejection-sampling fast path for `SimulateStrategy::playout`'s
    /// uniform rollouts -- same idea as `AtariGo::random_action`: draw a
    /// random cell and run the `GoEngine`-backed `State::valid`/`is_ko`
    /// checks on just that one cell instead of probing every empty cell via
    /// `generate_actions`, falling back to the full enumeration once
    /// `max_attempts` misses in a row (bounds cost when legal placements are
    /// sparse, and is also what correctly proves "no legal move").
    ///
    /// The swap-eligible state (exactly one stone on the board) is left to
    /// the `generate_actions` fallback unconditionally rather than folded
    /// into rejection sampling: it's already a single-stone board (cheap to
    /// enumerate), and giving `SWAP` its correct uniform weight against an
    /// a-priori-unknown count of legal placements isn't something rejection
    /// sampling over cells alone can do.
    fn random_action(state: &State, rng: &mut rand::rngs::SmallRng) -> Option<Move> {
        use rand::Rng;
        if state.can_swap && state.occupied().count_ones() == 1 {
            let mut actions = Vec::new();
            Self::generate_actions(state, &mut actions);
            return Some(actions[rng.gen_range(0..actions.len())]);
        }
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
            if valid && !state.is_ko(index, will_capture) {
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

    fn parse_action(state: &State, input: &str) -> Option<Self::A> {
        let n = state.black().rows();
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
            if col < n {
                if let Ok(row) = chars
                    .collect::<String>()
                    .trim()
                    .parse::<usize>()
                    .map(|x| x - 1)
                {
                    if row < n {
                        let index = row * n + col;
                        let (valid, will_capture) = state.valid(index);
                        let is_ko = state.is_ko(index, will_capture);
                        if valid && !is_ko {
                            return Some(Move::new(index as u16, will_capture));
                        } else {
                            eprintln!("invalid placement: (valid={valid}, is_ko={is_ko})");
                        }
                    } else {
                        eprintln!("row out of range: {row} must be >= 1 and <= {n}");
                    }
                }
            } else {
                eprintln!("col out of range: {col} must be >= 1 and <= {n}");
            }
        }
        None
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        if *action == Move::SWAP {
            "swap".into()
        } else {
            const COL_NAMES: &[u8] = b"ABCDEFGHIJKLMNOPQRST";
            let n = state.black().cols();
            let (row, col) = (action.0 as usize / n, action.0 as usize % n);
            format!("{}{}", COL_NAMES[col] as char, row + 1)
        }
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
impl mcts::strategies::mcts::render::NodeRender for State {}

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
    fn seeded_random_play(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = n * n * 8 + 32;

        for _ in 0..max_plies {
            if Gonnect::is_terminal(&state) {
                assert!(
                    Gonnect::winner(&state).is_some(),
                    "a terminal Gonnect state must have a winner (draws are not possible)"
                );
                return;
            }
            let mut actions = Vec::new();
            Gonnect::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = Gonnect::apply(state, &action);
        }
        panic!("Gonnect(n={n}) (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_gonnect_seeded_playouts_terminate() {
        for seed in 0..200 {
            seeded_random_play(5, seed);
        }
    }

    /// Same seeded-playout regression, but on a board size that spans
    /// multiple words (9x9 = 81 bits = 2 words), to prove the port to
    /// `bitboard::Board` didn't only work on the single-word case.
    #[test]
    fn test_gonnect_9x9_seeded_playouts_terminate() {
        for seed in 0..30 {
            seeded_random_play(9, seed);
        }
    }

    /// Exhaustively explore every reachable position from the empty 3x3
    /// board (small enough to enumerate fully) and check that every
    /// terminal position has a winner, every non-terminal position has a
    /// legal move, and the whole reachable state graph is finite -- i.e.
    /// there is no line of play that fails to terminate.
    #[test]
    fn test_gonnect_3x3_all_lines_terminate_with_a_winner() {
        let start = State::new(3);
        let mut seen: HashSet<State> = HashSet::new();
        let mut queue: VecDeque<State> = VecDeque::new();
        seen.insert(start.clone());
        queue.push_back(start);

        let mut explored = 0usize;
        while let Some(state) = queue.pop_front() {
            explored += 1;
            assert!(
                explored <= 500_000,
                "reachable-state graph is unexpectedly large -- possible non-termination"
            );

            if Gonnect::is_terminal(&state) {
                assert!(
                    Gonnect::winner(&state).is_some(),
                    "a terminal Gonnect state must have a winner (draws are not possible)"
                );
                continue;
            }

            let mut actions = Vec::new();
            Gonnect::generate_actions(&state, &mut actions);
            assert!(
                !actions.is_empty(),
                "generate_actions must never be empty for a non-terminal state"
            );

            for action in actions {
                let next = Gonnect::apply(state.clone(), &action);
                if seen.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Equivalence check: before retiring `check_go_move` as Gonnect's own
    // legality/capture path (now `GoEngine::check`/`play`, see `State::valid`/
    // `apply`), replay the same seeded-random-playout regression games above
    // and assert the engine-backed action set matches a `check_go_move`
    // oracle computed independently from the same board/turn/ko state at
    // every ply.

    /// Old-path oracle: legal actions computed directly from `check_go_move`
    /// plus the ko check, against a plain `black`/`white`/`ko_black`/
    /// `ko_white` set of boards, mirroring exactly what `State::valid`/
    /// `is_ko`/`generate_actions` did before the `GoEngine` port.
    fn old_path_actions(
        black: Bits,
        white: Bits,
        ko_black: Bits,
        ko_white: Bits,
        turn: Player,
        can_swap: bool,
    ) -> Vec<Move> {
        let occupied = black | white;
        let (player, opponent) = match turn {
            Player::Black => (black, white),
            Player::White => (white, black),
        };
        let (player_ko, opponent_ko) = match turn {
            Player::Black => (ko_black, ko_white),
            Player::White => (ko_white, ko_black),
        };
        let mut actions = Vec::new();
        if can_swap && occupied.count_ones() == 1 {
            actions.push(Move::SWAP);
        }
        for index in !occupied {
            let (valid, will_capture) = bitboard::check_go_move(player, opponent, index);
            if !valid {
                continue;
            }
            let mut seed = player.empty_like();
            seed.set_index(index);
            let would_be_player = player | seed;
            let would_be_opponent = opponent & !will_capture;
            let is_ko = player_ko == would_be_player && opponent_ko == would_be_opponent;
            if !is_ko {
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
        let max_plies = n * n * 8 + 32;

        for _ in 0..max_plies {
            if Gonnect::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            Gonnect::generate_actions(&state, &mut actions);
            let old_actions = old_path_actions(
                state.black(),
                state.white(),
                state.ko_black,
                state.ko_white,
                state.turn(),
                state.can_swap,
            );
            assert_eq!(
                actions, old_actions,
                "engine-backed action set diverged from the check_go_move oracle at seed {seed}"
            );
            let action = actions[rng.gen_range(0..actions.len())];
            state = Gonnect::apply(state, &action);
        }
        panic!("Gonnect(n={n}) (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_engine_backed_gonnect_matches_check_go_move_oracle() {
        for seed in 0..200 {
            seeded_random_play_matches_old_path(5, seed);
        }
        for seed in 0..30 {
            seeded_random_play_matches_old_path(9, seed);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // `random_action`'s rejection-sampling fast path must always agree with `generate_actions`'s
    // full enumeration: every draw is either `Move::NO_MOVE` when that's the only legal action, or
    // an action also present in `generate_actions`'s output (`SWAP` included, since that state is
    // left to the `generate_actions` fallback unconditionally).

    fn random_action_matches_generate_actions(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = n * n * 8 + 32;

        for _ in 0..max_plies {
            if Gonnect::is_terminal(&state) {
                return;
            }
            let mut actions = Vec::new();
            Gonnect::generate_actions(&state, &mut actions);
            // Draw several times from the same state to exercise both the
            // rejection-sampling success path and (near the end of the
            // game, when legal placements are sparse) its full-enumeration
            // fallback.
            for _ in 0..8 {
                let drawn = Gonnect::random_action(&state, &mut rng).expect(
                    "random_action must return Some whenever generate_actions is non-empty",
                );
                assert!(
                    actions.contains(&drawn),
                    "random_action drew {drawn:?}, not present in generate_actions {actions:?}"
                );
            }
            let action = actions[rng.gen_range(0..actions.len())];
            state = Gonnect::apply(state, &action);
        }
        panic!("Gonnect(n={n}) (seed {seed}) did not terminate within {max_plies} plies");
    }

    #[test]
    fn test_gonnect_random_action_matches_generate_actions() {
        for seed in 0..200 {
            random_action_matches_generate_actions(5, seed);
        }
        for seed in 0..30 {
            random_action_matches_generate_actions(9, seed);
        }
    }

    #[test]
    #[ignore = "flaky: unseeded MCTS playouts occasionally run for many minutes before a \
                connection forms -- observed hanging under a full-workspace `cargo test` run; \
                test_gonnect_seeded_playouts_terminate covers the same termination concern \
                deterministically"]
    fn test_gonnect_render() {
        let mut search = TreeSearch::<Gonnect, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(false)
                .max_iterations(20),
        );
        _ = search.choose_action(&State::new(3));
        render::render(&search);
    }
}
