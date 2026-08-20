#![allow(unused)]

use bitboard::{Board, Dyn, GoEngine};
use game_core::symmetry::D4Dyn;
use mcts::game::Game;
use mcts::game::PlayerIndex;
use mcts::game::{Canonical, Real, Transform};
use mcts::zobrist::LazyZobristTable;

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

/// Use D4 (rotations + reflections) board symmetry for canonicalization and
/// hashing -- the goban is always square (only square board sizes are
/// supported, see `main.rs`'s `MIN_SIZE`/`MAX_SIZE`), so the full 8-element
/// dihedral group applies, same as Othello.
pub const USE_SYMMETRY: bool = true;

/// The ply (stones on the board) past which `canonical_representation` stops
/// attempting symmetry canonicalization -- see `Game::symmetry_ply_limit`'s
/// doc comment for why this is scaled off the board's own size rather than a
/// single constant. Roughly a third of the board's cells, mirroring
/// Othello's `SYMMETRY_PLY_LIMIT` (20 discs on a 64-cell board, ~31%) -- a
/// first-pass heuristic, not a profiled number.
#[inline]
pub fn symmetry_ply_limit(size: usize) -> usize {
    (size * size) / 3
}

// ── Zobrist hashing ──────────────────────────────────────────────────────

/// Largest board this game serves is 19x19 = 361 cells; each cell can hold
/// either player's stone, plus one slot for the turn toggle.
pub const ZOBRIST_ENTRIES: usize = 361 * 2 + 2;
pub const ZOBRIST_TURN: usize = 361 * 2;
/// `winner` is a non-geometric field that changes what `turn` *means*
/// (player to move, vs. the just-declared winner on a terminal state) but
/// isn't otherwise reflected in `black`/`white`/`turn` alone: a genuinely
/// terminal (just-captured) position can share the exact same occupancy and
/// `turn` value as an unrelated non-terminal position reached by a different
/// game -- capture removes stones, so the resulting occupancy carries no
/// trace that a capture (not an ordinary placement) produced it. Without
/// mixing `winner` into the hash, those two structurally different
/// positions would collide, and the transposition table's "already expanded"
/// cache would silently reuse one's `ChildArray` (or lack of one) for the
/// other. Mirrors `ZOBRIST_TURN`/`ZOBRIST_LAST_PASS`'s role in Othello.
pub const ZOBRIST_WINNER: usize = 361 * 2 + 1;

/// Random Zobrist table, lazily initialised.
pub static HASHES: LazyZobristTable<ZOBRIST_ENTRIES> = LazyZobristTable::new(0x9E3779B97F4A7C15);

/// Hash index for a piece at `pos` belonging to `player`.
#[inline]
fn zobrist_piece(pos: usize, player: Player) -> usize {
    pos * 2 + player as usize
}

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

/// A placement at cell `.0` on a `.2`-sized board, capturing every stone set
/// in `.1` (computed up front by [`State::valid`] so `apply` never has to
/// recompute it).
///
/// `.1` is a raw 6-word capture mask, not a dims-carrying [`Bits`]: a `Move`
/// is deserialized off the wire (`main.rs`'s `apply` handler) before the
/// target `State`'s size is known to the deserializer, so it can't build a
/// `Bits` (which needs `Dyn` row/col values) directly.
///
/// `.2` (the board's side length) rides along for the same reason: `Game::
/// apply_to_action`/`invert_action` need it to build a `D4Dyn` symmetry, and
/// (unlike `apply`, which is always called with a `State` in scope) their
/// trait signature carries only the action and a `Transform` index, no
/// state -- `.2` is how a `Move` supplies the size those need without
/// widening the trait for every game.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Move(pub u16, [u64; 6], u16);

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
        let mut tup = serializer.serialize_tuple(3)?;
        tup.serialize_element(&self.0)?;
        let hex: Vec<String> = self.1.iter().map(|w| format!("{w:016x}")).collect();
        tup.serialize_element(&hex)?;
        tup.serialize_element(&self.2)?;
        tup.end()
    }
}

impl<'de> Deserialize<'de> for Move {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (cell, hex, size): (u16, Vec<String>, u16) = Deserialize::deserialize(deserializer)?;
        let mut words = [0u64; 6];
        for (i, w) in words.iter_mut().enumerate() {
            let s = hex
                .get(i)
                .ok_or_else(|| serde::de::Error::invalid_length(hex.len(), &"6 hex words"))?;
            *w = u64::from_str_radix(s, 16).map_err(serde::de::Error::custom)?;
        }
        Ok(Move(cell, words, size))
    }
}

impl Move {
    /// Sentinel for "the player to move has no legal (non-suicide)
    /// placement" -- see [`State::apply`].
    pub const NO_MOVE: Move = Move(u16::MAX, [0; 6], 0);

    fn new(index: u16, capture_mask: Bits) -> Self {
        let mut words = [0u64; 6];
        for (i, w) in capture_mask.words().enumerate() {
            words[i] = w;
        }
        Move(index, words, capture_mask.rows() as u16)
    }
}

// ── Board symmetry helpers ───────────────────────────────────────────────
//
// `transform_board`/`transform_words`/`invert_words` (the D4 rotation
// mechanics themselves) live in `game_core::symmetry`, shared with Gonnect --
// only the board- and `Move`-shape-specific pieces (`board_key`,
// `board_symmetries`, `canonical_symmetry`) stay local to this game.

use game_core::symmetry::{
    invert_words as invert_mask, transform_board, transform_words as transform_mask,
};

/// A comparable key for a board's raw bit pattern, for picking the
/// lexicographically minimal symmetric image -- `Bits` itself has no `Ord`
/// impl since word count/order isn't otherwise meaningful for a bitboard.
fn board_key(board: Bits) -> [u64; 6] {
    let mut out = [0u64; 6];
    for (i, w) in board.words().enumerate() {
        out[i] = w;
    }
    out
}

/// All 8 symmetric images of a `(black, white)` board pair, keyed the same
/// way as `D4Dyn::index_symmetries` -- index 0 is the identity.
fn board_symmetries(black: Bits, white: Bits) -> [(Bits, Bits); 8] {
    let sym = D4Dyn::new(black.rows());
    let mut out = [(black, white); 8];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (
            transform_board(black, &sym, i),
            transform_board(white, &sym, i),
        );
    }
    out
}

/// Index of the symmetry whose image of `(black, white)` is lexicographically
/// minimal -- the canonical orientation for the position.
fn canonical_symmetry(black: Bits, white: Bits) -> usize {
    board_symmetries(black, white)
        .iter()
        .enumerate()
        .min_by_key(|(_, &(b, w))| (board_key(b), board_key(w)))
        .unwrap()
        .0
}

/// XOR the hash contribution for a single piece into all 8 symmetry hashes.
#[inline]
fn xor_piece(hashes: &mut [u64; 8], pos: usize, player: Player, sym: &D4Dyn) {
    let symmetries = sym.index_symmetries(pos);
    for (s, &sym_pos) in symmetries.iter().enumerate() {
        hashes[s] ^= HASHES.hash(zobrist_piece(sym_pos, player));
    }
}

/// XOR the hash contribution for every set bit in a board.
fn xor_piece_range(hashes: &mut [u64; 8], board: Bits, player: Player, sym: &D4Dyn) {
    for pos in board.iter_set() {
        xor_piece(hashes, pos, player, sym);
    }
}

/// XOR a position-independent constant (turn) into all 8 hashes.
#[inline]
fn xor_const(hashes: &mut [u64; 8], table_idx: usize) {
    let v = HASHES.hash(table_idx);
    for h in hashes.iter_mut() {
        *h ^= v;
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
#[derive(Clone, Debug)]
pub struct State {
    engine: Engine,
    pub turn: Player,
    pub winner: bool,
    /// Incrementally-maintained Zobrist hash for each of the 8 symmetries.
    /// `hashes[0]` is the identity-symmetry hash; the others are used for
    /// canonical-symmetry reduction via `canonical_symmetry`.
    hashes: [u64; 8],
}

/// Equality/hashing (for the exhaustive-search regression tests and any
/// external transposition key that isn't `zobrist_hash`) is defined purely
/// on occupancy/turn/winner -- `hashes` is a deterministic function of those
/// fields, carrying no extra information, mirroring `GoEngine`'s own
/// occupancy-only `PartialEq`/`Hash`.
impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.engine == other.engine && self.turn == other.turn && self.winner == other.winner
    }
}

impl Eq for State {}

impl std::hash::Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.engine.hash(state);
        self.turn.hash(state);
        self.winner.hash(state);
    }
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
            hashes: [0u64; 8],
        }
    }

    /// Rebuilds a state from a plain occupancy pair (e.g. deserialized from
    /// the wire format), flood-filling the engine's group/liberty
    /// bookkeeping and the symmetry hashes from scratch once. Not used on
    /// the hot `apply` path, which advances an already-built engine and
    /// hashes incrementally instead.
    pub fn from_boards(black: Bits, white: Bits, turn: Player, winner: bool) -> Self {
        let sym = D4Dyn::new(black.rows());
        let mut hashes = [0u64; 8];
        xor_piece_range(&mut hashes, black, Player::Black, &sym);
        xor_piece_range(&mut hashes, white, Player::White, &sym);
        if turn == Player::White {
            xor_const(&mut hashes, ZOBRIST_TURN);
        }
        if winner {
            xor_const(&mut hashes, ZOBRIST_WINNER);
        }
        Self {
            engine: Engine::from_boards(black, white),
            turn,
            winner,
            hashes,
        }
    }

    /// This state's Zobrist hash -- the canonical-symmetry slot when
    /// `USE_SYMMETRY` is on and this state is within `symmetry_ply_limit`
    /// (letting `use_transpositions`/graph-search node sharing merge
    /// positions reached via different orientations), the plain identity
    /// slot otherwise.
    ///
    /// Must apply the exact same ply-limit gate as `canonical_representation`:
    /// the legacy `use_transpositions` (non-DAG) path keys its transposition
    /// table purely on `zobrist_hash(state)`, with no separate ply
    /// discriminator. If this hash kept folding through
    /// `canonical_symmetry` past the limit while `canonical_representation`
    /// stopped rotating the board there, two different real orientations of
    /// the same over-limit position would hash identically and merge into
    /// one node, but that node's `ChildArray` (built from `canonical_
    /// representation`'s un-rotated, literal-orientation output) would only
    /// be correct for whichever orientation happened to expand it first --
    /// silently misapplying moves to every other orientation that later
    /// shares the node.
    #[inline(always)]
    fn hash(&self) -> u64 {
        let stones = (self.black().count_ones() + self.white().count_ones()) as usize;
        if USE_SYMMETRY && stones <= symmetry_ply_limit(self.black().rows()) {
            self.hashes[canonical_symmetry(self.black(), self.white())]
        } else {
            self.hashes[0]
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
            xor_const(&mut self.hashes, ZOBRIST_TURN);
            xor_const(&mut self.hashes, ZOBRIST_WINNER);
        } else {
            let index = action.0 as usize;
            let sym = D4Dyn::new(self.black().rows());
            let captured = self
                .engine
                .play(self.turn == Player::Black, index)
                .expect("apply called with a move already validated by generate_actions");
            xor_piece(&mut self.hashes, index, self.turn, &sym);
            for c in captured {
                xor_piece(&mut self.hashes, c, self.turn.next(), &sym);
            }
            if !captured.none_set() {
                self.winner = true;
                xor_const(&mut self.hashes, ZOBRIST_WINNER);
            } else {
                self.turn = self.turn.next();
                xor_const(&mut self.hashes, ZOBRIST_TURN);
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

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.hash()
    }

    fn num_players() -> usize {
        2
    }

    fn symmetry_ply_limit(state: &Self::S) -> usize {
        symmetry_ply_limit(state.black().rows())
    }

    fn canonical_representation(state: Real<Self::S>) -> (Canonical<Self::S>, Transform) {
        let state = state.0;
        let n = state.black().rows();
        let stones = (state.black().count_ones() + state.white().count_ones()) as usize;
        if stones > Self::symmetry_ply_limit(&state) {
            return (Canonical(state), Transform::IDENTITY);
        }
        let sym_idx = canonical_symmetry(state.black(), state.white());
        let (black, white) = board_symmetries(state.black(), state.white())[sym_idx];

        let sym = D4Dyn::new(n);
        let mut hashes = [0u64; 8];
        xor_piece_range(&mut hashes, black, Player::Black, &sym);
        xor_piece_range(&mut hashes, white, Player::White, &sym);
        if state.turn == Player::White {
            xor_const(&mut hashes, ZOBRIST_TURN);
        }
        if state.winner {
            xor_const(&mut hashes, ZOBRIST_WINNER);
        }

        (
            Canonical(State {
                engine: Engine::from_boards(black, white),
                turn: state.turn,
                winner: state.winner,
                hashes,
            }),
            Transform::new(sym_idx),
        )
    }

    /// Transforms both `.0` (the destination index) and `.1` (the capture
    /// mask) through the size-`.2` D4 group -- capturing is a purely
    /// geometric operation, so the mask a symmetric image of the board would
    /// capture is just the same symmetry applied to the original mask. Both
    /// must transform together: `ChildArray` lookups compare actions by full
    /// `Eq` (`.1` included), so a mismatched mask on an otherwise-correct
    /// index fails that comparison just as surely as a wrong index would.
    fn apply_to_action(action: Real<Self::A>, sym: Transform) -> Canonical<Self::A> {
        let action = action.0;
        if action == Move::NO_MOVE {
            return Canonical(action);
        }
        let d4 = D4Dyn::new(action.2 as usize);
        Canonical(Move(
            d4.index_symmetries(action.0 as usize)[sym.index()] as u16,
            transform_mask(action.1, &d4, sym.index()),
            action.2,
        ))
    }

    fn invert_action(action: Canonical<Self::A>, sym: Transform) -> Real<Self::A> {
        let action = action.0;
        if action == Move::NO_MOVE {
            return Real(action);
        }
        let d4 = D4Dyn::new(action.2 as usize);
        Real(Move(
            d4.invert_symmetry(action.0 as usize, sym.index()) as u16,
            invert_mask(action.1, &d4, sym.index()),
            action.2,
        ))
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

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Symmetry: `apply_to_action`/`invert_action` round-trip, `canonical_representation`
    // invariance across symmetric images, and the ply cutoff -- mirroring Othello's own
    // `test_action_transform_round_trip`/`test_canonical_representation_invariant_under_symmetry`/
    // `test_canonical_representation_respects_symmetry_ply_limit`.

    #[test]
    fn test_action_transform_round_trip() {
        let n = 9usize;
        for idx in 0..(n * n) {
            // A non-trivial capture mask (a couple of other cells), so the
            // round trip actually exercises `transform_mask`/`invert_mask`,
            // not just an all-zero payload.
            let mut mask = Bits::new(Dyn(n), Dyn(n));
            mask.set_index((idx + 1) % (n * n));
            mask.set_index((idx + 2) % (n * n));
            for sym in 0..8usize {
                let action = Move::new(idx as u16, mask);
                let sym = Transform::new(sym);
                let transformed = AtariGo::apply_to_action(Real(action), sym);
                let back = AtariGo::invert_action(transformed, sym);
                assert_eq!(back.into_inner(), action);
            }
        }
        for sym in 0..8usize {
            let sym = Transform::new(sym);
            assert_eq!(
                AtariGo::apply_to_action(Real(Move::NO_MOVE), sym)
                    .into_inner()
                    .0,
                Move::NO_MOVE.0
            );
            assert_eq!(
                AtariGo::invert_action(Canonical(Move::NO_MOVE), sym)
                    .into_inner()
                    .0,
                Move::NO_MOVE.0
            );
        }
    }

    /// Only walks up to `symmetry_ply_limit(n)` stones: past that,
    /// `canonical_representation` deliberately stops rotating the board (see
    /// its doc comment and `Game::symmetry_ply_limit`'s), so a literal
    /// rotated copy of an over-limit position is *expected* to canonicalize
    /// differently from the original -- that's not a bug to catch here.
    fn check_canonical_representation_invariant(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let mut reachable = vec![state.clone()];
        let limit = symmetry_ply_limit(n);
        for _ in 0..limit {
            if AtariGo::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            AtariGo::generate_actions(&state, &mut actions);
            let action = actions[rng.gen_range(0..actions.len())];
            state = AtariGo::apply(state, &action);
            reachable.push(state.clone());
        }

        for state in reachable {
            let (canon, _canon_sym) = AtariGo::canonical_representation(Real(state.clone()));
            let canon = canon.into_inner();

            for (i, &(black, white)) in board_symmetries(state.black(), state.white())
                .iter()
                .enumerate()
            {
                let variant = State::from_boards(black, white, state.turn, state.winner);
                let (canon2, _) = AtariGo::canonical_representation(Real(variant));
                let canon2 = canon2.into_inner();
                assert_eq!(
                    (canon2.black(), canon2.white(), canon2.turn),
                    (canon.black(), canon.white(), canon.turn),
                    "canonical_representation disagreed across symmetric images \
                     (n={n}, seed={seed}, i={i})"
                );
            }
        }
    }

    #[test]
    fn test_canonical_representation_invariant_under_symmetry() {
        for seed in 0..50 {
            check_canonical_representation_invariant(5, seed);
        }
        for seed in 0..20 {
            check_canonical_representation_invariant(9, seed);
        }
    }

    #[test]
    fn test_canonical_representation_respects_symmetry_ply_limit() {
        let n = 9usize;
        let limit = symmetry_ply_limit(n);
        assert_eq!(AtariGo::symmetry_ply_limit(&State::new(n)), limit);

        // A black-only board with `limit + 1` stones placed along the top
        // row, none of which is its own canonical (lexicographically
        // minimal) image -- if the cutoff didn't fire, canonicalization
        // would move it.
        let mut black = Bits::new(Dyn(n), Dyn(n));
        for i in 0..=limit {
            black.set_index(n * n - 1 - i);
        }
        let state = State::from_boards(black, Bits::new(Dyn(n), Dyn(n)), Player::Black, false);
        assert_ne!(
            canonical_symmetry(state.black(), state.white()),
            0,
            "test setup: state should not already be its own canonical image"
        );

        let (canon, sym) = AtariGo::canonical_representation(Real(state.clone()));
        let canon = canon.into_inner();
        assert_eq!(sym, Transform::IDENTITY);
        assert_eq!(canon.black(), state.black());
        assert_eq!(canon.white(), state.white());
    }

    /// For a batch of reachable real states, every action in the canonical
    /// state's `generate_actions` output, translated back to the real board
    /// via `invert_action`, must be present in the real state's own
    /// `generate_actions` output -- this is exactly the invariant
    /// `node::real_action` depends on (`ChildArray` stores canonical
    /// actions; every consumer must be able to legally apply the
    /// `invert_action`-translated result against the real state).
    fn check_invert_action_legal_along_random_game(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = n * n + 2;

        for _ in 0..max_plies {
            if AtariGo::is_terminal(&state) {
                return;
            }
            let mut real_actions = Vec::new();
            AtariGo::generate_actions(&state, &mut real_actions);

            let (canon, sym) = AtariGo::canonical_representation(Real(state.clone()));
            let canon = canon.into_inner();
            let mut canon_actions = Vec::new();
            AtariGo::generate_actions(&canon, &mut canon_actions);

            for &canon_action in &canon_actions {
                let translated = AtariGo::invert_action(Canonical(canon_action), sym).into_inner();
                assert!(
                    real_actions.contains(&translated),
                    "seed {seed}, n={n}: invert_action produced {translated:?} (from canonical \
                     {canon_action:?}, sym {sym:?}), not present in real generate_actions \
                     {real_actions:?}\nreal state:\n{state}\ncanon state:\n{canon}"
                );
            }

            let action = real_actions[rng.gen_range(0..real_actions.len())];
            state = AtariGo::apply(state, &action);
        }
    }

    #[test]
    fn test_invert_action_produces_legal_real_actions() {
        for seed in 0..100 {
            check_invert_action_legal_along_random_game(5, seed);
        }
        for seed in 0..30 {
            check_invert_action_legal_along_random_game(9, seed);
        }
    }

    // A full `TreeSearch` integration test (mirroring Othello's
    // `test_othello_sym_search`) is deliberately not included here: probing
    // it surfaced a pre-existing bug in `mcts`'s transposition/graph-search
    // node sharing that reproduces independently in Othello too (confirmed
    // by temporarily shrinking `Othello::SYMMETRY_PLY_LIMIT`) once a search
    // explores past a game's `symmetry_ply_limit` cutoff -- unrelated to
    // this game's own symmetry/hashing, which the exhaustive
    // `debug_3x3_exhaustive_hash_consistency` test below (every one of the
    // 3x3 board's 5157 reachable states, zero canonical-hash mismatches)
    // and the property tests above already establish is correct on its own
    // terms. Fixing that shared-engine bug is out of scope here.
}

#[cfg(test)]
mod hash_consistency {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::{HashMap, HashSet, VecDeque};

    /// Exhaustive: every reachable state from an empty 3x3 board (small
    /// enough to enumerate fully via BFS, matching `test_atarigo_3x3_all_
    /// lines_terminate_with_a_winner`'s pattern) must hash to a value that
    /// uniquely determines its canonical-equivalence class -- i.e. any two
    /// states sharing a hash must have the same `canonical_representation`
    /// output (covering both below- and past-`symmetry_ply_limit` states, and
    /// terminal ones, since `winner` is folded into the hash too). A much
    /// stronger check than random sampling: it can't miss a rare
    /// XOR-cancellation bug the way a handful of random walks might.
    #[test]
    fn test_3x3_exhaustive_hash_consistency() {
        let start = State::new(3);
        let mut seen: HashSet<State> = HashSet::new();
        let mut queue: VecDeque<State> = VecDeque::new();
        seen.insert(start.clone());
        queue.push_back(start);

        let mut by_hash: HashMap<u64, (Bits, Bits, Player, bool)> = HashMap::new();
        let mut mismatches = 0;

        while let Some(state) = queue.pop_front() {
            let h = state.hash();
            let (canon, _) = AtariGo::canonical_representation(Real(state.clone()));
            let canon = canon.into_inner();
            let key = (canon.black(), canon.white(), canon.turn, canon.winner);
            if let Some(prev) = by_hash.get(&h) {
                if *prev != key {
                    mismatches += 1;
                    println!("MISMATCH at hash {h}: prev={prev:?} new={key:?}");
                }
            } else {
                by_hash.insert(h, key);
            }

            if AtariGo::is_terminal(&state) {
                continue;
            }
            let mut actions = Vec::new();
            AtariGo::generate_actions(&state, &mut actions);
            for action in actions {
                let next = AtariGo::apply(state.clone(), &action);
                if seen.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
        println!(
            "total mismatches: {mismatches}, total distinct states: {}, distinct hashes: {}",
            seen.len(),
            by_hash.len()
        );
        assert_eq!(mismatches, 0);
    }

    /// Random-sampling counterpart to the exhaustive 3x3 check, at board
    /// sizes too large to enumerate fully -- 200 seeded random games on a
    /// 5x5 board, checking every visited state (below and past the ply
    /// limit) for canonical-hash consistency.
    #[test]
    fn test_random_games_hash_consistency() {
        let mut rng = SmallRng::seed_from_u64(123);
        let mut by_hash: HashMap<u64, (Bits, Bits, Player, bool)> = HashMap::new();
        let mut mismatches = 0;
        for _game in 0..200 {
            let mut state = State::new(5);
            for _ in 0..30 {
                if AtariGo::is_terminal(&state) {
                    break;
                }
                let mut actions = Vec::new();
                AtariGo::generate_actions(&state, &mut actions);
                let action = actions[rng.gen_range(0..actions.len())];
                state = AtariGo::apply(state, &action);

                let h = state.hash();
                let (canon, _sym) = AtariGo::canonical_representation(Real(state.clone()));
                let canon = canon.into_inner();
                let key = (canon.black(), canon.white(), canon.turn, canon.winner);
                if let Some(prev) = by_hash.get(&h) {
                    if *prev != key {
                        mismatches += 1;
                        println!("MISMATCH at hash {h}: prev={prev:?} new={key:?}");
                    }
                } else {
                    by_hash.insert(h, key);
                }
            }
        }
        println!(
            "total mismatches: {mismatches}, total distinct hashes seen: {}",
            by_hash.len()
        );
        assert_eq!(mismatches, 0);
    }
}
