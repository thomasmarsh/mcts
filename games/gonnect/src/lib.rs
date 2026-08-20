#![allow(unused)]

pub mod book;

use bitboard::{Board, Dyn, GoEngine};
use game_core::symmetry::{invert_words, transform_board, transform_words, D4Dyn};
use mcts::game::Game;
use mcts::game::PlayerIndex;
use mcts::game::{Canonical, Real, Transform};
use mcts::zobrist::LazyZobristTable;

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

/// Use D4 (rotations + reflections) board symmetry for canonicalization and
/// hashing -- the goban is always square (only square board sizes are
/// supported, see `main.rs`'s `SUPPORTED_SIZES`), so the full 8-element
/// dihedral group applies, same as Othello/AtariGo.
pub const USE_SYMMETRY: bool = true;

/// The ply (stones on the board) past which `canonical_representation` stops
/// attempting symmetry canonicalization -- see `Game::symmetry_ply_limit`'s
/// doc comment for why this is scaled off the board's own size rather than a
/// single constant. Roughly a third of the board's cells, mirroring
/// AtariGo's/Othello's own heuristics -- a first-pass number, not profiled.
#[inline]
pub fn symmetry_ply_limit(size: usize) -> usize {
    (size * size) / 3
}

// ── Zobrist hashing ──────────────────────────────────────────────────────

/// A cell can independently belong to any of 4 boards: `black`, `white`, and
/// the ko snapshots `ko_black`/`ko_white` -- see `State`'s doc comment for
/// why the ko boards are geometric (must rotate with the rest) rather than
/// derived from `black`/`white` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    Black = 0,
    White = 1,
    KoBlack = 2,
    KoWhite = 3,
}

/// Largest board this game serves is 19x19 = 361 cells, 4 channels per cell,
/// plus one slot each for `turn`, `can_swap`, and `winner`.
pub const ZOBRIST_ENTRIES: usize = 361 * 4 + 3;
pub const ZOBRIST_TURN: usize = 361 * 4;
pub const ZOBRIST_CAN_SWAP: usize = 361 * 4 + 1;
/// `winner` is a non-geometric field that changes what `turn` *means*
/// (player to move, vs. the just-declared winner on a terminal state) but
/// isn't otherwise reflected in the board channels alone -- see AtariGo's
/// identical `ZOBRIST_WINNER` doc comment for why omitting this causes real
/// hash collisions between unrelated terminal/non-terminal positions.
pub const ZOBRIST_WINNER: usize = 361 * 4 + 2;

/// Random Zobrist table, lazily initialised.
pub static HASHES: LazyZobristTable<ZOBRIST_ENTRIES> = LazyZobristTable::new(0xB5297A4D3C1E9F27);

/// Hash index for a cell at `pos` on channel `channel`.
#[inline]
fn zobrist_cell(pos: usize, channel: Channel) -> usize {
    pos * 4 + channel as usize
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
///
/// `.2` (the board's side length) rides along for the same reason: `Game::
/// apply_to_action`/`invert_action` need it to build a `D4Dyn` symmetry, and
/// (unlike `apply`, which is always called with a `State` in scope) their
/// trait signature carries only the action and a `Transform` index, no
/// state -- `.2` is how a `Move` supplies the size those need without
/// widening the trait for every game.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Move(u16, [u64; 6], u16);

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
    pub const SWAP: Move = Move(u16::MAX, [0; 6], 0);
    pub const NO_MOVE: Move = Move(u16::MAX - 1, [0; 6], 0);

    fn new(index: u16, capture_mask: Bits) -> Self {
        let mut words = [0u64; 6];
        for (i, w) in capture_mask.words().enumerate() {
            words[i] = w;
        }
        Move(index, words, capture_mask.rows() as u16)
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

// ── Board symmetry / Zobrist helpers ─────────────────────────────────────

/// XOR the hash contribution for a single cell into all 8 symmetry hashes.
#[inline]
fn xor_cell(hashes: &mut [u64; 8], pos: usize, channel: Channel, sym: &D4Dyn) {
    let symmetries = sym.index_symmetries(pos);
    for (s, &sym_pos) in symmetries.iter().enumerate() {
        hashes[s] ^= HASHES.hash(zobrist_cell(sym_pos, channel));
    }
}

/// XOR the hash contribution for every set bit of `board` on `channel`.
fn xor_board(hashes: &mut [u64; 8], board: Bits, channel: Channel, sym: &D4Dyn) {
    for pos in board.iter_set() {
        xor_cell(hashes, pos, channel, sym);
    }
}

/// XOR a position-independent constant (turn, can_swap, winner) into all 8
/// hashes.
#[inline]
fn xor_const(hashes: &mut [u64; 8], table_idx: usize) {
    let v = HASHES.hash(table_idx);
    for h in hashes.iter_mut() {
        *h ^= v;
    }
}

/// Rebuilds all 8 symmetry hashes from scratch -- the counterpart to
/// AtariGo's/Othello's `from_position`/`from_boards`, used whenever
/// incremental maintenance isn't worth the bookkeeping (deserializing off
/// the wire, `Game::canonical_representation`, the rare `SWAP` move).
fn rebuild_hashes(
    black: Bits,
    white: Bits,
    ko_black: Bits,
    ko_white: Bits,
    turn: Player,
    can_swap: bool,
    winner: bool,
) -> [u64; 8] {
    let sym = D4Dyn::new(black.rows());
    let mut hashes = [0u64; 8];
    xor_board(&mut hashes, black, Channel::Black, &sym);
    xor_board(&mut hashes, white, Channel::White, &sym);
    xor_board(&mut hashes, ko_black, Channel::KoBlack, &sym);
    xor_board(&mut hashes, ko_white, Channel::KoWhite, &sym);
    if turn == Player::White {
        xor_const(&mut hashes, ZOBRIST_TURN);
    }
    if can_swap {
        xor_const(&mut hashes, ZOBRIST_CAN_SWAP);
    }
    if winner {
        xor_const(&mut hashes, ZOBRIST_WINNER);
    }
    hashes
}

/// All 8 symmetric images of a `(black, white, ko_black, ko_white)` board
/// quadruple, keyed the same way as `D4Dyn::index_symmetries` -- index 0 is
/// the identity.
fn board_symmetries(
    black: Bits,
    white: Bits,
    ko_black: Bits,
    ko_white: Bits,
) -> [(Bits, Bits, Bits, Bits); 8] {
    let sym = D4Dyn::new(black.rows());
    let mut out = [(black, white, ko_black, ko_white); 8];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (
            transform_board(black, &sym, i),
            transform_board(white, &sym, i),
            transform_board(ko_black, &sym, i),
            transform_board(ko_white, &sym, i),
        );
    }
    out
}

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

/// Index of the symmetry whose image of `(black, white)` is lexicographically
/// minimal -- the canonical orientation for the position. Deliberately keyed
/// on `(black, white)` alone, not the ko boards too: the chosen symmetry is
/// applied uniformly to every geometric field regardless, so including
/// `ko_black`/`ko_white` in the tie-break would only complicate the
/// comparison without changing which orientation gets picked for equivalent
/// `(black, white)` pairs reached via different histories.
fn canonical_symmetry(black: Bits, white: Bits, ko_black: Bits, ko_white: Bits) -> usize {
    board_symmetries(black, white, ko_black, ko_white)
        .iter()
        .enumerate()
        .min_by_key(|(_, &(b, w, _, _))| (board_key(b), board_key(w)))
        .unwrap()
        .0
}

/// Not `Copy`: `Engine`'s group/liberty bookkeeping is `Vec`-backed (a
/// `Dyn`-dimensioned board has no compile-time cell count to size a fixed
/// array with -- see `bitboard::go::GoEngine`'s doc comment), so `apply`
/// clones rather than implicitly copies.
///
/// `ko_black`/`ko_white` (the `black()`/`white()` snapshot from just before
/// the most recent placement, used by [`is_ko`](Self::is_ko)) are geometric
/// fields -- board patterns that must rotate/reflect along with `black`/
/// `white` under `Game::canonical_representation`, not carried through
/// unchanged the way `turn`/`can_swap`/`winner` are.
#[derive(Clone, Debug)]
pub struct State {
    engine: Engine,
    ko_black: Bits,
    ko_white: Bits,
    turn: Player,
    can_swap: bool,
    winner: bool,
    /// Incrementally-maintained Zobrist hash for each of the 8 symmetries.
    /// `hashes[0]` is the identity-symmetry hash; the others are used for
    /// canonical-symmetry reduction via `canonical_symmetry`.
    hashes: [u64; 8],
}

/// Equality/hashing (for the exhaustive-search regression tests and any
/// external transposition key that isn't `zobrist_hash`) is defined purely
/// on the fields that determine future play -- `hashes` is a deterministic
/// function of those, carrying no extra information, mirroring `GoEngine`'s
/// own occupancy-only `PartialEq`/`Hash`.
impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.engine == other.engine
            && self.ko_black == other.ko_black
            && self.ko_white == other.ko_white
            && self.turn == other.turn
            && self.can_swap == other.can_swap
            && self.winner == other.winner
    }
}

impl Eq for State {}

impl std::hash::Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.engine.hash(state);
        self.ko_black.hash(state);
        self.ko_white.hash(state);
        self.turn.hash(state);
        self.can_swap.hash(state);
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
        let ones = !Bits::new(Dyn(size), Dyn(size));
        Self::from_parts(
            Bits::new(Dyn(size), Dyn(size)),
            Bits::new(Dyn(size), Dyn(size)),
            ones,
            ones,
            Player::default(),
            true,
            false,
        )
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
        let hashes = rebuild_hashes(black, white, ko_black, ko_white, turn, can_swap, winner);
        Self {
            engine: Engine::from_boards(black, white),
            ko_black,
            ko_white,
            turn,
            can_swap,
            winner,
            hashes,
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

    /// This state's Zobrist hash -- the canonical-symmetry slot when
    /// `USE_SYMMETRY` is on and this state is within `symmetry_ply_limit`
    /// (letting `use_transpositions`/graph-search node sharing merge
    /// positions reached via different orientations), the plain identity
    /// slot otherwise. Must apply the exact same ply-limit gate as
    /// `Gonnect::canonical_representation` -- see AtariGo's identical
    /// `State::hash` doc comment for why a mismatched gate corrupts
    /// `use_transpositions`'s legacy (non-DAG) transposition table.
    #[inline(always)]
    fn hash(&self) -> u64 {
        let stones = (self.black().count_ones() + self.white().count_ones()) as usize;
        if USE_SYMMETRY && stones <= symmetry_ply_limit(self.black().rows()) {
            self.hashes
                [canonical_symmetry(self.black(), self.white(), self.ko_black, self.ko_white)]
        } else {
            self.hashes[0]
        }
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

        // Rebuilt from scratch rather than maintained incrementally: unlike
        // AtariGo/Othello's single black/white pair, a Gonnect move can
        // touch 4 geometric channels at once (a placement moves both ko
        // boards *and* the captured-stone channel; `SWAP` swaps the entire
        // black/white channels outright), and this game's board sizes (up
        // to 19x19) keep the O(stones) rebuild cheap relative to the
        // `GoEngine::play` call just above it.
        self.hashes = rebuild_hashes(
            self.black(),
            self.white(),
            self.ko_black,
            self.ko_white,
            self.turn,
            self.can_swap,
            self.winner,
        );

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

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.hash()
    }

    fn symmetry_ply_limit(state: &Self::S) -> usize {
        symmetry_ply_limit(state.black().rows())
    }

    fn canonical_representation(state: Real<Self::S>) -> (Canonical<Self::S>, Transform) {
        let state = state.0;
        let stones = (state.black().count_ones() + state.white().count_ones()) as usize;
        if stones > Self::symmetry_ply_limit(&state) {
            return (Canonical(state), Transform::IDENTITY);
        }
        let sym_idx =
            canonical_symmetry(state.black(), state.white(), state.ko_black, state.ko_white);
        let (black, white, ko_black, ko_white) =
            board_symmetries(state.black(), state.white(), state.ko_black, state.ko_white)[sym_idx];

        (
            Canonical(State::from_parts(
                black,
                white,
                ko_black,
                ko_white,
                state.turn,
                state.can_swap,
                state.winner,
            )),
            Transform::new(sym_idx),
        )
    }

    /// Only `.0` (the destination index) and `.1` (the capture mask) are
    /// transformed through the size-`.2` D4 group; `Move::SWAP`/`Move::
    /// NO_MOVE` are fixed points of every symmetry, mirroring how Othello
    /// treats `Move::PASS`. See AtariGo's identical `apply_to_action` doc
    /// comment for why the mask must transform alongside the index.
    fn apply_to_action(action: Real<Self::A>, sym: Transform) -> Canonical<Self::A> {
        let action = action.0;
        if action == Move::NO_MOVE || action == Move::SWAP {
            return Canonical(action);
        }
        let d4 = D4Dyn::new(action.2 as usize);
        Canonical(Move(
            d4.index_symmetries(action.0 as usize)[sym.index()] as u16,
            transform_words(action.1, &d4, sym.index()),
            action.2,
        ))
    }

    fn invert_action(action: Canonical<Self::A>, sym: Transform) -> Real<Self::A> {
        let action = action.0;
        if action == Move::NO_MOVE || action == Move::SWAP {
            return Real(action);
        }
        let d4 = D4Dyn::new(action.2 as usize);
        Real(Move(
            d4.invert_symmetry(action.0 as usize, sym.index()) as u16,
            invert_words(action.1, &d4, sym.index()),
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

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Symmetry: `apply_to_action`/`invert_action` round-trip, `canonical_representation`
    // invariance across symmetric images, the ply cutoff, and that a canonical action translated
    // back via `invert_action` is always legal against the real state -- mirroring AtariGo's own
    // symmetry test suite.

    #[test]
    fn test_action_transform_round_trip() {
        let n = 9usize;
        for idx in 0..(n * n) {
            let mut mask = Bits::new(Dyn(n), Dyn(n));
            mask.set_index((idx + 1) % (n * n));
            mask.set_index((idx + 2) % (n * n));
            for sym in 0..8usize {
                let action = Move::new(idx as u16, mask);
                let sym = Transform::new(sym);
                let transformed = Gonnect::apply_to_action(Real(action), sym);
                let back = Gonnect::invert_action(transformed, sym);
                assert_eq!(back.into_inner(), action);
            }
        }
        for sentinel in [Move::NO_MOVE, Move::SWAP] {
            for sym in 0..8usize {
                let sym = Transform::new(sym);
                assert_eq!(
                    Gonnect::apply_to_action(Real(sentinel), sym).into_inner(),
                    sentinel
                );
                assert_eq!(
                    Gonnect::invert_action(Canonical(sentinel), sym).into_inner(),
                    sentinel
                );
            }
        }
    }

    fn check_canonical_representation_invariant(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let mut reachable = vec![state.clone()];
        let limit = symmetry_ply_limit(n);
        for _ in 0..limit {
            if Gonnect::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            Gonnect::generate_actions(&state, &mut actions);
            let action = actions[rng.gen_range(0..actions.len())];
            state = Gonnect::apply(state, &action);
            reachable.push(state.clone());
        }

        for state in reachable {
            let (canon, _sym) = Gonnect::canonical_representation(Real(state.clone()));
            let canon = canon.into_inner();

            for &(black, white, ko_black, ko_white) in
                board_symmetries(state.black(), state.white(), state.ko_black, state.ko_white)
                    .iter()
            {
                let variant = State::from_parts(
                    black,
                    white,
                    ko_black,
                    ko_white,
                    state.turn,
                    state.can_swap,
                    state.winner,
                );
                let (canon2, _) = Gonnect::canonical_representation(Real(variant));
                let canon2 = canon2.into_inner();
                assert_eq!(
                    (canon2.black(), canon2.white(), canon2.turn),
                    (canon.black(), canon.white(), canon.turn),
                    "canonical_representation disagreed across symmetric images (n={n}, seed={seed})"
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
        assert_eq!(Gonnect::symmetry_ply_limit(&State::new(n)), limit);

        // A black-only board with `limit + 1` stones placed along the top
        // row, none of which is its own canonical (lexicographically
        // minimal) image -- if the cutoff didn't fire, canonicalization
        // would move it.
        let mut black = Bits::new(Dyn(n), Dyn(n));
        for i in 0..=limit {
            black.set_index(n * n - 1 - i);
        }
        let ones = !Bits::new(Dyn(n), Dyn(n));
        let state = State::from_parts(
            black,
            Bits::new(Dyn(n), Dyn(n)),
            ones,
            ones,
            Player::Black,
            false,
            false,
        );
        assert_ne!(
            canonical_symmetry(state.black(), state.white(), state.ko_black, state.ko_white),
            0,
            "test setup: state should not already be its own canonical image"
        );

        let (canon, sym) = Gonnect::canonical_representation(Real(state.clone()));
        let canon = canon.into_inner();
        assert_eq!(sym, Transform::IDENTITY);
        assert_eq!(canon.black(), state.black());
        assert_eq!(canon.white(), state.white());
    }

    fn check_invert_action_legal_along_random_game(n: usize, seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::new(n);
        let max_plies = n * n * 8 + 32;

        for _ in 0..max_plies {
            if Gonnect::is_terminal(&state) {
                return;
            }
            let mut real_actions = Vec::new();
            Gonnect::generate_actions(&state, &mut real_actions);

            let (canon, sym) = Gonnect::canonical_representation(Real(state.clone()));
            let canon = canon.into_inner();
            let mut canon_actions = Vec::new();
            Gonnect::generate_actions(&canon, &mut canon_actions);

            for &canon_action in &canon_actions {
                let translated = Gonnect::invert_action(Canonical(canon_action), sym).into_inner();
                assert!(
                    real_actions.contains(&translated),
                    "seed {seed}, n={n}: invert_action produced {translated:?} (from canonical \
                     {canon_action:?}, sym {sym:?}), not present in real generate_actions \
                     {real_actions:?}\nreal state:\n{state}\ncanon state:\n{canon}"
                );
            }

            let action = real_actions[rng.gen_range(0..real_actions.len())];
            state = Gonnect::apply(state, &action);
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
}

#[cfg(test)]
mod hash_consistency {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::{HashMap, HashSet, VecDeque};

    /// A canonicalized state's geometric + non-geometric fields, used as the
    /// "same equivalence class" comparison key in the hash-consistency
    /// checks below.
    type CanonKey = (Bits, Bits, Bits, Bits, Player, bool, bool);

    /// Exhaustive: every reachable state from an empty 3x3 board must hash
    /// to a value that uniquely determines its canonical-equivalence class
    /// -- i.e. any two states sharing a hash must have the same
    /// `canonical_representation` output. Covers below- and
    /// past-`symmetry_ply_limit` states, `SWAP`, and terminal positions
    /// (`winner`/`can_swap` are folded into the hash too), mirroring
    /// AtariGo's `test_3x3_exhaustive_hash_consistency`.
    #[test]
    fn test_3x3_exhaustive_hash_consistency() {
        let start = State::new(3);
        let mut seen: HashSet<State> = HashSet::new();
        let mut queue: VecDeque<State> = VecDeque::new();
        seen.insert(start.clone());
        queue.push_back(start);

        let mut by_hash: HashMap<u64, CanonKey> = HashMap::new();
        let mut mismatches = 0;
        let mut explored = 0usize;

        while let Some(state) = queue.pop_front() {
            explored += 1;
            assert!(
                explored <= 500_000,
                "reachable-state graph is unexpectedly large -- possible non-termination"
            );

            let h = state.hash();
            let (canon, _) = Gonnect::canonical_representation(Real(state.clone()));
            let canon = canon.into_inner();
            let key = (
                canon.black(),
                canon.white(),
                canon.ko_black,
                canon.ko_white,
                canon.turn,
                canon.can_swap,
                canon.winner,
            );
            if let Some(prev) = by_hash.get(&h) {
                if *prev != key {
                    mismatches += 1;
                    println!("MISMATCH at hash {h}: prev={prev:?} new={key:?}");
                }
            } else {
                by_hash.insert(h, key);
            }

            if Gonnect::is_terminal(&state) {
                continue;
            }
            let mut actions = Vec::new();
            Gonnect::generate_actions(&state, &mut actions);
            for action in actions {
                let next = Gonnect::apply(state.clone(), &action);
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
    /// sizes too large to enumerate fully.
    #[test]
    fn test_random_games_hash_consistency() {
        let mut rng = SmallRng::seed_from_u64(123);
        let mut by_hash: HashMap<u64, CanonKey> = HashMap::new();
        let mut mismatches = 0;
        for _game in 0..200 {
            let mut state = State::new(5);
            for _ in 0..30 {
                if Gonnect::is_terminal(&state) {
                    break;
                }
                let mut actions = Vec::new();
                Gonnect::generate_actions(&state, &mut actions);
                let action = actions[rng.gen_range(0..actions.len())];
                state = Gonnect::apply(state, &action);

                let h = state.hash();
                let (canon, _sym) = Gonnect::canonical_representation(Real(state.clone()));
                let canon = canon.into_inner();
                let key = (
                    canon.black(),
                    canon.white(),
                    canon.ko_black,
                    canon.ko_white,
                    canon.turn,
                    canon.can_swap,
                    canon.winner,
                );
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
