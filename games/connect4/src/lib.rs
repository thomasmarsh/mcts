#![allow(unused)]

//! Connect Four: two players drop discs into one of `C` columns; a disc
//! always settles on top of whatever's already in that column (gravity), and
//! the first player to line up four of their own discs -- horizontally,
//! vertically, or on either diagonal -- wins. Draws happen when the board
//! fills with no such line.
//!
//! The board is a `bitboard::Board<u64, Const<R>, Const<C>>` (standard 6x7 =
//! 42 cells, well under the 64-bit word), row-major with row 0 at the
//! bottom -- matching `Breakthrough`'s convention, where `shift_south`
//! already moves *toward* row 0. A column's fill height is derived on
//! demand from the occupied board rather than tracked as separate state
//! (`State::height`): gravity guarantees a column has no gaps, so the
//! count of contiguous occupied cells from row 0 upward *is* the height,
//! and deriving it removes a field that symmetry canonicalization would
//! otherwise have to keep in sync with the boards it mirrors.
//!
//! Symmetry: unlike a square board (Othello/Gonnect's D4) or a
//! non-square board with no gravity (`game_core::symmetry::KleinFour`),
//! Connect Four admits only a left-right mirror -- flipping rows would
//! swap which end of a column gravity pulls toward, which isn't a symmetry
//! of the rules at all. That's the two-element group C2 (a.k.a. D1):
//! identity and column-mirror (`col -> C - 1 - col`), i.e.
//! `game_core::symmetry::ColMirror<C>` -- `KleinFour`'s subgroup for exactly
//! this "only one flip is valid" case. This crate only adds board-level
//! helpers (`mirror_board`/`board_symmetries`/`canonical_symmetry`) on top
//! of it, following the same "one array slot per symmetry element"
//! hashing/canonicalization pattern as `ttt`'s `D4Symmetry<3>` and Gonnect's
//! `D4Dyn`, just with 2 slots instead of 8.

mod heuristic;

use bitboard::{Board, Const};
use game_core::display::{RectangularBoard, RectangularBoardDisplay};
use game_core::symmetry::{ColMirror, SymmetryGroup};
use mcts::game::{Canonical, Game, PlayerIndex, Real, Transform};
use mcts::zobrist::LazyZobristTable;

pub use heuristic::Heuristic;

use serde::Serialize;
use std::fmt;

pub type BitBoard<const R: usize, const C: usize> = Board<u64, Const<R>, Const<C>>;

/// Number of symmetry elements Connect Four admits: identity + column-mirror
/// (see this crate's module doc comment).
pub const NUM_SYMMETRIES: usize = 2;

/// Whether `canonical_representation` actually canonicalizes, or just
/// returns the state unchanged at `Transform::IDENTITY` -- a single flag so
/// the behavior can be A/B'd without touching every call site, mirroring
/// `ttt`'s/Gonnect's own `USE_SYMMETRY` constant.
pub const USE_SYMMETRY: bool = true;

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

/// A drop into column `.0`.
#[derive(Clone, Copy, Serialize, Debug, Hash, PartialEq, Eq)]
pub struct Move(pub u8);

impl Move {
    #[inline(always)]
    pub fn col(self) -> usize {
        self.0 as usize
    }
}

// ── Symmetry helpers ─────────────────────────────────────────────────────
//
// The cell-index-level group is `game_core::symmetry::ColMirror<C>`; the
// helpers here just lift it to board pairs and Zobrist hash slots.

/// The `[identity, mirror]` images of a cell index -- element `s` is what a
/// piece placed there contributes to `State::hashes[s]`, and what
/// `Game::apply_to_action`/`invert_action` look up by `Transform::index()`.
#[inline]
fn index_symmetries<const C: usize>(i: usize) -> [usize; NUM_SYMMETRIES] {
    ColMirror::<C>::index_symmetries(i)
}

/// Mirror every set cell of a board left-right.
#[inline]
fn mirror_board<const R: usize, const C: usize>(b: BitBoard<R, C>) -> BitBoard<R, C> {
    let mut out = b.empty_like();
    for idx in b.iter_set() {
        out.set_index(ColMirror::<C>::apply_index(idx, 1));
    }
    out
}

/// The `[identity, mirror]` images of a `(black, white)` board pair, keyed
/// the same way as `index_symmetries` -- index 0 is the identity.
#[inline]
fn board_symmetries<const R: usize, const C: usize>(
    black: BitBoard<R, C>,
    white: BitBoard<R, C>,
) -> [(BitBoard<R, C>, BitBoard<R, C>); NUM_SYMMETRIES] {
    [(black, white), (mirror_board(black), mirror_board(white))]
}

/// A comparable key for a board's raw bits, for picking the
/// lexicographically minimal symmetric image.
#[inline]
fn board_key<const R: usize, const C: usize>(b: BitBoard<R, C>) -> u64 {
    b.bits()
}

/// Index of the symmetry whose image of `(black, white)` is
/// lexicographically minimal -- the canonical orientation for the position.
#[inline]
fn canonical_symmetry<const R: usize, const C: usize>(
    black: BitBoard<R, C>,
    white: BitBoard<R, C>,
) -> usize {
    board_symmetries(black, white)
        .iter()
        .enumerate()
        .min_by_key(|(_, &(b, w))| (board_key(b), board_key(w)))
        .unwrap()
        .0
}

// ── Zobrist hashing ──────────────────────────────────────────────────────

/// `State` is backed by a `u64` bitboard, so no board size this game
/// supports can exceed 64 cells -- sized for the worst case rather than
/// parameterized per `R, C`, mirroring `game_breakthrough::zobrist`.
const MAX_CELLS: usize = 64;
const HASHES_LEN: usize = 2 * MAX_CELLS;
static HASHES: LazyZobristTable<HASHES_LEN> = LazyZobristTable::new(0xC4C4_0E5C_0110_5E11);

#[inline]
fn cell_zobrist(index: usize, player: Player) -> u64 {
    debug_assert!(index < MAX_CELLS);
    HASHES.hash(2 * index + player.to_index())
}

/// Rebuilds all `NUM_SYMMETRIES` hashes from scratch -- the counterpart to
/// `State::apply`'s incremental update, used by `canonical_representation`
/// (whose canonicalized board didn't reach its shape via a move sequence to
/// replay) and by any caller reconstructing a `State` from a bare board.
/// A move's color is recoverable from which board (`black`/`white`) the
/// piece sits on, so -- like `ttt`'s `HashedPosition` -- there's no separate
/// "whose turn" contribution to fold in: two states with the same occupied
/// cells and colors are the same state, full stop, and hash identically.
fn rebuild_hashes<const R: usize, const C: usize>(
    black: BitBoard<R, C>,
    white: BitBoard<R, C>,
) -> [u64; NUM_SYMMETRIES] {
    let mut hashes = [0u64; NUM_SYMMETRIES];
    for idx in black.iter_set() {
        for (s, &sym_idx) in index_symmetries::<C>(idx).iter().enumerate() {
            hashes[s] ^= cell_zobrist(sym_idx, Player::Black);
        }
    }
    for idx in white.iter_set() {
        for (s, &sym_idx) in index_symmetries::<C>(idx).iter().enumerate() {
            hashes[s] ^= cell_zobrist(sym_idx, Player::White);
        }
    }
    hashes
}

// ── Win detection ─────────────────────────────────────────────────────────

/// True if `b` contains four in a row in any of the four line directions
/// (vertical, horizontal, and both diagonals). Classic bit-parallel trick:
/// `m1 = b & step(b)` marks the start of every run of >= 2 in `step`'s
/// direction; `m1 & step(step(m1))` marks the start of every run of >= 4
/// (two >= 2 runs, chained). Each `step` call re-masks at the board's own
/// walls (see `Board::shift_north`/etc.), so this is correct at any `R, C`,
/// not just when a run happens to fit without wrapping.
#[inline]
fn four_in_a_row<const R: usize, const C: usize>(b: BitBoard<R, C>) -> bool {
    type B<const R: usize, const C: usize> = BitBoard<R, C>;
    let run = |step: fn(B<R, C>) -> B<R, C>| {
        let m1 = b & step(b);
        let m2 = m1 & step(step(m1));
        !m2.none_set()
    };
    run(B::<R, C>::shift_north)
        || run(B::<R, C>::shift_east)
        || run(B::<R, C>::shift_northeast)
        || run(B::<R, C>::shift_northwest)
}

// ── State ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State<const R: usize, const C: usize> {
    black: BitBoard<R, C>,
    white: BitBoard<R, C>,
    turn: Player,
    winner: bool,
    /// Incrementally-maintained Zobrist hash for each symmetry element --
    /// `hashes[0]` is the identity-symmetry hash, `hashes[1]` the mirrored
    /// one.
    hashes: [u64; NUM_SYMMETRIES],
}

impl<const R: usize, const C: usize> Default for State<R, C> {
    fn default() -> Self {
        debug_assert!(R * C <= MAX_CELLS, "board must fit in a u64 bitboard");
        debug_assert!(R >= 4 && C >= 4, "board must be able to hold a 4-in-a-row");
        Self {
            black: BitBoard::EMPTY,
            white: BitBoard::EMPTY,
            turn: Player::Black,
            winner: false,
            hashes: [0; NUM_SYMMETRIES],
        }
    }
}

impl<const R: usize, const C: usize> State<R, C> {
    pub fn black(&self) -> BitBoard<R, C> {
        self.black
    }

    pub fn white(&self) -> BitBoard<R, C> {
        self.white
    }

    pub fn turn(&self) -> Player {
        self.turn
    }

    pub fn has_winner(&self) -> bool {
        self.winner
    }

    /// Builds a state directly from its boards, rebuilding the symmetry
    /// hashes from scratch -- used by `canonical_representation` (whose
    /// mirrored board didn't arrive via `apply`) and any caller
    /// reconstructing a state from wire data.
    pub fn from_parts(
        black: BitBoard<R, C>,
        white: BitBoard<R, C>,
        turn: Player,
        winner: bool,
    ) -> Self {
        Self {
            black,
            white,
            turn,
            winner,
            hashes: rebuild_hashes(black, white),
        }
    }

    #[inline(always)]
    fn occupied(&self) -> BitBoard<R, C> {
        self.black | self.white
    }

    #[inline(always)]
    fn player(&self, player: Player) -> BitBoard<R, C> {
        match player {
            Player::Black => self.black,
            Player::White => self.white,
        }
    }

    /// The number of discs already stacked in `col` -- gravity guarantees
    /// no gaps, so this is just how far up the column stays occupied from
    /// row 0 (see this module's doc comment on why that's derived rather
    /// than tracked as its own field).
    #[inline]
    fn height(&self, col: usize) -> usize {
        let occupied = self.occupied();
        (0..R)
            .take_while(|&row| occupied.get_index(row * C + col))
            .count()
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.occupied().count_ones() as usize == R * C
    }

    fn moves(&self, actions: &mut Vec<Move>) {
        if self.winner {
            return;
        }
        for col in 0..C {
            if self.height(col) < R {
                actions.push(Move(col as u8));
            }
        }
    }

    #[inline]
    fn apply(&mut self, action: &Move) -> Self {
        let col = action.col();
        let row = self.height(col);
        debug_assert!(row < R, "apply called on a full column");
        let index = row * C + col;

        let mut player = self.player(self.turn);
        player.set_index(index);
        match self.turn {
            Player::Black => self.black = player,
            Player::White => self.white = player,
        }

        for (s, &sym_idx) in index_symmetries::<C>(index).iter().enumerate() {
            self.hashes[s] ^= cell_zobrist(sym_idx, self.turn);
        }

        if four_in_a_row(player) {
            self.winner = true;
        } else {
            self.turn = self.turn.next();
        }

        *self
    }

    /// This state's Zobrist hash -- the canonical-symmetry slot when
    /// `USE_SYMMETRY` is on, the plain identity slot otherwise.
    #[inline(always)]
    fn hash(&self) -> u64 {
        if USE_SYMMETRY {
            self.hashes[canonical_symmetry(self.black, self.white)]
        } else {
            self.hashes[0]
        }
    }
}

#[derive(Clone)]
pub struct Connect4<const R: usize, const C: usize>;

/// The traditional 7-wide, 6-tall board.
pub type Standard = Connect4<6, 7>;

impl<const R: usize, const C: usize> Game for Connect4<R, C> {
    type S = State<R, C>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<R, C>, action: &Move) -> State<R, C> {
        state.apply(action)
    }

    fn generate_actions(state: &State<R, C>, actions: &mut Vec<Move>) {
        state.moves(actions);
    }

    fn is_terminal(state: &State<R, C>) -> bool {
        state.winner || state.is_full()
    }

    fn player_to_move(state: &State<R, C>) -> Player {
        state.turn
    }

    fn winner(state: &State<R, C>) -> Option<Player> {
        if state.winner {
            Some(state.turn)
        } else {
            None
        }
    }

    fn parse_action(state: &Self::S, input: &str) -> Option<Self::A> {
        let col = input.trim().parse::<usize>().ok()?.checked_sub(1)?;
        if col < C && state.height(col) < R {
            Some(Move(col as u8))
        } else {
            None
        }
    }

    fn notation(_state: &Self::S, action: &Self::A) -> String {
        format!("{}", action.col() + 1)
    }

    fn num_players() -> usize {
        2
    }

    fn zobrist_hash(state: &State<R, C>) -> u64 {
        state.hash()
    }

    fn canonical_representation(state: Real<Self::S>) -> (Canonical<Self::S>, Transform) {
        let state = state.0;
        let sym = canonical_symmetry(state.black, state.white);
        let (black, white) = board_symmetries(state.black, state.white)[sym];
        (
            Canonical(State::from_parts(black, white, state.turn, state.winner)),
            Transform::new(sym),
        )
    }

    fn apply_to_action(action: Real<Self::A>, sym: Transform) -> Canonical<Self::A> {
        let col = action.0.col();
        let mirrored = ColMirror::<C>::apply_index(col, sym.index());
        Canonical(Move(mirrored as u8))
    }

    fn invert_action(action: Canonical<Self::A>, sym: Transform) -> Real<Self::A> {
        let col = action.0.col();
        let mirrored = ColMirror::<C>::invert_index(col, sym.index());
        Real(Move(mirrored as u8))
    }
}

impl<const R: usize, const C: usize> RectangularBoard for State<R, C> {
    const NUM_DISPLAY_ROWS: usize = R;
    const NUM_DISPLAY_COLS: usize = C;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        if self.black.get(row, col) {
            'X'
        } else if self.white.get(row, col) {
            'O'
        } else {
            '.'
        }
    }
}

impl<const R: usize, const C: usize> fmt::Display for State<R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::util::random_play;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::collections::{HashMap, HashSet, VecDeque};

    #[test]
    fn test_connect4() {
        random_play::<Standard>();
    }

    /// A hand-built vertical win: Black drops four times into column 0
    /// uninterrupted (White plays elsewhere each turn) and must be declared
    /// the winner on the fourth drop, without advancing the turn further.
    #[test]
    fn test_vertical_win() {
        let mut state = State::<6, 7>::default();
        for &col in &[0u8, 1, 0, 1, 0, 1, 0] {
            if Standard::is_terminal(&state) {
                break;
            }
            state = Standard::apply(state, &Move(col));
        }
        assert!(state.has_winner());
        assert_eq!(Standard::winner(&state), Some(Player::Black));
    }

    /// Same shape, but a horizontal win along row 0.
    #[test]
    fn test_horizontal_win() {
        let mut state = State::<6, 7>::default();
        // Black plays columns 0,1,2,3 on row 0; White plays column 4 (a
        // fresh column each time, well clear of black's row) in between so
        // it never blocks or wins first.
        for &col in &[0u8, 4, 1, 4, 2, 4, 3] {
            if Standard::is_terminal(&state) {
                break;
            }
            state = Standard::apply(state, &Move(col));
        }
        assert!(state.has_winner());
        assert_eq!(Standard::winner(&state), Some(Player::Black));
    }

    /// A diagonal win: Black's winning cells (0,0),(1,1),(2,2),(3,3) each
    /// need White (or a harmless Black "waste" move in an unrelated column
    /// 6) to build up the right height underneath first, so moves are
    /// scripted turn by turn rather than played randomly. Column 6 absorbs
    /// the turns where Black has no diagonal cell ready to play yet, without
    /// creating a 4-in-a-row of its own (at most 3 stacked there).
    #[test]
    fn test_diagonal_win() {
        let mut state = State::<6, 7>::default();
        let columns: &[u8] = &[0, 1, 1, 2, 6, 2, 2, 3, 6, 3, 6, 3, 3];
        for &col in columns {
            assert!(
                !Standard::is_terminal(&state),
                "won before the scripted diagonal completed"
            );
            state = Standard::apply(state, &Move(col));
        }
        assert!(state.has_winner());
        assert_eq!(Standard::winner(&state), Some(Player::Black));
        for (row, col) in [(0, 0), (1, 1), (2, 2), (3, 3)] {
            assert!(
                state.black().get(row, col),
                "expected Black at ({row},{col})"
            );
        }
    }

    /// A full, drawn board (no 4-in-a-row for either color) must report a
    /// terminal state with no winner. Rather than hand-deriving a specific
    /// filled board (easy to get subtly wrong -- an early attempt at this
    /// produced an accidental diagonal win, since a naive alternating fill
    /// pattern is exactly a checkerboard, and a checkerboard's diagonals
    /// are always monochromatic), search random self-play games on a small
    /// board across many seeds for one that happens to fill without either
    /// side winning -- small enough, and enough seeds, that this reliably
    /// finds one while still exercising the real generate_actions/apply/
    /// is_terminal path end to end.
    #[test]
    fn test_draw() {
        type Small = Connect4<4, 5>;
        for seed in 0..2000u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut state = State::<4, 5>::default();
            while !Small::is_terminal(&state) {
                let mut actions = Vec::new();
                Small::generate_actions(&state, &mut actions);
                let action = actions[rng.gen_range(0..actions.len())];
                state = Small::apply(state, &action);
            }
            if !state.has_winner() {
                assert_eq!(Small::winner(&state), None);
                return;
            }
        }
        panic!("no drawn game found in 2000 random seeds on a 4x5 board");
    }

    /////////////////////////////////////////////////////////////////////
    // Symmetry: mirror round-trip, canonical-representation invariance,
    // and hash consistency -- mirroring `ttt`'s/Gonnect's own symmetry
    // test suites at the 2-element scale this game's mirror group has.

    #[test]
    fn test_action_transform_round_trip() {
        for col in 0..7u8 {
            for sym in 0..NUM_SYMMETRIES {
                let sym = Transform::new(sym);
                let action = Move(col);
                let transformed = Standard::apply_to_action(Real(action), sym);
                let back = Standard::invert_action(transformed, sym);
                assert_eq!(back.into_inner(), action);
            }
        }
    }

    fn check_canonical_representation_invariant(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::<6, 7>::default();
        let mut reachable = vec![state];
        for _ in 0..20 {
            if Standard::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            Standard::generate_actions(&state, &mut actions);
            let action = actions[rng.gen_range(0..actions.len())];
            state = Standard::apply(state, &action);
            reachable.push(state);
        }

        for state in reachable {
            let (canon, _sym) = Standard::canonical_representation(Real(state));
            let canon = canon.into_inner();

            for &(black, white) in board_symmetries(state.black(), state.white()).iter() {
                let variant = State::from_parts(black, white, state.turn(), state.has_winner());
                let (canon2, _) = Standard::canonical_representation(Real(variant));
                let canon2 = canon2.into_inner();
                assert_eq!(
                    (canon2.black(), canon2.white(), canon2.turn()),
                    (canon.black(), canon.white(), canon.turn()),
                    "canonical_representation disagreed across symmetric images (seed={seed})"
                );
            }
        }
    }

    #[test]
    fn test_canonical_representation_invariant_under_symmetry() {
        for seed in 0..50 {
            check_canonical_representation_invariant(seed);
        }
    }

    fn check_invert_action_legal_along_random_game(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = State::<6, 7>::default();

        for _ in 0..20 {
            if Standard::is_terminal(&state) {
                return;
            }
            let mut real_actions = Vec::new();
            Standard::generate_actions(&state, &mut real_actions);

            let (canon, sym) = Standard::canonical_representation(Real(state));
            let canon = canon.into_inner();
            let mut canon_actions = Vec::new();
            Standard::generate_actions(&canon, &mut canon_actions);

            for &canon_action in &canon_actions {
                let translated = Standard::invert_action(Canonical(canon_action), sym).into_inner();
                assert!(
                    real_actions.contains(&translated),
                    "seed {seed}: invert_action produced {translated:?}, not present in \
                     generate_actions {real_actions:?}"
                );
            }

            let action = real_actions[rng.gen_range(0..real_actions.len())];
            state = Standard::apply(state, &action);
        }
    }

    #[test]
    fn test_invert_action_produces_legal_real_actions() {
        for seed in 0..100 {
            check_invert_action_legal_along_random_game(seed);
        }
    }

    /// Every reachable position (up to a shallow ply cap, exhaustively) must
    /// hash to a value that uniquely determines its canonical-equivalence
    /// class -- i.e. any two states sharing a hash must canonicalize to the
    /// same `(black, white, turn)`, mirroring `ttt`'s/Gonnect's own
    /// hash-consistency checks.
    #[test]
    fn test_exhaustive_hash_consistency_shallow() {
        let start = State::<4, 4>::default();
        let mut seen: HashSet<State<4, 4>> = HashSet::new();
        let mut queue: VecDeque<(State<4, 4>, usize)> = VecDeque::new();
        seen.insert(start);
        queue.push_back((start, 0));

        let mut by_hash: HashMap<u64, (BitBoard<4, 4>, BitBoard<4, 4>, Player)> = HashMap::new();
        let mut mismatches = 0;
        let max_ply = 10;

        while let Some((state, ply)) = queue.pop_front() {
            let h = Connect4::<4, 4>::zobrist_hash(&state);
            let (canon, _) = Connect4::<4, 4>::canonical_representation(Real(state));
            let canon = canon.into_inner();
            let key = (canon.black(), canon.white(), canon.turn());
            if let Some(prev) = by_hash.get(&h) {
                if *prev != key {
                    mismatches += 1;
                }
            } else {
                by_hash.insert(h, key);
            }

            if Connect4::<4, 4>::is_terminal(&state) || ply >= max_ply {
                continue;
            }
            let mut actions = Vec::new();
            Connect4::<4, 4>::generate_actions(&state, &mut actions);
            for action in actions {
                let next = Connect4::<4, 4>::apply(state, &action);
                if seen.insert(next) {
                    queue.push_back((next, ply + 1));
                }
            }
        }
        assert_eq!(mismatches, 0);
    }
}
