//! Phantom (4, 4, 4): free placement on a 4x4 grid, four in a row (row,
//! column, or either main diagonal) wins, and neither player can see the
//! opponent's marks. The board is otherwise an ordinary m,n,k game --
//! `games/ttt` generalized in board size, not in rules.
//!
//! The only information a player ever gains about the opponent's marks
//! comes from attempting to place at a cell that turns out to already be
//! occupied: the attempt is rejected (the board is unchanged and the
//! attempting player's turn continues rather than passing), and the
//! attempting player now knows that cell is occupied, though not by which
//! piece -- with only two players, "not mine" already means "theirs".
//! [`Position`] therefore carries the ground-truth board (needed for
//! referee bookkeeping: [`Position::winner`] and terminality are always
//! ground-truth facts, never perspective-dependent) plus, per player, a
//! bitmask of cells that player has discovered are occupied this way.
//!
//! [`Phantom::generate_actions`] is defined purely in terms of the board
//! field, with no reference to the known-occupied masks: it returns every
//! currently-empty cell. That single definition is correct both for a
//! search that cheats by running directly against the ground-truth state
//! (every board-empty cell really is legal, so it never gets a placement
//! rejected) and for search that respects hidden information via
//! [`Phantom::determinize`], which produces a fully-determined *guess* at
//! the board consistent with the mover's own knowledge -- once that guess
//! exists, "board-empty" within it already means "legal given what the
//! mover knows", and no further filtering is needed. The known-occupied
//! masks only matter to `determinize` itself, which must never guess an
//! opponent mark away from a cell the mover has already learned holds one.
//!
//! Because a rejected placement leaves the mover to move again,
//! `Phantom::alternating_moves` reports `false` -- turn order here isn't
//! strict ping-pong between the two players.
//!
//! This board is small enough (16 cells) and has few enough symmetries
//! worth the bookkeeping that this first implementation skips symmetry
//! canonicalization and Zobrist hashing entirely (both default to no-ops on
//! `Game`), matching e.g. `games/nim`/`games/breakthrough`. Adding either
//! later would need the per-player known-occupied masks transformed
//! consistently with the board under whatever symmetry element is applied,
//! not just the board itself.

use game_core::display::{RectangularBoard, RectangularBoardDisplay};
use mcts::game::{Game, PlayerIndex};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::Serialize;
use std::fmt;

pub const NUM_ROWS: usize = 4;
pub const NUM_COLS: usize = 4;
pub const NUM_CELLS: usize = NUM_ROWS * NUM_COLS;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    X,
    O,
}

impl PlayerIndex for Piece {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

impl Piece {
    pub fn next(self) -> Piece {
        match self {
            Piece::X => Piece::O,
            Piece::O => Piece::X,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move(pub u8);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position {
    pub turn: Piece,
    /// Ground truth: both players' real marks, 2 bits per cell (`00` empty,
    /// `01` X, `10` O), row-major, cell `i`'s bits at `i << 1`.
    pub board: u32,
    /// Per player, a bitmask (bit `i` = cell `i`) of cells that player has
    /// personally discovered are occupied by attempting to place there and
    /// being rejected. Always a subset of the opponent's true marks on
    /// `board` -- a player never needs to "discover" their own marks.
    pub known_occupied: [u16; 2],
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

impl Position {
    pub fn new() -> Self {
        Self {
            turn: Piece::X,
            board: 0,
            known_occupied: [0, 0],
        }
    }

    pub fn get(&self, index: usize) -> Option<Piece> {
        match (self.board >> (index << 1)) & 0b11 {
            0b00 => None,
            0b01 => Some(Piece::X),
            0b10 => Some(Piece::O),
            _ => unreachable!(),
        }
    }

    /// Overwrites cell `index` outright (clearing whatever was there
    /// first), unlike a plain OR-in -- needed because `determinize` has to
    /// rewrite ambiguous cells in either direction (clear a guess that
    /// didn't pan out, or place a fresh one), not just fill previously-empty
    /// cells the way ordinary gameplay does.
    fn write(&mut self, index: usize, piece: Option<Piece>) {
        let shift = index << 1;
        self.board &= !(0b11u32 << shift);
        if let Some(p) = piece {
            self.board |= ((p as u32) + 1) << shift;
        }
    }

    pub fn winner(&self) -> Option<Piece> {
        for win in LINES {
            debug_assert_eq!(win.count_ones(), 4);
            if win & self.board == win {
                return Some(Piece::X);
            } else if win & (self.board >> 1) == win {
                return Some(Piece::O);
            }
        }
        None
    }

    fn is_filled(&self) -> bool {
        let pairs = 0b01010101_01010101_01010101_01010101u32;
        (self.board | (self.board >> 1)) & pairs == pairs
    }

    pub fn gen_moves(&self, actions: &mut Vec<Move>) {
        for i in 0..NUM_CELLS {
            if self.get(i).is_none() {
                actions.push(Move(i as u8));
            }
        }
    }

    /// Attempts to place the mover's piece at `m`. Returns `true` if the
    /// cell was actually empty (the mark is placed and the turn passes),
    /// or `false` if it was already occupied (the board is left unchanged,
    /// the mover's `known_occupied` mask gains this cell, and the same
    /// player must choose again). Every legal action offered by
    /// `Phantom::generate_actions` against *this exact* `board` is
    /// guaranteed to succeed -- a rejection is only possible when `self` is
    /// the ground-truth state and `m` came from a search over a
    /// `determinize`d guess that turned out wrong.
    pub fn apply(&mut self, m: Move) -> bool {
        let mover = self.turn;
        let idx = m.0 as usize;
        if self.get(idx).is_some() {
            self.known_occupied[mover.to_index()] |= 1 << m.0;
            false
        } else {
            self.write(idx, Some(mover));
            self.turn = mover.next();
            true
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

/// Winning lines as bitmasks isolating each targeted cell's low bit (the
/// bit that's set for X but not O): the four rows, the four columns, and
/// both main diagonals. `winner` checks X by `win & board == win` and O by
/// shifting `board` right one bit first, so O's high bit lands in the same
/// low-bit positions the mask already targets -- one mask serves both
/// colors.
const LINES: [u32; 10] = [
    0b00000000_00000000_00000000_01010101, // row 0
    0b00000000_00000000_01010101_00000000, // row 1
    0b00000000_01010101_00000000_00000000, // row 2
    0b01010101_00000000_00000000_00000000, // row 3
    0b00000001_00000001_00000001_00000001, // col 0
    0b00000100_00000100_00000100_00000100, // col 1
    0b00010000_00010000_00010000_00010000, // col 2
    0b01000000_01000000_01000000_01000000, // col 3
    0b01000000_00010000_00000100_00000001, // main diagonal (0,0)-(3,3)
    0b00000001_00000100_00010000_01000000, // anti diagonal (0,3)-(3,0)
];

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct Phantom;

impl Game for Phantom {
    type S = Position;
    type A = Move;
    type P = Piece;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        state.gen_moves(actions);
    }

    fn apply(mut state: Self::S, m: &Self::A) -> Self::S {
        state.apply(*m);
        state
    }

    fn notation(_state: &Self::S, m: &Self::A) -> String {
        let x = m.0 as usize % NUM_COLS;
        let y = m.0 as usize / NUM_COLS;
        format!("({x}, {y})")
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.winner().is_some() || state.is_filled()
    }

    fn winner(state: &Self::S) -> Option<Piece> {
        if !Self::is_terminal(state) {
            unreachable!();
        }
        state.winner()
    }

    fn player_to_move(state: &Self::S) -> Piece {
        state.turn
    }

    fn has_hidden_information() -> bool {
        true
    }

    fn alternating_moves() -> bool {
        false
    }

    /// Produces a fully-determined guess at the board, consistent with
    /// whatever `player_to_move(state)` actually knows: their own marks and
    /// every cell they've discovered is occupied stay exactly as in the
    /// ground truth, and the remaining (ground-truth opponent mark count
    /// minus how many of those the mover already knows about) opponent
    /// marks are scattered uniformly among the still-ambiguous cells. The
    /// *count* of opponent marks is not hidden -- it's recoverable from ply
    /// parity -- only *which* ambiguous cells hold them is.
    fn determinize(mut state: Self::S, rng: &mut SmallRng) -> Self::S {
        let mover = state.turn;
        let mover_idx = mover.to_index();
        let opponent = mover.next();
        let known = state.known_occupied[mover_idx];

        let opponent_total = (0..NUM_CELLS)
            .filter(|&i| state.get(i) == Some(opponent))
            .count();
        let known_count = known.count_ones() as usize;
        debug_assert!(
            known_count <= opponent_total,
            "a cell the mover has learned is occupied must actually hold an opponent mark"
        );

        let mut ambiguous: Vec<usize> = (0..NUM_CELLS)
            .filter(|&i| state.get(i) != Some(mover) && (known >> i) & 1 == 0)
            .collect();
        ambiguous.shuffle(rng);

        for &i in &ambiguous {
            state.write(i, None);
        }
        for &i in ambiguous.iter().take(opponent_total - known_count) {
            state.write(i, Some(opponent));
        }

        state
    }
}

impl RectangularBoard for Position {
    const NUM_DISPLAY_ROWS: usize = NUM_ROWS;
    const NUM_DISPLAY_COLS: usize = NUM_COLS;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        match self.get(row * NUM_COLS + col) {
            None => '.',
            Some(Piece::X) => 'X',
            Some(Piece::O) => 'O',
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::{Phantom, Piece, Position, LINES, NUM_CELLS};
    use mcts::{
        game::{Game, PlayerIndex},
        strategies::{
            mcts::{node::NodeState, render, strategy, IsmctsMode, SearchConfig, TreeSearch},
            Search,
        },
        util::random_play,
    };
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_phantom_random_play() {
        random_play::<Phantom>();
    }

    #[test]
    fn line_masks_each_cover_exactly_four_cells() {
        for win in LINES {
            assert_eq!(win.count_ones(), 4);
        }
    }

    fn play(moves: &[u8]) -> Position {
        let mut state = Position::new();
        for &m in moves {
            let ok = state.apply(super::Move(m));
            assert!(ok, "move {m} unexpectedly rejected during a scripted line");
        }
        state
    }

    #[test]
    fn winner_detects_a_row() {
        // X: 0,1,2,3 (row 0); O: 4,5,6 in between X's moves.
        let state = play(&[0, 4, 1, 5, 2, 6, 3]);
        assert_eq!(state.winner(), Some(Piece::X));
        assert!(Phantom::is_terminal(&state));
    }

    #[test]
    fn winner_detects_a_column() {
        // X: 0,4,8,12 (col 0); O: 1,5,9 in between.
        let state = play(&[0, 1, 4, 5, 8, 9, 12]);
        assert_eq!(state.winner(), Some(Piece::X));
    }

    #[test]
    fn winner_detects_the_main_diagonal() {
        // X: 0,5,10,15; O: 1,2,3 in between.
        let state = play(&[0, 1, 5, 2, 10, 3, 15]);
        assert_eq!(state.winner(), Some(Piece::X));
    }

    #[test]
    fn winner_detects_the_anti_diagonal() {
        // X: 3,6,9,12; O: 0,1,2 in between.
        let state = play(&[3, 0, 6, 1, 9, 2, 12]);
        assert_eq!(state.winner(), Some(Piece::X));
    }

    // A full board with nobody having connected four -- found by seeded
    // random self-play rather than hand-laid-out (a hand-built checkerboard
    // fill is a classic trap here: constant-parity diagonals always end up
    // monochromatic, which is exactly the outcome this test needs to
    // avoid).
    #[test]
    fn a_full_board_can_end_without_a_winner() {
        let mut found = None;
        'seeds: for seed in 0u64..500 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut state = Position::new();
            let mut actions = Vec::new();
            loop {
                if Phantom::is_terminal(&state) {
                    if state.winner().is_none() {
                        found = Some(state);
                        break 'seeds;
                    }
                    break;
                }
                actions.clear();
                Phantom::generate_actions(&state, &mut actions);
                use rand::Rng;
                let m = actions[rng.gen_range(0..actions.len())];
                state.apply(m);
            }
        }
        let state = found.expect("no drawn game found across 500 seeds");
        assert!(state.winner().is_none());
        assert!(state.is_filled());
    }

    #[test]
    fn apply_places_on_an_empty_cell_and_advances_the_turn() {
        let mut state = Position::new();
        let ok = state.apply(super::Move(5));
        assert!(ok);
        assert_eq!(state.get(5), Some(Piece::X));
        assert_eq!(state.turn, Piece::O);
        assert_eq!(state.known_occupied, [0, 0]);
    }

    #[test]
    fn apply_rejects_an_occupied_cell_without_advancing_the_turn() {
        let mut state = Position::new();
        assert!(state.apply(super::Move(5))); // X takes cell 5.
        assert_eq!(state.turn, Piece::O);

        // O, unaware cell 5 is taken, attempts it and is rejected.
        let ok = state.apply(super::Move(5));
        assert!(!ok);
        assert_eq!(
            state.turn,
            Piece::O,
            "a rejected attempt does not pass the turn"
        );
        assert_eq!(
            state.get(5),
            Some(Piece::X),
            "the board is unchanged by a rejection"
        );
        assert_eq!(state.known_occupied[Piece::O.to_index()], 1 << 5);
        assert_eq!(state.known_occupied[Piece::X.to_index()], 0);
    }

    #[test]
    fn determinize_keeps_movers_own_marks_and_known_cells_fixed() {
        let mut state = Position::new();
        // X: 0, 3. O: 1. O then attempts both of X's cells and is rejected
        // both times, revealing them without changing whose turn it is --
        // so O ends up knowing the location of every X mark on the board,
        // and determinize has nothing left to guess.
        state.apply(super::Move(0));
        state.apply(super::Move(1));
        state.apply(super::Move(3));
        assert!(!state.apply(super::Move(0)));
        assert!(!state.apply(super::Move(3)));
        assert_eq!(state.turn, Piece::O);

        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..20 {
            let determinized = Phantom::determinize(state, &mut rng);
            assert_eq!(determinized.get(0), Some(Piece::X));
            assert_eq!(determinized.get(3), Some(Piece::X));
            assert_eq!(determinized.get(1), Some(Piece::O));
            assert_eq!(determinized.known_occupied, state.known_occupied);
        }
    }

    #[test]
    fn determinize_preserves_the_true_opponent_mark_count() {
        let mut state = Position::new();
        state.apply(super::Move(0)); // X
        state.apply(super::Move(1)); // O
        state.apply(super::Move(2)); // X
        state.apply(super::Move(6)); // O
        assert_eq!(state.turn, Piece::X);

        let true_opponent_count = (0..NUM_CELLS)
            .filter(|&i| state.get(i) == Some(Piece::O))
            .count();

        let mut rng = SmallRng::seed_from_u64(3);
        let mut saw_a_different_layout = false;
        for _ in 0..20 {
            let determinized = Phantom::determinize(state, &mut rng);
            let determinized_count = (0..NUM_CELLS)
                .filter(|&i| determinized.get(i) == Some(Piece::O))
                .count();
            assert_eq!(determinized_count, true_opponent_count);
            if determinized.board != state.board {
                saw_a_different_layout = true;
            }
        }
        assert!(
            saw_a_different_layout,
            "determinize never guessed a different opponent layout across 20 resamples"
        );
    }

    impl render::NodeRender for Position {}

    // End-to-end ISMCTS wiring: every iteration searches its own
    // `Phantom::determinize`d guess at the opponent's marks, so the root's
    // `ChildArray` should widen and accumulate real availability counts.
    // Unlike a game where legality never depends on hidden information, a
    // chosen action here can legitimately be rejected against the real
    // state (that's the entire point of Phantom's information leak), so
    // this drives real self-play with a retry loop around rejections
    // instead of asserting the chosen action is always literally legal.
    #[test]
    fn ismcts_self_play_retries_rejections_and_tracks_availability() {
        let mut search: TreeSearch<Phantom, strategy::Ucb1> = TreeSearch::new().config(
            SearchConfig::new()
                .ismcts_mode(IsmctsMode::SingleTree)
                .max_iterations(40)
                .seed(11),
        );

        let mut state = Position::new();
        let mut saw_availability = false;
        for _ in 0..8 {
            if Phantom::is_terminal(&state) {
                break;
            }
            let action = search.choose_action(&state);

            let root = search.index.get(search.root_id);
            let children = root.children();
            assert!(children.is_growable());
            if let Some(idx) = (0..children.len()).find(|&i| children.action(i) == action) {
                if children.availability(idx) > 0 {
                    saw_availability = true;
                }
            }

            state.apply(action);
        }
        assert!(
            saw_availability,
            "ISMCTS never recorded availability for a chosen root action"
        );
    }

    #[test]
    fn ismcts_redeterminize_self_play_retries_rejections_and_tracks_availability() {
        let mut search: TreeSearch<Phantom, strategy::Ucb1> = TreeSearch::new().config(
            SearchConfig::new()
                .ismcts_mode(IsmctsMode::SingleTree)
                .ismcts_redeterminize(true)
                .max_iterations(40)
                .seed(17),
        );

        let mut state = Position::new();
        let mut saw_availability = false;
        for _ in 0..8 {
            if Phantom::is_terminal(&state) {
                break;
            }
            let action = search.choose_action(&state);

            let root = search.index.get(search.root_id);
            let children = root.children();
            assert!(children.is_growable());
            if let Some(idx) = (0..children.len()).find(|&i| children.action(i) == action) {
                if children.availability(idx) > 0 {
                    saw_availability = true;
                }
            }

            state.apply(action);
        }
        assert!(
            saw_availability,
            "re-determinizing ISMCTS never recorded availability for a chosen root action"
        );
    }

    // MO-ISMCTS (`IsmctsMode::MultiTree`): one tree per player descended
    // together every iteration (see `SearchConfig::ismcts_mode`'s doc
    // comment) -- checks the same plumbing the `SingleTree` tests above do
    // (growable root, real availability accumulating, self-play runs to
    // completion without panicking), against Phantom specifically since
    // it's this workspace's correctness gate for the algorithm (its own
    // published Phantom (4, 4, 4) ranking is what MO-ISMCTS is meant to
    // reproduce).
    #[test]
    fn multi_tree_ismcts_self_play_retries_rejections_and_tracks_availability() {
        let mut search: TreeSearch<Phantom, strategy::Ucb1> = TreeSearch::new().config(
            SearchConfig::new()
                .ismcts_mode(IsmctsMode::MultiTree)
                .max_iterations(40)
                .seed(11),
        );

        let mut state = Position::new();
        let mut saw_availability = false;
        for _ in 0..8 {
            if Phantom::is_terminal(&state) {
                break;
            }
            let action = search.choose_action(&state);

            let root = search.index.get(search.root_id);
            let children = root.children();
            assert!(children.is_growable());
            if let Some(idx) = (0..children.len()).find(|&i| children.action(i) == action) {
                if children.availability(idx) > 0 {
                    saw_availability = true;
                }
            }

            state.apply(action);
        }
        assert!(
            saw_availability,
            "MO-ISMCTS never recorded availability for a chosen root action"
        );
    }

    // Regression test for a real crash found while running strength
    // comparisons: `Node::expand`'s `OnceLock` only ever resolves once, so
    // if the very first call to touch the root happened to run against a
    // `Phantom::determinize`d guess rather than the literal board, an
    // unlucky guess (the opponent's guessed marks already forming a win,
    // even though the real board doesn't) could permanently mark an
    // ongoing, non-terminal root position `Terminal` -- `select_final_action`
    // has no fallback for a `Terminal` root and panics trying to read its
    // (nonexistent) children. Fixed by eagerly expanding the root against
    // the literal state before any iteration runs (the root's own position
    // is never hidden from any player, unlike every other node). Setting
    // `max_iterations(0)` isolates exactly that eager expansion -- no
    // iteration ever runs, so this can only pass if the root was expanded
    // up front from the real board.
    #[test]
    fn ismcts_root_expands_from_the_literal_state_before_any_iteration() {
        for mode in [IsmctsMode::SingleTree, IsmctsMode::MultiTree] {
            let mut search: TreeSearch<Phantom, strategy::Ucb1> = TreeSearch::new().config(
                SearchConfig::new()
                    .ismcts_mode(mode)
                    .max_iterations(0)
                    .seed(1),
            );
            let state = Position::new();
            let _ = search.choose_action(&state);

            let root = search.index.get(search.root_id);
            let Some(NodeState::Expanded(children)) = root.status() else {
                panic!(
                    "root should be Expanded from the literal state with zero iterations run, \
                     found {:?} instead",
                    root.status()
                );
            };
            let mut legal = Vec::new();
            Phantom::generate_actions(&state, &mut legal);
            assert_eq!(children.len(), legal.len());
        }
    }
}
