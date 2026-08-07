use crate::display::{RectangularBoard, RectangularBoardDisplay};
use crate::game::{Game, PlayerIndex};
use crate::zobrist::LazyZobristTable;
use serde::{Deserialize, Serialize};
use std::fmt;

const USE_SYMMETRY: bool = false;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Position {
    pub turn: Piece,
    pub board: u32,
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
        }
    }

    pub fn set(&mut self, index: usize, piece: Piece) {
        self.board |= ((piece as u32) + 1) << (index << 1);
    }

    pub fn get(&self, index: usize) -> Option<Piece> {
        match (self.board >> (index << 1)) & 0b11 {
            0b00 => None,
            0b01 => Some(Piece::X),
            0b10 => Some(Piece::O),
            _ => unreachable!(),
        }
    }

    pub fn winner(&self) -> Option<Piece> {
        for win in [
            0b000000_000000_010101u32,
            0b000000_010101_000000,
            0b010101_000000_000000,
            0b000001_000001_000001,
            0b000100_000100_000100,
            0b010000_010000_010000,
            0b010000_000100_000001,
            0b000001_000100_010000,
        ] {
            debug_assert_eq!(win.count_ones(), 3);
            if win & self.board == win {
                return Some(Piece::X);
            } else if win & (self.board >> 1) == win {
                return Some(Piece::O);
            }
        }
        None
    }

    fn is_filled(&self) -> bool {
        let pairs = 0b010101_010101_010101;
        (self.board | (self.board >> 1)) & pairs == pairs
    }

    pub fn gen_moves(&self, actions: &mut Vec<Move>) {
        for i in 0..9 {
            if self.get(i).is_none() {
                actions.push(Move(i as u8));
            }
        }
    }

    pub fn apply(&mut self, m: Move) {
        assert!(self.get(m.0 as usize).is_none());
        self.set(m.0 as usize, self.turn);
        self.turn = self.turn.next();
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub const NUM_SYMMETRIES: usize = 8;

pub mod sym {
    use super::NUM_SYMMETRIES;

    const H: [usize; 9] = [6, 7, 8, 3, 4, 5, 0, 1, 2];
    const V: [usize; 9] = [2, 1, 0, 5, 4, 3, 8, 7, 6];
    const D: [usize; 9] = [8, 5, 2, 7, 4, 1, 6, 3, 0];

    #[inline]
    pub fn index_symmetries(i: usize, symmetries: &mut [usize; NUM_SYMMETRIES]) {
        symmetries[0] = i;
        symmetries[1] = H[i];
        symmetries[2] = V[i];
        symmetries[3] = D[i];
        symmetries[4] = V[H[i]];
        symmetries[5] = D[H[i]];
        symmetries[6] = D[V[i]];
        symmetries[7] = D[V[H[i]]];
    }

    #[inline]
    pub fn invert_symmetry(i: usize, symmetry_index: usize) -> usize {
        match symmetry_index {
            0 => i,
            1 => H[i],
            2 => V[i],
            3 => D[i],
            4 => H[V[i]],
            5 => H[D[i]],
            6 => V[D[i]],
            7 => H[V[D[i]]],
            _ => unreachable!("Invalid symmetry index"),
        }
    }

    #[inline]
    pub fn board_symmetries(board: u32, symmetries: &mut [u32; NUM_SYMMETRIES]) {
        debug_assert!(symmetries.iter().all(|x| *x == 0));

        symmetries[0] = board;
        (0..9).for_each(|i| {
            let p = (board >> (i << 1)) & 0b11;
            symmetries[1] |= p << (H[i] * 2);
            symmetries[2] |= p << (V[i] * 2);
            symmetries[3] |= p << (D[i] * 2);
            symmetries[4] |= p << (V[H[i]] * 2);
            symmetries[5] |= p << (D[H[i]] * 2);
            symmetries[6] |= p << (D[V[i]] * 2);
            symmetries[7] |= p << (D[V[H[i]]] * 2);
        });
    }

    #[inline]
    pub fn canonical_symmetry(board: u32) -> usize {
        let mut sym = [0; 8];
        board_symmetries(board, &mut sym);
        sym.iter().enumerate().min_by_key(|(_, &v)| v).unwrap().0
    }
}

////////////////////////////////////////////////////////////////////////////////////////

// 9 playable positions * 2 players
const NUM_MOVES: usize = 18;

static HASHES: LazyZobristTable<NUM_MOVES> = LazyZobristTable::new(0xFEAAE62226597B38);

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct HashedPosition {
    pub position: Position,
    pub(crate) hashes: [u64; 8],
}

impl HashedPosition {
    pub fn new() -> Self {
        Self {
            position: Position::new(),
            hashes: [0; 8],
        }
    }
}

impl Default for HashedPosition {
    fn default() -> Self {
        Self::new()
    }
}

impl HashedPosition {
    /// Rebuilds a `HashedPosition` from a bare `Position` with no move-order
    /// history -- needed by the stateless game server, which receives a
    /// client-supplied position on every request rather than
    /// replaying moves through `apply` one at a time.
    ///
    /// XOR is commutative, so `apply`'s incremental `hashes[s] ^=
    /// HASHES.hash((index << 1) | turn)` per move can be reproduced from the
    /// final board alone, in any order -- as long as each cell's `turn` value
    /// at the time it was placed is recoverable. It is: `apply` always sets
    /// `board[index] = self.turn` before advancing the turn, so the piece
    /// sitting on a filled cell today *is* the turn value that placed it.
    pub fn from_position(position: Position) -> Self {
        let mut hashed = Self {
            position,
            hashes: [0; 8],
        };
        let mut symmetries = [0usize; NUM_SYMMETRIES];
        for i in 0..9 {
            if let Some(piece) = position.get(i) {
                sym::index_symmetries(i, &mut symmetries);
                for (s, index) in symmetries.iter().enumerate() {
                    hashed.hashes[s] ^= HASHES.hash((index << 1) | piece as usize);
                }
            }
        }
        hashed
    }

    #[inline]
    fn apply(&mut self, m: Move) {
        let mut symmetries = [0; NUM_SYMMETRIES];
        sym::index_symmetries(m.0 as usize, &mut symmetries);
        for (i, index) in symmetries.iter().enumerate() {
            self.hashes[i] ^= HASHES.hash((index << 1) | self.position.turn as usize);
        }
        self.position.apply(m);
    }

    #[inline(always)]
    fn hash(&self) -> u64 {
        if USE_SYMMETRY {
            self.hashes[sym::canonical_symmetry(self.position.board)]
        } else {
            self.hashes[0]
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct TicTacToe;

impl Game for TicTacToe {
    type S = HashedPosition;
    type A = Move;
    type P = Piece;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        state.position.gen_moves(actions);
    }

    fn apply(mut state: Self::S, m: &Self::A) -> Self::S {
        state.apply(*m);
        state
    }

    fn notation(_state: &Self::S, m: &Self::A) -> String {
        let x = m.0 % 3;
        let y = m.0 / 3;
        format!("({}, {})", x, y)
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.position.winner().is_some() || state.position.is_filled()
    }

    fn winner(state: &Self::S) -> Option<Piece> {
        if !Self::is_terminal(state) {
            unreachable!();
        }

        state.position.winner()
    }

    fn player_to_move(state: &Self::S) -> Piece {
        state.position.turn
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.hash()
    }
}

impl RectangularBoard for HashedPosition {
    const NUM_DISPLAY_ROWS: usize = 3;
    const NUM_DISPLAY_COLS: usize = 3;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        match self.position.get(row * 3 + col) {
            None => '.',
            Some(Piece::X) => 'X',
            Some(Piece::O) => 'O',
        }
    }
}

impl fmt::Display for HashedPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use super::{HashedPosition, Move, TicTacToe};
    use crate::{
        game::Game,
        strategies::{
            mcts::{node::QInit, render, strategy, SearchConfig, TreeSearch},
            Search,
        },
        util::random_play,
    };

    #[test]
    fn test_ttt() {
        random_play::<TicTacToe>();
    }

    // `HashedPosition::from_position` must reproduce exactly what
    // incremental `apply` calls would have produced -- this is what lets a
    // stateless server (server/adapters/ttt.rs) rebuild a client-supplied
    // position without knowing the move order that reached it.
    #[test]
    fn test_from_position_matches_incremental_hash() {
        let mut state = HashedPosition::new();
        for m in [4u8, 0, 8, 1, 2] {
            state = TicTacToe::apply(state, &Move(m));
            assert_eq!(HashedPosition::from_position(state.position), state);
        }
    }

    #[test]
    fn test_symmetries() {
        if USE_SYMMETRY {
            let mut unhashed = FxHashSet::default();
            let mut hashed = FxHashSet::default();
            let mut n = 0;

            let mut stack = vec![HashedPosition::new()];
            let mut actions = Vec::new();
            while let Some(state) = stack.pop() {
                unhashed.insert(state.position.board);
                hashed.insert(state.hash());
                n += 1;

                if !TicTacToe::is_terminal(&state) {
                    actions.clear();
                    TicTacToe::generate_actions(&state, &mut actions);
                    actions.iter().for_each(|action| {
                        stack.push(TicTacToe::apply(state, action));
                    });
                }
            }

            println!("num positions seen: {n}");
            println!("distinct: {}", unhashed.len());
            println!("distinct w/symmetry: {}", hashed.len());

            // There are 5478 distinct Tic-tac-toe positions, ignoring symmetries.
            assert_eq!(unhashed.len(), 5478);

            // There are 765 unique Tic-tac-toe positions, observing symmetries.
            assert_eq!(hashed.len(), 765);
        }
    }

    use proptest::prelude::*;

    // #[inline]
    // pub fn invert_symmetry(i: usize, symmetry_index: usize) -> usize {
    //     match symmetry_index {
    //         0 => i,
    //         1 => H[i],
    //         2 => V[i],
    //         3 => D[i],
    //         4 => V[H[i]],
    //         5 => D[H[i]],
    //         6 => D[V[i]],
    //         7 => V[D[H[i]]],
    //         _ => unreachable!("Invalid symmetry index"),
    //     }
    // }

    // #[inline]
    // pub fn board_symmetries(board: u32, symmetries: &mut [u32; NUM_SYMMETRIES]) {
    //     debug_assert!(symmetries.iter().all(|x| *x == 0));

    // Define a property-based test for inversion
    use super::*;
    proptest! {

        #[test]
        fn test_idempotent_sym(original_index in 0..9usize, symmetry_used in 0..8usize) {
            // Apply the symmetry
            println!("index: {original_index}");
            println!("symmetry: {symmetry_used}");
            let mut xs = [0; NUM_SYMMETRIES];
            sym::index_symmetries(original_index, &mut xs);
            let transformed_index = xs[symmetry_used];
            println!("index': {transformed_index}");

            // Invert the symmetry
            let inverted_index = sym::invert_symmetry(transformed_index, symmetry_used);
            println!("index'-1: {inverted_index}");

            // Check if the inversion gives back the original index
            prop_assert_eq!(inverted_index, original_index);
        }
    }

    impl render::NodeRender for HashedPosition {}

    #[test]
    fn test_ttt_sym_search() {
        type TS = TreeSearch<TicTacToe, strategy::Ucb1>;
        let mut ts = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(3000)
                .q_init(QInit::Loss)
                .use_transpositions(true),
        );
        let state = HashedPosition::default();
        _ = ts.choose_action(&state);
        println!("hits: {}", ts.table.hits.load(std::sync::atomic::Ordering::Relaxed));

        assert!(ts.table.hits.load(std::sync::atomic::Ordering::Relaxed) > 0);
        render::render_trans(&ts, &HashedPosition::default());
    }

    // Regression test for the player-to-move perspective bug: tree descent
    // and rollout must score every position from the perspective of whoever
    // is actually to move there, not the real root's mover.
    //
    // Position (X to move), reached via moves 0, 4, 8, 1:
    //   X O .
    //   . O .
    //   . . X
    //
    // O already threatens to win immediately at cell 7 (column 1: 1,4,7).
    // Move(7) is the *only* move that doesn't lose outright (a brute-force
    // negamax solve confirms: Move(7) = draw, every other legal move = loss
    // one ply later when O completes that column). Move(6) in particular
    // looks tempting -- it forks two lines of its own (6-7-8 and 0-3-6) --
    // but it ignores O's threat and loses to O simply taking 7, which both
    // wins for O and incidentally blocks X's fork. A search that scores
    // O's replies from the root player's perspective instead of O's own
    // will systematically under-explore O's crushing reply and can easily
    // rate the flashy Move(6) fork above the correct, unglamorous Move(7)
    // block.
    fn must_block_position() -> HashedPosition {
        let mut state = HashedPosition::new();
        for m in [0u8, 4, 8, 1] {
            state = TicTacToe::apply(state, &Move(m));
        }
        state
    }

    #[test]
    fn test_ucb1_finds_forced_block() {
        type TS = TreeSearch<TicTacToe, strategy::Ucb1>;
        let state = must_block_position();
        let mut ts = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(5000)
                .q_init(QInit::Loss)
                .seed(42),
        );

        let action = ts.choose_action(&state);
        assert_eq!(action, Move(7));
    }

    #[test]
    fn test_ucb1dm_finds_forced_block() {
        type TS = TreeSearch<TicTacToe, strategy::Ucb1DM>;
        let state = must_block_position();
        let mut ts = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(5000)
                .q_init(QInit::Loss)
                .seed(42),
        );

        let action = ts.choose_action(&state);
        assert_eq!(action, Move(7));
    }

    // MCTS-Solver A/B on the same forced-block position: with the solver
    // enabled, the tree becomes fully solvable well within budget (the
    // remaining game is only a handful of plies deep from here), so
    // `choose_action` should stop once the root is proven rather than
    // burning the whole `max_iterations` budget -- while still finding the
    // same correct move. With the solver left at its default `false`, this
    // is the untouched plain-UCT path and should run every iteration
    // (matching `test_ucb1_finds_forced_block` above).
    #[test]
    fn test_mcts_solver_finds_forced_block_and_terminates_early() {
        type TS = TreeSearch<TicTacToe, strategy::Ucb1>;
        let state = must_block_position();

        let mut solved = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(5000)
                .q_init(QInit::Loss)
                .use_mcts_solver(true)
                .seed(42),
        );
        let action = solved.choose_action(&state);
        assert_eq!(action, Move(7));
        let solved_iters = solved
            .stats
            .iter_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            solved_iters < 5000,
            "solver should stop once the root is proven, used {solved_iters} iterations"
        );

        let mut unsolved = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(5000)
                .q_init(QInit::Loss)
                .seed(42),
        );
        let action = unsolved.choose_action(&state);
        assert_eq!(action, Move(7));
        let unsolved_iters = unsolved
            .stats
            .iter_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            unsolved_iters, 5000,
            "without the solver, the full iteration budget should still run"
        );
    }

    #[test]
    fn test_mcts_solver_tree_parallel_finds_forced_block_and_terminates_early() {
        type TS = TreeSearch<TicTacToe, strategy::Ucb1>;
        let state = must_block_position();

        let mut solved = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(5000)
                .num_tree_threads(4)
                .q_init(QInit::Loss)
                .use_mcts_solver(true)
                .seed(42),
        );
        let action = solved.choose_action(&state);
        assert_eq!(action, Move(7));
        let solved_iters = solved
            .stats
            .iter_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            solved_iters < 5000,
            "tree-parallel solver should stop once the shared root is proven, \
             used {solved_iters} iterations"
        );

        let mut unsolved = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(5000)
                .num_tree_threads(4)
                .q_init(QInit::Loss)
                .seed(42),
        );
        let action = unsolved.choose_action(&state);
        assert_eq!(action, Move(7));
        let unsolved_iters = unsolved
            .stats
            .iter_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            unsolved_iters, 5000,
            "without the solver, the full iteration budget should still run across \
             all worker threads"
        );
    }
}
