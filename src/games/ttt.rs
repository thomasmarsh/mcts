use crate::display::{RectangularBoard, RectangularBoardDisplay};
use crate::game::{Game, PlayerIndex, Symmetry, ZobristHash};
use crate::zobrist::{LazyZobristTable, ZobristKey};
use serde::Serialize;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    X,
    O,
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

    pub fn depth(&self) -> usize {
        self.board.count_ones() as usize
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub const NUM_SYMMETRIES: usize = 8;

pub mod sym {
    use super::NUM_SYMMETRIES;

    // Maps a 3x3 index from a base symmetry to an index in another symmetry
    const SYMMETRIES: [[usize; 9]; 8] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8], // base
        [2, 5, 8, 1, 4, 7, 0, 3, 6], // rot 90
        [8, 7, 6, 5, 4, 3, 2, 1, 0], // rot 180
        [6, 3, 0, 7, 4, 1, 8, 5, 2], // rot 270
        [6, 7, 8, 3, 4, 5, 0, 1, 2], // flip h
        [2, 1, 0, 5, 4, 3, 8, 7, 6], // flip v
        [0, 3, 6, 1, 4, 7, 2, 5, 8], // flip diag
        [8, 5, 2, 7, 4, 1, 6, 3, 0], // flip anti-diag
    ];

    // All are self-inverses except 90 and 270 rotations
    pub const INVERSE: [usize; 8] = [0, 3, 2, 1, 4, 5, 6, 7];

    // Index by [from][to] to get a symmetry to make the desired transform
    pub const TRANSFORM: [[usize; 8]; 8] = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [3, 0, 1, 2, 6, 7, 5, 4],
        [2, 3, 0, 1, 5, 4, 7, 6],
        [1, 2, 3, 0, 7, 6, 4, 5],
        [4, 6, 5, 7, 0, 2, 1, 3],
        [5, 7, 4, 6, 2, 0, 3, 1],
        [6, 5, 7, 4, 3, 1, 0, 2],
        [7, 4, 6, 5, 1, 3, 2, 0],
    ];

    pub fn board_symmetry(board: u32, symmetry: usize) -> u32 {
        let mut result = 0;

        // We could do this in fewer operations with more code with some shifts
        // and some branching. We could also trying to be smart about this by
        // early terminating when we know we have a winner (e.g., by taking
        // highest bits set); that  does not result in faster code, perhaps
        // because of branch prediction failure. This parallel loop is fine.
        (0..9).for_each(|i| {
            let p = (board >> (i << 1)) & 0b11;
            assert_eq!(transform_index(i, 0, symmetry), SYMMETRIES[symmetry][i]);
            result |= p << (SYMMETRIES[symmetry][i] << 1);
        });
        result
    }

    // Converts a u32 bitboard (where positions occupy 2 bits) to all possible symmetries
    #[inline]
    pub fn board_symmetries(board: u32, symmetries: &mut [u32; NUM_SYMMETRIES]) {
        debug_assert!(symmetries.iter().all(|x| *x == 0));

        // We could do this in fewer operations with more code with some shifts
        // and some branching. We could also trying to be smart about this by
        // early terminating when we know we have a winner (e.g., by taking
        // highest bits set); that  does not result in faster code, perhaps
        // because of branch prediction failure. This parallel loop is fine.
        symmetries[0] = board;
        (0..9).for_each(|i| {
            let p = (board >> (i << 1)) & 0b11;
            (1..8).for_each(|j| {
                assert_eq!(transform_index(i, 0, j), SYMMETRIES[j][i]);
                symmetries[j] |= p << (SYMMETRIES[j][i] << 1);
            })
        });
    }

    // Converts an index into a bitboard into all indices for each symmetry.
    #[inline]
    pub fn index_symmetries(i: usize, symmetries: &mut [usize; NUM_SYMMETRIES]) {
        (0..8).for_each(|j| {
            symmetries[j] = SYMMETRIES[j][i];
        });
    }

    // Identifies the canonical symmetry and the relative symmetry index for a given board
    #[inline]
    pub fn identify_symmetry(board: u32) -> (usize, usize) {
        let mut symmetries = [0; NUM_SYMMETRIES];
        board_symmetries(board, &mut symmetries);

        // let mut canonical_symmetry = 0;
        // let mut max_values = [0; 9];

        // // Find the maximum values for each square
        // (0..9).for_each(|i| {
        //     (0..NUM_SYMMETRIES).for_each(|s| {
        //         if symmetries[s] >> (i * 2) & 0b11 > max_values[i] {
        //             max_values[i] = symmetries[s] >> (i * 2) & 0b11;
        //         }
        //     });
        // });

        // // Determine the canonical symmetry
        // for s in 1..NUM_SYMMETRIES {
        //     let mut is_canonical = true;
        //     for i in 0..9 {
        //         if symmetries[s] >> (i * 2) & 0b11 != max_values[i] {
        //             is_canonical = false;
        //             break;
        //         }
        //     }
        //     if is_canonical {
        //         canonical_symmetry = s;
        //         break;
        //     }
        // }

        let canonical_symmetry = symmetries
            .into_iter()
            .enumerate()
            .max_by_key(|(i, v)| (*v, *i))
            .unwrap()
            .0;

        // Determine the relative symmetry index
        let relative_symmetry = INVERSE[canonical_symmetry];

        (canonical_symmetry, relative_symmetry)
    }

    // Given an index and a symmetry, places it in the inverse symmetry
    #[inline]
    pub fn invert_symmetry(index: usize, symmetry: usize) -> usize {
        SYMMETRIES[INVERSE[symmetry]][index]
    }

    // Given an index and a symmetry, places it in the symmetry
    #[inline]
    pub fn transform_index(index: usize, from: usize, to: usize) -> usize {
        SYMMETRIES[TRANSFORM[from][to]][index]
    }

    #[inline]
    pub fn transform_board(board: u32, from: usize, to: usize) -> u32 {
        let s = TRANSFORM[from][to];
        let mut bs = [0; 8];
        board_symmetries(board, &mut bs);
        assert_eq!(bs[s], board_symmetry(board, s));
        bs[s]
    }

    #[inline]
    pub fn invert_board(board: u32, symmetry: usize) -> u32 {
        let s = INVERSE[symmetry];
        let mut bs = [0; 8];
        board_symmetries(board, &mut bs);
        bs[s]
    }
}

////////////////////////////////////////////////////////////////////////////////////////

// 9 playable positions * 2 players
const NUM_MOVES: usize = 18;

static HASHES: LazyZobristTable<NUM_MOVES, 9> = LazyZobristTable::new(0xFEAAE62226597B38);

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct HashedPosition {
    pub position: Position,
    pub(crate) hashes: [ZobristKey; 8],
}

impl HashedPosition {
    pub fn new() -> Self {
        Self {
            position: Position::new(),
            hashes: [ZobristKey::new(); 8],
        }
    }
}

impl Default for HashedPosition {
    fn default() -> Self {
        Self::new()
    }
}

impl HashedPosition {
    #[inline]
    fn apply(&mut self, m: Move) {
        let mut symmetries = [0; NUM_SYMMETRIES];
        sym::index_symmetries(m.0 as usize, &mut symmetries);
        for (i, index) in symmetries.iter().enumerate() {
            let hash_index = (index << 1) | self.position.turn as usize;
            HASHES.apply(hash_index, self.position.depth(), &mut self.hashes[i]);
        }
        self.position.apply(m);
    }

    #[inline(always)]
    fn hash(&self) -> ZobristHash<u64> {
        let (canonical, relative) = sym::identify_symmetry(self.position.board);
        ZobristHash {
            hash: self.hashes[canonical].state,
            symmetry: relative.into(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct TicTacToe;

impl Game for TicTacToe {
    type S = HashedPosition;
    type A = Move;
    type K = u64;

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

    fn winner(state: &Self::S) -> Option<PlayerIndex> {
        if !Self::is_terminal(state) {
            unreachable!();
        }

        state.position.winner().map(|x| (x as usize).into())
    }

    fn player_to_move(state: &Self::S) -> PlayerIndex {
        (state.position.turn as usize).into()
    }

    fn zobrist_hash(state: &Self::S) -> ZobristHash<Self::K> {
        state.hash()
    }

    fn canonicalize_action(state: &Self::S, action: Self::A) -> Self::A {
        let (canonical, relative) = sym::identify_symmetry(state.position.board);
        let base = sym::transform_index(action.0 as usize, 0, canonical);
        Move(base as u8)
    }

    fn relativize_action(state: &Self::S, action: Self::A) -> Self::A {
        let (canonical, relative) = sym::identify_symmetry(state.position.board);
        let base = sym::transform_index(action.0 as usize, 0, canonical);
        let rel = sym::transform_index(base, 0, relative);
        Move(rel as u8)
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

fn debug_print(board: u32) {
    let h = HashedPosition {
        position: Position {
            turn: Piece::X,
            board,
        },
        hashes: [ZobristKey::new(); 8],
    };
    println!("{}", h);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashSet;

    use super::{HashedPosition, TicTacToe};
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

    #[test]
    fn test_symmetries() {
        let mut unhashed = FxHashSet::default();
        let mut hashed = FxHashSet::default();
        let mut n = 0;

        let mut stack = vec![HashedPosition::new()];
        let mut actions = Vec::new();
        while let Some(state) = stack.pop() {
            unhashed.insert(state.position.board);
            hashed.insert(state.hash().hash);
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

    #[test]
    fn test_symmetry_transform() {
        (0..9).for_each(|i| {
            (0..8).for_each(|a| {
                (0..8).for_each(|b| {
                    let j = sym::transform_index(i, 0, a);
                    let k = sym::transform_index(j, a, b);
                    let l = sym::transform_index(k, b, 0);
                    assert_eq!(i, l);
                });
            });
        });
    }

    use proptest::prelude::*;

    fn valid_bitboard() -> impl Strategy<Value = u32> {
        (0u32..(1 << 18)).prop_filter("Valid bitboard value", |&b| {
            for i in 0..9 {
                let mask = 0b11 << (i * 2);
                let segment = (b & mask) >> (i * 2);
                if segment == 0b11 {
                    return false;
                }
            }
            true
        })
    }

    proptest! {
        #[test]
        fn test_symmetry_canonical(input in valid_bitboard()) {
            let (canonical, relative) = sym::identify_symmetry(input);

            let base = sym::transform_board(input, 0, canonical);
            let rel = sym::transform_board(base, 0, relative);
            assert_eq!(rel, input);

            (0..9).for_each(|i| {
                let base = sym::transform_index(i, 0, canonical);
                let ident = sym::transform_index(base, 0, relative);
                assert_eq!(ident, i);
            });
        }

        #[test]
        fn test_symmetry_transform_board(b0 in valid_bitboard()) {
            (0..8).for_each(|a| {
                (0..8).for_each(|b| {
                    let b1 = sym::transform_board(b0, 0, a);
                    let b2 = sym::transform_board(b1, a, b);
                    let b3 = sym::transform_board(b2, b, 0);
                    assert_eq!(b3, b0);
                });
            });
        }

        #[test]
        fn test_idempotent_symmetry_board(board in valid_bitboard()) {
            let mut bs = [0; NUM_SYMMETRIES];
            sym::board_symmetries(board, &mut bs);
            (0..8).for_each(|j| {
                // Apply the symmetry
                let transformed = sym::transform_board(board, 0, j);
                // Invert the symmetry
                let inverted = sym::invert_board(transformed, j);

                // Check if the inversion gives back the original
                assert_eq!(inverted, board);

            });
        }

    }

    #[test]
    fn test_invert_sym() {
        (0..8).for_each(|i| {
            assert_eq!(sym::INVERSE[i], sym::TRANSFORM[i][0]);
        })
    }

    #[test]
    fn test_idempotent_symmetry_index() {
        (0..9).for_each(|original_index| {
            (0..8).for_each(|symmetry_used| {
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
                assert_eq!(inverted_index, original_index);
            });
        });
    }

    impl render::NodeRender for HashedPosition {}

    #[test]
    fn test_ttt_sym_search() {
        type TS = TreeSearch<TicTacToe, strategy::Ucb1>;
        let mut ts = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(3_000_000)
                .q_init(QInit::Infinity)
                .use_transpositions(true),
        );
        let state = HashedPosition::default();
        _ = ts.choose_action(&state);
        println!("hits: {}", ts.table.hits);

        assert!(ts.table.hits > 0);
        render::render_trans(&ts, &HashedPosition::default());
    }
}
