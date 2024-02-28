use crate::display::{RectangularBoard, RectangularBoardDisplay};
use crate::game::{Game, PlayerIndex};
use crate::zobrist::LazyZobristTable;
use serde::Serialize;
use std::fmt;

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

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct HashedPosition {
    pub position: Position,
    pub(crate) hashes: [u64; 8],
}

impl HashedPosition {
    pub fn new() -> Self {
        Self {
            position: Position::new(),
            hashes: [HASHES.initial(); 8],
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
            self.hashes[i] ^= HASHES.hash((index << 1) | self.position.turn as usize);
        }
        self.position.apply(m);
    }

    #[inline(always)]
    fn hash(&self) -> u64 {
        self.hashes[sym::canonical_symmetry(self.position.board)]
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

    use super::{HashedPosition, TicTacToe};
    use crate::{
        game::Game,
        strategies::{
            mcts::{render, strategy, SearchConfig, TreeSearch},
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

    impl render::NodeRender for HashedPosition {}

    #[test]
    fn test_ttt_sym_search() {
        type TS = TreeSearch<TicTacToe, strategy::Ucb1>;
        let mut ts = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(20)
                .q_init(crate::strategies::mcts::node::UnvisitedValueEstimate::Loss)
                .use_transpositions(true),
        );
        let state = HashedPosition::default();
        _ = ts.choose_action(&state);
        println!("hits: {}", ts.table.hits);

        assert!(ts.table.hits > 0);
        render::render_trans(&ts);
    }
}
