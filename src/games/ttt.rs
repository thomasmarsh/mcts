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

const BOARD_LEN: usize = 9;

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
        let value = match piece {
            Piece::X => 0b01,
            Piece::O => 0b10,
        };

        self.board |= value << (index << 1);
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
            0b000000_000000_010101,
            0b000000_010101_000000,
            0b010101_000000_000000,
            0b000001_000001_000000,
            0b000100_000100_000100,
            0b010000_010000_010000,
            0b010000_000100_000001,
            0b000001_000100_010000,
        ] {
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

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct HashedPosition {
    pub position: Position,
    pub(crate) hash: u64,
}

impl HashedPosition {
    pub fn new() -> Self {
        Self {
            position: Position::new(),
            hash: 0,
        }
    }
}

impl Default for HashedPosition {
    fn default() -> Self {
        Self::new()
    }
}

impl HashedPosition {
    fn apply(&mut self, m: Move) {
        self.hash ^= HASHES.action_hash((m.0 << 1) as usize | self.position.turn as usize);
        self.position.apply(m);
    }
}

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
        RectangularBoardDisplay(self).fmt(f)?;
        writeln!(f, "is_filled: {}", self.position.is_filled())
    }
}

const NUM_MOVES: usize = BOARD_LEN * 2;
const MAX_DEPTH: usize = 9;

static HASHES: LazyZobristTable<NUM_MOVES, MAX_DEPTH> = LazyZobristTable::new(0xFEAAE62226597B38);

#[cfg(test)]
mod tests {
    use super::TicTacToe;
    use crate::util::random_play;

    #[test]
    fn test_ttt() {
        random_play::<TicTacToe>();
    }
}
