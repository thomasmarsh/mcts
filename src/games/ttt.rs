use crate::game::{Game, Winner};

use std::{fmt::Display, ops::Div};

#[derive(Clone, Copy, PartialEq)]
enum Piece {
    X,
    O,
}

impl Piece {
    fn next(self) -> Piece {
        match self {
            Piece::X => Piece::O,
            Piece::O => Piece::X,
        }
    }
}

const BOARD_LEN: usize = 9;

#[derive(Clone, Copy)]
pub struct Move(pub u8);

#[derive(Clone, Copy)]
pub struct Position {
    pub turn: Piece,
    pub board: [Option<Piece>; BOARD_LEN],
}

impl Position {
    pub fn new() -> Self {
        Self {
            turn: Piece::X,
            board: [None; BOARD_LEN],
        }
    }

    pub fn winner(&self) -> Option<Piece> {
        let check = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ];

        for (a, b, c) in check {
            let ax = self.board[a];
            let bx = self.board[b];
            let cx = self.board[c];

            if ax.is_some() && ax == bx && bx == cx {
                return ax;
            }
        }

        None
    }

    pub fn gen_moves(&self) -> Vec<Move> {
        self.board
            .into_iter()
            .enumerate()
            .filter(|(_, piece)| piece.is_none())
            .map(|(i, _)| Move(i as u8))
            .collect::<Vec<Move>>()
    }

    pub fn apply(&mut self, m: Move) {
        assert!(self.board[m.0 as usize].is_none());
        self.board[m.0 as usize] = Some(self.turn);
        self.turn = self.turn.next();
    }
}

impl Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..9 {
            match self.board[i] {
                Some(Piece::X) => write!(f, " X")?,
                Some(Piece::O) => write!(f, " O")?,
                None => write!(f, " .")?,
            }
            if (i + 1) % 3 == 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

pub struct TicTacToe;

#[derive(Clone, Copy)]
pub struct HashedPosition {
    pub position: Position,
    hash: u64,
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
        self.hash ^= HASHES[(m.0 << 1) as usize | self.position.turn as usize];
        self.position.apply(m);
    }
}

impl Game for TicTacToe {
    type S = HashedPosition;
    type M = Move;

    fn generate_moves(state: &Self::S, moves: &mut Vec<Self::M>) {
        moves.extend(state.position.gen_moves())
    }

    fn apply(state: &mut Self::S, m: Self::M) -> Option<Self::S> {
        let mut tmp = state.clone();
        tmp.apply(m);
        Some(tmp)
    }

    fn notation(_state: &Self::S, m: Self::M) -> Option<String> {
        let x = m.0 % 3;
        let y = m.0 / 3;
        Some(format!("({}, {})", x, y))
    }

    fn undo(state: &mut Self::S, m: Self::M) {
        state.position.board[m.0 as usize] = None
    }

    fn get_winner(state: &Self::S) -> Option<Winner> {
        state
            .position
            .winner()
            .map(|x| {
                if x == state.position.turn {
                    Winner::PlayerToMove
                } else {
                    Winner::PlayerJustMoved
                }
            })
            .or_else(|| {
                if state.position.gen_moves().is_empty() {
                    Some(Winner::Draw)
                } else {
                    None
                }
            })
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.hash
    }
}

const HASHES: [u64; BOARD_LEN * 2] = [
    0xFEAAE62226597B38,
    0x36CE71B949976A40,
    0x5CC3B44974898A3F,
    0xC9CDBA14D63CD1A5,
    0xB0D6E4CAC682A58B,
    0x0F71B6F72EECF09E,
    0xDE16109EC19E1A28,
    0x0575879F44F30B68,
    0x2A4E85C28F6D50D2,
    0x0EBF01E9C0DAAD57,
    0x0C5BD5F40C96FC69,
    0x4C67B789C5C5442B,
    0x0F8928C057283D2E,
    0x20AA167E48D874E0,
    0x49765C9A3FD19766,
    0x0C649A5927A4705F,
    0x762A61CA14D1297A,
    0x97FE5DDB4E75CC70,
];
