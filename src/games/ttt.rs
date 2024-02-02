use crate::game::Game;
use std::fmt::Display;

#[derive(Clone, Copy, PartialEq, Debug)]
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

const BOARD_LEN: usize = 9;

#[derive(Clone, Copy, Debug)]
pub struct Move(pub u8);

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Position {
    pub turn: Piece,
    pub board: [Option<Piece>; BOARD_LEN],
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

#[derive(Clone, Copy, PartialEq, Debug)]
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
    type P = Piece;

    fn gen_moves(state: &Self::S) -> Vec<Self::M> {
        state.position.gen_moves()
    }

    fn apply(state: &Self::S, m: Self::M) -> Self::S {
        let mut tmp = *state;
        tmp.apply(m);
        tmp
    }

    fn notation(_state: &Self::S, m: &Self::M) -> String {
        let x = m.0 % 3;
        let y = m.0 / 3;
        format!("({}, {})", x, y)
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.position.winner().is_some() || state.position.board.iter().all(|x| x.is_some())
    }

    fn empty_move(_state: &Self::S) -> Self::M {
        Move(0)
    }

    fn winner(state: &Self::S) -> Option<Self::P> {
        if !Self::is_terminal(state) {
            unreachable!();
        }

        state.position.winner()
    }

    fn player_to_move(state: &Self::S) -> Self::P {
        state.position.turn
    }

    fn get_reward(init_state: &Self::S, term_state: &Self::S) -> i32 {
        if !Self::is_terminal(term_state) {
            panic!();
        }

        let winner = Self::winner(term_state);

        if winner.is_some() {
            if Some(Self::player_to_move(init_state)) == winner {
                1
            } else {
                -100
            }
        } else {
            0
        }
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
