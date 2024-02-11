use crate::game::{Game, PlayerIndex};
use serde::Serialize;
use std::fmt::Display;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Player {
    First,
    Second,
}

impl Player {
    fn next(&self) -> Player {
        match self {
            Player::First => Player::Second,
            Player::Second => Player::First,
        }
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    R,
    Y,
    G,
}

const BOARD_LEN: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move(pub u8);

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Position {
    pub turn: Player,
    pub winner: Option<Player>,
    pub board: [Option<Piece>; BOARD_LEN],
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

fn next(current: Option<Piece>) -> Option<Piece> {
    match current {
        None => Some(Piece::R),
        Some(Piece::R) => Some(Piece::Y),
        Some(Piece::Y) => Some(Piece::G),
        Some(Piece::G) => None,
    }
}

impl Position {
    pub fn new() -> Self {
        Self {
            turn: Player::First,
            winner: None,
            board: [None; BOARD_LEN],
        }
    }

    pub fn check_winner(&mut self) {
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
                self.winner = Some(self.turn);
            }
        }
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
        assert!(self.board[m.0 as usize] != Some(Piece::G));
        self.board[m.0 as usize] = next(self.board[m.0 as usize]);
        self.check_winner();
        self.turn = self.turn.next();
    }
}

impl Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..9 {
            match self.board[i] {
                Some(Piece::R) => write!(f, " R")?,
                Some(Piece::Y) => write!(f, " Y")?,
                Some(Piece::G) => write!(f, " G")?,
                None => write!(f, " .")?,
            }
            if (i + 1) % 3 == 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

pub struct TttTrafficLights;

impl Game for TttTrafficLights {
    type S = Position;
    type A = Move;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        actions.extend(state.gen_moves());
    }

    fn apply(state: Self::S, m: &Self::A) -> Self::S {
        let mut tmp = state;
        tmp.apply(*m);
        tmp
    }

    fn notation(_state: &Self::S, m: &Self::A) -> String {
        let x = m.0 % 3;
        let y = m.0 / 3;
        format!("({}, {})", x, y)
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.winner.is_some() || state.board.iter().all(|x| x.is_some())
    }

    fn winner(state: &Self::S) -> Option<Player> {
        if !Self::is_terminal(state) {
            unreachable!();
        }

        state.winner
    }

    fn player_to_move(state: &Self::S) -> Player {
        state.turn
    }
}
