// Described in https://arxiv.org/abs/0801.0579
//
// This game is interesting from a number of perspectives:
//
//  - The order of play is not strictly alternating
//
//  - bidding for the right to move can be applied to any N player game, so
//    could be even used as a Game decorator.
//
//  - Because we don't want the hidden bids to affect the rollouts we have
//    to determinize the results.

use crate::game::{Game, PlayerIndex};
use rand::rngs::SmallRng;
use rand::Rng;
use serde::Serialize;
use std::{cmp::Ordering, fmt::Display};

impl PlayerIndex for Piece {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

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

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Player {
    chips: u16,
    bid: u16,
}

impl Player {
    fn bid_moves(&self) -> Vec<Move> {
        (0..=self.chips).map(Move::Bid).collect()
    }

    fn bid(&mut self, n: u16) {
        debug_assert!(self.bid == 0);
        debug_assert!(self.chips >= n);
        self.chips -= n;
        self.bid = n;
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Phase {
    BidX,
    BidO,
    Tie,
    PlayX,
    PlayO,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize)]
pub enum TiebreakChoice {
    Use,
    Keep,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize)]
pub enum Move {
    Bid(u16),
    Place(u8),
    Tiebreak(TiebreakChoice),
}

const BOARD_LEN: usize = 9;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BiddingTicTacToe {
    pub board: [Option<Piece>; BOARD_LEN],
    pub x: Player,
    pub o: Player,
    pub tiebreaker: Piece,
    pub phase: Phase,
}

impl Default for BiddingTicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl BiddingTicTacToe {
    pub fn new() -> Self {
        Self {
            board: [None; BOARD_LEN],
            x: Player { chips: 100, bid: 0 },
            o: Player { chips: 100, bid: 0 },
            tiebreaker: Piece::O,
            phase: Phase::BidX,
        }
    }

    fn pick_x(&mut self) {
        self.phase = Phase::PlayX;
        self.o.chips += self.o.bid + self.x.bid;
        self.o.bid = 0;
        self.x.bid = 0;
    }

    fn pick_o(&mut self) {
        self.phase = Phase::PlayO;
        self.x.chips += self.o.bid + self.x.bid;
        self.o.bid = 0;
        self.x.bid = 0;
    }

    fn referee(&mut self) {
        match self.x.bid.cmp(&self.o.bid) {
            Ordering::Equal => self.phase = Phase::Tie,
            Ordering::Greater => self.pick_x(),
            Ordering::Less => self.pick_o(),
        }
    }

    fn tiebreak(&mut self, choice: TiebreakChoice) {
        let picked = match choice {
            TiebreakChoice::Use => {
                self.tiebreaker = self.tiebreaker.next();
                self.tiebreaker.next()
            }
            TiebreakChoice::Keep => self.tiebreaker.next(),
        };
        match picked {
            Piece::X => self.pick_x(),
            Piece::O => self.pick_o(),
        }
    }

    fn place(&mut self, pos: usize) {
        assert!(self.board[pos].is_none());
        match self.phase {
            Phase::PlayX => self.board[pos] = Some(Piece::X),
            Phase::PlayO => self.board[pos] = Some(Piece::O),
            _ => unreachable!(),
        }
        self.phase = Phase::BidX;
    }

    pub fn apply(&mut self, m: Move) {
        match m {
            Move::Bid(n) => match self.phase {
                Phase::BidX => {
                    self.x.bid(n);
                    self.phase = Phase::BidO;
                }
                Phase::BidO => {
                    self.o.bid(n);
                    self.referee();
                }
                _ => unreachable!(),
            },
            Move::Tiebreak(choice) => self.tiebreak(choice),
            Move::Place(pos) => self.place(pos as usize),
        }
    }

    pub fn gen_moves(&self) -> Vec<Move> {
        match self.phase {
            Phase::BidX => self.x.bid_moves(),
            Phase::BidO => self.o.bid_moves(),
            Phase::Tie => vec![
                Move::Tiebreak(TiebreakChoice::Use),
                Move::Tiebreak(TiebreakChoice::Keep),
            ],
            Phase::PlayX => self.board_moves(),
            Phase::PlayO => self.board_moves(),
        }
    }

    fn board_moves(&self) -> Vec<Move> {
        self.board
            .into_iter()
            .enumerate()
            .filter(|(_, piece)| piece.is_none())
            .map(|(i, _)| Move::Place(i as u8))
            .collect::<Vec<Move>>()
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

    fn player_to_move(&self) -> Piece {
        match self.phase {
            Phase::BidX => Piece::X,
            Phase::BidO => Piece::O,
            Phase::Tie => self.tiebreaker,
            Phase::PlayX => Piece::X,
            Phase::PlayO => Piece::O,
        }
    }
}

impl Display for BiddingTicTacToe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--")?;
        writeln!(f, "phase: {:?}", self.phase)?;
        writeln!(f, "X: chips={} bid={}", self.x.chips, self.x.bid)?;
        writeln!(f, "O: chips={} bid={}", self.o.chips, self.o.bid)?;
        writeln!(f, "tiebreaker: {:?}", self.tiebreaker)?;
        writeln!(f, "board:")?;
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

impl Game for BiddingTicTacToe {
    type S = BiddingTicTacToe;
    type A = Move;
    type P = Piece;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        actions.extend(state.gen_moves());
    }

    fn apply(mut state: Self::S, m: &Self::A) -> Self::S {
        state.apply(*m);
        state
    }

    fn determinize(state: Self::S, rng: &mut SmallRng) -> Self::S {
        let mut state = state;
        // Not sure this is enough to hide all the bid information. I think
        // we introduce bias by not modeling simultaneous moves directly. But
        // this is a start.
        if state.phase == Phase::BidO {
            let chips = state.x.chips + state.x.bid;
            let n = rng.gen_range(0..=chips);
            state.x.chips = n;
            state.x.bid = chips - n;
        }
        state
    }

    fn notation(_state: &Self::S, m: &Self::A) -> String {
        match m {
            Move::Bid(n) => format!("Bid({})", n),
            Move::Place(pos) => {
                let x = pos % 3;
                let y = pos / 3;
                format!("({}, {})", x, y)
            }
            Move::Tiebreak(TiebreakChoice::Use) => "Tiebreak:Use".into(),
            Move::Tiebreak(TiebreakChoice::Keep) => "Tiebreak:Keep".into(),
        }
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.winner().is_some() || state.board.iter().all(|x| x.is_some())
    }

    fn winner(state: &Self::S) -> Option<Piece> {
        if !Self::is_terminal(state) {
            unreachable!();
        }

        state.winner()
    }

    fn player_to_move(state: &Self::S) -> Piece {
        state.player_to_move()
    }
}
