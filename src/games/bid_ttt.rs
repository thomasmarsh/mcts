// Described in https://arxiv.org/abs/0801.0579
//
// This game is interesting from a number of perspectives:
//
//  - The order of play is not strictly alternating
//
//  - bidding for the right to move can be applied to any N player game, so
//    could be even used as a Game decorator.

use crate::game::Game;
use std::{cmp::Ordering, fmt::Display};

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

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TiebreakChoice {
    Use,
    Keep,
}

#[derive(Clone, Copy, PartialEq, Debug)]
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

        // if self.gen_moves().is_empty() {
        // return Some(self.player_to_move().next());
        // }

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
    type M = Move;
    type P = Piece;

    fn gen_moves(state: &Self::S) -> Vec<Self::M> {
        state.gen_moves()
    }

    fn apply(state: &Self::S, m: Self::M) -> Self::S {
        let mut tmp = *state;
        tmp.apply(m);
        tmp
    }

    fn notation(_state: &Self::S, m: &Self::M) -> String {
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
        //|| state.gen_moves().is_empty()
    }

    fn winner(state: &Self::S) -> Option<Self::P> {
        if !Self::is_terminal(state) {
            unreachable!();
        }

        state.winner()
    }

    fn player_to_move(state: &Self::S) -> Self::P {
        state.player_to_move()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::flat_mc::FlatMonteCarloStrategy;
    use crate::strategies::mcts::TreeSearch;
    use crate::strategies::Strategy;

    type AgentMCTS = TreeSearch<BiddingTicTacToe>;
    type AgentFlat = FlatMonteCarloStrategy<BiddingTicTacToe>;

    #[test]
    fn test_bid_ttt() {
        // NOTE: Flat MC is terrible at this game.
        let mut flat = AgentFlat::new().verbose();
        flat.set_samples_per_move(1000);

        let mut mcts = AgentMCTS::new();
        mcts.config.verbose = true;

        let mut state = BiddingTicTacToe::new();
        while !BiddingTicTacToe::is_terminal(&state) {
            println!("{}", state);
            let player = BiddingTicTacToe::player_to_move(&state);
            let m = match player {
                Piece::X => mcts.choose_move(&state),
                Piece::O => flat.choose_move(&state),
            }
            .unwrap();
            println!(
                "move: {:?} {}",
                BiddingTicTacToe::player_to_move(&state),
                BiddingTicTacToe::notation(&state, &m)
            );
            state.apply(m);
        }

        println!("{}", state);
        println!("winner: {:?}", state.winner());
    }
}
