#![allow(unused)]

use super::bitboard;
use super::bitboard::BitBoard;
use crate::display::RectangularBoard;
use crate::display::RectangularBoardDisplay;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::zobrist::LazyZobristTable;

use serde::Serialize;
use std::fmt;
use std::mem::swap;

#[derive(Copy, Clone, Serialize, Debug, Default)]
pub enum Player {
    #[default]
    Black,
    White,
}

impl Player {
    fn next(self) -> Player {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Copy, Serialize, Debug, Hash, PartialEq, Eq)]
pub struct Move(u8, u64);

impl Move {
    const SWAP: Move = Move(0xff, 0);
    const NO_MOVE: Move = Move(0xfe, 0);
}

#[derive(Clone, Copy, Serialize, Debug)]
pub struct State<const N: usize> {
    black: BitBoard<N, N>,
    white: BitBoard<N, N>,
    ko_black: BitBoard<N, N>,
    ko_white: BitBoard<N, N>,
    turn: Player,
    can_swap: bool,
    winner: bool,
    hash: u64,
}

impl<const N: usize> Default for State<N> {
    fn default() -> Self {
        Self {
            black: BitBoard::default(),
            white: BitBoard::default(),
            ko_black: BitBoard::ONES,
            ko_white: BitBoard::ONES,
            turn: Player::default(),
            can_swap: true,
            winner: false,
            hash: 0,
        }
    }
}

static HASHES: LazyZobristTable<128> = LazyZobristTable::new(0x50223C7);

impl<const N: usize> State<N> {
    // fn rehash(&mut self) {
    //     self.hash = 0;
    //     for i in self.black {
    //         self.hash ^= HASHES.hash(i << 1);
    //     }

    //     for i in self.white {
    //         self.hash ^= HASHES.hash((i << 1) | 1);
    //     }
    // }

    // fn update_hash(&mut self, action: &Move) {
    //     if action.1 > 0 {
    //         self.rehash()
    //     } else {
    //         self.hash ^= HASHES.hash(((action.0 as usize) << 1) | self.turn as usize);
    //     }
    // }

    #[inline(always)]
    fn occupied(&self) -> BitBoard<N, N> {
        self.black | self.white
    }

    #[inline(always)]
    fn player(&self, player: Player) -> BitBoard<N, N> {
        match player {
            Player::Black => self.black,
            Player::White => self.white,
        }
    }

    #[inline(always)]
    fn player_ko(&self, player: Player) -> BitBoard<N, N> {
        match player {
            Player::Black => self.ko_black,
            Player::White => self.ko_white,
        }
    }

    #[inline(always)]
    fn color(&self, index: usize) -> Player {
        debug_assert!(self.occupied().get(index));
        if self.black.get(index) {
            Player::Black
        } else {
            debug_assert!(self.white.get(index));
            Player::White
        }
    }

    #[inline]
    fn is_ko(&self, index: usize, will_capture: BitBoard<N, N>) -> bool {
        let player = self.player(self.turn) | BitBoard::from_index(index);
        let opponent = self.player(self.turn.next()) & !will_capture;
        let player_ko = self.player_ko(self.turn);
        let opponent_ko = self.player_ko(self.turn.next());
        player_ko == player && opponent_ko == opponent
    }

    #[inline]
    fn valid(&self, index: usize) -> (bool, BitBoard<N, N>) {
        if (self.black | self.white).get(index) {
            return (false, BitBoard::EMPTY);
        }
        bitboard::check_go_move::<N, N>(
            self.player(self.turn),
            self.player(self.turn.next()),
            index,
        )
    }

    #[inline]
    fn apply(&mut self, action: &Move) -> Self {
        if *action == Move::NO_MOVE {
            self.winner = true;
        } else if *action == Move::SWAP {
            swap(&mut self.black, &mut self.white);
            self.can_swap = false;
        } else {
            debug_assert!(!self.occupied().get(action.0 as usize));
            let index = action.0 as usize;
            let player = self.player(self.turn) | BitBoard::from_index(index);
            let opponent = self.player(self.turn.next());
            self.ko_black = self.black;
            self.ko_white = self.white;
            match self.turn {
                Player::Black => {
                    self.black = player;
                    self.white = opponent & !BitBoard::new(action.1);
                }
                Player::White => {
                    self.white = player;
                    self.black = opponent & !BitBoard::new(action.1);
                }
            }
            if player.has_opposite_connection4(index) {
                self.winner = true;
            }
        }
        if self.can_swap && self.occupied().count_ones() == 1 {
            self.can_swap = false;
        }
        // self.update_hash(action);
        if !self.winner {
            self.turn = self.turn.next();
        }

        *self
    }
}

#[derive(Clone)]
pub struct Gonnect<const N: usize>;

impl<const N: usize> Game for Gonnect<N> {
    type S = State<N>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<N>, action: &Move) -> State<N> {
        state.apply(action)
    }

    fn generate_actions(state: &State<N>, actions: &mut Vec<Move>) {
        if state.can_swap && state.occupied().count_ones() == 1 {
            actions.push(Move::SWAP);
        }
        for index in !state.occupied() {
            let (valid, will_capture) = state.valid(index);
            if valid && !state.is_ko(index, will_capture) {
                actions.push(Move(index as u8, will_capture.get_raw()))
            }
        }
        if actions.is_empty() {
            actions.push(Move::NO_MOVE);
        }
    }

    fn is_terminal(state: &State<N>) -> bool {
        state.winner
    }

    fn player_to_move(state: &State<N>) -> Player {
        state.turn
    }

    fn winner(state: &State<N>) -> Option<Player> {
        if state.winner {
            Some(state.turn)
        } else {
            None
        }
    }

    fn parse_action(state: &State<N>, input: &str) -> Option<Self::A> {
        if input.trim() == "swap" {
            if state.can_swap && state.occupied().count_ones() == 1 {
                return Some(Move::SWAP);
            } else {
                eprintln!("invalid move");
                return None;
            }
        }
        let mut chars = input.chars();

        if let Some(file) = chars.next() {
            let col = file.to_ascii_uppercase() as usize - 'A' as usize;
            if col < N {
                if let Ok(row) = chars
                    .collect::<String>()
                    .trim()
                    .parse::<usize>()
                    .map(|x| x - 1)
                {
                    if row < N {
                        let index = BitBoard::<N, N>::to_index(row, col);
                        let (valid, will_capture) = state.valid(index);
                        let is_ko = state.is_ko(index, will_capture);
                        if valid && !is_ko {
                            return Some(Move(index as u8, will_capture.get_raw()));
                        } else {
                            eprintln!("invalid placement: (valid={valid}, is_ko={is_ko})");
                        }
                    } else {
                        eprintln!("row out of range: {row} must be >= 1 and <= {N}");
                    }
                }
            } else {
                eprintln!("col out of range: {col} must be >= 1 and <= {N}");
            }
        }
        None
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        if *action == Move::SWAP {
            "swap".into()
        } else {
            const COL_NAMES: &[u8] = b"ABCDEFGH";
            let (row, col) = BitBoard::<N, N>::to_coord(action.0 as usize);
            format!("{}{}", COL_NAMES[col] as char, row + 1)
        }
    }

    fn num_players() -> usize {
        2
    }
}

impl<const N: usize> RectangularBoard for State<N> {
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        if self.black.get_at(row, col) {
            'X'
        } else if self.white.get_at(row, col) {
            'O'
        } else {
            '.'
        }
    }
}

impl<const N: usize> fmt::Display for State<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

struct MovesDisplay<const N: usize>(State<N>);

impl<const N: usize> RectangularBoard for MovesDisplay<N> {
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        let mut actions = Vec::new();
        Gonnect::generate_actions(&self.0, &mut actions);
        let mut found = false;
        for action in &actions {
            let (r, c) = BitBoard::<N, N>::to_coord(action.0 as usize);
            if r == row && c == col {
                found = true;
            }
        }

        if self.0.black.get_at(row, col) {
            'X'
        } else if self.0.white.get_at(row, col) {
            'O'
        } else if found {
            '+'
        } else {
            '.'
        }
    }
}

#[cfg(test)]
impl<const N: usize> crate::strategies::mcts::render::NodeRender for State<N> {}

#[cfg(test)]
mod tests {
    use crate::{
        strategies::{
            mcts::{node::QInit, render, strategy, SearchConfig, TreeSearch},
            Search,
        },
        util::random_play,
    };

    use super::*;

    #[test]
    fn test_gonnect() {
        random_play::<Gonnect<6>>();
    }

    #[test]
    fn test_render() {
        let mut search = TreeSearch::<Gonnect<8>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Draw)
                .max_iterations(20),
        );
        _ = search.choose_action(&State::default());
        render::render(&search);
    }
}
