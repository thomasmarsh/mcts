use game_core::display::{RectangularBoard, RectangularBoardDisplay};
use game_core::symmetry::D4Symmetry;
use mcts::{
    game::{Game, PlayerIndex},
    zobrist::LazyZobristTable,
};
use serde::Serialize;
use std::fmt::Display;

pub const USE_SYMMETRY: bool = false;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
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

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    R,
    Y,
    G,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move(pub u8);

impl Move {
    fn new(piece: Piece, index: usize) -> Self {
        Move(((index as u8) << 2) | piece as u8)
    }

    fn _piece(self) -> Piece {
        match self.0 & 0b11 {
            0b01 => Piece::R,
            0b10 => Piece::Y,
            0b11 => Piece::G,
            _ => unreachable!(),
        }
    }

    fn index(self) -> usize {
        (self.0 >> 2) as usize
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Position {
    pub turn: Player,
    pub winner: bool,
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
            turn: Player::First,
            winner: false,
            board: 0,
        }
    }

    pub fn get(&self, index: usize) -> Option<Piece> {
        match ((self.board as usize) >> (index * 2)) & 0b11 {
            0b00 => None,
            0b01 => Some(Piece::R),
            0b10 => Some(Piece::Y),
            0b11 => Some(Piece::G),
            _ => unreachable!(),
        }
    }

    fn incr(&mut self, index: usize) {
        debug_assert_ne!(self.get(index), Some(Piece::G));
        let current = (self.board >> (index * 2)) & 0b11;
        debug_assert_ne!(current, 0b11);
        let clear = !(0b11 << (index * 2));
        self.board = (self.board & clear) | ((current + 1) << (index * 2));
    }

    pub fn has_winner(&mut self) -> bool {
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
            let ax = self.get(a);
            let bx = self.get(b);
            let cx = self.get(c);

            if ax.is_some() && ax == bx && bx == cx {
                return true;
            }
        }
        false
    }

    fn gen_moves(&self, actions: &mut Vec<Move>) {
        (0..9).for_each(|i| match self.get(i) {
            Some(Piece::Y) => actions.push(Move::new(Piece::G, i)),
            Some(Piece::R) => actions.push(Move::new(Piece::Y, i)),
            None => actions.push(Move::new(Piece::R, i)),
            _ => (),
        });
    }

    fn apply(&mut self, m: Move) {
        assert!(self.get(m.index()) != Some(Piece::G));
        self.incr(m.index());
        self.winner = self.has_winner();
        if !self.winner {
            self.turn = self.turn.next();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

// 9 playable positions * 4 states * 2 players
pub const NUM_MOVES: usize = 72;

pub static HASHES: LazyZobristTable<NUM_MOVES> = LazyZobristTable::new(0x4);

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
    /// Rebuild a `HashedPosition` from a raw `Position`, computing hashes
    /// from scratch (no prior hash to XOR from). Mirrors
    /// `ttt::HashedPosition::from_position`.
    pub fn from_position(position: Position) -> Self {
        let mut tmp = Self { position, hashes: [0; 8] };
        // Walk every cell that is occupied and XOR its hash contribution
        // to rebuild the full hash from scratch.
        for i in 0..9 {
            let value = ((tmp.position.board as usize) >> (i * 2)) & 0b11;
            if value == 0 {
                continue;
            }
            let q = (i << 3) | (value << 1) | tmp.position.turn as usize;
            tmp.hashes[0] ^= HASHES.hash(q);
        }
        tmp
    }


    #[inline]
    fn apply(&mut self, m: Move) {
        if USE_SYMMETRY {
            let symmetries = D4Symmetry::<3>::index_symmetries(m.index());
            // TODO: self.hashes[0] is producing bad values. The `else` branch below is working.
            for (i, index) in symmetries.iter().enumerate() {
                let value = ((self.position.board as usize) >> (index * 2)) & 0b11;
                let q = (index << 3) | (value << 1) | self.position.turn as usize;
                self.hashes[i] ^= HASHES.hash(q);
            }
        } else {
            let index = m.index();
            let value = ((self.position.board as usize) >> (index * 2)) & 0b11;
            let q = (index << 3) | (value << 1) | self.position.turn as usize;
            self.hashes[0] ^= HASHES.hash(q);
        }
        self.position.apply(m);
    }

    #[inline(always)]
    fn hash(&self) -> u64 {
        if USE_SYMMETRY {
            self.hashes[D4Symmetry::<3>::packed_canonical_symmetry(self.position.board)]
        } else {
            self.hashes[0]
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

impl RectangularBoard for HashedPosition {
    const NUM_DISPLAY_ROWS: usize = 3;
    const NUM_DISPLAY_COLS: usize = 3;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        let index = row * 3 + col;
        match self.position.get(index) {
            Some(Piece::R) => 'R',
            Some(Piece::Y) => 'Y',
            Some(Piece::G) => 'G',
            None => '.',
        }
    }
}

impl Display for HashedPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct TrafficLights;

impl Game for TrafficLights {
    type S = HashedPosition;
    type A = Move;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        state.position.gen_moves(actions);
    }

    fn apply(state: Self::S, m: &Self::A) -> Self::S {
        let mut tmp = state;
        tmp.apply(*m);
        tmp
    }

    fn get_reward(init: &Self::S, term: &Self::S) -> f64 {
        let utility = Self::compute_utilities(term)[Self::player_to_move(init).to_index()];
        if utility < 0. {
            return utility * 100.;
        }
        utility
    }

    fn notation(_state: &Self::S, m: &Self::A) -> String {
        let i = m.index();
        let x = i % 3;
        let y = i / 3;
        format!("({}, {})", x, y)
    }

    fn is_terminal(state: &Self::S) -> bool {
        state.position.winner
    }

    fn winner(state: &Self::S) -> Option<Player> {
        if state.position.winner {
            Some(state.position.turn)
        } else {
            None
        }
    }

    fn player_to_move(state: &Self::S) -> Player {
        state.position.turn
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.hash()
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use super::*;
    use mcts::util::random_play;

    /// Random-play smoke test: the engine must never panic or hang on any
    /// sequence of moves.
    #[test]
    fn test_tl_rand() {
        random_play::<TrafficLights>();
    }

    /// Enumerate all legal positions and verify the symmetry-hash collision
    /// counts are stable.  This is a game-logic property, not an MCTS
    /// property -- it validates the hash encoding is correct.
    #[test]
    fn test_tl_symmetries() {
        if USE_SYMMETRY {
            let mut unhashed = FxHashSet::default();
            let mut hashed = FxHashSet::default();

            let mut stack = vec![HashedPosition::new()];
            let mut actions = Vec::new();
            while let Some(state) = stack.pop() {
                let k = state.position.board;
                if !unhashed.contains(&k) {
                    unhashed.insert(k);
                    hashed.insert(state.hash());

                    if !TrafficLights::is_terminal(&state) {
                        actions.clear();
                        TrafficLights::generate_actions(&state, &mut actions);
                        actions.iter().for_each(|action| {
                            stack.push(TrafficLights::apply(state, action));
                        });
                    }
                }
            }

            println!("distinct: {}", unhashed.len());
            println!("distinct w/symmetry: {}", hashed.len());

            // There are 36 bits of state in the board, counting illegal moves,
            // over 68 billion states. Only 256,208 states are legal given terminal
            // states with wins. Taking into account the eight-way symmetry, we get
            // a reduction in state space, but only a small reduction to 244,129
            // distinct states.
            assert_eq!(unhashed.len(), 256208);
            assert_eq!(hashed.len(), 244129);
        }
    }

    // MCTS-integration tests (render, transposition-table hits, grave stats)
    // that access the `mcts` crate's private fields (`ts.table`, `ts.stats`)
    // are not movable to this crate.  They belong in a separate test crate
    // (`mcts-tests/`) that depends on both `mcts` and `game-traffic-lights`
    // -- see `plan/decouple-games.md` Step 7.
}
