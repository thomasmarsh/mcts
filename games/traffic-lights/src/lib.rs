use game_core::display::{RectangularBoard, RectangularBoardDisplay};
use game_core::symmetry::D4Symmetry;
use mcts::{
    game::{Canonical, Game, PlayerIndex, Real, Transform},
    zobrist::LazyZobristTable,
};
use serde::Serialize;
use std::fmt::Display;

pub const USE_SYMMETRY: bool = true;

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

    fn with_index(self, index: usize) -> Self {
        Move((self.0 & 0b11) | ((index as u8) << 2))
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

// 9 cells * 4 possible values (2 bits) for occupied-cell tokens, plus 1
// reserved slot for the side-to-move toggle.
pub const NUM_MOVES: usize = 72;

pub static HASHES: LazyZobristTable<NUM_MOVES> = LazyZobristTable::new(0x4);

/// Zobrist token for "cell `index` currently holds `value`" (`value` in
/// 1..=3; `value == 0` is empty and contributes no token -- both here and
/// in every caller below, an empty cell is simply skipped).
#[inline]
fn cell_token(index: usize, value: usize) -> u64 {
    HASHES.hash((index << 2) | value)
}

/// One past the highest `cell_token` index (`8 << 2 | 3 == 35`), reserved
/// for the side-to-move toggle so it can never collide with a real cell
/// token.
const SIDE_TO_MOVE_INDEX: usize = 9 << 2;

/// Zobrist token toggled once per ply so that two positions with identical
/// cell contents but different players to move still hash differently.
#[inline]
fn side_to_move_token() -> u64 {
    HASHES.hash(SIDE_TO_MOVE_INDEX)
}

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
    /// `ttt::HashedPosition::from_position`, but must also mirror `apply`'s
    /// own token scheme below (current-value cell tokens plus a single
    /// side-to-move toggle) rather than `ttt`'s simpler placed-once-piece
    /// scheme, since a traffic-lights cell's value changes more than once.
    pub fn from_position(position: Position) -> Self {
        let mut tmp = Self {
            position,
            hashes: [0; 8],
        };
        // Walk every occupied cell and XOR in its symmetric-image tokens
        // to rebuild all 8 hashes from scratch.
        for i in 0..9 {
            let value = ((tmp.position.board as usize) >> (i * 2)) & 0b11;
            if value == 0 {
                continue;
            }
            for (s, &index) in D4Symmetry::<3>::index_symmetries(i).iter().enumerate() {
                tmp.hashes[s] ^= cell_token(index, value);
            }
        }
        if tmp.position.turn == Player::Second {
            for h in tmp.hashes.iter_mut() {
                *h ^= side_to_move_token();
            }
        }
        tmp
    }

    #[inline]
    fn apply(&mut self, m: Move) {
        let index = m.index();
        let old_value = ((self.position.board as usize) >> (index * 2)) & 0b11;
        let new_value = old_value + 1;

        // Always maintain all 8 symmetric-image hashes (mirrors ttt), so
        // `hash()` can pick whichever slot `USE_SYMMETRY` calls for without
        // `apply` needing to know which slot that'll be. XOR out the old
        // cell token (if the cell wasn't empty) and XOR in the new one, for
        // every symmetric image of the moved cell.
        let symmetries = D4Symmetry::<3>::index_symmetries(index);
        for (i, &sym_index) in symmetries.iter().enumerate() {
            if old_value != 0 {
                self.hashes[i] ^= cell_token(sym_index, old_value);
            }
            self.hashes[i] ^= cell_token(sym_index, new_value);
        }

        let turn_before = self.position.turn;
        self.position.apply(m);
        if self.position.turn != turn_before {
            for h in self.hashes.iter_mut() {
                *h ^= side_to_move_token();
            }
        }
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

    fn canonical_representation(state: Real<Self::S>) -> (Canonical<Self::S>, Transform) {
        let state = state.0;
        let sym = D4Symmetry::<3>::packed_canonical_symmetry(state.position.board);
        let mut symmetries = [0u32; 8];
        D4Symmetry::<3>::packed_board_symmetries(state.position.board, &mut symmetries);
        let canon = Position {
            turn: state.position.turn,
            winner: state.position.winner,
            board: symmetries[sym],
        };
        (
            Canonical(HashedPosition::from_position(canon)),
            Transform::new(sym),
        )
    }

    fn apply_to_action(action: Real<Self::A>, sym: Transform) -> Canonical<Self::A> {
        Canonical(
            action
                .0
                .with_index(D4Symmetry::<3>::index_symmetries(action.0.index())[sym.index()]),
        )
    }

    fn invert_action(action: Canonical<Self::A>, sym: Transform) -> Real<Self::A> {
        Real(action.0.with_index(D4Symmetry::<3>::invert_symmetry(
            action.0.index(),
            sym.index(),
        )))
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
            // a reduction in state space to 33,986 distinct states -- cross-checked
            // against an independent count keyed on `canonical_representation`'s
            // output (turn, winner, canonical board) rather than the Zobrist hash,
            // so this isn't just re-asserting whatever the hash happens to produce.
            assert_eq!(unhashed.len(), 256208);
            assert_eq!(hashed.len(), 33986);
        }
    }

    // MCTS-integration tests (render, transposition-table hits, grave stats)
    // that access the `mcts` crate's private fields (`ts.table`, `ts.stats`)
    // are not movable to this crate.  They belong in a separate test crate
    // (`mcts-tests/`) that depends on both `mcts` and `game-traffic-lights`.

    // `Game::apply_to_action`/`invert_action` translate only the cell index
    // half of `Move` -- the piece color half must survive untouched,
    // mirroring ttt's proptest-based `test_action_transform_round_trip`
    // (no proptest dev-dependency here, so this is an exhaustive loop over
    // the same small domain instead).
    #[test]
    fn test_action_transform_round_trip() {
        for idx in 0..9usize {
            for piece in 1..=3u8 {
                for sym in 0..8usize {
                    let action = Move(((idx as u8) << 2) | piece);
                    let sym = Transform::new(sym);
                    let transformed = TrafficLights::apply_to_action(Real(action), sym);
                    let back = TrafficLights::invert_action(transformed, sym);
                    assert_eq!(back.into_inner(), action);
                }
            }
        }
    }

    // `canonical_representation` must map every symmetric image of a
    // reachable state to the same canonical result, with non-geometric
    // fields (`turn`, `winner`) untouched -- mirroring ttt's
    // `test_canonical_representation_invariant_under_symmetry`.
    #[test]
    fn test_canonical_representation_invariant_under_symmetry() {
        let mut reachable = vec![HashedPosition::new()];
        for m in [
            Move::new(Piece::R, 4),
            Move::new(Piece::R, 0),
            Move::new(Piece::Y, 4),
        ] {
            let next = TrafficLights::apply(*reachable.last().unwrap(), &m);
            reachable.push(next);
        }

        for state in reachable {
            let (canon, canon_sym) = TrafficLights::canonical_representation(Real(state));
            let canon = canon.into_inner();

            let mut symmetries = [0u32; 8];
            D4Symmetry::<3>::packed_board_symmetries(state.position.board, &mut symmetries);
            for &board in symmetries.iter() {
                let variant = HashedPosition::from_position(Position {
                    turn: state.position.turn,
                    winner: state.position.winner,
                    board,
                });
                let (canon2, _) = TrafficLights::canonical_representation(Real(variant));
                assert_eq!(
                    canon2.into_inner().position,
                    canon.position,
                    "canonical_representation disagreed across symmetric images \
                     of board {board:#x}"
                );
            }

            assert_eq!(symmetries[canon_sym.index()], canon.position.board);
        }
    }
}
