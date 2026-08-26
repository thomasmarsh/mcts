//! Akron (pyramidal connection game on `pyramid::Pyramid`).
//!
//! This is a placement-only skeleton: a player either has pieces left in
//! their pile or doesn't, and every pile placement lands on an empty
//! board-level (level 0) hole -- see [`Action::Add`]'s doc comment for why
//! placement is restricted to level 0 here even though `Pyramid::can_place`
//! itself allows any supported cell. There is no movement
//! (`Action::Move`), no over/under cut rule, and no real win condition yet
//! (`Game::winner` always returns `None`, and `Game::is_terminal` is just
//! "the player to move has no pile left and no legal placement") -- those
//! land in later phases on top of this scaffold.

use std::fmt;

use bitboard::Dyn;
use mcts::game::{Game, PlayerIndex};
use pyramid::Pyramid;
use serde::{Deserialize, Serialize};

pub mod connectivity;

/// Smallest supported base width -- same range as `games/margo`, since both
/// sit on the same `pyramid::Pyramid` foundation.
pub const MIN_N: usize = 4;

/// Largest supported base width -- fixes `Cells`' storage width (`[u64;
/// 7]`, since `pyramid::total_cells(10) == 385` needs 7 words).
pub const MAX_N: usize = 10;

/// `State::default()`'s board size -- the published rules' "advanced"
/// 10x10/50-marble option is `MAX_N`; 8x8/32 is the standard size, but this
/// crate follows `games/margo`'s own default choice of 7 for consistency
/// across the pyramidal games.
pub const DEFAULT_N: usize = 7;

type Cells = Pyramid<[u64; 7], Dyn>;

/// A player's starting pile size for a base-`n` board: `n^2 / 2`, "enough
/// to cover the board surface" per the published rules (an 8x8 board gets
/// 32 marbles per player; this crate's other supported sizes scale the same
/// way, rounding down for odd `n`, which the published rules don't cover
/// directly).
pub const fn pile_size(n: usize) -> u32 {
    (n * n / 2) as u32
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Default, Hash, PartialEq, Eq)]
pub enum Player {
    #[default]
    White,
    Black,
}

impl Player {
    fn next(self) -> Player {
        match self {
            Player::White => Player::Black,
            Player::Black => Player::White,
        }
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

/// A move: place a piece from the mover's pile. `.0` is the flat pyramid
/// index of a level-0 (board-level) cell -- see `pyramid::Pyramid::index`/
/// `to_coord`. Pile placements never land above level 0 (the published
/// rules: "pieces added from the pile must be placed directly on the board
/// and not stacked on existing pieces") -- unlike `Pyramid::can_place`
/// itself, which allows any supported cell regardless of how it's reached,
/// `State::generate_actions` filters candidates to level 0 specifically for
/// this action, since `can_place` has no notion of "placed from pile" vs.
/// "relocated". A moved piece is the only way a piece ever reaches a
/// higher level.
#[derive(Copy, Clone, Serialize, Deserialize, Debug, Hash, PartialEq, Eq)]
pub enum Action {
    Add(u16),
}

/// Board state: `occupied` is every placed piece regardless of colour;
/// `black` marks which of those cells belong to Black -- White's pieces are
/// `occupied & !black`, derived rather than stored separately so the two
/// boards can't drift out of sync (same split `games/margo::State` uses).
/// `white_pile`/`black_pile` count each player's remaining unplaced pieces
/// (see [`pile_size`]); a player with an empty pile can no longer add,
/// though this phase has no other action that would let them play on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    occupied: Cells,
    black: Cells,
    white_pile: u32,
    black_pile: u32,
    turn: Player,
}

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_N)
    }
}

impl State {
    /// A fresh empty base-`n` board, both piles full. `n` must be within
    /// `MIN_N..=MAX_N`.
    pub fn new(n: usize) -> Self {
        assert!(
            (MIN_N..=MAX_N).contains(&n),
            "Akron board size must be between {MIN_N} and {MAX_N}, got {n}"
        );
        Self {
            occupied: Cells::new(Dyn(n)),
            black: Cells::new(Dyn(n)),
            white_pile: pile_size(n),
            black_pile: pile_size(n),
            turn: Player::default(),
        }
    }

    #[inline]
    pub fn is_occupied(&self, index: usize) -> bool {
        self.occupied.get_index(index)
    }

    #[inline]
    pub fn is_black(&self, index: usize) -> bool {
        self.black.get_index(index)
    }

    #[inline]
    pub fn is_white(&self, index: usize) -> bool {
        self.is_occupied(index) && !self.is_black(index)
    }

    #[inline]
    pub fn turn(&self) -> Player {
        self.turn
    }

    /// This board's base width -- see `MIN_N`/`MAX_N`.
    #[inline]
    pub fn n(&self) -> usize {
        self.occupied.n()
    }

    /// Total addressable cells for this board's size (see
    /// `pyramid::total_cells`).
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.occupied.total_cells()
    }

    /// Remaining unplaced pieces for `player` -- see [`pile_size`].
    #[inline]
    pub fn pile(&self, player: Player) -> u32 {
        match player {
            Player::White => self.white_pile,
            Player::Black => self.black_pile,
        }
    }

    /// Every occupied cell's flat index, for a wire adapter to serialize.
    pub fn occupied_indices(&self) -> Vec<usize> {
        self.occupied.iter_set().collect()
    }

    /// Every Black-occupied cell's flat index, for a wire adapter to
    /// serialize.
    pub fn black_indices(&self) -> Vec<usize> {
        self.black.iter_set().collect()
    }
}

#[derive(Clone)]
pub struct Akron;

impl Game for Akron {
    type S = State;
    type A = Action;
    type P = Player;

    fn apply(mut state: State, action: &Action) -> State {
        let Action::Add(index) = *action;
        let index = index as usize;
        debug_assert!(
            state.occupied.to_coord(index).2 == 0,
            "Action::Add must target a level-0 cell"
        );
        debug_assert!(
            !state.is_occupied(index),
            "action generated by generate_actions must be legal"
        );
        state.occupied.set_index(index);
        match state.turn {
            Player::White => {
                state.white_pile -= 1;
            }
            Player::Black => {
                state.black.set_index(index);
                state.black_pile -= 1;
            }
        }
        state.turn = state.turn.next();
        state
    }

    fn generate_actions(state: &State, actions: &mut Vec<Action>) {
        if state.pile(state.turn) == 0 {
            return;
        }
        let n = state.occupied.n();
        for index in 0..(n * n) {
            if !state.is_occupied(index) {
                actions.push(Action::Add(index as u16));
            }
        }
    }

    /// The player to move has no legal placement, either because their
    /// pile is empty or every level-0 cell is occupied. This is not yet the
    /// full end condition -- there is no over/under cut rule or win
    /// condition to check for here.
    fn is_terminal(state: &State) -> bool {
        if state.pile(state.turn) == 0 {
            return true;
        }
        let n = state.occupied.n();
        (0..(n * n)).all(|index| state.is_occupied(index))
    }

    fn player_to_move(state: &State) -> Player {
        state.turn
    }

    /// No win condition is implemented yet -- placement alone never wins.
    fn winner(_state: &State) -> Option<Player> {
        None
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        let Action::Add(index) = *action;
        let (col, row, _level) = state.occupied.to_coord(index as usize);
        format!("({col},{row})")
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for level in 0..self.occupied.n() {
            let side = self.occupied.level_side(level);
            writeln!(f, "L{level}:")?;
            for row in (0..side).rev() {
                for col in 0..side {
                    let index = self.occupied.index(col, row, level);
                    let ch = if !self.is_occupied(index) {
                        '.'
                    } else if self.is_black(index) {
                        'X'
                    } else {
                        'O'
                    };
                    write!(f, "{ch} ")?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::util::random_play;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    fn random_play_smoke_test() {
        random_play::<Akron>();
    }

    #[test]
    fn add_actions_only_target_level_zero_cells() {
        let state = State::new(DEFAULT_N);
        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        let n = state.n();
        assert_eq!(actions.len(), n * n);
        for action in actions {
            let Action::Add(index) = action;
            let (_, _, level) = state.occupied.to_coord(index as usize);
            assert_eq!(level, 0, "Add must only target level-0 cells");
        }
    }

    #[test]
    fn occupied_cell_is_not_offered_again() {
        let mut state = State::new(DEFAULT_N);
        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        let first = actions[0];
        state = Akron::apply(state, &first);

        actions.clear();
        Akron::generate_actions(&state, &mut actions);
        let Action::Add(placed) = first;
        assert!(
            !actions.contains(&Action::Add(placed)),
            "a just-occupied cell must not be offered again"
        );
        assert_eq!(actions.len(), state.n() * state.n() - 1);
    }

    #[test]
    fn pile_exhaustion_ends_the_game() {
        // A small board (n = 4, pile_size = 8 per player) so exhausting
        // both piles by legal play is cheap and deterministic.
        let mut rng = SmallRng::seed_from_u64(0);
        let mut state = State::new(4);
        assert_eq!(pile_size(4), 8);

        let mut plies = 0;
        while !Akron::is_terminal(&state) {
            let mut actions = Vec::new();
            Akron::generate_actions(&state, &mut actions);
            assert!(!actions.is_empty(), "non-terminal state must have a move");
            let action = actions[rng.gen_range(0..actions.len())];
            state = Akron::apply(state, &action);
            plies += 1;
            assert!(plies <= state.total_cells() + 2, "game should have ended");
        }

        // Terminal because a pile ran out, or because level 0 filled up
        // (n^2 = 16 is even, so both 8-piece piles exactly fill it) --
        // either way, the player to move must have no legal placement.
        assert_eq!(state.pile(state.turn()), 0);
    }

    #[test]
    fn rejects_board_size_outside_supported_range() {
        assert!(std::panic::catch_unwind(|| State::new(MIN_N - 1)).is_err());
        assert!(std::panic::catch_unwind(|| State::new(MAX_N + 1)).is_err());
    }
}
