//! Akron (pyramidal connection game on `pyramid::Pyramid`).
//!
//! A player either adds a piece from their pile ([`Action::Add`], always
//! level 0 -- see its doc comment for why) or relocates one of their own
//! pieces already on the board ([`Action::Move`]). There is no over/under
//! cut rule and no real win condition yet (`Game::winner` always returns
//! `None`, and `Game::is_terminal` is just "the player to move has no pile
//! left and the board's level-0 surface is full") -- those land in later
//! phases on top of this scaffold. `Action::Move` legality already uses
//! [`connectivity::Groups`]' cut-aware connectivity (not raw touching
//! adjacency) for the "destination must touch the mover's own group" check,
//! since that module already exists; it isn't yet used for the win
//! condition itself.

use std::fmt;

use bitboard::{Adjacency, Dyn};
use mcts::game::{Game, PlayerIndex};
use pyramid::{get_adjacency, Pyramid};
use serde::{Deserialize, Serialize};

pub mod connectivity;
use connectivity::Groups;

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
    /// Relocate the mover's own piece from `.0` to `.1` (both flat pyramid
    /// indices), per the published rules' movement clause: the destination
    /// must be empty, supported, and touch some other cell already in the
    /// mover's own connected group (using [`connectivity::Groups`]'
    /// cut-aware connectivity, excluding the mover's own vacated cell and
    /// anything the resulting cascade itself relocates this turn -- see
    /// `State::move_destinations`). A piece that supports exactly one other
    /// piece drags that piece down to fill the vacated gap
    /// (`pyramid::Pyramid::relocate`'s cascade), possibly recursively.
    Move(u16, u16),
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

    /// The chain of cells a relocation starting at `from` would vacate,
    /// read-only (mirrors, without mutating, the drop chain
    /// `pyramid::Pyramid::relocate`'s cascade performs internally):
    /// `chain[0]` is `from` itself, and each following entry is the single
    /// dependent that drops into the previous entry's gap, one level up.
    /// `None` if `from` (or some link further up the chain) is pinned --
    /// two or more dependents, so there's no single unambiguous drop.
    /// Every entry except the last ends up reoccupied once the whole
    /// cascade settles; only `chain.last()` is actually empty afterwards.
    fn vacated_chain(&self, from: usize) -> Option<Vec<usize>> {
        let mut chain = vec![from];
        let (mut col, mut row, mut level) = self.occupied.to_coord(from);
        loop {
            match self.occupied.dependents(col, row, level).as_slice() {
                [] => return Some(chain),
                &[(dc, dr)] => {
                    col = dc;
                    row = dr;
                    level += 1;
                    chain.push(self.occupied.index(col, row, level));
                }
                _ => return None,
            }
        }
    }

    /// Whether the piece at `from` is a *freedom* of its own group: removing
    /// it doesn't break the rest of that group's connectivity into more than
    /// one piece (a singleton, or already-disconnected, piece is trivially a
    /// freedom). Purely a derived property of the current board -- see the
    /// published rules' "Freedoms" clause -- computed by actually trying the
    /// removal against [`connectivity::Groups`]' cut-aware connectivity.
    fn is_freedom(&self, from: usize) -> bool {
        let mut before = Groups::compute(self.n(), &self.occupied, &self.black);
        let mut after_occupied = self.occupied;
        after_occupied.clear_index(from);
        let mut after = Groups::compute(self.n(), &after_occupied, &self.black);
        let members: Vec<usize> = (0..self.total_cells())
            .filter(|&i| i != from && before.same_group(from, i))
            .collect();
        members.windows(2).all(|w| after.same_group(w[0], w[1]))
    }

    /// Legal destinations for relocating the piece at `from`, per the
    /// published rules' movement clause: an empty cell, not part of this
    /// turn's own cascade (`chain`), that touches some *other*, not-also-
    /// moving-this-turn cell already in `from`'s own cut-aware connected
    /// group -- then confirmed physically placeable (support, and a
    /// successful cascade with no pinning further up) via
    /// `pyramid::Pyramid::relocate` itself against a scratch copy.
    fn move_destinations(&self, from: usize, chain: &[usize]) -> Vec<usize> {
        let n = self.n();
        let adjacency = get_adjacency(n);
        let mut groups = Groups::compute(n, &self.occupied, &self.black);
        let color = self.is_black(from);

        let mut candidates: Vec<usize> = Vec::new();
        for anchor in 0..self.total_cells() {
            if anchor == from
                || !self.is_occupied(anchor)
                || self.is_black(anchor) != color
                || chain.contains(&anchor)
                || !groups.same_group(from, anchor)
            {
                continue;
            }
            for to in adjacency.neighbors(anchor) {
                if self.is_occupied(to) || chain.contains(&to) || candidates.contains(&to) {
                    continue;
                }
                let mut trial = self.occupied;
                if trial.relocate(self.occupied.to_coord(from), self.occupied.to_coord(to)) {
                    candidates.push(to);
                }
            }
        }
        candidates
    }
}

#[derive(Clone)]
pub struct Akron;

impl Game for Akron {
    type S = State;
    type A = Action;
    type P = Player;

    fn apply(mut state: State, action: &Action) -> State {
        match *action {
            Action::Add(index) => {
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
            }
            Action::Move(from, to) => {
                let (from, to) = (from as usize, to as usize);
                let color_from = state.is_black(from);
                debug_assert!(
                    state.is_occupied(from) && color_from == (state.turn == Player::Black),
                    "action generated by generate_actions must be legal"
                );
                // `chain` (see `State::vacated_chain`) is the pillar of
                // single-dependents above `from` that the cascade drops one
                // level each -- its colours have to shift in lockstep with
                // `relocate`'s occupancy-only cascade, since `black` is a
                // separate bitboard `relocate` knows nothing about. Capture
                // each link's colour before mutating anything.
                let chain = state
                    .vacated_chain(from)
                    .expect("action generated by generate_actions must be legal");
                let chain_colors: Vec<bool> = chain.iter().map(|&i| state.is_black(i)).collect();
                let ok = state
                    .occupied
                    .relocate(state.occupied.to_coord(from), state.occupied.to_coord(to));
                debug_assert!(ok, "action generated by generate_actions must be legal");
                for i in 1..chain.len() {
                    if chain_colors[i] {
                        state.black.set_index(chain[i - 1]);
                    } else {
                        state.black.clear_index(chain[i - 1]);
                    }
                }
                // The chain's last link is the one cell that ends up
                // genuinely empty once the whole cascade settles.
                state.black.clear_index(*chain.last().unwrap());
                if color_from {
                    state.black.set_index(to);
                } else {
                    state.black.clear_index(to);
                }
            }
        }
        state.turn = state.turn.next();
        state
    }

    fn generate_actions(state: &State, actions: &mut Vec<Action>) {
        if state.pile(state.turn) > 0 {
            let n = state.occupied.n();
            for index in 0..(n * n) {
                if !state.is_occupied(index) {
                    actions.push(Action::Add(index as u16));
                }
            }
        }

        let color = state.turn == Player::Black;
        for from in 0..state.total_cells() {
            if !state.is_occupied(from) || state.is_black(from) != color {
                continue;
            }
            let Some(chain) = state.vacated_chain(from) else {
                continue;
            };
            if !state.is_freedom(from) {
                continue;
            }
            for to in state.move_destinations(from, &chain) {
                actions.push(Action::Move(from as u16, to as u16));
            }
        }
    }

    /// The player to move has no pile left, or -- ignoring `Action::Move`
    /// entirely -- every level-0 cell is occupied. This deliberately does
    /// not account for `Action::Move`'s availability: with movement in
    /// play but no win condition or repetition rule wired in yet (both
    /// land in a later phase), treating "no legal move at all" as the real
    /// terminal condition here would let a game with an emptied pile but
    /// ongoing relocation freedom run for a very long time (or, without a
    /// repetition rule, not terminate by this criterion at all) -- exactly
    /// what this crate's `#[test]`s must not do (see this repo's rule on
    /// keeping `cargo test --lib` fast). A player whose pile is empty
    /// simply loses their turn's options here for now.
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
        match *action {
            Action::Add(index) => {
                let (col, row, _level) = state.occupied.to_coord(index as usize);
                format!("({col},{row})")
            }
            Action::Move(from, to) => {
                let (fc, fr, fl) = state.occupied.to_coord(from as usize);
                let (tc, tr, tl) = state.occupied.to_coord(to as usize);
                format!("({fc},{fr},{fl})->({tc},{tr},{tl})")
            }
        }
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
            let Action::Add(index) = action else {
                panic!("empty board must only offer Add actions");
            };
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
        let Action::Add(placed) = first else {
            panic!("first move on an empty board must be an Add");
        };
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

    /// Builds a bare `State` for hand-placed movement fixtures: an empty
    /// `n`-board with arbitrary piles (irrelevant to `Action::Move`) and
    /// `turn` set so the caller's mover is legal to act.
    fn state_with(n: usize, turn: Player) -> State {
        State {
            occupied: Pyramid::new(Dyn(n)),
            black: Pyramid::new(Dyn(n)),
            white_pile: pile_size(n),
            black_pile: pile_size(n),
            turn,
        }
    }

    fn place(state: &mut State, col: usize, row: usize, level: usize, black: bool) -> usize {
        state.occupied.set(col, row, level);
        if black {
            state.black.set(col, row, level);
        }
        state.occupied.index(col, row, level)
    }

    /// A relocation that leaves the mover's own group connected is legal
    /// and, once applied, actually moves the piece.
    #[test]
    fn legal_relocation_that_preserves_group_connectivity() {
        let mut state = state_with(4, Player::White);
        // A row of three touching White pieces: (2,0,0) is an endpoint, so
        // removing it leaves (0,0,0)-(1,0,0) still connected.
        place(&mut state, 0, 0, 0, false);
        place(&mut state, 1, 0, 0, false);
        let from = place(&mut state, 2, 0, 0, false);
        let to = state.occupied.index(1, 1, 0);

        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        assert!(
            actions.contains(&Action::Move(from as u16, to as u16)),
            "an endpoint piece must be able to relocate to a cell touching the rest of its group"
        );

        let state = Akron::apply(state, &Action::Move(from as u16, to as u16));
        assert!(!state.is_occupied(from), "the source cell is now empty");
        assert!(state.is_occupied(to) && !state.is_black(to));
        assert_eq!(state.turn(), Player::Black);
    }

    /// A relocation that would disconnect the mover's own group -- moving
    /// the middle piece out of a three-in-a-row chain, with no other path
    /// between the two endpoints -- is never offered, per the published
    /// rules' "Freedoms" clause.
    #[test]
    fn rejected_relocation_that_would_break_group_connectivity() {
        let mut state = state_with(4, Player::White);
        place(&mut state, 0, 0, 0, false);
        let middle = place(&mut state, 1, 0, 0, false);
        place(&mut state, 2, 0, 0, false);

        assert!(
            !state.is_freedom(middle),
            "removing the middle piece disconnects the two endpoints"
        );

        let mut actions = Vec::new();
        Akron::generate_actions(&state, &mut actions);
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::Move(from, _) if *from as usize == middle)),
            "a non-freedom piece must never be offered as a Move source"
        );
    }

    /// Relocating a piece that uniquely supports another piece drags the
    /// dependent down to fill the vacated gap -- including when the
    /// dependent belongs to the *other* player, exercising the colour-bit
    /// bookkeeping `Game::apply`'s `Action::Move` arm has to do in lockstep
    /// with `pyramid::Pyramid::relocate`'s occupancy-only cascade.
    #[test]
    fn relocation_triggers_a_cascade_drop() {
        let mut state = state_with(4, Player::White);
        // A White 2x2 base block supporting a single Black dependent --
        // support is colour-blind, so this is physically valid even though
        // the dependent belongs to the other player.
        place(&mut state, 0, 0, 0, false);
        place(&mut state, 1, 0, 0, false);
        place(&mut state, 0, 1, 0, false);
        let from = place(&mut state, 1, 1, 0, false);
        let dependent = place(&mut state, 0, 0, 1, true);
        let to = state.occupied.index(2, 0, 0);

        assert_eq!(state.vacated_chain(from), Some(vec![from, dependent]));

        let state = Akron::apply(state, &Action::Move(from as u16, to as u16));

        assert!(
            state.is_occupied(from) && state.is_black(from),
            "the Black dependent drops down to fill the vacated gap"
        );
        assert!(
            !state.is_occupied(dependent),
            "the dependent's old cell is genuinely empty now"
        );
        assert!(
            state.is_occupied(to) && !state.is_black(to),
            "the White mover lands at `to`"
        );
    }

    /// A piece that only touches the rest of the mover's group through a
    /// cell that is *itself* dropping this turn (as another player's piece
    /// resting on the mover, in this fixture) has no legal destination at
    /// all: that cell can't be used as the "touches a connected cell"
    /// anchor, per the published rules' cascade caveat, even though it
    /// would otherwise be one (and even though some other, physically
    /// supported cell touches it).
    #[test]
    fn cascade_dropped_piece_cannot_anchor_a_destination() {
        let mut state = state_with(4, Player::White);
        let from = place(&mut state, 0, 0, 0, false);
        let dependent = place(&mut state, 0, 0, 1, true);
        // Support for a same-level neighbour of `dependent`, entirely in
        // Black -- physically valid (support doesn't care about colour) but
        // contributes no White anchor of its own.
        place(&mut state, 1, 0, 0, true);
        place(&mut state, 2, 0, 0, true);
        place(&mut state, 1, 1, 0, true);
        place(&mut state, 2, 1, 0, true);
        let would_be_destination = state.occupied.index(1, 0, 1);

        assert!(
            state.occupied.can_place(1, 0, 1),
            "the candidate destination must be physically placeable"
        );
        assert!(
            get_adjacency(state.n())
                .neighbors(dependent)
                .into_iter()
                .any(|n| n == would_be_destination),
            "the candidate destination must genuinely touch the dropping piece"
        );

        let chain = state
            .vacated_chain(from)
            .expect("a single dependent cascades cleanly");
        assert!(
            state.is_freedom(from),
            "the sole White piece is trivially a freedom"
        );
        let destinations = state.move_destinations(from, &chain);
        assert!(
            !destinations.contains(&would_be_destination),
            "a cell reachable only through this turn's own dropping piece must not be offered"
        );
        assert!(
            destinations.is_empty(),
            "with the dropping piece excluded, this isolated mover has no anchor left at all"
        );
    }
}
