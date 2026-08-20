//! Margo (pyramidal marble game) on a base-`n` pyramid (`pyramid::Pyramid`).
//!
//! Board size is a runtime parameter, not a distinct compiled type per size
//! (mirroring `games/gonnect`'s variable board size): the published
//! recommended sizes run 4x4 ("Spargo", the Shibumi-set size) through 10x10+
//! ("Extreme"), with 7x7 ("Standard") the most common recommendation and
//! this crate's default. `MAX_N` (10) fixes the storage width (`[u64; 7]`,
//! 448 bits, comfortably covering `total_cells(10) == 385`); every smaller
//! size reuses the same monomorphization with a runtime-sized `Pyramid`.
//!
//! A piece may be placed on any empty, supported cell (`Pyramid::can_place`).
//! Capture is Go-style over the *visible* (non-buried, non-zombie)
//! touching-adjacency graph: after a placement, any enemy group without a
//! liberty is removed -- except a member pinned by a capturing-colour piece
//! resting directly on top of it, which survives in place as a "zombie"
//! (still occupying its cell and still counted for scoring, but permanently
//! excluded from future connectivity, like a buried piece). Suicide is
//! illegal unless the placement itself captures enough to give the new
//! piece a liberty, and a piece hidden by another piece directly above it
//! doesn't count toward any group's connectivity or liberties even though
//! it's still physically on the board. There is no ko or swap rule yet. The
//! game ends when the player to move has no legal placement, and the winner
//! is whoever has more pieces on the board (a draw if equal).

use std::fmt;

use bitboard::{Board, Dyn};
use mcts::game::{Game, PlayerIndex};
use pyramid::{Pyramid, TouchingAdjacency};
use serde::{Deserialize, Serialize};

/// Smallest supported base width -- "Spargo", the size a physical Shibumi
/// ball set (4x4x4) plays Margo on.
pub const MIN_N: usize = 4;

/// Largest supported base width -- the published rules' "Extreme" size.
/// Fixes `Cells`/`GoBoard`'s storage width (`[u64; 7]`, since
/// `pyramid::total_cells(10) == 385` needs 7 words); every smaller size
/// reuses the same storage at runtime, unused high bits simply staying zero.
pub const MAX_N: usize = 10;

/// `State::default()`'s board size -- "Standard", the rules' own
/// recommendation for most games.
pub const DEFAULT_N: usize = 7;

type Cells = Pyramid<[u64; 7], Dyn>;

/// A flat bitset addressed by the same index as `Cells`, used only as a
/// container for [`resolve_captures`]'s Go-style flood fill over
/// `TouchingAdjacency` -- a single-row `Dyn` board, since the pyramid's flat
/// index has no natural row/col split of its own.
type GoBoard = Board<[u64; 7], Dyn, Dyn>;

fn go_board(cells: usize) -> GoBoard {
    GoBoard::new(Dyn(1), Dyn(cells))
}

/// The visible (non-buried, non-zombie) subset of `black`/`white` occupancy,
/// split into per-colour [`GoBoard`]s -- the input [`resolve_captures`] runs
/// its touching-adjacency flood fill over. Buried and zombie cells are
/// dropped entirely (neither colour): buried because a piece hidden by an
/// occluder "does not count in any connection", zombie because a captured
/// group's pinned survivor is permanently excluded from connectivity the
/// same way (see the module docs' zombie summary).
fn visible_boards(occupied: &Cells, black: &Cells, zombie: &Cells) -> (GoBoard, GoBoard) {
    let cells = occupied.total_cells();
    let mut black_board = go_board(cells);
    let mut white_board = go_board(cells);
    for index in 0..cells {
        if !occupied.get_index(index) {
            continue;
        }
        let (col, row, level) = occupied.to_coord(index);
        if occupied.is_buried(col, row, level) || zombie.get_index(index) {
            continue;
        }
        if black.get_index(index) {
            black_board.set_index(index);
        } else {
            white_board.set_index(index);
        }
    }
    (black_board, white_board)
}

/// Go-style legality/capture check for a stone already placed at `index` in
/// `own` (mirrors `bitboard::check_go_move`'s algorithm exactly, but over
/// `TouchingAdjacency`'s table via `table_flood`/`table_neighbor_mask`
/// instead of `Board::flood4`'s rectangular shift math, and taking `own`
/// with the candidate stone already set rather than seeding it internally --
/// a placement can newly bury an existing, already-placed piece elsewhere
/// (occluding it from two levels up), so `own`/`opp` are always rebuilt
/// from scratch against the post-placement buried set rather than patched
/// incrementally, leaving no separate pre-placement board to seed from).
/// Returns the mask
/// of opponent cells to capture if legal, or `None` if the placement is
/// suicide.
///
/// Closing a captured cell's last liberty by occupying its dependent two
/// levels above one of its own supporters has the same retroactive-burial
/// effect: the newly-placed piece buries that supporter (it's exactly the
/// occluder position for it), which was also one of the captured cell's own
/// neighbors, permanently reopening a liberty. This makes some captures --
/// specifically, ones whose last liberty is two levels above an existing
/// piece the captured cell also depends on -- structurally unreachable, not
/// just hard to set up by hand (see `apply_captures_splits_pinned_and_
/// removed_members`, which tests the zombie/removal split directly against
/// a hand-built capture mask rather than fighting this to build one).
fn resolve_captures(
    own: GoBoard,
    opp: GoBoard,
    index: usize,
    adjacency: &TouchingAdjacency,
) -> Option<GoBoard> {
    debug_assert!(own.get_index(index));
    debug_assert!(!opp.get_index(index));
    let occupied = own | opp;
    let group = bitboard::table_flood(own, adjacency, index);
    let group_adjacent = bitboard::table_neighbor_mask(group, adjacency);
    let empty_adjacent = !occupied & group_adjacent;
    let safe = !empty_adjacent.none_set();

    let occupied_adjacent = occupied & group_adjacent;
    let mut seen = own.empty_like();
    let mut will_capture = own.empty_like();
    for point in occupied_adjacent {
        debug_assert!(
            opp.get_index(point),
            "own's flood already covers same-colour neighbors"
        );
        if !seen.get_index(point) {
            let enemy_group = bitboard::table_flood(opp, adjacency, point);
            seen |= enemy_group;
            let enemy_adjacent = bitboard::table_neighbor_mask(enemy_group, adjacency);
            let enemy_empty_adjacent = !occupied & enemy_adjacent;
            if enemy_empty_adjacent.none_set() {
                will_capture |= enemy_group;
            }
        }
    }

    (safe || !will_capture.none_set()).then_some(will_capture)
}

/// Splits a computed capture mask into zombies and outright removals: a
/// captured cell survives in place, keeping its colour, iff a
/// capturing-colour piece rests directly on top of it (one of its own
/// dependents belongs to the mover) -- removing it would leave that piece
/// floating. `mover_black` identifies the capturing colour. Factored out of
/// [`State::resolve_place`] so the zombie/removal split can be unit tested
/// directly against a hand-built capture mask, without needing a
/// [`resolve_captures`]-derived one -- mirroring how
/// `placement_that_captures_to_gain_a_liberty_is_accepted` tests
/// `resolve_captures` directly when a full `State`-reachable scenario isn't
/// tractable to construct by hand.
fn apply_captures(
    mut occupied: Cells,
    mut black: Cells,
    mut zombie: Cells,
    captured: GoBoard,
    mover_black: bool,
) -> (Cells, Cells, Cells) {
    for cell in captured {
        let (c, r, l) = occupied.to_coord(cell);
        let pinned_by_capturer = occupied
            .dependents(c, r, l)
            .into_iter()
            .any(|(dc, dr)| black.get(dc, dr, l + 1) == mover_black);
        if pinned_by_capturer {
            zombie.set_index(cell);
        } else {
            occupied.clear_index(cell);
            black.clear_index(cell);
        }
    }
    (occupied, black, zombie)
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

/// Place a piece at flat pyramid index `.0` (see `pyramid::Pyramid::index`/
/// `to_coord`). `u16` because `total_cells(MAX_N) == 385` overflows `u8`.
#[derive(Copy, Clone, Serialize, Deserialize, Debug, Hash, PartialEq, Eq)]
pub struct Place(pub u16);

/// Board state: `occupied` is every placed piece regardless of colour (the
/// bitset `Pyramid::can_place`/`is_supported` operate over, since support is
/// colour-agnostic); `black` marks which of those cells belong to Black --
/// White's pieces are `occupied & !black`, derived rather than stored
/// separately so the two boards can't drift out of sync with each other.
/// `zombie` marks cells that survived capture pinned by a capturing-colour
/// piece resting directly on top (see the module docs) -- still set in
/// `occupied`/`black`, but permanently excluded from visible connectivity
/// the same way a buried cell is, since the support structure that pins them
/// never un-forms. Board size lives entirely in `occupied`/`black`'s own
/// `Dyn` dimension (`Pyramid::n`), not a separate field, so it can never
/// disagree with them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    occupied: Cells,
    black: Cells,
    zombie: Cells,
    turn: Player,
}

impl Default for State {
    fn default() -> Self {
        Self::new(DEFAULT_N)
    }
}

impl State {
    /// A fresh empty base-`n` board. `n` must be within `MIN_N..=MAX_N`.
    pub fn new(n: usize) -> Self {
        assert!(
            (MIN_N..=MAX_N).contains(&n),
            "Margo board size must be between {MIN_N} and {MAX_N}, got {n}"
        );
        Self {
            occupied: Cells::new(Dyn(n)),
            black: Cells::new(Dyn(n)),
            zombie: Cells::new(Dyn(n)),
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
    pub fn is_zombie(&self, index: usize) -> bool {
        self.zombie.get_index(index)
    }

    #[inline]
    pub fn turn(&self) -> Player {
        self.turn
    }

    /// Attempts to place the player to move's piece at flat `index`,
    /// returning the resulting `(occupied, black, zombie)` boards (captures
    /// already applied) if legal, or `None` if the cell isn't a legal
    /// placement (unsupported/occupied) or would be suicide.
    fn resolve_place(
        &self,
        index: usize,
        adjacency: &TouchingAdjacency,
    ) -> Option<(Cells, Cells, Cells)> {
        let (col, row, level) = self.occupied.to_coord(index);
        if !self.occupied.can_place(col, row, level) {
            return None;
        }
        let mover_black = self.turn == Player::Black;

        let mut new_occupied = self.occupied;
        new_occupied.set_index(index);
        let mut new_black = self.black;
        if mover_black {
            new_black.set_index(index);
        }

        let (black_board, white_board) = visible_boards(&new_occupied, &new_black, &self.zombie);
        let (own_board, opp_board) = if mover_black {
            (black_board, white_board)
        } else {
            (white_board, black_board)
        };

        let captured = if new_occupied.is_buried(col, row, level) {
            // A newly-placed piece that's instantly buried takes part in no
            // connection at all -- neither at risk of suicide nor able to
            // capture anything -- so it's trivially legal with no captures.
            own_board.empty_like()
        } else {
            resolve_captures(own_board, opp_board, index, adjacency)?
        };

        let (new_occupied, new_black, new_zombie) =
            apply_captures(new_occupied, new_black, self.zombie, captured, mover_black);

        Some((new_occupied, new_black, new_zombie))
    }

    fn piece_count(&self, player: Player) -> u32 {
        match player {
            Player::Black => self.black.count_ones(),
            Player::White => self.occupied.count_ones() - self.black.count_ones(),
        }
    }
}

#[derive(Clone)]
pub struct Margo;

impl Game for Margo {
    type S = State;
    type A = Place;
    type P = Player;

    fn apply(mut state: State, action: &Place) -> State {
        let adjacency = TouchingAdjacency::new(state.occupied.n());
        let (new_occupied, new_black, new_zombie) = state
            .resolve_place(action.0 as usize, &adjacency)
            .expect("action generated by generate_actions must be legal");
        state.occupied = new_occupied;
        state.black = new_black;
        state.zombie = new_zombie;
        state.turn = state.turn.next();
        state
    }

    fn generate_actions(state: &State, actions: &mut Vec<Place>) {
        let adjacency = TouchingAdjacency::new(state.occupied.n());
        for index in 0..state.occupied.total_cells() {
            if !state.is_occupied(index) && state.resolve_place(index, &adjacency).is_some() {
                actions.push(Place(index as u16));
            }
        }
    }

    fn player_to_move(state: &State) -> Player {
        state.turn
    }

    /// Piece-count race: the player with more pieces on the board wins once
    /// the mover has no legal placement left. Equal counts are a draw
    /// (`None`), matching the published rule ("draw if equal").
    fn winner(state: &State) -> Option<Player> {
        let black = state.piece_count(Player::Black);
        let white = state.piece_count(Player::White);
        match black.cmp(&white) {
            std::cmp::Ordering::Greater => Some(Player::Black),
            std::cmp::Ordering::Less => Some(Player::White),
            std::cmp::Ordering::Equal => None,
        }
    }

    fn notation(state: &Self::S, action: &Self::A) -> String {
        let (col, row, level) = state.occupied.to_coord(action.0 as usize);
        format!("({col},{row},L{level})")
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

    fn can_place(state: &State, index: usize) -> bool {
        let (col, row, level) = state.occupied.to_coord(index);
        state.occupied.can_place(col, row, level)
    }

    #[test]
    fn random_play_smoke_test() {
        random_play::<Margo>();
    }

    #[test]
    fn base_level_needs_no_support() {
        let state = State::new(DEFAULT_N);
        assert!(can_place(&state, state.occupied.index(0, 0, 0)));
    }

    #[test]
    fn higher_level_needs_all_four_supporters() {
        let mut state = State::new(DEFAULT_N);
        let target = state.occupied.index(0, 0, 1);
        assert!(!can_place(&state, target));

        let adjacency = TouchingAdjacency::new(state.occupied.n());
        for &(col, row) in &[(0, 0), (1, 0), (0, 1)] {
            let idx = state.occupied.index(col, row, 0);
            let (occupied, black, zombie) = state.resolve_place(idx, &adjacency).unwrap();
            state.occupied = occupied;
            state.black = black;
            state.zombie = zombie;
            state.turn = state.turn.next();
        }
        assert!(
            !can_place(&state, target),
            "three of four supporters is not enough"
        );

        let idx = state.occupied.index(1, 1, 0);
        let (occupied, black, zombie) = state.resolve_place(idx, &adjacency).unwrap();
        state.occupied = occupied;
        state.black = black;
        state.zombie = zombie;
        assert!(can_place(&state, target));
    }

    #[test]
    fn occupied_cell_cannot_be_placed_into_again() {
        let mut state = State::new(DEFAULT_N);
        let adjacency = TouchingAdjacency::new(state.occupied.n());
        let idx = state.occupied.index(2, 2, 0);
        let (occupied, black, zombie) = state.resolve_place(idx, &adjacency).unwrap();
        state.occupied = occupied;
        state.black = black;
        state.zombie = zombie;
        assert!(!can_place(&state, idx));
    }

    #[test]
    fn rejects_board_size_outside_supported_range() {
        assert!(std::panic::catch_unwind(|| State::new(MIN_N - 1)).is_err());
        assert!(std::panic::catch_unwind(|| State::new(MAX_N + 1)).is_err());
    }

    #[test]
    fn every_recommended_board_size_supports_random_play() {
        for n in MIN_N..=MAX_N {
            let mut rng = SmallRng::seed_from_u64(n as u64);
            let mut state = State::new(n);
            let max_plies = state.occupied.total_cells() + 2;
            for _ in 0..max_plies {
                if Margo::is_terminal(&state) {
                    break;
                }
                let mut actions = Vec::new();
                Margo::generate_actions(&state, &mut actions);
                assert!(
                    !actions.is_empty(),
                    "n={n}: no legal moves on a non-terminal state"
                );
                let action = actions[rng.gen_range(0..actions.len())];
                state = Margo::apply(state, &action);
            }
        }
    }

    /// A single-member group with one liberty is captured when its last
    /// liberty is filled -- but here that liberty happens to be the group's
    /// own dependent, so the closing black stone rests directly on top of
    /// it, pinning it: the captured piece stays on the board as a zombie
    /// (original colour kept, but excluded from future connectivity) rather
    /// than being removed. Constructed via direct field construction (per
    /// the new-game skill's guidance), not simulated play, so the scenario
    /// is exact regardless of `generate_actions`'s own behaviour.
    ///
    /// Uses the base corner `(0, 0, 0)` rather than an interior cell: a
    /// touching-adjacency neighbor set includes not just the (up to four)
    /// same-level orthogonal cells but also the (up to four) level-`+1`
    /// cells resting on top -- an interior level-0 cell has as many as four
    /// of those extra vertical neighbors, but a corner cell has exactly one
    /// (`(0, 0, 1)`, its only possible dependent -- see
    /// `pyramid::dependent_positions`), and that one is only ever fillable
    /// once `(0, 0, 0)` itself is occupied (since it's one of `(0, 0, 1)`'s
    /// four supporters). So surrounding a corner needs exactly its two
    /// lateral neighbors filled, then the capturing move is the corner's own
    /// dependent -- newly placeable precisely because the target itself
    /// supports it, and unavoidably a pin: a captured cell's last liberty
    /// can only ever be closed by occupying its dependent (every other
    /// neighbor a corner cell has is a lateral, already accounted for), and
    /// closing that specific slot with the mover's own colour is exactly
    /// what "pinned" means. See [`apply_captures`]'s own tests for the
    /// plain-removal case (a captured cell whose dependent, if any, isn't
    /// occupied by the mover) -- reproducing that end-to-end through a real
    /// `State` turns out to need a fully captured multi-level group, and
    /// closing one always buries an existing piece it also depends on (see
    /// `resolve_captures`'s neighboring doc comment on this), reopening a
    /// liberty no further move can close.
    #[test]
    fn scripted_zombie_survives_pinned_capture() {
        let mut state = State::new(DEFAULT_N);
        let white_cell = state.occupied.index(0, 0, 0);
        state.occupied.set_index(white_cell);

        let black_cells = [(1, 0, 0), (0, 1, 0), (1, 1, 0)];
        for &(col, row, level) in &black_cells {
            let idx = state.occupied.index(col, row, level);
            state.occupied.set_index(idx);
            state.black.set_index(idx);
        }
        state.turn = Player::Black;

        let adjacency = TouchingAdjacency::new(state.occupied.n());
        let last = state.occupied.index(0, 0, 1);
        let (new_occupied, new_black, new_zombie) = state.resolve_place(last, &adjacency).unwrap();

        assert!(
            new_occupied.get_index(white_cell),
            "pinned white stone must survive as a zombie, not be removed"
        );
        assert!(
            !new_black.get_index(white_cell),
            "a zombie keeps its original colour"
        );
        assert!(
            new_zombie.get_index(white_cell),
            "captured-but-pinned stone must be marked zombie"
        );
        assert!(
            new_occupied.get_index(last),
            "black's placement must remain"
        );
        assert!(new_black.get_index(last));
    }

    /// [`apply_captures`]'s zombie/removal split, exercised directly against
    /// a hand-built capture mask rather than one [`resolve_captures`]
    /// derives from a real encirclement: two unrelated white cells, one with
    /// a black piece resting directly on top of it (pinned), one with no
    /// piece above it at all (not pinned). Landed this way -- rather than as
    /// a single `State`-reachable captured *group* with one pinned and one
    /// plain-removed member -- because every way of constructing that
    /// requires closing a captured cell's liberty by occupying its
    /// dependent, and doing that two levels above any base cell always
    /// buries an existing supporter that same cell also depends on for its
    /// own liberties (see `resolve_captures`'s doc comment), permanently
    /// reopening a liberty no further move can close. `apply_captures`
    /// itself has no such constraint -- it only inspects a precomputed
    /// capture mask.
    #[test]
    fn apply_captures_splits_pinned_and_removed_members() {
        let mut occupied = Cells::new(Dyn(DEFAULT_N));
        let mut black = Cells::new(Dyn(DEFAULT_N));

        let pinned = occupied.index(0, 0, 0);
        let pin = occupied.index(0, 0, 1);
        let removed = occupied.index(2, 2, 0);

        occupied.set_index(pinned);
        occupied.set_index(pin);
        occupied.set_index(removed);
        black.set_index(pin);

        let mut captured = go_board(occupied.total_cells());
        captured.set_index(pinned);
        captured.set_index(removed);

        let (new_occupied, new_black, new_zombie) =
            apply_captures(occupied, black, Cells::new(Dyn(DEFAULT_N)), captured, true);

        assert!(
            new_occupied.get_index(pinned),
            "pinned member stays on the board"
        );
        assert!(
            !new_black.get_index(pinned),
            "a zombie keeps its original colour"
        );
        assert!(new_zombie.get_index(pinned));

        assert!(
            !new_occupied.get_index(removed),
            "unpinned member is removed"
        );
        assert!(!new_zombie.get_index(removed));
    }

    /// A buried piece (hidden by an occluder two levels up) must not count
    /// toward its own colour's group liberties: constructed directly with a
    /// buried white stone that would otherwise have zero visible liberties
    /// if it counted, but here isn't even part of any group, so it's simply
    /// absent from `visible_boards`'s output.
    #[test]
    fn buried_piece_excluded_from_visible_boards() {
        let mut state = State::new(DEFAULT_N);
        // Build a full pyramid tip around (1, 1) so a piece at level 0
        // becomes buried: occluder sits at (col - 1, row - 1, level + 2).
        // Buried target: (1, 1, 0); occluder: (0, 0, 2).
        let base = [(1, 1), (2, 1), (1, 2), (2, 2)];
        for &(col, row) in &base {
            let idx = state.occupied.index(col, row, 0);
            state.occupied.set_index(idx);
        }
        let mid = [(1, 1), (2, 1), (1, 2)];
        for &(col, row) in &mid {
            let idx = state.occupied.index(col, row, 1);
            state.occupied.set_index(idx);
        }
        let occluder = state.occupied.index(0, 0, 2);
        state.occupied.set_index(occluder);

        let target = state.occupied.index(1, 1, 0);
        assert!(
            state.occupied.is_buried(1, 1, 0),
            "test setup must actually bury the target"
        );

        let (black_board, white_board) =
            visible_boards(&state.occupied, &state.black, &state.zombie);
        assert!(!black_board.get_index(target));
        assert!(!white_board.get_index(target));
    }

    /// Placing into a spot with zero liberties and no capture is illegal.
    ///
    /// Uses the apex of a `MIN_N` (4x4) pyramid rather than a base cell: the
    /// apex's *only* touching neighbors are its four level-2 supporters (no
    /// same-level neighbors -- the apex level is a single cell -- and
    /// nothing rests above it), and by the time it's placeable at all those
    /// four must already be filled (support requirement), so "surround the
    /// apex" is just "fill its four supporters" with no lateral cells to
    /// separately account for. With those four supporters left with open
    /// liberties of their own below (level 1 is untouched here), capturing
    /// them isn't possible either, so placing white at the apex is plain
    /// suicide.
    #[test]
    fn suicide_placement_rejected() {
        let mut state = State::new(MIN_N);
        let supporters = [(0, 0, 2), (1, 0, 2), (0, 1, 2), (1, 1, 2)];
        for &(col, row, level) in &supporters {
            let idx = state.occupied.index(col, row, level);
            state.occupied.set_index(idx);
            state.black.set_index(idx);
        }
        state.turn = Player::White;

        let adjacency = TouchingAdjacency::new(state.occupied.n());
        let apex = state.occupied.index(0, 0, 3);
        assert!(state.resolve_place(apex, &adjacency).is_none());
    }

    /// A placement that itself captures an enemy group is legal even though
    /// the placed stone would otherwise have zero liberties, since the
    /// capture opens up a liberty for it.
    ///
    /// Exercises [`resolve_captures`] directly rather than through
    /// `State`/`Cells`: the apex is the *only* cell whose full neighbor set
    /// can be closed off at all (every other cell has a dependent above it
    /// that, per the support rule, can't possibly be occupied yet, since it
    /// needs the candidate cell itself as one of its own four supporters --
    /// so every non-apex placement always keeps a guaranteed-empty neighbor
    /// and can never be true suicide), but even the apex's four supporters
    /// are so densely cross-connected through the touching graph that
    /// closing every one of their liberties requires occupying essentially
    /// the entire rest of the board (verified empirically for `MIN_N`'s
    /// 30-cell board, not derived by hand) -- and on a real `Cells` board
    /// that dense a fill buries several cells retroactively (a placement
    /// can occlude an existing piece two levels below it), reopening
    /// liberties `resolve_captures` alone doesn't know about. Working
    /// directly against a synthetic `GoBoard` pair sidesteps that
    /// unrelated interaction and isolates the thing this test is actually
    /// about: the "safe or captures" legality check itself, over the real
    /// `TouchingAdjacency` table.
    #[test]
    fn placement_that_captures_to_gain_a_liberty_is_accepted() {
        let n = MIN_N;
        let cells = pyramid::total_cells(n);
        let apex = pyramid::index(n, 0, 0, n - 1);

        let mut own = go_board(cells);
        own.set_index(apex);
        let mut opp = go_board(cells);
        for i in 0..cells {
            if i != apex {
                opp.set_index(i);
            }
        }

        let adjacency = TouchingAdjacency::new(n);
        let captured = resolve_captures(own, opp, apex, &adjacency)
            .expect("capturing placement must be legal");
        for i in 0..cells {
            if i != apex {
                assert!(
                    captured.get_index(i),
                    "every non-apex cell must be captured"
                );
            }
        }
        assert!(!captured.get_index(apex));
    }
}
