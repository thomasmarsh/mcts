//! Go-specific capture/liberty logic layered on top of the general-purpose
//! [`Board`]: a flood-based, from-scratch legality/capture check
//! ([`check_go_move`]) and an incremental engine ([`GoEngine`]) that
//! maintains group/liberty bookkeeping across [`GoEngine::play`] calls
//! instead of re-flooding the whole board on every probe.

use crate::adjacency::{table_flood, table_neighbor_mask, Adjacency, RectAdjacency, MAX_NEIGHBORS};
use crate::board::Board;
use crate::dim::Dim;
use crate::storage::Storage;

/// From-scratch legality/capture check: would a stone at `index` be legal
/// for `player` (given `opponent`'s stones), and what would it capture?
/// O(size of the touched groups) -- floods the candidate's own group and
/// every adjacent opponent group, from nothing cached. [`GoEngine::check`]
/// is the O(neighbors) incremental counterpart this exists to validate
/// (and to seed [`GoEngine::from_boards`]'s liberty counts).
pub fn check_go_move<S: Storage, R: Dim, C: Dim>(
    player: Board<S, R, C>,
    opponent: Board<S, R, C>,
    index: usize,
) -> (bool, Board<S, R, C>) {
    debug_assert!(!player.intersects(opponent));
    let occupied = player | opponent;
    debug_assert!(!occupied.get_index(index));

    let mut seed = player.empty_like();
    seed.set_index(index);
    let player = player | seed;
    let occupied = player | opponent;
    let group = player.flood4(index);
    let adjacent = group.adjacency_mask();
    let occupied_adjacent = occupied & adjacent;
    let empty_adjacent = !occupied & adjacent;

    // If we have adjacent empty positions we still have liberties.
    let safe = !empty_adjacent.none_set();

    let mut seen = player.empty_like();
    let mut will_capture = player.empty_like();
    for point in occupied_adjacent {
        // By definition, adjacent non-empty points must be the opponent.
        debug_assert!(occupied.get_index(point));
        debug_assert!(opponent.get_index(point));
        if !seen.get_index(point) {
            let group = opponent.flood4(point);
            let adjacent = group.adjacency_mask();
            let empty_adjacent = !occupied & adjacent;
            if empty_adjacent.none_set() {
                will_capture |= group;
            }
            seen |= group;
        }
    }

    (safe || !will_capture.none_set(), will_capture)
}

/// Incremental Go-style capture/liberty engine: union-find group tracking
/// with a per-group liberty *count* (not a full liberty bitboard -- see
/// below), maintained across [`play`](Self::play) calls instead of
/// re-flooded from scratch on every legality probe the way
/// [`check_go_move`] is. [`check`](Self::check) answers "would a stone at
/// this index be legal, and what would it capture?" in O(neighbors) time
/// using only the four orthogonal neighbor cells' cached group liberty
/// counts -- no flood fill -- which is the operation `generate_actions`
/// calls once per *candidate* empty cell. The O(group size) work
/// (relabeling merged groups, rescanning liberties after a capture) only
/// happens in [`play`](Self::play), once per move actually applied, not
/// once per candidate.
///
/// Group membership is a circular linked list per group (`chain_next`),
/// with `group_rep` giving every member's canonical representative cell
/// eagerly (updated for every relabeled cell at merge time, so lookups are
/// O(1), never a path walk). A representative's own `liberties` entry is
/// the group's liberty count; non-representative cells' entries are stale
/// and must not be read directly -- go through `group_rep` first.
///
/// A liberty *count* (rather than a per-group liberty bitboard) is enough
/// because merging groups or removing a captured group only ever needs a
/// boolean "is there at least one liberty left" for legality (any
/// contributing group having a liberty makes the merged group safe,
/// regardless of overlap with another contributing group's liberties -- see
/// [`check`](Self::check)) or an exact recount bounded by the touched
/// group's own size (see [`rescan_liberties`](Self::rescan_liberties)),
/// never a full-board liberty enumeration.
///
/// Generic over `Board`'s own `S`/`R`/`C` axes (storage backend, dimension
/// kind), so the same engine serves both a compile-time-sized game board
/// and a runtime-sized one -- unlike `game-core`'s original `GoEngine`,
/// there's no separate `CELLS` const generic to keep in sync with `N * M`:
/// the per-cell bookkeeping (`group_rep`/`chain_next`/`liberties`) is a
/// `Vec<u16>` sized from `board.len()` at construction time instead of a
/// fixed-size array, since a `Dyn`-dimensioned board has no compile-time
/// cell count to size an array with in the first place.
///
/// Also generic over `A: Adjacency`, defaulting to [`RectAdjacency`] (the
/// original hardcoded rectangular row/col shift math, reproduced exactly --
/// see `adjacency`'s regression tests). This is what lets a non-rectangular
/// board (e.g. pyramid's top-down "touching" table) reuse this engine's
/// union-find liberty bookkeeping wholesale: only the neighbor relation
/// changes, everything else (`black`/`white`/`occupied` storage, group/
/// liberty tables) stays a plain `Board<S, R, C>` addressed by flat index,
/// so a non-rectangular topology can ride on a `Board<S, Dyn, Dyn>` shaped
/// as one flat row (`rows = 1`) purely as a bitset container.
#[derive(Clone, Debug)]
pub struct GoEngine<S: Storage, R: Dim, C: Dim, A: Adjacency = RectAdjacency> {
    black: Board<S, R, C>,
    white: Board<S, R, C>,
    /// `black | white`, maintained incrementally by [`play`](Self::play)
    /// instead of re-ORed from scratch on every [`check`](Self::check) call
    /// -- `generate_actions` calls `check` once per candidate empty cell, so
    /// recomputing this OR per candidate (rather than once per move
    /// actually played) was pure waste.
    occupied: Board<S, R, C>,
    /// The cell-adjacency relation this engine's neighbor lookups go
    /// through -- `RectAdjacency` for an ordinary rectangular board, or a
    /// precomputed table for a non-rectangular one.
    adjacency: A,
    /// `group_rep[cell]` is the representative cell of `cell`'s group, or
    /// [`SENTINEL`](Self::SENTINEL) if `cell` is empty.
    group_rep: Vec<u16>,
    /// Circular linked list over a group's member cells; a singleton group
    /// points to itself. Undefined for empty cells.
    chain_next: Vec<u16>,
    /// `liberties[r]` is the liberty count of the group represented by `r`,
    /// valid only when `group_rep[r] == r`.
    liberties: Vec<u16>,
}

/// Which cell `play` happens to pick as a group's representative (and thus
/// the contents of `group_rep`/`chain_next`/`liberties` at non-representative
/// slots) depends on move order, not just the resulting position -- two
/// engines reached by different move sequences can have identical `black`/
/// `white` occupancy with different internal bookkeeping. Callers (transposition
/// tables, opening-book state indices, the exhaustive-search regression tests
/// in `atarigo`/`gonnect`) all mean "same position" when they compare states,
/// so equality/hashing here is defined purely on occupancy, matching what
/// `check`/`play` actually depend on externally -- the group/liberty tables
/// are a deterministic function of `black`/`white` and carry no extra
/// information once occupancy agrees.
impl<S: Storage, R: Dim, C: Dim, A: Adjacency> PartialEq for GoEngine<S, R, C, A> {
    fn eq(&self, other: &Self) -> bool {
        self.black == other.black && self.white == other.white
    }
}

impl<S: Storage, R: Dim, C: Dim, A: Adjacency> Eq for GoEngine<S, R, C, A> {}

impl<S: Storage, R: Dim, C: Dim, A: Adjacency> std::hash::Hash for GoEngine<S, R, C, A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for w in self.black.words() {
            w.hash(state);
        }
        for w in self.white.words() {
            w.hash(state);
        }
        self.black.rows().hash(state);
        self.black.cols().hash(state);
    }
}

impl<S: Storage, R: Dim, C: Dim> GoEngine<S, R, C, RectAdjacency> {
    pub fn new(rows: R, cols: C) -> Self {
        Self::new_with_adjacency(rows, cols, RectAdjacency::new(rows.get(), cols.get()))
    }

    /// Rebuilds an engine from a pair of occupancy boards by flood-filling
    /// every group once, from scratch -- O(board size), not incremental.
    /// Exists for tests (an independent construction path to cross-check
    /// [`play`](Self::play)'s incremental bookkeeping against) and for
    /// hydrating an engine from a plain `Board` pair (e.g. after
    /// deserializing a saved position), not for the hot path.
    pub fn from_boards(black: Board<S, R, C>, white: Board<S, R, C>) -> Self {
        let adjacency = RectAdjacency::new(black.rows(), black.cols());
        Self::from_boards_with_adjacency(black, white, adjacency)
    }
}

impl<S: Storage, R: Dim, C: Dim, A: Adjacency> GoEngine<S, R, C, A> {
    const SENTINEL: u16 = u16::MAX;

    /// Same as [`GoEngine::<S, R, C>::new`] but over an arbitrary `Adjacency`
    /// provider instead of the default rectangular one -- e.g. pyramid's
    /// precomputed top-down touching table.
    pub fn new_with_adjacency(rows: R, cols: C, adjacency: A) -> Self {
        let black = Board::new(rows, cols);
        let cells = black.len();
        debug_assert!(cells <= u16::MAX as usize, "board too large for u16 cells");
        Self {
            black,
            white: Board::new(rows, cols),
            occupied: Board::new(rows, cols),
            adjacency,
            group_rep: vec![Self::SENTINEL; cells],
            chain_next: vec![Self::SENTINEL; cells],
            liberties: vec![0; cells],
        }
    }

    /// Same as [`GoEngine::<S, R, C>::from_boards`] but over an arbitrary
    /// `Adjacency` provider -- see [`new_with_adjacency`](Self::new_with_adjacency).
    pub fn from_boards_with_adjacency(
        black: Board<S, R, C>,
        white: Board<S, R, C>,
        adjacency: A,
    ) -> Self {
        debug_assert!(!black.intersects(white));
        let occupied = black | white;
        let cells = black.len();
        debug_assert!(cells <= u16::MAX as usize, "board too large for u16 cells");
        let mut engine = Self {
            black,
            white,
            occupied,
            adjacency,
            group_rep: vec![Self::SENTINEL; cells],
            chain_next: vec![Self::SENTINEL; cells],
            liberties: vec![0; cells],
        };
        let mut assigned = black.empty_like();

        for start in 0..cells {
            if !occupied.get_index(start) || assigned.get_index(start) {
                continue;
            }
            let own_board = if black.get_index(start) { black } else { white };
            let group = table_flood(own_board, &engine.adjacency, start);
            assigned |= group;

            let members: Vec<usize> = group.iter_set().collect();
            for (i, &cell) in members.iter().enumerate() {
                engine.group_rep[cell] = start as u16;
                engine.chain_next[cell] = members[(i + 1) % members.len()] as u16;
            }

            let liberties = !occupied & table_neighbor_mask(group, &engine.adjacency);
            engine.liberties[start] = liberties.count_ones() as u16;
        }

        engine
    }

    pub fn black(&self) -> Board<S, R, C> {
        self.black
    }

    pub fn white(&self) -> Board<S, R, C> {
        self.white
    }

    #[inline]
    fn occupied(&self) -> Board<S, R, C> {
        self.occupied
    }

    #[inline]
    fn own_board(&self, black_to_move: bool) -> Board<S, R, C> {
        if black_to_move {
            self.black
        } else {
            self.white
        }
    }

    #[inline]
    fn opp_board(&self, black_to_move: bool) -> Board<S, R, C> {
        if black_to_move {
            self.white
        } else {
            self.black
        }
    }

    #[inline]
    fn rep(&self, cell: usize) -> Option<u16> {
        let r = self.group_rep[cell];
        (r != Self::SENTINEL).then_some(r)
    }

    /// The liberty count of the group occupying `cell`, or `None` if `cell`
    /// is empty. Exposed mainly for tests, which use it to compare two
    /// independently-built engines cell-by-cell without caring which cell
    /// each happened to pick as its group's representative.
    pub fn liberties_at(&self, cell: usize) -> Option<u32> {
        self.rep(cell).map(|r| self.liberties[r as usize] as u32)
    }

    /// All cells belonging to the group represented by `rep`, walking its
    /// chain. O(group size); only called for a group about to be captured
    /// or otherwise actually inspected, never per legality-check candidate.
    fn group_cells(&self, rep: u16) -> Board<S, R, C> {
        let mut mask = self.black.empty_like();
        let mut cell = rep;
        loop {
            mask.set_index(cell as usize);
            cell = self.chain_next[cell as usize];
            if cell == rep {
                break;
            }
        }
        mask
    }

    /// Recomputes the liberty count of the group represented by `rep` by
    /// walking its chain and counting distinct empty neighbor cells against
    /// the current board. A count can't be maintained by simple addition
    /// across a merge or capture (two groups can share a liberty), so this
    /// exact recount -- bounded by the touched group's own size, not the
    /// whole board -- is how [`play`](Self::play) keeps liberty counts
    /// correct after a merge or after a neighboring capture opens up new
    /// liberties.
    fn rescan_liberties(&mut self, rep: u16) {
        let occupied = self.occupied();
        let mut liberty_mask = self.black.empty_like();
        let mut cell = rep;
        loop {
            for nb in self.adjacency.neighbors(cell as usize) {
                if !occupied.get_index(nb) {
                    liberty_mask.set_index(nb);
                }
            }
            cell = self.chain_next[cell as usize];
            if cell == rep {
                break;
            }
        }
        self.liberties[rep as usize] = liberty_mask.count_ones() as u16;
    }

    /// Merges the group represented by `old_rep` into the group represented
    /// by `new_rep`: relabels every one of `old_rep`'s members to point to
    /// `new_rep` (walking its chain -- O(group size), unavoidable since
    /// `group_rep` lookups must stay O(1)), then splices the two circular
    /// chains into one with a single pointer swap.
    fn splice_and_relabel(&mut self, new_rep: u16, old_rep: u16) {
        let mut cell = old_rep;
        loop {
            self.group_rep[cell as usize] = new_rep;
            cell = self.chain_next[cell as usize];
            if cell == old_rep {
                break;
            }
        }
        self.chain_next.swap(new_rep as usize, old_rep as usize);
    }

    /// Checks whether a stone at `index` would be legal for `black_to_move`
    /// and what it would capture, without mutating `self`. Mirrors
    /// [`check_go_move`]'s signature and semantics exactly, but in
    /// O(neighbors) using cached per-group liberty counts instead of
    /// flooding the candidate's would-be group and every adjacent opponent
    /// group from scratch.
    ///
    /// Safety only needs a boolean "is there a liberty somewhere", not an
    /// exact merged count: if any contributing own-color neighbor group has
    /// a liberty other than `index` (i.e. `liberties > 1`, since `index`
    /// being empty and adjacent to that group makes it one of the group's
    /// liberties already), or `index` itself borders an empty cell, the
    /// merged group has at least that one liberty regardless of how much
    /// its contributors' liberty sets overlap elsewhere.
    #[inline]
    pub fn check(&self, black_to_move: bool, index: usize) -> (bool, Board<S, R, C>) {
        debug_assert!(!self.occupied().get_index(index));
        let own_board = self.own_board(black_to_move);
        let opp_board = self.opp_board(black_to_move);
        let occupied = self.occupied();

        let mut safe = false;
        let mut opp_reps_seen = [Self::SENTINEL; MAX_NEIGHBORS];
        let mut n_opp_seen = 0usize;
        let mut will_capture = self.black.empty_like();

        for nb in self.adjacency.neighbors(index) {
            if !occupied.get_index(nb) {
                safe = true;
                continue;
            }
            let r = self
                .rep(nb)
                .expect("occupied cell must have a group representative");
            if opp_board.get_index(nb) {
                if !opp_reps_seen[..n_opp_seen].contains(&r) {
                    opp_reps_seen[n_opp_seen] = r;
                    n_opp_seen += 1;
                    if self.liberties[r as usize] == 1 {
                        will_capture |= self.group_cells(r);
                    }
                }
            } else {
                debug_assert!(own_board.get_index(nb));
                if self.liberties[r as usize] > 1 {
                    safe = true;
                }
            }
        }

        (safe || !will_capture.none_set(), will_capture)
    }

    /// Plays a stone at `index` for `black_to_move` if legal, returning the
    /// captured mask (possibly empty) and updating group/liberty bookkeeping
    /// incrementally. Returns `None` (no mutation) if illegal.
    pub fn play(&mut self, black_to_move: bool, index: usize) -> Option<Board<S, R, C>> {
        let (legal, will_capture) = self.check(black_to_move, index);
        if !legal {
            return None;
        }

        if black_to_move {
            self.black.set_index(index);
        } else {
            self.white.set_index(index);
        }
        self.occupied.set_index(index);
        self.group_rep[index] = index as u16;
        self.chain_next[index] = index as u16;

        for cell in will_capture {
            if black_to_move {
                self.white.clear_index(cell);
            } else {
                self.black.clear_index(cell);
            }
            self.occupied.clear_index(cell);
            self.group_rep[cell] = Self::SENTINEL;
        }

        let own_board = self.own_board(black_to_move);
        let mut own_reps_seen = [Self::SENTINEL; MAX_NEIGHBORS];
        let mut n_own_seen = 0usize;
        for nb in self.adjacency.neighbors(index) {
            if nb != index && own_board.get_index(nb) {
                let r = self
                    .rep(nb)
                    .expect("occupied own-color cell must have a group representative");
                if r != index as u16 && !own_reps_seen[..n_own_seen].contains(&r) {
                    own_reps_seen[n_own_seen] = r;
                    n_own_seen += 1;
                    self.splice_and_relabel(index as u16, r);
                }
            }
        }
        self.rescan_liberties(index as u16);

        let opp_board = self.opp_board(black_to_move);
        let mut opp_reps_seen = [Self::SENTINEL; MAX_NEIGHBORS];
        let mut n_opp_seen = 0usize;
        for nb in self.adjacency.neighbors(index) {
            if opp_board.get_index(nb) {
                let r = self
                    .rep(nb)
                    .expect("occupied opponent cell must have a group representative");
                if !opp_reps_seen[..n_opp_seen].contains(&r) {
                    opp_reps_seen[n_opp_seen] = r;
                    n_opp_seen += 1;
                    self.liberties[r as usize] -= 1;
                }
            }
        }

        // Every surviving group bordering a captured cell just gained a
        // liberty there -- rescan those exactly (a captured cell can border
        // more than one liberty of the same group, so `+= captured_neighbor
        // count` would overcount) rather than guess. A bitboard (not a
        // fixed-size array) tracks which representatives have already been
        // rescanned since an arbitrarily large capture can border an
        // arbitrary number of distinct surviving groups.
        let mut rescanned = self.black.empty_like();
        rescanned.set_index(index);
        for cell in will_capture {
            for nb in self.adjacency.neighbors(cell) {
                if let Some(r) = self.rep(nb) {
                    if !rescanned.get_index(r as usize) {
                        rescanned.set_index(r as usize);
                        self.rescan_liberties(r);
                    }
                }
            }
        }

        Some(will_capture)
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::{Const, Dyn};
    use proptest::prelude::*;

    #[test]
    fn empty_board_every_move_legal_and_captures_nothing() {
        let engine: GoEngine<u64, Const<5>, Const<5>> = GoEngine::new(Const, Const);
        for cell in 0..25 {
            let (legal, captured) = engine.check(true, cell);
            assert!(legal);
            assert!(captured.none_set());
        }
    }

    #[test]
    fn single_stone_surrounded_is_suicide() {
        // Black stone at center of a plus-shape of white stones with no
        // other liberties: playing the center for black must be illegal
        // (suicide, no capture).
        type E = GoEngine<u64, Const<3>, Const<3>>;
        let mut engine = E::new(Const, Const);
        // White occupies the four orthogonal neighbors of center (1,1):
        // (0,1), (2,1), (1,0), (1,2). Their own outer liberties keep them
        // alive so black's move at center doesn't capture anything.
        for (r, c) in [(0, 1), (2, 1), (1, 0), (1, 2)] {
            engine.play(false, r * 3 + c).unwrap();
        }
        let center = 4;
        let (legal, captured) = engine.check(true, center);
        assert!(!legal);
        assert!(captured.none_set());
    }

    #[test]
    fn capturing_a_single_stone() {
        // Black stones at (0,1), (1,0), (1,2), (2,1) surround white at
        // (1,1) except one liberty at... on a 3x3 board (1,1) has exactly
        // those four neighbors, so placing the fourth captures white.
        type E = GoEngine<u64, Const<3>, Const<3>>;
        let mut engine = E::new(Const, Const);
        let white_center = 4;
        engine.play(false, white_center).unwrap();
        for (r, c) in [(0, 1), (1, 0), (1, 2)] {
            engine.play(true, r * 3 + c).unwrap();
        }
        let last = 2 * 3 + 1;
        let captured = engine.play(true, last).unwrap();
        assert_eq!(captured.count_ones(), 1);
        assert!(captured.get_index(white_center));
        assert!(!engine.white().get_index(white_center));
        assert!(!engine.black().get_index(white_center));
    }

    #[test]
    fn dyn_dims_agree_with_const_dims() {
        // Same position, same moves, played on a `Const`-dimensioned engine
        // and a `Dyn`-dimensioned one -- both storage backends this crate
        // supports must produce identical results, exercising the `Vec`-
        // backed bookkeeping's dynamic sizing path.
        type EConst = GoEngine<u64, Const<3>, Const<3>>;
        type EDyn = GoEngine<u64, Dyn, Dyn>;
        let mut a = EConst::new(Const, Const);
        let mut b = EDyn::new(Dyn(3), Dyn(3));

        let moves = [
            (false, 4), // (1,1)
            (true, 1),  // (0,1)
            (true, 3),  // (1,0)
            (true, 5),  // (1,2)
            (true, 7),  // (2,1)
        ];
        for &(black_to_move, index) in &moves {
            let ra = a.play(black_to_move, index);
            let rb = b.play(black_to_move, index);
            assert_eq!(ra.is_some(), rb.is_some());
            if let (Some(ca), Some(cb)) = (ra, rb) {
                assert_eq!(ca.count_ones(), cb.count_ones());
            }
        }
        assert_eq!(
            a.black().iter_set().collect::<Vec<_>>(),
            b.black().iter_set().collect::<Vec<_>>()
        );
        assert_eq!(
            a.white().iter_set().collect::<Vec<_>>(),
            b.white().iter_set().collect::<Vec<_>>()
        );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Oracle 1: an engine rebuilt from an arbitrary random board (via
    // `from_boards`, itself independent of `play`'s incremental bookkeeping)
    // must agree with `check_go_move`'s flood-based legality/capture output
    // at every empty candidate cell.

    /// True if `board` contains a group with zero liberties against
    /// `other` -- a state that can never arise from legal play (such a
    /// group would already have been captured the moment it reached zero
    /// liberties), so arbitrary randomly-generated boards must be filtered
    /// to exclude it before comparing against `check_go_move`/`GoEngine`,
    /// both of which only promise agreement on boards actually reachable
    /// through play.
    fn has_dead_group<S: Storage, R: Dim, C: Dim>(
        board: Board<S, R, C>,
        other: Board<S, R, C>,
    ) -> bool {
        let occupied = board | other;
        let mut seen = board.empty_like();
        for start in board {
            if seen.get_index(start) {
                continue;
            }
            let group = board.flood4(start);
            seen |= group;
            let liberties = !occupied & group.adjacency_mask();
            if liberties.none_set() {
                return true;
            }
        }
        false
    }

    fn check_against_flood_oracle<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
        rows: R,
        cols: C,
        black_bits: &[usize],
        white_bits: &[usize],
    ) {
        let n = rows.get() * cols.get();
        let mut black: Board<S, R, C> = Board::new(rows, cols);
        let mut white: Board<S, R, C> = Board::new(rows, cols);
        for &i in black_bits {
            let i = i % n;
            if !white.get_index(i) {
                black.set_index(i);
            }
        }
        for &i in white_bits {
            let i = i % n;
            if !black.get_index(i) {
                white.set_index(i);
            }
        }
        if has_dead_group(black, white) || has_dead_group(white, black) {
            return;
        }

        let engine = GoEngine::<S, R, C>::from_boards(black, white);

        for index in 0..n {
            if black.get_index(index) || white.get_index(index) {
                continue;
            }
            for &black_to_move in &[true, false] {
                let (player, opponent) = if black_to_move {
                    (black, white)
                } else {
                    (white, black)
                };
                let (expected_legal, expected_capture) = check_go_move(player, opponent, index);
                let (legal, capture) = engine.check(black_to_move, index);
                assert_eq!(legal, expected_legal, "legality mismatch at {index}");
                assert_eq!(
                    capture, expected_capture,
                    "capture mask mismatch at {index}"
                );
            }
        }
    }

    // Oracle 2: replay a sequence of moves (skipping any the flood oracle
    // deems illegal) through both the incremental engine and a plain Board
    // pair updated via `check_go_move`. After every move, engine occupancy
    // must match the reference boards, the returned capture mask must
    // match, and every occupied cell's liberty count must match a fresh
    // `from_boards` rebuild -- which validates `play`'s merge/rescan
    // bookkeeping without caring which cell either path picked as a group's
    // representative.

    fn check_play_sequence_against_oracle<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
        rows: R,
        cols: C,
        moves: &[(bool, usize)],
    ) {
        let n = rows.get() * cols.get();
        let mut engine: GoEngine<S, R, C> = GoEngine::new(rows, cols);
        let mut ref_black: Board<S, R, C> = Board::new(rows, cols);
        let mut ref_white: Board<S, R, C> = Board::new(rows, cols);

        for &(black_to_move, raw_index) in moves {
            let index = raw_index % n;
            if ref_black.get_index(index) || ref_white.get_index(index) {
                continue;
            }
            let (player, opponent) = if black_to_move {
                (ref_black, ref_white)
            } else {
                (ref_white, ref_black)
            };
            let (legal, expected_capture) = check_go_move(player, opponent, index);
            let played = engine.play(black_to_move, index);

            if !legal {
                assert!(
                    played.is_none(),
                    "engine allowed an illegal move at {index}"
                );
                continue;
            }
            let captured = played.unwrap_or_else(|| {
                panic!("engine rejected a move the flood oracle deemed legal at {index}")
            });
            assert_eq!(
                captured, expected_capture,
                "capture mask mismatch at {index}"
            );

            if black_to_move {
                ref_black.set_index(index);
                ref_white &= !expected_capture;
            } else {
                ref_white.set_index(index);
                ref_black &= !expected_capture;
            }

            assert_eq!(
                engine.black(),
                ref_black,
                "black board mismatch after {index}"
            );
            assert_eq!(
                engine.white(),
                ref_white,
                "white board mismatch after {index}"
            );

            let rebuilt = GoEngine::<S, R, C>::from_boards(ref_black, ref_white);
            for cell in 0..n {
                assert_eq!(
                    engine.liberties_at(cell),
                    rebuilt.liberties_at(cell),
                    "liberty count mismatch at {cell} after playing {index}"
                );
            }
        }
    }

    macro_rules! oracle_tests {
        ($mod_name:ident, $storage:ty, $dim_kind:ident, $n:expr, $m:expr, $cells:expr) => {
            mod $mod_name {
                use super::*;

                proptest! {
                    #![proptest_config(ProptestConfig::with_cases(64))]

                    #[test]
                    fn flood_oracle(
                        black_bits in proptest::collection::vec(0usize..$cells, 0..40),
                        white_bits in proptest::collection::vec(0usize..$cells, 0..40),
                    ) {
                        oracle_tests!(@dims $dim_kind, $n, $m, rows, cols);
                        check_against_flood_oracle::<$storage, _, _>(rows, cols, &black_bits, &white_bits);
                    }

                    #[test]
                    fn play_sequence(
                        moves in proptest::collection::vec(
                            (any::<bool>(), 0usize..$cells),
                            0..60,
                        ),
                    ) {
                        oracle_tests!(@dims $dim_kind, $n, $m, rows, cols);
                        check_play_sequence_against_oracle::<$storage, _, _>(rows, cols, &moves);
                    }
                }
            }
        };
        (@dims const, $n:expr, $m:expr, $rows:ident, $cols:ident) => {
            let $rows: Const<$n> = Const;
            let $cols: Const<$m> = Const;
        };
        (@dims dyn, $n:expr, $m:expr, $rows:ident, $cols:ident) => {
            let $rows = Dyn($n);
            let $cols = Dyn($m);
        };
    }

    // Sub-word boards (fit in one word).
    oracle_tests!(oracle_3x3, u64, const, 3, 3, 9);
    oracle_tests!(oracle_6x6, u64, const, 6, 6, 36);
    // Multi-word boards, matching Gonnect/AtariGo's real supported sizes.
    oracle_tests!(oracle_9x9, [u64; 2], const, 9, 9, 81);
    oracle_tests!(oracle_13x13, [u64; 3], const, 13, 13, 169);
    // Multi-word board outside today's supported sizes, exercising a shape
    // no current game exercises yet -- larger than 13x13 but still
    // multi-word, unlike the sub-word boards above.
    oracle_tests!(oracle_11x11, [u64; 2], const, 11, 11, 121);
    // `Dyn` dims, proving the flood oracle and the incremental engine agree
    // on the runtime-sized path a variable-size game would actually use.
    oracle_tests!(oracle_9x9_dyn, [u64; 2], dyn, 9, 9, 81);
    oracle_tests!(oracle_13x13_dyn, [u64; 3], dyn, 13, 13, 169);
}
