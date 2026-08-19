use crate::bigbitboard::BigBitBoard;

/// Incremental Go-style capture/liberty engine: union-find group tracking
/// with a per-group liberty *count* (not a full liberty bitboard -- see
/// below), maintained across [`play`](Self::play) calls instead of
/// re-flooded from scratch on every legality probe the way
/// [`bigbitboard::check_go_move`] is. [`check`](Self::check) answers
/// "would a stone at this index be legal, and what would it capture?" in
/// O(neighbors) time using only the four orthogonal neighbor cells' cached
/// group liberty counts -- no flood fill -- which is the operation
/// `generate_actions` calls once per *candidate* empty cell. The O(group
/// size) work (relabeling merged groups, rescanning liberties after a
/// capture) only happens in [`play`](Self::play), once per move actually
/// applied, not once per candidate.
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
/// Generic over `N` (board is `N x N`) and `WORDS` the same way
/// `BigBitBoard` is, plus `CELLS = N * N` for the same reason `BigBitBoard`
/// takes `WORDS` as an independent parameter: stable Rust can't derive an
/// array length from another const generic, so callers supply it and
/// [`CHECK_CELLS`](Self::CHECK_CELLS) catches a mismatch at
/// monomorphization.
#[derive(Clone, Copy, Debug)]
pub struct GoEngine<const N: usize, const WORDS: usize, const CELLS: usize> {
    black: BigBitBoard<N, N, WORDS>,
    white: BigBitBoard<N, N, WORDS>,
    /// `black | white`, maintained incrementally by [`play`](Self::play)
    /// instead of re-ORed from scratch on every [`check`](Self::check) call
    /// -- `generate_actions` calls `check` once per candidate empty cell, so
    /// recomputing this OR per candidate (rather than once per move
    /// actually played) was pure waste.
    occupied: BigBitBoard<N, N, WORDS>,
    /// `group_rep[cell]` is the representative cell of `cell`'s group, or
    /// [`SENTINEL`](Self::SENTINEL) if `cell` is empty.
    group_rep: [u16; CELLS],
    /// Circular linked list over a group's member cells; a singleton group
    /// points to itself. Undefined for empty cells.
    chain_next: [u16; CELLS],
    /// `liberties[r]` is the liberty count of the group represented by `r`,
    /// valid only when `group_rep[r] == r`.
    liberties: [u16; CELLS],
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
impl<const N: usize, const WORDS: usize, const CELLS: usize> PartialEq
    for GoEngine<N, WORDS, CELLS>
{
    fn eq(&self, other: &Self) -> bool {
        self.black == other.black && self.white == other.white
    }
}

impl<const N: usize, const WORDS: usize, const CELLS: usize> Eq for GoEngine<N, WORDS, CELLS> {}

impl<const N: usize, const WORDS: usize, const CELLS: usize> std::hash::Hash
    for GoEngine<N, WORDS, CELLS>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.black.hash(state);
        self.white.hash(state);
    }
}

impl<const N: usize, const WORDS: usize, const CELLS: usize> GoEngine<N, WORDS, CELLS> {
    const SENTINEL: u16 = u16::MAX;

    /// Referenced by every constructor path so a mis-sized `CELLS` fails to
    /// compile at the call site's monomorphization, rather than silently
    /// indexing out of bounds.
    const CHECK_CELLS: () = assert!(
        CELLS == N * N,
        "GoEngine::<N, WORDS, CELLS>: CELLS must equal N * N"
    );

    pub const fn new() -> Self {
        let () = Self::CHECK_CELLS;
        Self {
            black: BigBitBoard::EMPTY,
            white: BigBitBoard::EMPTY,
            occupied: BigBitBoard::EMPTY,
            group_rep: [Self::SENTINEL; CELLS],
            chain_next: [Self::SENTINEL; CELLS],
            liberties: [0; CELLS],
        }
    }

    /// Rebuilds an engine from a pair of occupancy boards by flood-filling
    /// every group once, from scratch -- O(board size), not incremental.
    /// Exists for tests (an independent construction path to cross-check
    /// [`play`](Self::play)'s incremental bookkeeping against) and for
    /// hydrating an engine from a plain `BigBitBoard` pair (e.g. after
    /// deserializing a saved position), not for the hot path.
    pub fn from_boards(black: BigBitBoard<N, N, WORDS>, white: BigBitBoard<N, N, WORDS>) -> Self {
        debug_assert!(!black.intersects(white));
        let mut engine = Self::new();
        engine.black = black;
        engine.white = white;
        let occupied = black | white;
        engine.occupied = occupied;
        let mut assigned = BigBitBoard::<N, N, WORDS>::EMPTY;

        for start in 0..CELLS {
            if !occupied.get(start) || assigned.get(start) {
                continue;
            }
            let own_board = if black.get(start) { black } else { white };
            let group = own_board.flood4(start);
            assigned |= group;

            let members: Vec<usize> = group.collect();
            for (i, &cell) in members.iter().enumerate() {
                engine.group_rep[cell] = start as u16;
                engine.chain_next[cell] = members[(i + 1) % members.len()] as u16;
            }

            let liberties = !occupied & group.adjacency_mask();
            engine.liberties[start] = liberties.count_ones() as u16;
        }

        engine
    }

    pub fn black(&self) -> BigBitBoard<N, N, WORDS> {
        self.black
    }

    pub fn white(&self) -> BigBitBoard<N, N, WORDS> {
        self.white
    }

    #[inline]
    fn occupied(&self) -> BigBitBoard<N, N, WORDS> {
        self.occupied
    }

    #[inline]
    fn own_board(&self, black_to_move: bool) -> BigBitBoard<N, N, WORDS> {
        if black_to_move {
            self.black
        } else {
            self.white
        }
    }

    #[inline]
    fn opp_board(&self, black_to_move: bool) -> BigBitBoard<N, N, WORDS> {
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

    /// Up to four orthogonal neighbor indices of `index`, `None` past a
    /// board edge. Matches `BigBitBoard`'s north = row+1 / east = col+1
    /// convention (see its `shift_north`/`shift_east`).
    #[inline]
    fn neighbors(index: usize) -> [Option<usize>; 4] {
        let (row, col) = BigBitBoard::<N, N, WORDS>::to_coord(index);
        [
            (row + 1 < N).then(|| BigBitBoard::<N, N, WORDS>::to_index(row + 1, col)),
            (col + 1 < N).then(|| BigBitBoard::<N, N, WORDS>::to_index(row, col + 1)),
            (row > 0).then(|| BigBitBoard::<N, N, WORDS>::to_index(row - 1, col)),
            (col > 0).then(|| BigBitBoard::<N, N, WORDS>::to_index(row, col - 1)),
        ]
    }

    /// All cells belonging to the group represented by `rep`, walking its
    /// chain. O(group size); only called for a group about to be captured
    /// or otherwise actually inspected, never per legality-check candidate.
    fn group_cells(&self, rep: u16) -> BigBitBoard<N, N, WORDS> {
        let mut mask = BigBitBoard::<N, N, WORDS>::EMPTY;
        let mut cell = rep;
        loop {
            mask.set(cell as usize);
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
        let mut liberty_mask = BigBitBoard::<N, N, WORDS>::EMPTY;
        let mut cell = rep;
        loop {
            for nb in Self::neighbors(cell as usize).into_iter().flatten() {
                if !occupied.get(nb) {
                    liberty_mask.set(nb);
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
    /// [`bigbitboard::check_go_move`]'s signature and semantics exactly, but
    /// in O(neighbors) using cached per-group liberty counts instead of
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
    pub fn check(&self, black_to_move: bool, index: usize) -> (bool, BigBitBoard<N, N, WORDS>) {
        debug_assert!(!self.occupied().get(index));
        let own_board = self.own_board(black_to_move);
        let opp_board = self.opp_board(black_to_move);
        let occupied = self.occupied();

        let mut safe = false;
        let mut opp_reps_seen = [Self::SENTINEL; 4];
        let mut n_opp_seen = 0usize;
        let mut will_capture = BigBitBoard::<N, N, WORDS>::EMPTY;

        for nb in Self::neighbors(index).into_iter().flatten() {
            if !occupied.get(nb) {
                safe = true;
                continue;
            }
            let r = self
                .rep(nb)
                .expect("occupied cell must have a group representative");
            if opp_board.get(nb) {
                if !opp_reps_seen[..n_opp_seen].contains(&r) {
                    opp_reps_seen[n_opp_seen] = r;
                    n_opp_seen += 1;
                    if self.liberties[r as usize] == 1 {
                        will_capture |= self.group_cells(r);
                    }
                }
            } else {
                debug_assert!(own_board.get(nb));
                if self.liberties[r as usize] > 1 {
                    safe = true;
                }
            }
        }

        (safe || !will_capture.is_empty(), will_capture)
    }

    /// Plays a stone at `index` for `black_to_move` if legal, returning the
    /// captured mask (possibly empty) and updating group/liberty bookkeeping
    /// incrementally. Returns `None` (no mutation) if illegal.
    pub fn play(&mut self, black_to_move: bool, index: usize) -> Option<BigBitBoard<N, N, WORDS>> {
        let (legal, will_capture) = self.check(black_to_move, index);
        if !legal {
            return None;
        }

        if black_to_move {
            self.black.set(index);
        } else {
            self.white.set(index);
        }
        self.occupied.set(index);
        self.group_rep[index] = index as u16;
        self.chain_next[index] = index as u16;

        for cell in will_capture {
            if black_to_move {
                self.white.clear(cell);
            } else {
                self.black.clear(cell);
            }
            self.occupied.clear(cell);
            self.group_rep[cell] = Self::SENTINEL;
        }

        let own_board = self.own_board(black_to_move);
        let mut own_reps_seen = [Self::SENTINEL; 4];
        let mut n_own_seen = 0usize;
        for nb in Self::neighbors(index).into_iter().flatten() {
            if nb != index && own_board.get(nb) {
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
        let mut opp_reps_seen = [Self::SENTINEL; 4];
        let mut n_opp_seen = 0usize;
        for nb in Self::neighbors(index).into_iter().flatten() {
            if opp_board.get(nb) {
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
        let mut rescanned = BigBitBoard::<N, N, WORDS>::EMPTY;
        rescanned.set(index);
        for cell in will_capture {
            for nb in Self::neighbors(cell).into_iter().flatten() {
                if let Some(r) = self.rep(nb) {
                    if !rescanned.get(r as usize) {
                        rescanned.set(r as usize);
                        self.rescan_liberties(r);
                    }
                }
            }
        }

        Some(will_capture)
    }
}

impl<const N: usize, const WORDS: usize, const CELLS: usize> Default for GoEngine<N, WORDS, CELLS> {
    fn default() -> Self {
        Self::new()
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bigbitboard;
    use proptest::prelude::*;

    #[test]
    fn check_cells_matches_n_times_n() {
        let _ = GoEngine::<3, 1, 9>::new();
        let _ = GoEngine::<9, 2, 81>::new();
        let _ = GoEngine::<19, 6, 361>::new();
    }

    /// Measures `GoEngine`'s actual `size_of` at Gonnect/AtariGo's three
    /// supported sizes against a plain `BigBitBoard` pair (today's `State`
    /// content, modulo a couple of small scalar fields). Printed with
    /// `--nocapture`; the assertions just pin the byte counts so a future
    /// change to the representation shows up as a failing test, not a
    /// silent regression.
    #[test]
    fn state_size_at_supported_board_sizes() {
        use std::mem::size_of;

        let sizes: [(&str, usize, usize); 3] = [
            (
                "9x9",
                size_of::<GoEngine<9, 2, 81>>(),
                size_of::<BigBitBoard<9, 9, 2>>() * 2,
            ),
            (
                "13x13",
                size_of::<GoEngine<13, 3, 169>>(),
                size_of::<BigBitBoard<13, 13, 3>>() * 2,
            ),
            (
                "19x19",
                size_of::<GoEngine<19, 6, 361>>(),
                size_of::<BigBitBoard<19, 19, 6>>() * 2,
            ),
        ];

        for (label, engine_bytes, plain_pair_bytes) in sizes {
            println!(
                "{label}: GoEngine = {engine_bytes} bytes, plain black+white BigBitBoard pair = {plain_pair_bytes} bytes"
            );
        }

        // Measured (not the plan's rough estimate): 81 cells * 3 u16 arrays
        // (group_rep + chain_next + liberties) is 486 bytes of raw payload,
        // plus the black/white/occupied `BigBitBoard` triple, plus a couple
        // of bytes of struct alignment padding. Pinned here so a
        // representation change shows up as a failing test.
        assert_eq!(sizes[0].1, 536); // 9x9
        assert_eq!(sizes[1].1, 1088); // 13x13
        assert_eq!(sizes[2].1, 2312); // 19x19: ~2.3 KB, vs. today's 96-byte pair
    }

    #[test]
    fn empty_board_every_move_legal_and_captures_nothing() {
        type E = GoEngine<5, 1, 25>;
        let engine = E::new();
        for cell in 0..25 {
            let (legal, captured) = engine.check(true, cell);
            assert!(legal);
            assert!(captured.is_empty());
        }
    }

    #[test]
    fn single_stone_surrounded_is_suicide() {
        // Black stone at center of a plus-shape of white stones with no
        // other liberties: playing the center for black must be illegal
        // (suicide, no capture).
        type E = GoEngine<3, 1, 9>;
        let mut engine = E::new();
        // White occupies the four orthogonal neighbors of center (1,1):
        // (0,1), (2,1), (1,0), (1,2). Their own outer liberties keep them
        // alive so black's move at center doesn't capture anything.
        for (r, c) in [(0, 1), (2, 1), (1, 0), (1, 2)] {
            let idx = E::to_index_for_test(r, c);
            engine.play(false, idx).unwrap();
        }
        let center = E::to_index_for_test(1, 1);
        let (legal, captured) = engine.check(true, center);
        assert!(!legal);
        assert!(captured.is_empty());
    }

    #[test]
    fn capturing_a_single_stone() {
        // Black stones at (0,1), (1,0), (1,2), (2,1) surround white at
        // (1,1) except one liberty at... on a 3x3 board (1,1) has exactly
        // those four neighbors, so placing the fourth captures white.
        type E = GoEngine<3, 1, 9>;
        let mut engine = E::new();
        let white_center = E::to_index_for_test(1, 1);
        engine.play(false, white_center).unwrap();
        for (r, c) in [(0, 1), (1, 0), (1, 2)] {
            engine.play(true, E::to_index_for_test(r, c)).unwrap();
        }
        let last = E::to_index_for_test(2, 1);
        let captured = engine.play(true, last).unwrap();
        assert_eq!(captured.count_ones(), 1);
        assert!(captured.get(white_center));
        assert!(!engine.white().get(white_center));
        assert!(!engine.black().get(white_center));
    }

    impl<const N: usize, const WORDS: usize, const CELLS: usize> GoEngine<N, WORDS, CELLS> {
        fn to_index_for_test(row: usize, col: usize) -> usize {
            BigBitBoard::<N, N, WORDS>::to_index(row, col)
        }
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
    fn has_dead_group<const N: usize, const WORDS: usize>(
        board: BigBitBoard<N, N, WORDS>,
        other: BigBitBoard<N, N, WORDS>,
    ) -> bool {
        let occupied = board | other;
        let mut seen = BigBitBoard::<N, N, WORDS>::EMPTY;
        for start in board {
            if seen.get(start) {
                continue;
            }
            let group = board.flood4(start);
            seen |= group;
            let liberties = !occupied & group.adjacency_mask();
            if liberties.is_empty() {
                return true;
            }
        }
        false
    }

    fn check_against_flood_oracle<const N: usize, const WORDS: usize, const CELLS: usize>(
        black_bits: &[usize],
        white_bits: &[usize],
    ) {
        let n = N * N;
        let mut black = BigBitBoard::<N, N, WORDS>::EMPTY;
        let mut white = BigBitBoard::<N, N, WORDS>::EMPTY;
        for &i in black_bits {
            let i = i % n;
            if !white.get(i) {
                black.set(i);
            }
        }
        for &i in white_bits {
            let i = i % n;
            if !black.get(i) {
                white.set(i);
            }
        }
        if has_dead_group(black, white) || has_dead_group(white, black) {
            return;
        }

        let engine = GoEngine::<N, WORDS, CELLS>::from_boards(black, white);

        for index in 0..n {
            if black.get(index) || white.get(index) {
                continue;
            }
            for &black_to_move in &[true, false] {
                let (player, opponent) = if black_to_move {
                    (black, white)
                } else {
                    (white, black)
                };
                let (expected_legal, expected_capture) =
                    bigbitboard::check_go_move::<N, WORDS>(player, opponent, index);
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
    // deems illegal) through both the incremental engine and a plain
    // BigBitBoard pair updated via `check_go_move`. After every move,
    // engine occupancy must match the reference boards, the returned
    // capture mask must match, and every occupied cell's liberty count must
    // match a fresh `from_boards` rebuild -- which validates `play`'s
    // merge/rescan bookkeeping without caring which cell either path picked
    // as a group's representative.

    fn check_play_sequence_against_oracle<
        const N: usize,
        const WORDS: usize,
        const CELLS: usize,
    >(
        moves: &[(bool, usize)],
    ) {
        let n = N * N;
        let mut engine = GoEngine::<N, WORDS, CELLS>::new();
        let mut ref_black = BigBitBoard::<N, N, WORDS>::EMPTY;
        let mut ref_white = BigBitBoard::<N, N, WORDS>::EMPTY;

        for &(black_to_move, raw_index) in moves {
            let index = raw_index % n;
            if ref_black.get(index) || ref_white.get(index) {
                continue;
            }
            let (player, opponent) = if black_to_move {
                (ref_black, ref_white)
            } else {
                (ref_white, ref_black)
            };
            let (legal, expected_capture) =
                bigbitboard::check_go_move::<N, WORDS>(player, opponent, index);
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
                ref_black.set(index);
                ref_white &= !expected_capture;
            } else {
                ref_white.set(index);
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

            let rebuilt = GoEngine::<N, WORDS, CELLS>::from_boards(ref_black, ref_white);
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
        ($mod_name:ident, $n:expr, $words:expr, $cells:expr) => {
            mod $mod_name {
                use super::*;

                proptest! {
                    #![proptest_config(ProptestConfig::with_cases(64))]

                    #[test]
                    fn flood_oracle(
                        black_bits in proptest::collection::vec(0usize..$cells, 0..40),
                        white_bits in proptest::collection::vec(0usize..$cells, 0..40),
                    ) {
                        check_against_flood_oracle::<$n, $words, $cells>(&black_bits, &white_bits);
                    }

                    #[test]
                    fn play_sequence(
                        moves in proptest::collection::vec(
                            (any::<bool>(), 0usize..$cells),
                            0..60,
                        ),
                    ) {
                        check_play_sequence_against_oracle::<$n, $words, $cells>(&moves);
                    }
                }
            }
        };
    }

    // Sub-word board (9 bits, fits one word).
    oracle_tests!(oracle_3x3, 3, 1, 9);
    // Multi-word boards, matching Gonnect/AtariGo's real supported sizes.
    oracle_tests!(oracle_9x9, 9, 2, 81);
    oracle_tests!(oracle_13x13, 13, 3, 169);
}
