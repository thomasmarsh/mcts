//! Flat bitset storage for pyramidal (Shibumi-family) boards: an `n`x`n`
//! square base with `n` stacked levels, level `k` an `(n-k)`x`(n-k)` square
//! (base at `k = 0`, a single-cell apex at `k = n - 1`). Every level-`k`
//! piece physically sits centered over a 2x2 block of level-`(k-1)`
//! positions, but that support relation -- and everything built on top of it
//! (placement/removal legality, top-down connectivity, symmetry, game rules)
//! -- is out of scope here: this crate is purely the storage layer (index
//! math, get/set/iterate over the flat bitset), replacing `games/shibumi`'s
//! dead N=4-only stub with the same `bitboard::Storage`-backed approach
//! generalized to `n` in `2..=10`.

use bitboard::{Board, Dim, Storage};

pub mod adjacency;
pub mod symmetry;
pub use adjacency::{get_adjacency, touching_neighbors, TouchingAdjacency};
pub use symmetry::PyramidD4;

/// Sum of squares `1^2 + 2^2 + ... + x^2` -- the total cell count of an
/// `x`-level pyramid (`x` = base width). Closed form of
/// `(0..x).map(|l| (x - l) * (x - l)).sum()`.
#[inline(always)]
const fn sum_squares(x: usize) -> usize {
    x * (x + 1) * (2 * x + 1) / 6
}

/// The side length of level `level`'s square (level 0 = the `n`x`n` base,
/// level `n - 1` = the single-cell apex).
#[inline(always)]
pub const fn level_side(n: usize, level: usize) -> usize {
    n - level
}

/// The number of cells in level `level`.
#[inline(always)]
pub const fn level_size(n: usize, level: usize) -> usize {
    let side = level_side(n, level);
    side * side
}

/// The flat index of level `level`'s first cell: every level below it
/// (strictly larger squares, since levels shrink going up) has already been
/// counted.
#[inline(always)]
pub const fn level_offset(n: usize, level: usize) -> usize {
    sum_squares(n) - sum_squares(n - level)
}

/// The total cell count of a base-`n` pyramid: `n(n+1)(2n+1)/6`.
#[inline(always)]
pub const fn total_cells(n: usize) -> usize {
    sum_squares(n)
}

/// Whether `(col, row, level)` is a valid cell of a base-`n` pyramid --
/// see `Pyramid::in_bounds`.
#[inline(always)]
pub const fn in_bounds(n: usize, col: usize, row: usize, level: usize) -> bool {
    level < n && col < level_side(n, level) && row < level_side(n, level)
}

/// The flat index of `(col, row, level)` -- see `Pyramid::index`.
#[inline(always)]
pub const fn index(n: usize, col: usize, row: usize, level: usize) -> usize {
    level_offset(n, level) + row * level_side(n, level) + col
}

/// The inverse of `index` -- see `Pyramid::to_coord`.
pub fn to_coord(n: usize, index: usize) -> (usize, usize, usize) {
    debug_assert!(index < total_cells(n));
    let mut level = 0;
    while level + 1 < n && level_offset(n, level + 1) <= index {
        level += 1;
    }
    let side = level_side(n, level);
    let offset = index - level_offset(n, level);
    (offset % side, offset / side, level)
}

/// The (up to four) level-`(level + 1)` candidate positions that would rest
/// on `(col, row, level)` if occupied -- the inverse of `Pyramid::supporters`,
/// bounds-checked but (unlike `Pyramid::dependents`) not filtered by
/// occupancy, since the geometric relation it describes doesn't depend on
/// any particular board's state.
pub fn dependent_positions(n: usize, col: usize, row: usize, level: usize) -> Vec<(usize, usize)> {
    if level + 1 >= n {
        return Vec::new();
    }
    let above_side = level_side(n, level + 1);
    let mut out = Vec::new();
    for c in col.saturating_sub(1)..=col {
        for r in row.saturating_sub(1)..=row {
            if c < above_side && r < above_side {
                out.push((c, r));
            }
        }
    }
    out
}

/// A flat pyramidal bitset backed by `S` (mirrors `bitboard::Board`'s
/// `S: Storage` -- `u64` for the N=4 Shibumi-family fast path, since 30
/// cells fits a single word; `[u64; W]` for larger `n`, up to `[u64; 7]` for
/// `n = 10`'s 385 cells). `N` picks whether the base width is compile-time
/// (`Const<N>`) or runtime (`Dyn`), the same split `bitboard::Board` uses for
/// its `R`/`C`.
///
/// Cells are addressed by `(col, row, level)`: `level` 0 is the base,
/// `level` `n - 1` the single-cell apex; within a level, `(col, row)` is
/// row-major over that level's `(n - level)`-side square, matching
/// `bitboard::Board`'s own `row * cols + col` convention so a level can later
/// be extracted into an ordinary `Board` by a plain contiguous copy.
#[derive(Clone, Copy, Debug)]
pub struct Pyramid<S: Storage, N: Dim> {
    bits: S,
    n: N,
}

impl<S: Storage, N: Dim> Pyramid<S, N> {
    #[inline(always)]
    pub fn new(n: N) -> Self {
        Self { bits: S::zero(), n }
    }

    #[inline(always)]
    pub fn n(&self) -> usize {
        self.n.get()
    }

    /// The number of stacked levels (equal to the base width `n`).
    #[inline(always)]
    pub fn levels(&self) -> usize {
        self.n()
    }

    #[inline(always)]
    pub fn total_cells(&self) -> usize {
        total_cells(self.n())
    }

    #[inline(always)]
    pub fn level_side(&self, level: usize) -> usize {
        level_side(self.n(), level)
    }

    #[inline(always)]
    pub fn level_size(&self, level: usize) -> usize {
        level_size(self.n(), level)
    }

    #[inline(always)]
    pub fn level_offset(&self, level: usize) -> usize {
        level_offset(self.n(), level)
    }

    #[inline(always)]
    pub fn in_bounds(&self, col: usize, row: usize, level: usize) -> bool {
        in_bounds(self.n(), col, row, level)
    }

    #[inline(always)]
    pub fn index(&self, col: usize, row: usize, level: usize) -> usize {
        debug_assert!(self.in_bounds(col, row, level));
        index(self.n(), col, row, level)
    }

    /// The inverse of `index` -- see the free function of the same name.
    pub fn to_coord(&self, flat_index: usize) -> (usize, usize, usize) {
        to_coord(self.n(), flat_index)
    }

    /// Gets a single bit by its flat index -- see `index`.
    #[inline(always)]
    pub fn get_index(&self, index: usize) -> bool {
        (self.bits.word(index / 64) >> (index % 64)) & 1 != 0
    }

    /// Sets a single bit by its flat index -- see `index`.
    #[inline(always)]
    pub fn set_index(&mut self, index: usize) {
        *self.bits.word_mut(index / 64) |= 1u64 << (index % 64);
    }

    /// Clears a single bit by its flat index -- see `index`.
    #[inline(always)]
    pub fn clear_index(&mut self, index: usize) {
        *self.bits.word_mut(index / 64) &= !(1u64 << (index % 64));
    }

    #[inline(always)]
    pub fn get(&self, col: usize, row: usize, level: usize) -> bool {
        self.get_index(self.index(col, row, level))
    }

    #[inline(always)]
    pub fn set(&mut self, col: usize, row: usize, level: usize) {
        let index = self.index(col, row, level);
        self.set_index(index);
    }

    #[inline(always)]
    pub fn clear(&mut self, col: usize, row: usize, level: usize) {
        let index = self.index(col, row, level);
        self.clear_index(index);
    }

    pub fn count_ones(&self) -> u32 {
        (0..S::CAPACITY_WORDS)
            .map(|w| self.bits.word(w).count_ones())
            .sum()
    }

    /// The raw backing words, low word first -- for a caller that needs to
    /// fold over every word generically (e.g. hashing or an ordering key)
    /// independent of `S`'s concrete layout. Mirrors `bitboard::Board::words`.
    pub fn words(&self) -> impl Iterator<Item = u64> + '_ {
        (0..S::CAPACITY_WORDS).map(move |w| self.bits.word(w))
    }

    /// Iterates set flat indices in ascending order -- same trailing-zeros
    /// idiom as `bitboard::Board::iter_set`.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        (0..S::CAPACITY_WORDS).flat_map(move |w| {
            let mut word = self.bits.word(w);
            std::iter::from_fn(move || {
                if word == 0 {
                    None
                } else {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some(w * 64 + bit)
                }
            })
        })
    }
    /// Extracts level `level` into an ordinary `bitboard::Board`, so lateral
    /// operations (`flood4`, `adjacency_mask`, D4 symmetry, ...) can run on a
    /// single level unmodified rather than reimplemented against the flat
    /// pyramid storage. `LD` is the extracted board's own dim kind (`Const`
    /// for a known level side, `Dyn` otherwise) -- the caller supplies
    /// `side`, since a single level's side (`N - level`) isn't in general the
    /// same type as `Pyramid`'s own `N`. `LS` is the extracted board's
    /// storage; it must be large enough for `side * side` bits, which is
    /// smaller than `Self`'s own total-cell storage since a level is always a
    /// strict subset of the whole pyramid.
    ///
    /// Cells line up by construction: within a level, `Pyramid::index` is
    /// `level_offset(level) + row * side + col`, exactly matching
    /// `Board`'s own `row * cols + col` row-major convention (see this
    /// struct's own doc comment) -- so extraction is a plain contiguous copy
    /// of the level's bit range, not a coordinate remap.
    pub fn level_board<LS: Storage, LD: Dim>(&self, level: usize, side: LD) -> Board<LS, LD, LD> {
        debug_assert_eq!(side.get(), self.level_side(level));
        let offset = self.level_offset(level);
        let mut board: Board<LS, LD, LD> = Board::new(side, side);
        for local in 0..self.level_size(level) {
            if self.get_index(offset + local) {
                board.set_index(local);
            }
        }
        board
    }

    /// The inverse of `level_board`: writes `board`'s bits back into level
    /// `level`'s range of the flat pyramid storage, overwriting whatever was
    /// there before (both sets and clears, so a bit a caller cleared on the
    /// extracted board is reflected here too, not just newly-set bits).
    pub fn set_level_board<LS: Storage, LD: Dim>(
        &mut self,
        level: usize,
        board: &Board<LS, LD, LD>,
    ) {
        debug_assert_eq!(board.rows(), self.level_side(level));
        debug_assert_eq!(board.cols(), self.level_side(level));
        let offset = self.level_offset(level);
        for local in 0..self.level_size(level) {
            if board.get_index(local) {
                self.set_index(offset + local);
            } else {
                self.clear_index(offset + local);
            }
        }
    }

    /// The (up to four) level-`(level - 1)` cells that physically support a
    /// piece at `(col, row, level)`: a level-`k` piece sits centered over
    /// the 2x2 block of level-`(k - 1)` cells at those coordinates, since
    /// level `k - 1` is exactly one wider than level `k` and every in-range
    /// `(col, row)` at level `k` therefore has all four of `(col, row)`,
    /// `(col + 1, row)`, `(col, row + 1)`, `(col + 1, row + 1)` in range at
    /// level `k - 1`. Empty for `level == 0` (the base, which always rests
    /// on the table).
    pub fn supporters(&self, col: usize, row: usize, level: usize) -> Vec<(usize, usize)> {
        if level == 0 {
            Vec::new()
        } else {
            vec![
                (col, row),
                (col + 1, row),
                (col, row + 1),
                (col + 1, row + 1),
            ]
        }
    }

    /// Whether `(col, row, level)` rests on a fully occupied 2x2 block
    /// below it -- the physical precondition for a piece to be there at
    /// all. The base (`level == 0`) is always supported.
    pub fn is_supported(&self, col: usize, row: usize, level: usize) -> bool {
        level == 0
            || self
                .supporters(col, row, level)
                .into_iter()
                .all(|(c, r)| self.get(c, r, level - 1))
    }

    /// Whether a piece can legally be placed at `(col, row, level)`: in
    /// bounds, currently empty, and supported.
    pub fn can_place(&self, col: usize, row: usize, level: usize) -> bool {
        self.in_bounds(col, row, level)
            && !self.get(col, row, level)
            && self.is_supported(col, row, level)
    }

    /// The occupied level-`(level + 1)` cells resting directly on `(col,
    /// row, level)` -- the inverse of `supporters`: `(col, row, level)` is
    /// one of their four support cells. Empty above the apex.
    pub fn dependents(&self, col: usize, row: usize, level: usize) -> Vec<(usize, usize)> {
        dependent_positions(self.n(), col, row, level)
            .into_iter()
            .filter(|&(c, r)| self.get(c, r, level + 1))
            .collect()
    }

    /// Whether anything rests directly on `(col, row, level)` at all --
    /// the piece can be taken off the board outright only when this is
    /// false, since there would otherwise be nothing left to hold up
    /// whatever's above it.
    pub fn is_removable(&self, col: usize, row: usize, level: usize) -> bool {
        self.dependents(col, row, level).is_empty()
    }

    /// Whether `(col, row, level)` can be relocated to some other legal
    /// position without leaving anything floating. A piece with nothing
    /// resting on it is always movable. A piece supporting exactly one
    /// other piece is still movable: that single piece has an unambiguous
    /// place to drop into (the vacated cell) once the mover leaves. A piece
    /// supporting two or more others is pinned and cannot move at all --
    /// there is only one physical gap left behind, and it can't hold more
    /// than one dropped piece.
    pub fn is_movable(&self, col: usize, row: usize, level: usize) -> bool {
        self.dependents(col, row, level).len() <= 1
    }

    /// Whether `(col, row, level)` is buried: hidden from directly above by
    /// a piece at `(col - 1, row - 1, level + 2)`, the *only* position whose
    /// projected center exactly coincides with this one (see this crate's
    /// `adjacency` module docs for the geometric derivation). A buried piece
    /// still physically touches its neighbors -- `adjacency::touching_neighbors`
    /// doesn't know or care about burial -- but Margo/Akron's win-condition
    /// connectivity is defined over *visible* (non-buried) pieces only, per
    /// their published rules ("buried pieces do not count in any
    /// connection").
    pub fn is_buried(&self, col: usize, row: usize, level: usize) -> bool {
        level + 2 < self.n()
            && col > 0
            && row > 0
            && self.in_bounds(col - 1, row - 1, level + 2)
            && self.get(col - 1, row - 1, level + 2)
    }
}

impl<S: Storage, N: Dim> PartialEq for Pyramid<S, N> {
    fn eq(&self, other: &Self) -> bool {
        self.n() == other.n()
            && (0..S::CAPACITY_WORDS).all(|w| self.bits.word(w) == other.bits.word(w))
    }
}

impl<S: Storage, N: Dim> Eq for Pyramid<S, N> {}

#[cfg(test)]
mod tests {
    use super::*;
    use bitboard::{Const, Dyn};
    use proptest::prelude::*;

    // Array-backed oracle: an independent, obviously-correct model checked
    // against `Pyramid` -- same pattern `bitboard::board`'s own oracle tests
    // use, deliberately not sharing any arithmetic with `level_offset`/
    // `index`/`to_coord` themselves.

    /// Counts cells level-by-level, row-major within each level, until it
    /// reaches `(col, row, level)` -- a brute-force coordinate walk.
    fn naive_index(n: usize, col: usize, row: usize, level: usize) -> usize {
        let mut idx = 0;
        for l in 0..level {
            let side = n - l;
            idx += side * side;
        }
        let side = n - level;
        idx + row * side + col
    }

    fn naive_total_cells(n: usize) -> usize {
        (0..n).map(|l| (n - l) * (n - l)).sum()
    }

    fn all_coords(n: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for level in 0..n {
            let side = n - level;
            for row in 0..side {
                for col in 0..side {
                    out.push((col, row, level));
                }
            }
        }
        out
    }

    fn check_index_against_oracle<S: Storage, N: Dim>(n: N) {
        let pyramid: Pyramid<S, N> = Pyramid::new(n);
        assert_eq!(pyramid.total_cells(), naive_total_cells(n.get()));
        for (col, row, level) in all_coords(n.get()) {
            let expected = naive_index(n.get(), col, row, level);
            assert_eq!(
                pyramid.index(col, row, level),
                expected,
                "index mismatch at ({col},{row},{level})"
            );
            assert_eq!(
                pyramid.to_coord(expected),
                (col, row, level),
                "to_coord mismatch at index {expected}"
            );
        }
    }

    fn check_get_set_against_oracle<S: Storage, N: Dim>(n: N, sets: &[(usize, usize, usize)]) {
        let total = total_cells(n.get());
        let mut oracle = vec![false; total];
        let mut pyramid: Pyramid<S, N> = Pyramid::new(n);
        for &(col, row, level) in sets {
            if !pyramid.in_bounds(col, row, level) {
                continue;
            }
            oracle[pyramid.index(col, row, level)] = true;
            pyramid.set(col, row, level);
        }

        for (index, &expected) in oracle.iter().enumerate() {
            let (col, row, level) = pyramid.to_coord(index);
            assert_eq!(
                pyramid.get(col, row, level),
                expected,
                "get mismatch at index {index}"
            );
        }

        assert_eq!(
            pyramid.count_ones() as usize,
            oracle.iter().filter(|&&b| b).count(),
            "count_ones mismatch"
        );

        let mut got: Vec<usize> = pyramid.iter_set().collect();
        got.sort_unstable();
        let expected: Vec<usize> = (0..total).filter(|&i| oracle[i]).collect();
        assert_eq!(got, expected, "iterated set indices mismatch");
    }

    #[test]
    fn index_and_to_coord_match_oracle_every_n_in_range() {
        for n in 2..=10usize {
            check_index_against_oracle::<[u64; 7], Dyn>(Dyn(n));
        }
    }

    #[test]
    fn index_and_to_coord_match_oracle_const_4() {
        check_index_against_oracle::<u64, Const<4>>(Const);
    }

    proptest! {
        #[test]
        fn get_set_matches_oracle_dyn(
            n in 2usize..=10,
            triples in proptest::collection::vec((0usize..10, 0usize..10, 0usize..10), 0..100),
        ) {
            check_get_set_against_oracle::<[u64; 7], Dyn>(Dyn(n), &triples);
        }

        #[test]
        fn get_set_matches_oracle_const_4(
            triples in proptest::collection::vec((0usize..4, 0usize..4, 0usize..4), 0..60),
        ) {
            check_get_set_against_oracle::<u64, Const<4>>(Const, &triples);
        }
    }

    #[test]
    fn total_cells_sizing_table() {
        // 4x4-base Shibumi-family pyramids (30 cells) up through the
        // largest supported base, 10x10 (385 cells) -- all comfortably
        // within a `[u64; 7]` storage backend.
        assert_eq!(total_cells(4), 30);
        assert_eq!(total_cells(6), 91);
        assert_eq!(total_cells(8), 204);
        assert_eq!(total_cells(10), 385);
    }

    #[test]
    fn const_and_dyn_agree_at_the_same_size() {
        let mut a: Pyramid<u64, Const<4>> = Pyramid::new(Const);
        let mut b: Pyramid<[u64; 7], Dyn> = Pyramid::new(Dyn(4));
        for &(col, row, level) in &[(0, 0, 0), (1, 2, 1), (0, 0, 3)] {
            a.set(col, row, level);
            b.set(col, row, level);
        }
        assert_eq!(
            a.iter_set().collect::<Vec<_>>(),
            b.iter_set().collect::<Vec<_>>()
        );
    }

    // Level extraction/write-back round trips, checked against the same
    // coordinate-walk oracle style as the index/get/set tests above, plus a
    // check that a lateral op (`flood4`) behaves identically whether run
    // against a hand-built `Board` or one extracted from a `Pyramid`.

    fn check_level_board_matches_pyramid<S: Storage, N: Dim>(n: N, sets: &[(usize, usize, usize)]) {
        let mut pyramid: Pyramid<S, N> = Pyramid::new(n);
        for &(col, row, level) in sets {
            if pyramid.in_bounds(col, row, level) {
                pyramid.set(col, row, level);
            }
        }

        for level in 0..n.get() {
            let side = pyramid.level_side(level);
            let board: bitboard::Board<[u64; 2], bitboard::Dyn, bitboard::Dyn> =
                pyramid.level_board(level, bitboard::Dyn(side));
            for row in 0..side {
                for col in 0..side {
                    assert_eq!(
                        board.get(row, col),
                        pyramid.get(col, row, level),
                        "level {level} ({col},{row}) mismatch"
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn level_board_matches_pyramid_dyn(
            n in 2usize..=10,
            triples in proptest::collection::vec((0usize..10, 0usize..10, 0usize..10), 0..100),
        ) {
            check_level_board_matches_pyramid::<[u64; 7], Dyn>(Dyn(n), &triples);
        }

        #[test]
        fn level_board_matches_pyramid_const_4(
            triples in proptest::collection::vec((0usize..4, 0usize..4, 0usize..4), 0..60),
        ) {
            check_level_board_matches_pyramid::<u64, Const<4>>(Const, &triples);
        }
    }

    #[test]
    fn set_level_board_round_trips_through_extraction() {
        let mut pyramid: Pyramid<u64, Const<4>> = Pyramid::new(Const);
        pyramid.set(0, 0, 0);
        pyramid.set(3, 3, 0);
        pyramid.set(1, 1, 1);

        // Extract level 0 (the 4x4 base), mutate the extracted board (clear
        // one bit, set another), and write it back -- the flat pyramid must
        // reflect exactly the extracted board's final state, and other
        // levels must be untouched.
        let mut level0: bitboard::Board<u64, Const<4>, Const<4>> = pyramid.level_board(0, Const);
        assert!(level0.get(0, 0));
        level0.clear(0, 0);
        level0.set(2, 2);
        pyramid.set_level_board(0, &level0);

        assert!(!pyramid.get(0, 0, 0));
        assert!(pyramid.get(3, 3, 0));
        assert!(pyramid.get(2, 2, 0));
        assert!(
            pyramid.get(1, 1, 1),
            "level 1 must be untouched by writing back level 0"
        );
    }

    // Support, placement, and (im)movability primitives, hand-verified
    // against small N (2, 3, 4) manual diagrams rather than an oracle,
    // since these relations (unlike index math) aren't independently
    // re-derivable from a naive coordinate walk -- they *are* the
    // definition being tested.

    #[test]
    fn base_is_always_supported() {
        let pyramid: Pyramid<u64, Const<4>> = Pyramid::new(Const);
        assert!(pyramid.supporters(2, 1, 0).is_empty());
        assert!(pyramid.is_supported(2, 1, 0));
        assert!(pyramid.can_place(2, 1, 0));
    }

    #[test]
    fn higher_level_needs_all_four_supporters() {
        let mut pyramid: Pyramid<u64, Const<4>> = Pyramid::new(Const);
        assert_eq!(
            pyramid.supporters(0, 0, 1),
            vec![(0, 0), (1, 0), (0, 1), (1, 1)]
        );
        assert!(!pyramid.is_supported(0, 0, 1));
        assert!(!pyramid.can_place(0, 0, 1));

        pyramid.set(0, 0, 0);
        pyramid.set(1, 0, 0);
        pyramid.set(0, 1, 0);
        assert!(
            !pyramid.is_supported(0, 0, 1),
            "three of four supporters is not enough"
        );

        pyramid.set(1, 1, 0);
        assert!(pyramid.is_supported(0, 0, 1));
        assert!(pyramid.can_place(0, 0, 1));

        pyramid.set(0, 0, 1);
        assert!(
            !pyramid.can_place(0, 0, 1),
            "an occupied cell can't be placed into again"
        );
    }

    #[test]
    fn dependents_is_the_inverse_of_supporters() {
        // n = 3: level 0 is 3x3, level 1 is 2x2, level 2 is the apex.
        let mut pyramid: Pyramid<u64, Const<3>> = Pyramid::new(Const);
        for &(c, r) in &[(0, 0), (1, 0), (0, 1), (1, 1)] {
            pyramid.set(c, r, 0);
        }
        pyramid.set(0, 0, 1);

        assert_eq!(pyramid.dependents(0, 0, 0), vec![(0, 0)]);
        // (1, 0, 0) is a supporter of both level-1 (0, 0) and (1, 0), but
        // only (0, 0) is occupied.
        assert_eq!(pyramid.dependents(1, 0, 0), vec![(0, 0)]);
        assert_eq!(pyramid.dependents(0, 1, 0), vec![(0, 0)]);
        assert_eq!(pyramid.dependents(1, 1, 0), vec![(0, 0)]);
        // A corner far from the placed level-1 piece has no dependents.
        assert!(pyramid.dependents(2, 2, 0).is_empty());
        // The apex has no level above it.
        assert!(pyramid.dependents(0, 0, 2).is_empty());
    }

    #[test]
    fn removable_and_movable_agree_when_nothing_rests_above() {
        let mut pyramid: Pyramid<u64, Const<4>> = Pyramid::new(Const);
        pyramid.set(0, 0, 0);
        assert!(pyramid.is_removable(0, 0, 0));
        assert!(pyramid.is_movable(0, 0, 0));
    }

    #[test]
    fn single_dependent_is_pinned_but_movable() {
        // n = 2: level 0 is the 2x2 base, level 1 is the single apex cell,
        // which is the only piece that can ever rest on a level-0 cell here.
        let mut pyramid: Pyramid<u64, Const<2>> = Pyramid::new(Const);
        pyramid.set(0, 0, 0);
        pyramid.set(1, 0, 0);
        pyramid.set(0, 1, 0);
        pyramid.set(1, 1, 0);
        pyramid.set(0, 0, 1);

        for &(c, r) in &[(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_eq!(pyramid.dependents(c, r, 0), vec![(0, 0)]);
            assert!(
                !pyramid.is_removable(c, r, 0),
                "the apex rests on every base cell"
            );
            assert!(
                pyramid.is_movable(c, r, 0),
                "exactly one piece above still has an unambiguous place to drop"
            );
        }
    }

    #[test]
    fn two_or_more_dependents_are_pinned_and_immovable() {
        // n = 3, with both level-1 cells that rest on level-0 (1, 1)
        // occupied: (1, 1, 0) supports level-1 (0, 0) and (1, 1) both.
        let mut pyramid: Pyramid<u64, Const<3>> = Pyramid::new(Const);
        for &(c, r) in &[
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ] {
            pyramid.set(c, r, 0);
        }
        pyramid.set(0, 0, 1);
        pyramid.set(1, 1, 1);

        let deps = pyramid.dependents(1, 1, 0);
        assert_eq!(
            deps.len(),
            2,
            "(1,1,0) supports two level-1 cells: {deps:?}"
        );
        assert!(!pyramid.is_removable(1, 1, 0));
        assert!(
            !pyramid.is_movable(1, 1, 0),
            "two pieces can't both drop into one vacated gap"
        );
    }

    #[test]
    fn flood4_on_extracted_level_matches_lateral_adjacency() {
        // A connected L-shape plus one diagonal-only (unconnected-by-4) cell
        // on level 0 of an N=4 pyramid -- proves the extracted level behaves
        // exactly like an ordinary rectangular board's `flood4`, not just
        // that individual bits round-trip.
        let mut pyramid: Pyramid<u64, Const<4>> = Pyramid::new(Const);
        for &(col, row) in &[(0, 0), (0, 1), (1, 1)] {
            pyramid.set(col, row, 0);
        }
        pyramid.set(3, 3, 0); // diagonal-only from the L-shape's bounding area

        let level0: bitboard::Board<u64, Const<4>, Const<4>> = pyramid.level_board(0, Const);
        let start = bitboard::Board::<u64, Const<4>, Const<4>>::to_index(0, 0);
        let flood = level0.flood4(start);

        assert!(flood.get(0, 0));
        assert!(flood.get(1, 0));
        assert!(flood.get(1, 1));
        assert!(!flood.get(3, 3), "flood4 must not cross a non-adjacent gap");
        assert_eq!(flood.count_ones(), 3);
    }

    // Cross-check against the published rules: Span's page reports that a
    // complete 4x4-base (Shibumi-family) pyramid has exactly five "buried"
    // (non-visible) balls out of its 30 total, and that visible balls'
    // connectivity is what wins the game -- see `pyramid::adjacency`'s
    // module docs for the geometric derivation this falls out of.

    #[test]
    fn full_n4_pyramid_has_exactly_five_buried_balls() {
        let mut pyramid: Pyramid<u64, Const<4>> = Pyramid::new(Const);
        for level in 0..4 {
            let side = pyramid.level_side(level);
            for row in 0..side {
                for col in 0..side {
                    pyramid.set(col, row, level);
                }
            }
        }
        assert_eq!(pyramid.total_cells(), 30);

        let buried: Vec<(usize, usize, usize)> = (0..pyramid.total_cells())
            .map(|i| pyramid.to_coord(i))
            .filter(|&(c, r, l)| pyramid.is_buried(c, r, l))
            .collect();
        assert_eq!(
            buried.len(),
            5,
            "expected exactly five buried balls (Span's reported figure for a full 4x4 pyramid), got {buried:?}"
        );

        // The four level-0 cells directly under each level-2 ball, plus the
        // one level-1 cell directly under the apex.
        for &(c, r) in &[(1, 1), (1, 2), (2, 1), (2, 2)] {
            assert!(buried.contains(&(c, r, 0)), "({c},{r},0) should be buried");
        }
        assert!(buried.contains(&(1, 1, 1)), "(1,1,1) should be buried");
    }

    #[test]
    fn corner_and_apex_are_never_buried() {
        let mut pyramid: Pyramid<u64, Const<4>> = Pyramid::new(Const);
        for level in 0..4 {
            let side = pyramid.level_side(level);
            for row in 0..side {
                for col in 0..side {
                    pyramid.set(col, row, level);
                }
            }
        }
        assert!(!pyramid.is_buried(0, 0, 0), "no occluder can sit off-board");
        assert!(!pyramid.is_buried(0, 0, 3), "the apex has nothing above it");
    }
}
