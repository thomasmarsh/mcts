//! A neighbor-relation abstraction letting [`crate::GoEngine`]'s incremental
//! group/liberty bookkeeping (and any future opposite-connection-style span
//! traversal) run over any adjacency relation, not just a rectangular
//! `Board`'s own shift-based 4-neighbor math. [`RectAdjacency`] reproduces
//! that original rectangular math exactly (see `go.rs`'s regression tests,
//! which check it against `Board::adjacency_mask` bit-for-bit); pyramid's
//! `TouchingAdjacency` (wrapping its precomputed top-down "touching" table)
//! is the other real instance, letting Margo/Akron reuse this crate's
//! union-find liberty bookkeeping over a non-rectangular topology instead of
//! reimplementing it.

use crate::board::Board;
use crate::dim::Dim;
use crate::storage::Storage;

/// Upper bound on how many neighbors any single cell can have, across every
/// `Adjacency` implementation this crate knows about -- sized to fit a
/// pyramid cell's worst case (up to four same-level neighbors, up to four
/// cells it supports, up to four cells that support it), well above a
/// rectangular board's fixed four. Lets `GoEngine` dedupe neighbor group
/// representatives in a fixed-size stack array instead of allocating,
/// regardless of which `Adjacency` implementation it's instantiated over.
pub const MAX_NEIGHBORS: usize = 12;

/// A small, non-allocating list of up to [`MAX_NEIGHBORS`] neighbor indices,
/// returned by [`Adjacency::neighbors`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NeighborList {
    buf: [u32; MAX_NEIGHBORS],
    len: u8,
}

impl NeighborList {
    /// Builds a list from an arbitrary iterator of neighbor indices -- for
    /// an `Adjacency` implementation outside this crate (e.g. pyramid's
    /// precomputed touching table) that has no `Board` shift arithmetic to
    /// derive neighbors from directly.
    pub fn from_neighbors(iter: impl IntoIterator<Item = usize>) -> Self {
        let mut out = Self::default();
        for i in iter {
            out.push(i);
        }
        out
    }

    #[inline]
    fn push(&mut self, index: usize) {
        debug_assert!(
            (self.len as usize) < MAX_NEIGHBORS,
            "neighbor list overflow: more than {MAX_NEIGHBORS} neighbors"
        );
        self.buf[self.len as usize] = index as u32;
        self.len += 1;
    }
}

/// Iterator over a [`NeighborList`]'s entries.
pub struct NeighborListIter {
    buf: [u32; MAX_NEIGHBORS],
    len: u8,
    pos: u8,
}

impl Iterator for NeighborListIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.pos < self.len {
            let v = self.buf[self.pos as usize] as usize;
            self.pos += 1;
            Some(v)
        } else {
            None
        }
    }
}

impl IntoIterator for NeighborList {
    type Item = usize;
    type IntoIter = NeighborListIter;

    #[inline]
    fn into_iter(self) -> NeighborListIter {
        NeighborListIter {
            buf: self.buf,
            len: self.len,
            pos: 0,
        }
    }
}

/// A cell-adjacency relation: which (up to [`MAX_NEIGHBORS`]) cells neighbor
/// a given flat index. `GoEngine` and the table-driven traversal helpers
/// below are generic over this instead of hardcoding row/col shift
/// arithmetic, so the same incremental engine serves both a rectangular
/// `Board` (via [`RectAdjacency`]) and a precomputed adjacency table (e.g.
/// pyramid's top-down "touching" graph).
pub trait Adjacency {
    fn neighbors(&self, index: usize) -> NeighborList;
}

/// Rectangular 4-neighbor adjacency (north/east/south/west), matching
/// `Board`'s own `row * cols + col` indexing exactly -- the default
/// `GoEngine` provider, reproducing the pre-Phase-5 hardcoded shift-style
/// neighbor math so existing rectangular games are unaffected.
#[derive(Clone, Copy, Debug)]
pub struct RectAdjacency {
    rows: usize,
    cols: usize,
}

impl RectAdjacency {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
}

impl Adjacency for RectAdjacency {
    #[inline]
    fn neighbors(&self, index: usize) -> NeighborList {
        let (row, col) = (index / self.cols, index % self.cols);
        let mut out = NeighborList::default();
        if row + 1 < self.rows {
            out.push((row + 1) * self.cols + col);
        }
        if col + 1 < self.cols {
            out.push(row * self.cols + col + 1);
        }
        if row > 0 {
            out.push((row - 1) * self.cols + col);
        }
        if col > 0 {
            out.push(row * self.cols + col - 1);
        }
        out
    }
}

/// Table-driven flood fill: BFS over `mask`'s set bits under `adjacency`'s
/// neighbor relation, seeded from `start` -- the non-rectangular counterpart
/// to `Board::flood4`, used by `GoEngine::from_boards` so its from-scratch
/// group rebuild works identically whether `adjacency` is [`RectAdjacency`]
/// or a precomputed table like pyramid's touching-neighbor graph.
pub fn table_flood<S: Storage, R: Dim, C: Dim, A: Adjacency>(
    mask: Board<S, R, C>,
    adjacency: &A,
    start: usize,
) -> Board<S, R, C> {
    let mut flood = mask.empty_like();
    if !mask.get_index(start) {
        return flood;
    }
    flood.set_index(start);
    let mut stack = vec![start];
    while let Some(cell) = stack.pop() {
        for nb in adjacency.neighbors(cell) {
            if mask.get_index(nb) && !flood.get_index(nb) {
                flood.set_index(nb);
                stack.push(nb);
            }
        }
    }
    flood
}

/// Table-driven counterpart to `Board::adjacency_mask`: every cell adjacent
/// to (but not a member of) `mask`, under `adjacency`'s neighbor relation.
pub fn table_neighbor_mask<S: Storage, R: Dim, C: Dim, A: Adjacency>(
    mask: Board<S, R, C>,
    adjacency: &A,
) -> Board<S, R, C> {
    let mut out = mask.empty_like();
    for cell in mask {
        for nb in adjacency.neighbors(cell) {
            if !mask.get_index(nb) {
                out.set_index(nb);
            }
        }
    }
    out
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::{Const, Dyn};

    /// `RectAdjacency::neighbors` must agree, cell by cell, with the
    /// shift-based `Board::adjacency_mask` -- the two are independent
    /// derivations of the same rectangular 4-neighbor relation and must
    /// stay bit-identical.
    fn check_matches_adjacency_mask<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
        rows: R,
        cols: C,
    ) {
        let board: Board<S, R, C> = Board::new(rows, cols);
        let adjacency = RectAdjacency::new(rows.get(), cols.get());
        for index in 0..board.len() {
            let mut seed = board;
            seed.set_index(index);
            let expected: Vec<usize> = seed.adjacency_mask().iter_set().collect();
            let mut got: Vec<usize> = adjacency.neighbors(index).into_iter().collect();
            got.sort_unstable();
            assert_eq!(got, expected, "neighbor mismatch at index {index}");
        }
    }

    #[test]
    fn rect_adjacency_matches_board_adjacency_mask() {
        check_matches_adjacency_mask::<u64, _, _>(Const::<5>, Const::<5>);
        check_matches_adjacency_mask::<[u64; 2], _, _>(Const::<9>, Const::<9>);
        check_matches_adjacency_mask::<[u64; 2], _, _>(Dyn(9), Dyn(9));
        check_matches_adjacency_mask::<u64, _, _>(Dyn(3), Dyn(7));
    }

    #[test]
    fn table_flood_matches_flood4() {
        type S = [u64; 2];
        let mut board: Board<S, Dyn, Dyn> = Board::new(Dyn(9), Dyn(9));
        for (row, col) in [(0, 0), (0, 1), (1, 1), (3, 3), (3, 4), (5, 5)] {
            board.set(row, col);
        }
        let adjacency = RectAdjacency::new(board.rows(), board.cols());
        for start in board.iter_set() {
            let expected = board.flood4(start);
            let got = table_flood(board, &adjacency, start);
            assert_eq!(
                got, expected,
                "table_flood disagrees with flood4 seeded at {start}"
            );
        }
    }

    #[test]
    fn table_neighbor_mask_matches_adjacency_mask() {
        type S = [u64; 2];
        let mut board: Board<S, Dyn, Dyn> = Board::new(Dyn(9), Dyn(9));
        for (row, col) in [(0, 0), (0, 1), (1, 1), (3, 3), (3, 4), (5, 5)] {
            board.set(row, col);
        }
        let adjacency = RectAdjacency::new(board.rows(), board.cols());
        assert_eq!(
            table_neighbor_mask(board, &adjacency),
            board.adjacency_mask()
        );
    }
}
