//! Top-down "touching" adjacency: which cells of a base-`n` pyramid
//! physically touch which, derived from the sphere-packing geometry of the
//! stack -- independent of any board's actual occupancy, computed once per
//! `n` and cached as a flat table (the way `bitboard::Board` caches its wall
//! masks).
//!
//! # The geometry
//!
//! Every piece is a ball of diameter 1 (matching this crate's unit grid
//! spacing). A ball's projected center, in level-0's coordinate frame, is
//! `(col + (level + 1) / 2, row + (level + 1) / 2)` -- level 0 centers sit
//! at half-integer offsets `(col + 0.5, row + 0.5)`; a level-`k` piece sits
//! centered over the 2x2 block of level-`(k - 1)` positions that support it,
//! so each level up shifts the center by `(0.5, 0.5)`. A ball's height rises
//! by `sqrt(2) / 2` per level -- the vertical offset that keeps a resting
//! ball's center exactly one diameter (the touching distance) from each of
//! its four physical supporters, given their `(0.5, 0.5)` horizontal offset.
//!
//! Two balls touch iff the real 3-D distance between their centers equals
//! one diameter. Writing `dc`/`dr` for the (integer) column/row difference
//! and `dl` for the level difference, the horizontal offset contributed by
//! `dl` is `(dl / 2, dl / 2)` and the vertical offset is `dl * sqrt(2) / 2`,
//! so squared distance is:
//!
//! ```text
//! (dc + dl/2)^2 + (dr + dl/2)^2 + dl^2 / 2
//! ```
//!
//! Solving this equal to 1 for small integer `dc`, `dr`, `dl` (see this
//! module's `oracle_touching_neighbors` test, which does exactly this from
//! scratch with real `f64` coordinates rather than the closed-form solution
//! below) shows touching pairs are exactly:
//!
//! - `dl == 0`, `(dc, dr)` one of `(±1, 0)`/`(0, ±1)` -- same-level
//!   orthogonal (4-adjacency) neighbors. Diagonal same-level pairs
//!   (`dc, dr = ±1, ±1`) are `sqrt(2)` apart, not touching.
//! - `dl == ±1`, `(dc, dr)` one of the four combinations matching
//!   `Pyramid::supporters`/`dependents` -- a piece touches all (up to four)
//!   pieces that support it, and all (up to four) it in turn supports.
//! - `dl` with `|dl| >= 2` never touches: the vertical term alone
//!   (`dl^2 / 2 >= 2`) already exceeds the required squared distance of 1,
//!   so no horizontal offset can bring such a pair into contact.
//!
//! This matches the informal rule text for the Shibumi-family games this
//! crate targets ("balls are adjacent to any balls that they touch,
//! including flatly adjacent balls on the same level and supported balls
//! between levels" -- nestorgames' Shibumi rule book) and cross-checks
//! against Span's report of exactly five "non-visible" (buried) balls in a
//! complete 4x4 pyramid: see `Pyramid::is_buried`, whose `dl == 2`,
//! `dc = dr = -1` occlusion case falls directly out of the same center
//! formula (it's the unique position whose center exactly coincides with
//! the occluded ball's, i.e. horizontal offset zero) but is a distinct,
//! occupancy-dependent relation from the touching graph built here -- a
//! buried piece still physically touches its neighbors.

use bitboard::{Adjacency, NeighborList};

use crate::{dependent_positions, index, level_side, total_cells};

/// For every flat cell index of a base-`n` pyramid, the flat indices of
/// cells that would physically touch it if both were occupied -- see this
/// module's docs for the geometric derivation. Purely geometric: does not
/// depend on any board's occupancy, so it's safe to compute once per `n`
/// and reuse across boards/games.
pub fn touching_neighbors(n: usize) -> Vec<Vec<usize>> {
    let total = total_cells(n);
    let mut neighbors = vec![Vec::new(); total];

    for level in 0..n {
        let side = level_side(n, level);
        for row in 0..side {
            for col in 0..side {
                let here = index(n, col, row, level);

                // Same-level orthogonal neighbors -- forward-only (col+1,
                // row+1) so each pair is visited once; both endpoints'
                // lists still get updated.
                if col + 1 < side {
                    let other = index(n, col + 1, row, level);
                    neighbors[here].push(other);
                    neighbors[other].push(here);
                }
                if row + 1 < side {
                    let other = index(n, col, row + 1, level);
                    neighbors[here].push(other);
                    neighbors[other].push(here);
                }

                // The (up to four) level-(level + 1) cells this one would
                // support -- each such pair is visited exactly once, when
                // processing the lower cell.
                for (c, r) in dependent_positions(n, col, row, level) {
                    let other = index(n, c, r, level + 1);
                    neighbors[here].push(other);
                    neighbors[other].push(here);
                }
            }
        }
    }

    neighbors
}

/// `bitboard::Adjacency` over a base-`n` pyramid's precomputed top-down
/// "touching" table (see [`touching_neighbors`]) -- lets `bitboard::GoEngine`
/// run its union-find liberty bookkeeping directly over the pyramid's
/// touching graph (conhex-like, not a plain grid) instead of the hardcoded
/// rectangular shift arithmetic `RectAdjacency` reproduces. Stores the table
/// itself, not just `n`, since `touching_neighbors` isn't cheap enough to
/// recompute on every neighbor lookup.
#[derive(Clone, Debug)]
pub struct TouchingAdjacency {
    table: Vec<Vec<usize>>,
}

impl TouchingAdjacency {
    pub fn new(n: usize) -> Self {
        Self {
            table: touching_neighbors(n),
        }
    }
}

impl Adjacency for TouchingAdjacency {
    #[inline]
    fn neighbors(&self, index: usize) -> NeighborList {
        NeighborList::from_neighbors(self.table[index].iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::to_coord;

    /// Independent from-scratch derivation: real 3-D center coordinates
    /// (see module docs) and a brute-force all-pairs distance check, sharing
    /// no arithmetic with `touching_neighbors`/`index`/`dependent_positions`
    /// beyond `to_coord` (needed only to recover `(col, row, level)` from a
    /// flat index for the distance formula, not to decide adjacency).
    fn oracle_touching_neighbors(n: usize) -> Vec<Vec<usize>> {
        let total = total_cells(n);
        let center = |col: usize, row: usize, level: usize| -> (f64, f64, f64) {
            let l = level as f64;
            (
                col as f64 + (l + 1.0) / 2.0,
                row as f64 + (l + 1.0) / 2.0,
                l * std::f64::consts::FRAC_1_SQRT_2,
            )
        };
        let centers: Vec<(f64, f64, f64)> = (0..total)
            .map(|i| {
                let (col, row, level) = to_coord(n, i);
                center(col, row, level)
            })
            .collect();

        let mut neighbors = vec![Vec::new(); total];
        for i in 0..total {
            for j in (i + 1)..total {
                let (xi, yi, zi) = centers[i];
                let (xj, yj, zj) = centers[j];
                let d2 = (xi - xj).powi(2) + (yi - yj).powi(2) + (zi - zj).powi(2);
                if (d2 - 1.0).abs() < 1e-9 {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
        }
        neighbors
    }

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn touching_neighbors_matches_geometric_oracle_every_n_in_range() {
        for n in 2..=10usize {
            let derived = touching_neighbors(n);
            let oracle = oracle_touching_neighbors(n);
            assert_eq!(derived.len(), oracle.len(), "n = {n}: cell count mismatch");
            for i in 0..derived.len() {
                assert_eq!(
                    sorted(derived[i].clone()),
                    sorted(oracle[i].clone()),
                    "n = {n}: neighbor mismatch at flat index {i} ({:?})",
                    to_coord(n, i)
                );
            }
        }
    }

    #[test]
    fn base_corner_touches_two_lateral_and_one_support_neighbor() {
        // n = 4: the level-0 corner (0, 0) has exactly two in-bounds lateral
        // neighbors ((1, 0) and (0, 1)) and supports exactly one level-1
        // cell ((0, 0)) -- degree 3, no more, no less.
        let n = 4;
        let neighbors = touching_neighbors(n);
        let here = index(n, 0, 0, 0);
        assert_eq!(neighbors[here].len(), 3);

        let expected = [index(n, 1, 0, 0), index(n, 0, 1, 0), index(n, 0, 0, 1)];
        let mut got = neighbors[here].clone();
        got.sort_unstable();
        let mut expected = expected.to_vec();
        expected.sort_unstable();
        assert_eq!(got, expected);
    }

    #[test]
    fn diagonal_same_level_neighbors_never_touch() {
        let n = 6;
        let neighbors = touching_neighbors(n);
        let a = index(n, 2, 2, 0);
        let b = index(n, 3, 3, 0);
        assert!(!neighbors[a].contains(&b));
        assert!(!neighbors[b].contains(&a));
    }

    #[test]
    fn no_touching_pair_two_or_more_levels_apart() {
        // Cross-check the closed-form claim directly: no entry in any
        // adjacency list differs from its own cell by 2 or more levels.
        for n in 2..=10usize {
            let neighbors = touching_neighbors(n);
            for (i, list) in neighbors.iter().enumerate() {
                let (_, _, li) = to_coord(n, i);
                for &j in list {
                    let (_, _, lj) = to_coord(n, j);
                    assert!(
                        li.abs_diff(lj) <= 1,
                        "n = {n}: {i} (level {li}) and {j} (level {lj}) are {} levels apart but touch",
                        li.abs_diff(lj)
                    );
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// `bitboard::GoEngine` reuse: proves `TouchingAdjacency` is a real, working
// `Adjacency` provider by driving the general incremental engine over it,
// cross-checked against a from-scratch reference that shares nothing with
// `GoEngine`/`TouchingAdjacency` but the raw `touching_neighbors` table
// itself -- so a game built on this pyramid crate can reuse `GoEngine`'s
// union-find liberty bookkeeping over the conhex-like touching graph
// instead of reimplementing it.

#[cfg(test)]
mod goengine_reuse {
    use std::collections::HashSet;

    use bitboard::{Board, Dyn, GoEngine};
    use proptest::prelude::*;

    use super::*;

    // n = 4: the Shibumi-family base size (30 cells), fitting a single u64.
    const N: usize = 4;

    fn cells() -> usize {
        total_cells(N)
    }

    /// From-scratch liberty count for the group containing `start` in
    /// `occupied_self`, against `occupied_other` -- plain BFS directly over
    /// `touching_neighbors(N)`, independent of `GoEngine`'s union-find
    /// bookkeeping and of `TouchingAdjacency`'s `Adjacency` impl.
    fn reference_liberties(
        table: &[Vec<usize>],
        occupied_self: &[bool],
        occupied_other: &[bool],
        start: usize,
    ) -> usize {
        let mut seen = vec![false; table.len()];
        let mut stack = vec![start];
        seen[start] = true;
        let mut liberties = HashSet::new();
        while let Some(cell) = stack.pop() {
            for &nb in &table[cell] {
                if occupied_self[nb] {
                    if !seen[nb] {
                        seen[nb] = true;
                        stack.push(nb);
                    }
                } else if !occupied_other[nb] {
                    liberties.insert(nb);
                }
            }
        }
        liberties.len()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn goengine_liberties_match_table_oracle(
            black_bits in proptest::collection::vec(0usize..cells(), 0..20),
            white_bits in proptest::collection::vec(0usize..cells(), 0..20),
        ) {
            let n = cells();
            let mut black_occ = vec![false; n];
            let mut white_occ = vec![false; n];
            for &i in &black_bits {
                if !white_occ[i] {
                    black_occ[i] = true;
                }
            }
            for &i in &white_bits {
                if !black_occ[i] {
                    white_occ[i] = true;
                }
            }

            let mut black: Board<u64, Dyn, Dyn> = Board::new(Dyn(1), Dyn(n));
            let mut white: Board<u64, Dyn, Dyn> = Board::new(Dyn(1), Dyn(n));
            for i in 0..n {
                if black_occ[i] {
                    black.set_index(i);
                }
                if white_occ[i] {
                    white.set_index(i);
                }
            }

            let table = touching_neighbors(N);
            let engine = GoEngine::from_boards_with_adjacency(black, white, TouchingAdjacency::new(N));

            for i in 0..n {
                if let Some(lib) = engine.liberties_at(i) {
                    let (self_occ, other_occ) = if black_occ[i] {
                        (&black_occ, &white_occ)
                    } else {
                        (&white_occ, &black_occ)
                    };
                    let expected = reference_liberties(&table, self_occ, other_occ, i);
                    prop_assert_eq!(lib as usize, expected, "cell {} liberty mismatch", i);
                }
            }
        }
    }

    #[test]
    fn play_a_capture_over_the_touching_table() {
        // The single-cell apex (level 3, degree = however many level-2
        // cells touch it) is the easiest cell to fully surround: play white
        // on every one of its neighbors, then black at the apex must find
        // zero liberties -- suicide, illegal, no capture. Fill in one more
        // black stone at a neighbor first (removing one white liberty
        // source) then replay to trigger an actual capture instead of just
        // a suicide check, proving `play`'s incremental capture path (not
        // just `check`'s legality path) works over the table.
        let apex = total_cells(N) - 1;
        let adjacency = TouchingAdjacency::new(N);
        let apex_neighbors = adjacency.table[apex].clone();
        assert!(
            !apex_neighbors.is_empty(),
            "apex must have at least one neighbor"
        );

        let mut engine: GoEngine<u64, Dyn, Dyn, _> =
            GoEngine::new_with_adjacency(Dyn(1), Dyn(total_cells(N)), TouchingAdjacency::new(N));

        // Black takes the apex first, alone -- always legal on an empty board.
        assert!(engine.play(true, apex).is_some());

        // White surrounds it on every touching neighbor. The last placement
        // must capture the lone black stone at the apex (assuming those
        // neighbors have liberties of their own, which they do on an
        // otherwise-empty n=4 pyramid).
        let mut captured_apex = false;
        for &nb in &apex_neighbors {
            if let Some(captured) = engine.play(false, nb) {
                if captured.get_index(apex) {
                    captured_apex = true;
                }
            }
        }
        assert!(captured_apex, "surrounding the apex must capture it");
        assert!(!engine.black().get_index(apex));
        assert!(!engine.white().get_index(apex));
    }
}
