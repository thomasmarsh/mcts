//! Over/under crossing geometry: which pairs of touching edges project onto
//! the exact same footprint in top-down `(x, y)` view, at different heights.
//! This is the geometric basis for Akron's over/under rule ("if a connection
//! crosses over an opponent's connection at any point, the uppermost
//! connection prevails; the lower connection is cut until the upper one is
//! removed").
//!
//! # The geometry
//!
//! A touching edge connects two cells whose projected centers (see
//! `adjacency`'s module docs for the center formula, `col + (level + 1) / 2`
//! per axis) differ by a fixed `(dc, dr)` for a given level difference `dl`
//! -- `(±1, 0)`/`(0, ±1)` for `dl == 0` (same-level), or one of
//! `(±0.5, ±0.5)` for `dl == 1` (support/dependent). Each axis's projected
//! coordinate is `col`/`row` plus half the level, so translating both
//! endpoints of an edge by the same `(delta_col, delta_row, delta_level)`
//! adds `delta_level / 2` to every projected coordinate on top of
//! `delta_col`/`delta_row`. The edge's projected footprint is therefore
//! unchanged exactly when `delta_col` and `delta_row` both equal the
//! negation of half `delta_level`. The smallest nonzero integer solution
//! sets `delta_level` to `2` and both `delta_col` and `delta_row` to `-1`:
//! shifting an edge two levels up and one cell up-left (in both `col` and
//! `row`) reproduces its own projected shape and position exactly, offset
//! only in height (by two levels' worth of `sqrt(2) / 2`, per `adjacency`'s
//! vertical term).
//!
//! This is the same invariant `Pyramid::is_buried` already uses for a
//! single occluded cell (`(col, row, level)` is hidden by `(col - 1, row -
//! 1, level + 2)`, the only position whose projected center exactly
//! coincides with its own), lifted from a single point to a full edge.
//! Akron has no buried-cell exclusion (unlike Margo); over/under crossing
//! is what plays that role instead, at the level of connections rather than
//! individual pieces.
//!
//! # Pillars, not pairs
//!
//! Applying the shift repeatedly produces a whole chain of touching edges
//! sharing one footprint, not just a single partner: a `dl == 1` (support)
//! edge's shift lands on the *next* support edge one level further up the
//! same diagonal, which is itself reachable from the original edge's own
//! upper endpoint by one more ordinary touching step (its own `dl == 1`
//! edge to the level above) -- so consecutive members of a footprint's
//! chain are touching edges themselves, each sharing an endpoint with the
//! next. A `dl == 0` (same-level) edge's shift, by contrast, lands two
//! levels up with no in-between member (same-level edges at different
//! levels never touch), so that chain's members are already pairwise
//! disjoint one shift apart.
//!
//! Call this whole chain -- every touching edge reachable from a given one
//! by repeated `(±1, ±1, ∓2)` shifts, in either direction -- a *pillar*. Two
//! edges of a pillar can only stand in an over/under crossing relation if
//! they don't share an endpoint: a shared endpoint is one physical piece,
//! which has a single colour, so the two edges could never belong to
//! opposing-coloured connections in the first place. Crossing needs two
//! independent connections passing the same point, not one connection
//! passing through it via two of its own edges. Within a support-edge
//! pillar that excludes only the immediately-adjacent chain member (one
//! level away); a same-level pillar's members are already all disjoint.
//! Every other, taller pairing within a pillar is a genuine crossing: the
//! rules text's "even higher level cut" (Figure 6, White re-cutting Black's
//! cut of White) is exactly a third pillar member landing above a pair that
//! already crossed.

use std::sync::OnceLock;

use rustc_hash::FxHashMap;

use crate::{in_bounds, index, to_coord};

/// A touching edge, as a canonical `(min, max)` pair of flat cell indices.
pub type Edge = (usize, usize);

/// The result type of [`crossing_table`]/[`get_crossing_table`]: each edge
/// mapped to its list of higher, endpoint-disjoint, footprint-sharing
/// partners (see module docs). `FxHashMap` (not `std::collections::HashMap`)
/// because `games/akron`'s `Groups::compute` looks this table up once per
/// touching edge on every connectivity rebuild -- a hot enough path that
/// `HashMap`'s default SipHash (DoS-resistant, but overkill for a small
/// internal `usize`-pair key with no adversarial input) shows up as real
/// self time under profiling; Fx's non-cryptographic multiply-xor hash is
/// materially cheaper per lookup and `Edge` has no untrusted input to
/// protect against.
pub type CrossingTable = FxHashMap<Edge, Vec<Edge>>;

/// Shifts `(col, row, level)` by the crossing invariant `(-1, -1, +2)`: the
/// unique translation that reproduces the same projected `(x, y)` footprint
/// two levels higher (see module docs). Returns `None` if the shifted
/// coordinate falls outside the base-`n` pyramid.
pub fn crossing_shift(
    n: usize,
    col: usize,
    row: usize,
    level: usize,
) -> Option<(usize, usize, usize)> {
    let level = level.checked_add(2)?;
    let col = col.checked_sub(1)?;
    let row = row.checked_sub(1)?;
    in_bounds(n, col, row, level).then_some((col, row, level))
}

/// The inverse of [`crossing_shift`]: `(+1, +1, -2)`, reproducing the same
/// footprint two levels lower. `None` if the shifted coordinate falls
/// outside the base-`n` pyramid (including a negative level).
fn crossing_unshift(
    n: usize,
    col: usize,
    row: usize,
    level: usize,
) -> Option<(usize, usize, usize)> {
    let level = level.checked_sub(2)?;
    let col = col + 1;
    let row = row + 1;
    in_bounds(n, col, row, level).then_some((col, row, level))
}

/// Applies a coordinate shift function to both endpoints of an edge (given
/// as flat indices), returning the shifted edge's flat indices, or `None` if
/// either endpoint's shift falls outside the pyramid.
fn shift_edge(
    n: usize,
    a: usize,
    b: usize,
    shift: impl Fn(usize, usize, usize, usize) -> Option<(usize, usize, usize)>,
) -> Option<(usize, usize)> {
    let (ac, ar, al) = to_coord(n, a);
    let (bc, br, bl) = to_coord(n, b);
    let (ac2, ar2, al2) = shift(n, ac, ar, al)?;
    let (bc2, br2, bl2) = shift(n, bc, br, bl)?;
    Some((index(n, ac2, ar2, al2), index(n, bc2, br2, bl2)))
}

/// The edge exactly two levels higher whose projected footprint coincides
/// with touching edge `(a, b)`'s (flat indices, either order) -- shifts both
/// endpoints by [`crossing_shift`]. This is one step of the edge's *pillar*
/// (see module docs), not necessarily disjoint from `(a, b)`: for a `dl ==
/// 1` edge the two share an endpoint one level up and are not a valid
/// crossing pair by themselves (see [`crossing_table`]).
pub fn crossing_partner_edge(n: usize, a: usize, b: usize) -> Option<(usize, usize)> {
    shift_edge(n, a, b, crossing_shift)
}

/// The chain of cells at levels `level`, `level ± 2`, `level ± 4`, ... (as
/// far as stays in bounds) that all share one projected `(x, y)` point --
/// one endpoint's-eye view of a pillar (see module docs). Ordered ascending
/// by level.
fn point_tower(n: usize, col: usize, row: usize, level: usize) -> Vec<(usize, usize, usize)> {
    let mut lowest = (col, row, level);
    while let Some(prev) = crossing_unshift(n, lowest.0, lowest.1, lowest.2) {
        lowest = prev;
    }

    let mut tower = vec![lowest];
    let mut current = lowest;
    while let Some(next) = crossing_shift(n, current.0, current.1, current.2) {
        tower.push(next);
        current = next;
    }
    tower
}

/// The full pillar containing touching edge `(a, b)` (flat indices), as a
/// list of touching edges in ascending height order (lowest level first).
/// Always contains at least `(a, b)` itself.
///
/// A same-level (`dl == 0`) edge's pillar is just its own `(±1, ±1, ∓2)`
/// shift chain, since same-level edges at different levels never touch each
/// other directly. A support (`dl == 1`) edge's pillar instead merges its
/// two endpoints' individual [`point_tower`]s (one at even levels relative
/// to the edge, one at odd) into a single level-ordered sequence and pairs
/// up consecutive members -- each such pair is a real touching edge (the
/// module docs' "even higher level cut" chain), not just same-parity jumps.
fn pillar_of(n: usize, a: usize, b: usize) -> Vec<(usize, usize)> {
    let (ac, ar, al) = to_coord(n, a);
    let (bc, br, bl) = to_coord(n, b);

    if al == bl {
        let mut lowest = (a, b);
        while let Some(prev) = shift_edge(n, lowest.0, lowest.1, crossing_unshift) {
            lowest = prev;
        }
        let mut chain = vec![lowest];
        let mut current = lowest;
        while let Some(next) = shift_edge(n, current.0, current.1, crossing_shift) {
            chain.push(next);
            current = next;
        }
        chain
    } else {
        let (lo, hi) = if al < bl {
            ((ac, ar, al), (bc, br, bl))
        } else {
            ((bc, br, bl), (ac, ar, al))
        };
        let mut merged: Vec<(usize, usize, usize)> = point_tower(n, lo.0, lo.1, lo.2)
            .into_iter()
            .chain(point_tower(n, hi.0, hi.1, hi.2))
            .collect();
        merged.sort_unstable_by_key(|&(_, _, level)| level);
        merged
            .windows(2)
            .map(|w| {
                (
                    index(n, w[0].0, w[0].1, w[0].2),
                    index(n, w[1].0, w[1].1, w[1].2),
                )
            })
            .collect()
    }
}

/// Whether touching edges `(a, b)` and `(c, d)` (flat indices) share a
/// physical piece -- see module docs on why a shared endpoint rules out a
/// crossing relation regardless of pillar membership.
fn shares_an_endpoint(a: usize, b: usize, c: usize, d: usize) -> bool {
    a == c || a == d || b == c || b == d
}

/// For every touching edge of a base-`n` pyramid (per
/// `adjacency::touching_neighbors`), every strictly-higher, endpoint-
/// disjoint edge sharing its projected footprint (its full pillar, minus
/// itself and any pillar-adjacent edge it shares an endpoint with) -- keyed
/// by canonical `(min, max)` flat-index pairs, values sorted ascending by
/// height (lowest crossing partner first). An edge with an empty list has no
/// crossing partner at all (it's alone in its pillar, e.g. within the top
/// two levels for a support edge).
pub fn crossing_table(n: usize) -> CrossingTable {
    let neighbors = crate::adjacency::touching_neighbors(n);
    let mut edges = Vec::new();
    for (a, list) in neighbors.iter().enumerate() {
        for &b in list {
            if a < b {
                edges.push((a, b));
            }
        }
    }

    let mut table = FxHashMap::default();
    for &(a, b) in &edges {
        let pillar = pillar_of(n, a, b);
        let here = pillar
            .iter()
            .position(|&e| e == (a, b))
            .expect("pillar_of always contains its own input edge");
        let partners: Vec<(usize, usize)> = pillar[here + 1..]
            .iter()
            .filter(|&&(c, d)| !shares_an_endpoint(a, b, c, d))
            .map(|&(c, d)| (c.min(d), c.max(d)))
            .collect();
        table.insert((a, b), partners);
    }
    table
}

/// Global cache of [`crossing_table`] results keyed by base width `n`,
/// mirroring `adjacency::get_adjacency`'s cache -- `crossing_table`
/// allocates and walks the full touching table on every call.
static CROSSING_CACHE: OnceLock<Vec<CrossingTable>> = OnceLock::new();

/// Returns a `&'static` reference to a precomputed [`crossing_table`] for
/// base width `n`. Panics if `n > 10` (no such board size is valid), same as
/// [`crate::adjacency::get_adjacency`].
pub fn get_crossing_table(n: usize) -> &'static CrossingTable {
    let tables = CROSSING_CACHE.get_or_init(|| (0..=10).map(crossing_table).collect());
    &tables[n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency::touching_neighbors;
    use std::collections::HashMap;
    use crate::total_cells;

    /// Independent from-scratch derivation: real 3-D center coordinates
    /// (same formula as `adjacency`'s own oracle test, not shared code) and
    /// a brute-force search over *all* pairs of touching edges for ones
    /// whose projected `(x, y)` endpoints coincide -- doesn't assume the
    /// `(-1, -1, +2)` shift, pillar-chasing, or any other part of
    /// `crossing_table`'s derivation, so this is a genuine geometric check
    /// of the claim rather than a restatement of it. Pairs sharing an
    /// endpoint are excluded -- see module docs on why that's a real
    /// geometric requirement (one physical piece has one colour), not an
    /// arbitrary filter.
    fn oracle_crossing_table(n: usize) -> HashMap<(usize, usize), Vec<(usize, usize)>> {
        let center = |col: usize, row: usize, level: usize| -> (f64, f64, f64) {
            let l = level as f64;
            (
                col as f64 + (l + 1.0) / 2.0,
                row as f64 + (l + 1.0) / 2.0,
                l * std::f64::consts::FRAC_1_SQRT_2,
            )
        };
        let projected_endpoints = |a: usize, b: usize| -> ((f64, f64), (f64, f64)) {
            let (ac, ar, al) = to_coord(n, a);
            let (bc, br, bl) = to_coord(n, b);
            let (ax, ay, _) = center(ac, ar, al);
            let (bx, by, _) = center(bc, br, bl);
            ((ax, ay), (bx, by))
        };
        let height = |a: usize, b: usize| -> f64 {
            let (ac, ar, al) = to_coord(n, a);
            let (bc, br, bl) = to_coord(n, b);
            let (_, _, az) = center(ac, ar, al);
            let (_, _, bz) = center(bc, br, bl);
            az.min(bz)
        };
        let close =
            |p: (f64, f64), q: (f64, f64)| (p.0 - q.0).abs() < 1e-9 && (p.1 - q.1).abs() < 1e-9;

        let neighbors = touching_neighbors(n);
        let mut edges = Vec::new();
        for (a, list) in neighbors.iter().enumerate() {
            for &b in list {
                if a < b {
                    edges.push((a, b));
                }
            }
        }

        let mut table = HashMap::new();
        for &(a, b) in &edges {
            let (ap, bp) = projected_endpoints(a, b);
            let mut partners: Vec<(usize, usize)> = edges
                .iter()
                .copied()
                .filter(|&(c, d)| {
                    if (a, b) == (c, d) || shares_an_endpoint(a, b, c, d) {
                        return false;
                    }
                    let (cp, dp) = projected_endpoints(c, d);
                    let same_orientation = close(ap, cp) && close(bp, dp);
                    let swapped_orientation = close(ap, dp) && close(bp, cp);
                    (same_orientation || swapped_orientation) && height(c, d) > height(a, b)
                })
                .map(|(c, d)| (c.min(d), c.max(d)))
                .collect();
            partners.sort_unstable_by(|&(c, d), &(e, f)| {
                height(c, d).partial_cmp(&height(e, f)).unwrap()
            });
            table.insert((a, b), partners);
        }
        table
    }

    #[test]
    fn crossing_table_matches_geometric_oracle_every_n_in_range() {
        for n in 2..=8usize {
            let derived: HashMap<Edge, Vec<Edge>> = crossing_table(n).into_iter().collect();
            let oracle = oracle_crossing_table(n);
            assert_eq!(derived, oracle, "n = {n}: crossing table mismatch");
        }
    }

    #[test]
    fn crossing_shift_matches_is_buried_for_single_points() {
        // The crossing shift is is_buried's own invariant, lifted from a
        // single point to an edge -- confirm the two agree pointwise for
        // every cell: placing a piece at the shifted coordinate must make
        // `is_buried` true at the original cell, and vice versa.
        use crate::Pyramid;
        use bitboard::Dyn;
        let n = 6;
        for i in 0..total_cells(n) {
            let (col, row, level) = to_coord(n, i);
            let shifted = crossing_shift(n, col, row, level);

            let mut pyramid: Pyramid<[u64; 7], Dyn> = Pyramid::new(Dyn(n));
            pyramid.set(col, row, level);
            match shifted {
                Some((sc, sr, sl)) => {
                    pyramid.set(sc, sr, sl);
                    assert!(
                        pyramid.is_buried(col, row, level),
                        "({col},{row},{level}): crossing_shift says {:?} occludes it, but is_buried disagrees",
                        (sc, sr, sl)
                    );
                }
                None => {
                    // No in-bounds occluder exists per crossing_shift --
                    // is_buried must independently agree that nothing can
                    // ever bury this cell, for any occupancy.
                    for level2 in 0..n {
                        for row2 in 0..pyramid.level_side(level2) {
                            for col2 in 0..pyramid.level_side(level2) {
                                let mut candidate = pyramid;
                                candidate.set(col2, row2, level2);
                                assert!(
                                    !candidate.is_buried(col, row, level),
                                    "({col},{row},{level}): crossing_shift says no occluder exists, \
                                     but ({col2},{row2},{level2}) buries it"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn edge_within_top_two_levels_has_no_crossing_partner() {
        let n = 5;
        // The apex (level n - 1) and everything at level n - 2 can never
        // have a crossing partner: the +2 shift always leaves the pyramid,
        // and there's nothing below to make it a taller pillar either.
        let apex = index(n, 0, 0, n - 1);
        let table = crossing_table(n);
        for &nb in &touching_neighbors(n)[apex] {
            let key = (apex.min(nb), apex.max(nb));
            assert_eq!(table[&key], Vec::new());
        }
    }

    #[test]
    fn a_known_diagonal_support_edge_pillar_has_two_disjoint_partners() {
        // n = 5: base-level (2, 2, 0) touches level-1 (1, 2, 1) (one of its
        // four dependents). Its pillar climbs (2,2,0)-(1,2,1)-(1,1,2)-
        // (0,1,3)-(0,0,4), alternating between two projected points every
        // level (see module docs). The immediately-adjacent pillar member
        // (1,1,2)-(1,2,1) shares endpoint (1,2,1) and is excluded; the two
        // taller members, (1,1,2)-(0,1,3) and (0,1,3)-(0,0,4), are both
        // genuine disjoint crossing partners, closest first.
        let n = 5;
        let a = index(n, 2, 2, 0);
        let b = index(n, 1, 2, 1);
        assert!(touching_neighbors(n)[a].contains(&b));

        let mid = (index(n, 1, 1, 2), index(n, 0, 1, 3));
        let top = (index(n, 0, 1, 3), index(n, 0, 0, 4));
        let expected = vec![
            (mid.0.min(mid.1), mid.0.max(mid.1)),
            (top.0.min(top.1), top.0.max(top.1)),
        ];

        assert_eq!(crossing_table(n)[&(a.min(b), a.max(b))], expected);
    }

    #[test]
    fn a_known_same_level_edge_pillar_partner_is_already_disjoint_one_shift_up() {
        // n = 6: level-1 same-level edge (1, 1, 1)-(2, 1, 1) shifts directly
        // to level-3 (0, 0, 3)-(1, 0, 3) -- same-level edges at different
        // levels never touch, so this first pillar step is already a valid
        // (disjoint) crossing partner, unlike the support-edge case above.
        let n = 6;
        let a = index(n, 1, 1, 1);
        let b = index(n, 2, 1, 1);
        assert!(touching_neighbors(n)[a].contains(&b));

        let expected_a = index(n, 0, 0, 3);
        let expected_b = index(n, 1, 0, 3);
        assert_eq!(
            crossing_table(n)[&(a.min(b), a.max(b))],
            vec![(expected_a.min(expected_b), expected_a.max(expected_b))]
        );
    }

    #[test]
    fn crossing_table_entries_are_touching_edges_on_both_sides() {
        // Every key and every value in the table must itself be a real
        // touching pair -- the shift wouldn't be meaningful otherwise.
        for n in 2..=8usize {
            let neighbors = touching_neighbors(n);
            let touches = |a: usize, b: usize| neighbors[a].contains(&b);
            for (&(a, b), partners) in &crossing_table(n) {
                assert!(touches(a, b), "n = {n}: key ({a}, {b}) doesn't touch");
                for &(c, d) in partners {
                    assert!(touches(c, d), "n = {n}: value ({c}, {d}) doesn't touch");
                }
            }
        }
    }

    #[test]
    fn crossing_partners_never_share_an_endpoint_with_their_key() {
        for n in 2..=8usize {
            for (&(a, b), partners) in &crossing_table(n) {
                for &(c, d) in partners {
                    assert!(
                        !shares_an_endpoint(a, b, c, d),
                        "n = {n}: ({a}, {b}) and ({c}, {d}) share an endpoint but are listed as crossing"
                    );
                }
            }
        }
    }
}
