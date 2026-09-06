//! Over/under-cut-aware group connectivity: given a base-`n` pyramid's
//! occupancy and a two-colour split, which same-coloured pieces count as
//! *connected* under the over/under rule ("if a connection crosses over an
//! opponent's connection at any point, the uppermost connection prevails;
//! the lower connection is cut until the upper one is removed") used by
//! Shibumi-family connection games such as Akron.
//!
//! Board-generic (`Pyramid<S, N>`, any storage/dimension) and independent of
//! any single game's `State` -- any future game in the same 4x4 Shibumi
//! family that adopts this rule can reuse this module directly instead of
//! reimplementing union-find/flood-fill/pillar-walking from scratch per
//! game, the way `games/akron`'s own `connectivity.rs` originally did (now a
//! thin re-export of this module).
//!
//! # Rebuild, not incremental
//!
//! [`Groups::compute`] recomputes the whole structure from scratch given a
//! board's occupancy/colour, the same choice `games/margo`'s `Groups::rebuild`
//! makes for its own (much rarer) capture-driven rebuilds. Margo mostly
//! avoids that cost by maintaining an incremental union-find that only a
//! capture or the swap rule invalidates; that split doesn't help here,
//! because *every* move that changes a pillar's occupancy can change which
//! connections in that same pillar are cut (a piece landing on, or leaving,
//! a pillar changes that pillar's whole over/under ordering, not just the
//! moved piece's own immediate neighbours), and a union-find has no cheap
//! way to un-union a group once an edge it depended on is invalidated. A
//! full rebuild is proportional to board size, which stays small at every
//! supported `n` (`crate::MAX_N` = 10, at most `crate::total_cells(10)` =
//! 385 cells) -- the same reasoning that lets Margo choose whole-board
//! rebuilds for its rarer case applies unconditionally here.
//!
//! # Determining whether an edge is cut
//!
//! A touching edge only matters for connectivity when both endpoints are
//! occupied by the same colour -- that's what makes it a "connection" at
//! all. For such an edge, [`crate::crossing::get_crossing_table`] gives
//! every strictly-higher, footprint-sharing partner in its *pillar*, already
//! sorted ascending by height. Walking that chain top-down, the edge is cut
//! exactly when the nearest active (occupied, same-coloured-at-its-own-two-
//! endpoints) ancestor in the chain belongs to the *other* colour -- an
//! active ancestor of the *same* colour does not cut it (own-colour
//! connections never contest each other), and an ancestor that is itself cut
//! does not act as a blocker for anything further down (a cut connection
//! isn't "the uppermost connection" at that point, so it can't cut on
//! behalf of whatever cut it). This is exactly the rules text's Figure 5/6
//! narrative: Black cuts White, then White cuts Black's cut and *restores*
//! the original White connection underneath -- the restored connection is
//! precisely "nearest uncut ancestor" skipping over the now-cut Black edge.

use bitboard::{Adjacency, Dim, Storage};

use crate::crossing::{get_crossing_table, CrossingTable, Edge};
use crate::{get_adjacency, Pyramid};

/// A same-coloured, both-endpoints-occupied touching edge's colour, or
/// `None` if the edge doesn't currently connect anything (either endpoint
/// empty, or the two endpoints hold opposite colours).
fn edge_color<S: Storage, N: Dim>(
    occupied: &Pyramid<S, N>,
    black: &Pyramid<S, N>,
    (a, b): Edge,
) -> Option<bool> {
    let color = |index: usize| -> Option<bool> {
        occupied.get_index(index).then(|| black.get_index(index))
    };
    let ca = color(a)?;
    let cb = color(b)?;
    (ca == cb).then_some(ca)
}

/// Whether `edge` (already known to be active, of colour `color`) is cut by
/// the over/under rule -- see the module docs for the "nearest uncut
/// ancestor" derivation.
fn is_cut<S: Storage, N: Dim>(
    occupied: &Pyramid<S, N>,
    black: &Pyramid<S, N>,
    table: &CrossingTable,
    edge: Edge,
    color: bool,
) -> bool {
    let empty: Vec<Edge> = Vec::new();
    let chain = table.get(&edge).unwrap_or(&empty);
    // `chain` is `edge`'s own pillar, strictly above it, ascending by
    // height. Walk from the top down so the nearest active ancestor is
    // resolved first; the first active member reached (topmost) is never
    // cut by definition, since nothing above it can be its blocker.
    let mut blocker: Option<bool> = None;
    for &above in chain.iter().rev() {
        if let Some(c) = edge_color(occupied, black, above) {
            if blocker.is_none_or(|b| b == c) {
                blocker = Some(c);
            }
            // A blocker of the opposite colour to `above` leaves `blocker`
            // unchanged: `above` is itself cut, so it can't extend its own
            // cutting effect further down the chain.
        }
    }
    blocker.is_some_and(|b| b != color)
}

/// Whether every cell in `seeds` (all occupied, all the same colour -- e.g.
/// a piece's own pre-removal group, minus that piece itself) is still
/// mutually reachable via same-coloured, non-cut touching edges on
/// `occupied`/`black` -- a flood fill seeded only from `seeds`, rather than
/// a whole-board [`Groups::compute`] followed by a pairwise `same_group`
/// check.
///
/// This computes exactly the same thing `Groups::compute` plus pairwise
/// `same_group` checks would (both apply the identical per-edge [`is_cut`]
/// test against the same `occupied`/`black`), just without first visiting
/// every other cell on the board -- including ones in unrelated groups, of
/// the other colour, or simply empty -- that a caller checking one
/// particular group's post-removal connectivity was never going to ask
/// about anyway. In particular, this is *not* a shortcut that trusts
/// `seeds`' pre-removal grouping: the flood fill re-derives reachability
/// from scratch against the post-removal board, so it correctly follows any
/// edge the removal itself newly uncut (a same-coloured piece elsewhere that
/// was cut off before the removed piece was gone can rejoin here), and
/// correctly fails to follow any edge the removal newly cut (removing a
/// piece can just as well *expose* a lower opposing connection it had been
/// shielding -- see [`is_cut`]'s "topmost active ancestor" rule: a
/// same-coloured piece directly beneath the removed one in a pillar, only
/// live because the removed piece was the nearer blocker, can lose that
/// shielding the moment it's gone).
pub fn survives_removal<S: Storage, N: Dim>(
    n: usize,
    occupied: &Pyramid<S, N>,
    black: &Pyramid<S, N>,
    seeds: &[usize],
) -> bool {
    if seeds.len() <= 1 {
        return true;
    }
    let table = get_crossing_table(n);
    let adjacency = get_adjacency(n);
    let color = black.get_index(seeds[0]);

    let mut visited = vec![false; crate::total_cells(n)];
    let mut stack = vec![seeds[0]];
    visited[seeds[0]] = true;
    while let Some(u) = stack.pop() {
        for v in adjacency.neighbors(u) {
            if visited[v] || !occupied.get_index(v) || black.get_index(v) != color {
                continue;
            }
            let edge = (u.min(v), u.max(v));
            if is_cut(occupied, black, table, edge, color) {
                continue;
            }
            visited[v] = true;
            stack.push(v);
        }
    }
    seeds.iter().all(|&s| visited[s])
}

/// Cut-aware group connectivity over a board's occupancy/colour, computed
/// fresh (see module docs on why this is a rebuild, not incremental state).
/// Union-find (disjoint-set, path compression + union by size) restricted to
/// occupied cells, unioning only same-coloured touching edges that survive
/// the over/under rule.
///
/// Besides the union-find itself, [`compute`](Self::compute) also records
/// each group's member list, so a caller that already has a `Groups` for the
/// current board (e.g. `games/akron`'s `State::is_freedom`/
/// `State::move_destinations`, both called once per candidate piece from
/// `Game::generate_actions`) can answer "which cells are in this piece's
/// group" in `O(1)` plus a `find`, instead of scanning every cell on the
/// board per candidate.
#[derive(Clone, Debug)]
pub struct Groups {
    parent: Vec<u32>,
    size: Vec<u32>,
    /// `Some(colour)` for an occupied cell, `None` for an empty one.
    color: Vec<Option<bool>>,
    /// Each group's members, indexed by that group's union-find root (as of
    /// the end of `compute` -- root *identity* is stable afterward even
    /// though `find`'s path compression keeps shortening the chains to it).
    /// A flat, `total`-sized `Vec` rather than a hash map: a root is always
    /// a plain cell index in `0..total`, so this is a direct index instead
    /// of a hash + probe; every non-root entry just sits empty.
    member_lists: Vec<Vec<usize>>,
}

impl Groups {
    /// Computes cut-aware connectivity for a base-`n` board's `occupied`/
    /// `black` pair.
    pub fn compute<S: Storage, N: Dim>(
        n: usize,
        occupied: &Pyramid<S, N>,
        black: &Pyramid<S, N>,
    ) -> Self {
        let total = crate::total_cells(n);
        let mut groups = Groups {
            parent: (0..total as u32).collect(),
            size: vec![1; total],
            color: (0..total)
                .map(|i| occupied.get_index(i).then(|| black.get_index(i)))
                .collect(),
            member_lists: vec![Vec::new(); total],
        };

        let table = get_crossing_table(n);
        let adjacency = get_adjacency(n);
        for a in 0..total {
            let Some(ca) = groups.color[a] else { continue };
            for b in adjacency.neighbors(a) {
                if b <= a {
                    // Each touching edge is undirected; process it once,
                    // from its lower-indexed endpoint.
                    continue;
                }
                let Some(cb) = groups.color[b] else { continue };
                if ca != cb {
                    continue;
                }
                let edge = (a, b);
                if is_cut(occupied, black, table, edge, ca) {
                    continue;
                }
                groups.union(a, b);
            }
        }

        for i in 0..total {
            if groups.color[i].is_some() {
                let root = groups.find(i);
                groups.member_lists[root].push(i);
            }
        }

        groups
    }

    /// `index`'s group's members (including `index` itself), via the member
    /// list [`compute`](Self::compute) built once for the whole board --
    /// `O(1)` plus a `find`, not the `O(total cells)` board scan a caller
    /// would otherwise need to answer "which cells share a group with
    /// this one".
    pub fn group_members(&mut self, index: usize) -> &[usize] {
        let root = self.find(index);
        &self.member_lists[root]
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] as usize != root {
            root = self.parent[root] as usize;
        }
        let mut cur = x;
        while self.parent[cur] as usize != root {
            let next = self.parent[cur] as usize;
            self.parent[cur] = root as u32;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (small, big) = if self.size[ra] < self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big as u32;
        self.size[big] += self.size[small];
    }

    /// The colour occupying `index`, or `None` if it's empty.
    pub fn color_of(&self, index: usize) -> Option<bool> {
        self.color[index]
    }

    /// Whether `a` and `b` are in the same connected (cut-aware) group.
    /// Both must be occupied by the same colour, or this is trivially
    /// `false`.
    pub fn same_group(&mut self, a: usize, b: usize) -> bool {
        self.color[a].is_some() && self.color[a] == self.color[b] && self.find(a) == self.find(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::to_coord;
    use bitboard::Dyn;

    type Cells = Pyramid<[u64; 7], Dyn>;

    fn cells(n: usize) -> Cells {
        Pyramid::new(Dyn(n))
    }

    fn set(
        occupied: &mut Cells,
        black: &mut Cells,
        col: usize,
        row: usize,
        level: usize,
        is_black: bool,
    ) {
        occupied.set(col, row, level);
        if is_black {
            black.set(col, row, level);
        }
    }

    /// The uncontested case: two touching same-coloured pieces with nothing
    /// else on the board are simply connected.
    #[test]
    fn touching_same_color_pieces_are_connected() {
        let n = 5;
        let mut occupied = cells(n);
        let mut black = cells(n);
        set(&mut occupied, &mut black, 2, 2, 0, true);
        set(&mut occupied, &mut black, 3, 2, 0, true);

        let mut groups = Groups::compute(n, &occupied, &black);
        let a = occupied.index(2, 2, 0);
        let b = occupied.index(3, 2, 0);
        assert!(groups.same_group(a, b));
    }

    /// Two same-coloured pieces that only touch via an edge cut by a
    /// strictly-higher opposing edge are *not* connected through that edge.
    /// Uses the exact pillar from `pyramid::crossing`'s own
    /// `a_known_diagonal_support_edge_pillar_has_two_disjoint_partners`
    /// test: base (2,2,0)-(1,2,1) touches, and its pillar's first disjoint
    /// crossing partner is (1,1,2)-(0,1,3).
    #[test]
    fn opposing_higher_edge_cuts_the_connection_below_it() {
        let n = 5;
        let mut occupied = cells(n);
        let mut black = cells(n);
        // White base edge, cut by a Black edge directly above it in the
        // same pillar.
        set(&mut occupied, &mut black, 2, 2, 0, false);
        set(&mut occupied, &mut black, 1, 2, 1, false);
        set(&mut occupied, &mut black, 1, 1, 2, true);
        set(&mut occupied, &mut black, 0, 1, 3, true);

        let mut groups = Groups::compute(n, &occupied, &black);
        let white_a = occupied.index(2, 2, 0);
        let white_b = occupied.index(1, 2, 1);
        assert!(
            !groups.same_group(white_a, white_b),
            "white connection should be cut"
        );

        let black_a = occupied.index(1, 1, 2);
        let black_b = occupied.index(0, 1, 3);
        assert!(
            groups.same_group(black_a, black_b),
            "black's own cutting edge stays connected"
        );
    }

    /// Re-cutting the cutter (Figures 5/6's narrative): adding an even
    /// higher White edge above the Black cut restores the original White
    /// connection underneath, because the Black edge that was cutting it is
    /// now itself cut and stops acting as a blocker. Uses a same-level
    /// pillar (per `pyramid::crossing`'s module docs, a same-level edge's
    /// pillar members are always endpoint-disjoint from each other, unlike
    /// a support edge's, whose consecutive members share a physical piece
    /// and so can't independently hold three alternating colours) so all
    /// three edges can genuinely be set to alternating colours.
    #[test]
    fn cutting_the_cutter_restores_the_original_connection() {
        let n = 8;
        let mut occupied = cells(n);
        let mut black = cells(n);
        // Bottom: White base edge...
        set(&mut occupied, &mut black, 3, 3, 1, false);
        set(&mut occupied, &mut black, 4, 3, 1, false);
        // ...cut by a Black edge two levels above, same footprint...
        set(&mut occupied, &mut black, 2, 2, 3, true);
        set(&mut occupied, &mut black, 3, 2, 3, true);
        // ...re-cut by a White edge two levels above that.
        set(&mut occupied, &mut black, 1, 1, 5, false);
        set(&mut occupied, &mut black, 2, 1, 5, false);

        let mut groups = Groups::compute(n, &occupied, &black);
        let white_a = occupied.index(3, 3, 1);
        let white_b = occupied.index(4, 3, 1);
        assert!(
            groups.same_group(white_a, white_b),
            "original white connection should be restored once its cutter is itself cut"
        );

        let black_a = occupied.index(2, 2, 3);
        let black_b = occupied.index(3, 2, 3);
        assert!(
            !groups.same_group(black_a, black_b),
            "black's cut should itself be cut"
        );

        let top_a = occupied.index(1, 1, 5);
        let top_b = occupied.index(2, 1, 5);
        assert!(
            groups.same_group(top_a, top_b),
            "the topmost edge is never cut"
        );
    }

    /// A crossing pillar where the higher edge is the *same* colour as the
    /// lower one never cuts anything -- own-colour connections don't
    /// contest each other.
    #[test]
    fn same_color_higher_edge_does_not_cut() {
        let n = 5;
        let mut occupied = cells(n);
        let mut black = cells(n);
        set(&mut occupied, &mut black, 2, 2, 0, true);
        set(&mut occupied, &mut black, 1, 2, 1, true);
        set(&mut occupied, &mut black, 1, 1, 2, true);
        set(&mut occupied, &mut black, 0, 1, 3, true);

        let mut groups = Groups::compute(n, &occupied, &black);
        let a = occupied.index(2, 2, 0);
        let b = occupied.index(1, 2, 1);
        assert!(groups.same_group(a, b));
    }

    /// An isolated piece with no same-coloured touching neighbour is its
    /// own singleton group, and is unrelated to any other piece.
    #[test]
    fn isolated_pieces_are_not_connected_to_anything() {
        let n = 5;
        let mut occupied = cells(n);
        let mut black = cells(n);
        set(&mut occupied, &mut black, 0, 0, 0, true);
        set(&mut occupied, &mut black, 4, 4, 0, true);

        let mut groups = Groups::compute(n, &occupied, &black);
        let a = occupied.index(0, 0, 0);
        let b = occupied.index(4, 4, 0);
        assert!(!groups.same_group(a, b));
        assert_eq!(groups.color_of(a), Some(true));

        let (col, row, level) = to_coord(n, occupied.index(1, 1, 0));
        assert!(!occupied.get(col, row, level));
    }
}
