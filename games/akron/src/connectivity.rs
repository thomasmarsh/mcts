//! Over/under-cut-aware group connectivity: given a board's occupancy and
//! colouring, which same-coloured pieces count as *connected* under Akron's
//! rule that a lower connection is cut wherever a strictly-higher opposing
//! connection shares its projected footprint (`pyramid::crossing`).
//!
//! # Rebuild, not incremental
//!
//! [`Groups::compute`] recomputes the whole structure from scratch given a
//! board's occupancy/colour, the same choice `games/margo`'s `Groups::rebuild`
//! makes for its own (much rarer) capture-driven rebuilds. Margo mostly
//! avoids that cost by maintaining an incremental union-find that only a
//! capture or the swap rule invalidates; that split doesn't help here,
//! because *every* Akron move -- an add or a relocate -- can change which
//! connections are cut (a piece landing on, or leaving, some pillar changes
//! that pillar's whole over/under ordering, not just the moved piece's own
//! immediate neighbours), so there is no common case left for an incremental
//! update to be cheaper than. A full rebuild is proportional to board size,
//! which stays small at every supported `n` (`pyramid::MAX_N` = 10, at most
//! `pyramid::total_cells(10)` = 385 cells) -- the same reasoning that let
//! Margo choose whole-board rebuilds for its rarer case applies unconditionally
//! here.
//!
//! # Determining whether an edge is cut
//!
//! A touching edge only matters for connectivity when both endpoints are
//! occupied by the same colour -- that's what makes it a "connection" at
//! all. For such an edge, [`pyramid::crossing::get_crossing_table`] gives
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

use bitboard::{Adjacency, Dyn};
use pyramid::crossing::{get_crossing_table, CrossingTable, Edge};
use pyramid::{get_adjacency, Pyramid};

type Cells = Pyramid<[u64; 7], Dyn>;

/// A same-coloured, both-endpoints-occupied touching edge's colour, or
/// `None` if the edge doesn't currently connect anything (either endpoint
/// empty, or the two endpoints hold opposite colours).
fn edge_color(occupied: &Cells, black: &Cells, (a, b): Edge) -> Option<bool> {
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
fn is_cut(occupied: &Cells, black: &Cells, table: &CrossingTable, edge: Edge, color: bool) -> bool {
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

/// Cut-aware group connectivity over a board's occupancy/colour, computed
/// fresh (see module docs on why this is a rebuild, not incremental state).
/// Union-find (disjoint-set, path compression + union by size) restricted to
/// occupied cells, unioning only same-coloured touching edges that survive
/// the over/under rule.
#[derive(Clone, Debug)]
pub struct Groups {
    parent: Vec<u32>,
    size: Vec<u32>,
    /// `Some(colour)` for an occupied cell, `None` for an empty one.
    color: Vec<Option<bool>>,
}

impl Groups {
    /// Computes cut-aware connectivity for a base-`n` board's `occupied`/
    /// `black` pair.
    pub fn compute(n: usize, occupied: &Cells, black: &Cells) -> Self {
        let total = pyramid::total_cells(n);
        let mut groups = Groups {
            parent: (0..total as u32).collect(),
            size: vec![1; total],
            color: (0..total)
                .map(|i| occupied.get_index(i).then(|| black.get_index(i)))
                .collect(),
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

        groups
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
    use pyramid::to_coord;

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
