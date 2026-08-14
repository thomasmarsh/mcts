//! The `Hex { Rhombus }` topology (see `DESIGN.md`'s "Topology model"): a `side x side` rhombus
//! of hexagonal cells, using the same row-major site indexing as [`super::Rect`] (`row * side +
//! col`), but with six-way adjacency instead of four/eight -- see
//! `game_core::bitboard::BitBoard::flood6` for the concrete shift set this relies on.

/// A `side x side` rhombus-shaped hex board (Ludii's `(hex Diamond <side>)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hex {
    pub side: usize,
}

/// One of a `Hex` rhombus's four straight edges. Named after the compass point each one faces
/// once the underlying `side x side` square is pictured rotated 45 degrees into a diamond, per
/// [`Hex::edge_for_compass`]'s doc comment -- not a claim about matching real Ludii's rendered
/// board geometry, only about being internally consistent between [`crate::core::interp`] and
/// its own oracle test (there's no existing `games/hex` crate to check against, per `DESIGN.md`'s
/// corpus notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    North,
    South,
    East,
    West,
}

impl Hex {
    /// Converts a site index into a (row, col) coordinate, matching [`super::Rect::to_coord`].
    pub fn to_coord(&self, site: usize) -> (usize, usize) {
        (site / self.side, site % self.side)
    }

    /// Every site along `edge`: row `side - 1` is `North` (the board's top row, per
    /// `game_core::bitboard::BitBoard`'s bottom-left origin), row `0` is `South`, column `side -
    /// 1` is `East`, column `0` is `West`.
    pub fn edge(&self, edge: Edge) -> Vec<usize> {
        let n = self.side;
        match edge {
            Edge::South => (0..n).collect(),
            Edge::North => (0..n).map(|c| (n - 1) * n + c).collect(),
            Edge::West => (0..n).map(|r| r * n).collect(),
            Edge::East => (0..n).map(|r| r * n + (n - 1)).collect(),
        }
    }

    /// Maps a `.lud` `(sites Side <compassDirection>)` compass point onto one of this rhombus's
    /// four edges. `lud/Hex.lud` names its four sides `NE`/`SE`/`SW`/`NW` (the diagonal compass
    /// points, since a diamond's sides don't face the cardinal directions) -- picture the
    /// `side x side` square rotated 45 degrees clockwise into a diamond: the square's North edge
    /// now faces NE, East faces SE, South faces SW, and West faces NW. `None` for any other
    /// compass point, since nothing in the corpus so far needs one.
    pub fn edge_for_compass(compass: crate::ast::types::CompassDirection) -> Option<Edge> {
        use crate::ast::types::CompassDirection as C;
        match compass {
            C::NE => Some(Edge::North),
            C::SE => Some(Edge::East),
            C::SW => Some(Edge::South),
            C::NW => Some(Edge::West),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_of_a_3x3_board() {
        let hex = Hex { side: 3 };
        assert_eq!(hex.edge(Edge::South), vec![0, 1, 2]);
        assert_eq!(hex.edge(Edge::North), vec![6, 7, 8]);
        assert_eq!(hex.edge(Edge::West), vec![0, 3, 6]);
        assert_eq!(hex.edge(Edge::East), vec![2, 5, 8]);
    }

    #[test]
    fn compass_mapping_pairs_opposite_edges() {
        use crate::ast::types::CompassDirection as C;
        // P1 ((sites Side NE) (sites Side SW)) and P2 ((sites Side NW) (sites Side SE)) must
        // each map to a pair of *opposite* edges, and all four edges must be distinct, for
        // "connects across the board" to mean anything.
        assert_eq!(Hex::edge_for_compass(C::NE), Some(Edge::North));
        assert_eq!(Hex::edge_for_compass(C::SW), Some(Edge::South));
        assert_eq!(Hex::edge_for_compass(C::NW), Some(Edge::West));
        assert_eq!(Hex::edge_for_compass(C::SE), Some(Edge::East));
    }

    #[test]
    fn other_compass_points_are_unsupported() {
        use crate::ast::types::CompassDirection as C;
        assert_eq!(Hex::edge_for_compass(C::N), None);
        assert_eq!(Hex::edge_for_compass(C::E), None);
    }

    #[test]
    fn to_coord_matches_row_major_indexing() {
        let hex = Hex { side: 3 };
        assert_eq!(hex.to_coord(4), (1, 1));
        assert_eq!(hex.to_coord(8), (2, 2));
    }
}
