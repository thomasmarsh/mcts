//! The `Hex` topology (see `DESIGN.md`'s "Topology model"): hexagonal cells packed into the same
//! row-major `side x side` site indexing as [`super::Rect`] (`row * side + col`), with six-way
//! adjacency instead of four/eight -- see `game_core::bitboard::BitBoard::flood6` for the
//! concrete shift set this relies on. Two shapes share this one underlying `side x side` grid and
//! its six-way adjacency unchanged, differing only in which sites are valid and how the board's
//! edges are named: [`HexShape::Rhombus`] (Ludii's `(hex Diamond <side>)`, used by Hex) uses every
//! site in the grid; [`HexShape::Triangle`] (Ludii's `(hex Triangle <side>)`, used by Y) restricts
//! to the upper-left triangular half (`row + col < side`) -- a triangular board is literally a
//! bounded subset of the same infinite hex lattice a rhombus board also samples from, so it needs
//! no new coordinate packing or adjacency, only a smaller valid-site mask (see [`Hex::valid_sites`])
//! and a different edge set (three sides meeting at three corners, instead of four).

/// One of the two board shapes this `side x side` grid can represent -- see the module doc for
/// why both share the same underlying indexing/adjacency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexShape {
    Rhombus,
    Triangle,
}

/// A `side x side` hex board, either a full rhombus or a triangular subset of one -- see
/// [`HexShape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hex {
    pub side: usize,
    pub shape: HexShape,
}

/// One of a `Hex { Rhombus }` board's four straight edges. Named after the compass point each one
/// faces once the underlying `side x side` square is pictured rotated 45 degrees into a diamond
/// (`crate::style_c`'s `(side NE|SE|SW|NW)` names these directly) -- not a claim about matching
/// real Ludii's rendered board geometry, only about being internally consistent between
/// [`crate::core::interp`] and its own oracle test (there's no existing `games/hex` crate to check
/// against, per `DESIGN.md`'s corpus notes). Only meaningful for [`HexShape::Rhombus`] -- see
/// [`TriangleEdge`] for the other shape's three edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    North,
    South,
    East,
    West,
}

/// One of a `Hex { Triangle }` board's three straight edges -- see the module doc for why a
/// triangle is a bounded subset of the same grid a rhombus uses, with its own edge set. Named
/// after the shape's own geometry (not a compass point, since a triangle doesn't have four sides
/// to name that way): `Bottom` is row `0`, `Left` is column `0`, `Hypotenuse` is the far diagonal
/// `row + col == side - 1`. Each pair of edges shares exactly one corner site -- `Bottom`/`Left`
/// share site `0`, `Bottom`/`Hypotenuse` share site `side - 1`, `Left`/`Hypotenuse` share site
/// `(side - 1) * side` -- the three-corner topology a real triangular Y board also has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangleEdge {
    Bottom,
    Left,
    Hypotenuse,
}

impl Hex {
    /// Converts a site index into a (row, col) coordinate, matching [`super::Rect::to_coord`].
    pub fn to_coord(&self, site: usize) -> (usize, usize) {
        (site / self.side, site % self.side)
    }

    /// Every site of this `side x side` grid that's actually part of the board -- every site for
    /// [`HexShape::Rhombus`], or the upper-left triangular half (`row + col < side`) for
    /// [`HexShape::Triangle`]. A move generator masks its `(sites Empty)` region against this so
    /// legal moves never land outside the board's real shape; `flood`/`connects` need no such
    /// mask themselves, since they only ever traverse sites a player has actually been allowed to
    /// place a stone on.
    pub fn valid_sites(&self) -> Vec<usize> {
        let n = self.side;
        match self.shape {
            HexShape::Rhombus => (0..n * n).collect(),
            HexShape::Triangle => (0..n)
                .flat_map(|r| (0..n - r).map(move |c| r * n + c))
                .collect(),
        }
    }

    /// Every site along `edge`: row `side - 1` is `North` (the board's top row, per
    /// `game_core::bitboard::BitBoard`'s bottom-left origin), row `0` is `South`, column `side -
    /// 1` is `East`, column `0` is `West`. Only meaningful for [`HexShape::Rhombus`] -- see
    /// [`Hex::triangle_edge`] for the other shape.
    pub fn edge(&self, edge: Edge) -> Vec<usize> {
        let n = self.side;
        match edge {
            Edge::South => (0..n).collect(),
            Edge::North => (0..n).map(|c| (n - 1) * n + c).collect(),
            Edge::West => (0..n).map(|r| r * n).collect(),
            Edge::East => (0..n).map(|r| r * n + (n - 1)).collect(),
        }
    }

    /// Every site along `edge` -- see [`TriangleEdge`]'s doc comment for which sites each of the
    /// three variants names. Only meaningful for [`HexShape::Triangle`].
    pub fn triangle_edge(&self, edge: TriangleEdge) -> Vec<usize> {
        let n = self.side;
        match edge {
            TriangleEdge::Bottom => (0..n).collect(),
            TriangleEdge::Left => (0..n).map(|r| r * n).collect(),
            TriangleEdge::Hypotenuse => (0..n).map(|r| r * n + (n - 1 - r)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_of_a_3x3_board() {
        let hex = Hex {
            side: 3,
            shape: HexShape::Rhombus,
        };
        assert_eq!(hex.edge(Edge::South), vec![0, 1, 2]);
        assert_eq!(hex.edge(Edge::North), vec![6, 7, 8]);
        assert_eq!(hex.edge(Edge::West), vec![0, 3, 6]);
        assert_eq!(hex.edge(Edge::East), vec![2, 5, 8]);
    }

    #[test]
    fn rhombus_valid_sites_is_the_whole_grid() {
        let hex = Hex {
            side: 3,
            shape: HexShape::Rhombus,
        };
        assert_eq!(hex.valid_sites(), (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn triangle_valid_sites_of_a_side_4_board() {
        let hex = Hex {
            side: 4,
            shape: HexShape::Triangle,
        };
        // row 0: cols 0-3 (all valid, row+col<4); row 1: cols 0-2; row 2: cols 0-1; row 3: col 0.
        assert_eq!(hex.valid_sites(), vec![0, 1, 2, 3, 4, 5, 6, 8, 9, 12]);
    }

    #[test]
    fn triangle_edges_of_a_side_4_board_share_exactly_one_corner_each() {
        let hex = Hex {
            side: 4,
            shape: HexShape::Triangle,
        };
        let bottom = hex.triangle_edge(TriangleEdge::Bottom);
        let left = hex.triangle_edge(TriangleEdge::Left);
        let hyp = hex.triangle_edge(TriangleEdge::Hypotenuse);
        assert_eq!(bottom, vec![0, 1, 2, 3]);
        assert_eq!(left, vec![0, 4, 8, 12]);
        assert_eq!(hyp, vec![3, 6, 9, 12]);

        let shared = |a: &[usize], b: &[usize]| -> Vec<usize> {
            a.iter().copied().filter(|s| b.contains(s)).collect()
        };
        assert_eq!(shared(&bottom, &left), vec![0]);
        assert_eq!(shared(&bottom, &hyp), vec![3]);
        assert_eq!(shared(&left, &hyp), vec![12]);
    }

    #[test]
    fn to_coord_matches_row_major_indexing() {
        let hex = Hex {
            side: 3,
            shape: HexShape::Rhombus,
        };
        assert_eq!(hex.to_coord(4), (1, 1));
        assert_eq!(hex.to_coord(8), (2, 2));
    }
}
