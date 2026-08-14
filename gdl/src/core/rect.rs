//! The `Rect` topology (see `DESIGN.md`'s "Topology model"): an N x M grid, single-bit
//! occupancy per cell, row-major site indices matching [`game_core::bitboard::BitBoard`]'s own
//! `row * cols + col` convention.

/// An `N x M` rectangular grid topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub rows: usize,
    pub cols: usize,
}

impl Rect {
    /// Converts a site index into a row and column.
    pub fn to_coord(&self, site: usize) -> (usize, usize) {
        (site / self.cols, site % self.cols)
    }

    /// Every maximal-length window of exactly `length` consecutive sites along the board's four
    /// line directions (horizontal, vertical, and both diagonals). This is the general form of
    /// what a fixed board reduces to as a handful of static site-index lists -- for a 3x3 board
    /// with `length == 3` it produces exactly the 8 rows/columns/diagonals a tic-tac-toe win
    /// check tests, but the same sliding-window logic works unchanged for any board size or
    /// line length (e.g. 5-in-a-row on a larger board).
    pub fn lines(&self, length: usize) -> Vec<Vec<usize>> {
        if length == 0 {
            return Vec::new();
        }
        let step = length as isize - 1;
        let mut lines = Vec::new();
        // (row delta, col delta) per step, one representative per line direction -- the reverse
        // of each (e.g. (0, -1) for horizontal) is deliberately omitted, since starting the scan
        // from every cell in the forward direction already covers each window exactly once.
        for &(dr, dc) in &[(0isize, 1isize), (1, 0), (1, 1), (1, -1)] {
            for start_row in 0..self.rows {
                for start_col in 0..self.cols {
                    let end_row = start_row as isize + dr * step;
                    let end_col = start_col as isize + dc * step;
                    if end_row < 0
                        || end_row >= self.rows as isize
                        || end_col < 0
                        || end_col >= self.cols as isize
                    {
                        continue;
                    }
                    let sites = (0..length as isize)
                        .map(|i| {
                            let r = (start_row as isize + dr * i) as usize;
                            let c = (start_col as isize + dc * i) as usize;
                            r * self.cols + c
                        })
                        .collect();
                    lines.push(sites);
                }
            }
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tic_tac_toe_lines() {
        let rect = Rect { rows: 3, cols: 3 };
        let lines = rect.lines(3);
        assert_eq!(lines.len(), 8);
        // 3 rows, 3 columns, 2 diagonals.
        assert!(lines.contains(&vec![0, 1, 2]));
        assert!(lines.contains(&vec![3, 4, 5]));
        assert!(lines.contains(&vec![6, 7, 8]));
        assert!(lines.contains(&vec![0, 3, 6]));
        assert!(lines.contains(&vec![1, 4, 7]));
        assert!(lines.contains(&vec![2, 5, 8]));
        assert!(lines.contains(&vec![0, 4, 8]));
        assert!(lines.contains(&vec![2, 4, 6]));
    }

    #[test]
    fn no_lines_longer_than_the_board() {
        let rect = Rect { rows: 3, cols: 3 };
        assert!(rect.lines(4).is_empty());
    }

    #[test]
    fn non_square_board() {
        // A 2x4 board has no length-3 diagonal, but does have length-3 horizontal runs.
        let rect = Rect { rows: 2, cols: 4 };
        let lines = rect.lines(3);
        assert_eq!(lines.len(), 4); // 2 per row * 2 rows, no verticals/diagonals fit.
        for line in &lines {
            assert_eq!(line.len(), 3);
        }
    }
}
