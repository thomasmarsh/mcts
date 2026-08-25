//! 2D raster acceleration for Margo: flattens the 3D pyramid into fixed-stride
//! per-level bitboards, then uses 2D bitwise operations for group flood fills
//! and liberty counting.
#![allow(dead_code)]
//!
//! # Connectivity model
//!
//! Each pyramid cell `(col, row, level)` maps to raster index `row * n + col`
//! in level-board `levels[level]`. The raster surface is the topmost occupied
//! piece in each column. Connectivity uses two edge types:
//!
//! 1. **Same-level 4-way**: `(c, r, L)` ↔ `(c±1, r, L)` and `(c, r±1, L)`.
//!    These map to 3D same-level orthogonal touching (distance = 1 diameter).
//! 2. **Support relation**: a piece touches its four supporters at level-1
//!    and its (up to four) dependents at level+1. The upper piece blocks
//!    supporters on the raster. Our flood explicitly queues these cross-level
//!    edges via "peel-back".
//!
//! Same-level diagonal (`±(n+1)` offset) is NOT touching (3D distance = √2).
//!
//! # Liberties
//!
//! Only empty level-0 cells count. A piece at level L projects to a level-0
//! footprint of `(L+1)×(L+1)` cells. The group's liberties are empty level-0
//! cells 4-way-adjacent to the union of all members' footprints.

use crate::{Cells, MAX_N};
use bitboard::{Board, Dyn};

const STRIDE: usize = MAX_N;
type Storage = [u64; 2];
pub type LevelBoard = Board<Storage, Dyn, Dyn>;

fn level_board(n: usize) -> LevelBoard {
    LevelBoard::new(Dyn(n), Dyn(n))
}

#[derive(Clone, Debug)]
pub struct Raster {
    levels: Vec<LevelBoard>,
    n: usize,
    /// Wall masks for shift operations — prevent bit wrap-around.
    east_wall: LevelBoard,
    west_wall: LevelBoard,
    south_wall: LevelBoard,
    north_wall: LevelBoard,
}

impl Raster {
    pub fn from_pyramid(n: usize, occupied: &Cells) -> Self {
        let mut levels: Vec<LevelBoard> = (0..n).map(|_| level_board(n)).collect();
        for idx in 0..occupied.total_cells() {
            if occupied.get_index(idx) {
                let (col, row, level) = occupied.to_coord(idx);
                levels[level].set_index(row * n + col);
            }
        }
        Raster {
            levels,
            n,
            east_wall: wall_col(n, n - 1),
            west_wall: wall_col(n, 0),
            south_wall: wall_row(n, n - 1),
            north_wall: wall_row(n, 0),
        }
    }

    /// Compute the top-down visibility mask. Not used by the fast-path
    /// flood/capture checks — callers who need it pay the O(n²) cost.
    pub fn surface(&self) -> LevelBoard {
        Self::compute_surface(self.n, &self.levels)
    }
    pub fn n(&self) -> usize {
        self.n
    }

    // ── surface computation ────────────────────────────────────────────

    fn compute_surface(n: usize, levels: &[LevelBoard]) -> LevelBoard {
        let mut surface = level_board(n);
        let mut blocked = level_board(n);
        let not_east = !wall_col(n, n - 1);
        let not_south = !wall_row(n, n - 1);

        for l in (0..n).rev() {
            let layer = &levels[l];
            let visible = *layer & !blocked;
            surface |= visible;
            if l > 0 {
                // A piece at (c,r) blocks (c,r), (c+1,r), (c,r+1), (c+1,r+1)
                // on the level below.
                let e = (*layer & not_east) << 1usize;
                let s = (*layer & not_south) << n;
                let se = (e & not_south) << n;
                blocked = *layer | e | s | se;
            }
        }
        surface
    }

    // ── shift helpers ──────────────────────────────────────────────────

    fn shift_e(&self, b: &LevelBoard) -> LevelBoard {
        (*b & !self.east_wall) << 1usize
    }
    fn shift_w(&self, b: &LevelBoard) -> LevelBoard {
        (*b & !self.west_wall) >> 1usize
    }
    fn shift_s(&self, b: &LevelBoard) -> LevelBoard {
        (*b & !self.south_wall) << self.n
    }
    fn shift_n(&self, b: &LevelBoard) -> LevelBoard {
        (*b & !self.north_wall) >> self.n
    }
    fn shift_se(&self, b: &LevelBoard) -> LevelBoard {
        self.shift_s(&self.shift_e(b))
    }
    fn shift_nw(&self, b: &LevelBoard) -> LevelBoard {
        self.shift_n(&self.shift_w(b))
    }

    // ── public API ─────────────────────────────────────────────────────

    pub fn raster_index(&self, col: usize, row: usize) -> usize {
        row * self.n + col
    }

    /// Place a piece at `(col, row, level)`. Only toggles the bit in
    /// `levels`. Returns the linear raster index.
    pub fn place(&mut self, col: usize, row: usize, level: usize) -> usize {
        let pos = self.raster_index(col, row);
        debug_assert!(level < self.n);
        self.levels[level].set_index(pos);
        pos
    }

    /// Remove a piece from `levels`.
    pub fn remove(&mut self, col: usize, row: usize, level: usize) {
        let pos = self.raster_index(col, row);
        self.levels[level].clear_index(pos);
    }

    /// Flood connected same-color pieces starting from `(col, row, level)`.
    /// Uses bulk bitwise expansion: same-level 4-way neighbors via shift
    /// stencils, cross-level support/dependent edges via composed shifts.
    /// Iterates to a fixed point across all levels simultaneously — each
    /// iteration processes all frontier cells in parallel.
    pub fn flood(
        &self,
        col: usize,
        row: usize,
        level: usize,
        color: &[LevelBoard],
    ) -> Vec<LevelBoard> {
        let n = self.n;
        let mut result: Vec<LevelBoard> = (0..n).map(|_| level_board(n)).collect();
        let sp = self.raster_index(col, row);
        result[level].set_index(sp);

        loop {
            let mut changed = false;

            // Same-level bulk expansion.
            for l in 0..n {
                let frontier = &result[l];
                if frontier.count_ones() == 0 {
                    continue;
                }
                let nbrs = self.shift_e(frontier)
                    | self.shift_w(frontier)
                    | self.shift_s(frontier)
                    | self.shift_n(frontier);
                let new_same = nbrs & self.levels[l] & color[l] & !result[l];
                if new_same.count_ones() > 0 {
                    result[l] |= new_same;
                    changed = true;
                }
            }

            // Cross-level: supporters at level-1.
            // A piece at (c,r,L) is supported by (c,r,L-1), (c+1,r,L-1),
            // (c,r+1,L-1), (c+1,r+1,L-1). In bulk: shift E, S, SE and OR.
            for l in 1..n {
                let above = &result[l];
                if above.count_ones() == 0 {
                    continue;
                }
                let sup = *above | self.shift_e(above) | self.shift_s(above) | self.shift_se(above);
                let new_sup = sup & self.levels[l - 1] & color[l - 1] & !result[l - 1];
                if new_sup.count_ones() > 0 {
                    result[l - 1] |= new_sup;
                    changed = true;
                }
            }

            // Cross-level: dependents at level+1.
            // A piece at (c,r,L) supports (c-1,r-1,L+1), (c,r-1,L+1),
            // (c-1,r,L+1), (c,r,L+1). In bulk: shift W, N, NW and OR.
            for l in 0..(n - 1) {
                let below = &result[l];
                if below.count_ones() == 0 {
                    continue;
                }
                let dep = *below | self.shift_w(below) | self.shift_n(below) | self.shift_nw(below);
                let new_dep = dep & self.levels[l + 1] & color[l + 1] & !result[l + 1];
                if new_dep.count_ones() > 0 {
                    result[l + 1] |= new_dep;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
        result
    }

    /// Count liberties of `group` (output of [`flood`]). Liberties are
    /// empty level-0 cells 4-way-adjacent to level-0 group members.
    /// Higher-level pieces contribute liberties only indirectly via their
    /// connected level-0 supporters — an isolated high-level piece has no
    /// liberty. (Margo rule: "freedoms only exist on the board level.")
    pub fn count_liberties(&self, group: &[LevelBoard]) -> usize {
        let level0 = &group[0];
        if level0.count_ones() == 0 {
            return 0;
        }
        let adjacent = self.shift_e(level0)
            | self.shift_w(level0)
            | self.shift_s(level0)
            | self.shift_n(level0);
        let empty = !self.levels[0];
        (adjacent & empty).count_ones() as usize
    }

    /// Find enemy cells touching `group`. Uses bulk shift stencils to
    /// compute the frontier of the group on each level, then masks with
    /// occupation and enemy color. Cross-level support/dependent edges
    /// are also handled via composed shifts.
    pub fn enemies_touching(
        &self,
        group: &[LevelBoard],
        color: &[LevelBoard],
    ) -> Vec<(usize, usize, usize)> {
        let n = self.n;
        let mut seeds = Vec::new();

        for l in 0..n {
            if group[l].count_ones() == 0 {
                continue;
            }

            // Same-level 4-way frontier.
            let frontier = self.shift_e(&group[l])
                | self.shift_w(&group[l])
                | self.shift_s(&group[l])
                | self.shift_n(&group[l]);
            let enemies = frontier & self.levels[l] & !color[l];
            for pos in enemies.iter_set() {
                seeds.push((pos % n, pos / n, l));
            }

            // Supporters at level-1: group[l] shifted E, S, SE.
            if l > 0 {
                let sup = group[l]
                    | self.shift_e(&group[l])
                    | self.shift_s(&group[l])
                    | self.shift_se(&group[l]);
                let enemies = sup & self.levels[l - 1] & !color[l - 1];
                for pos in enemies.iter_set() {
                    seeds.push((pos % n, pos / n, l - 1));
                }
            }

            // Dependents at level+1: group[l] shifted W, N, NW.
            if l + 1 < n {
                let dep = group[l]
                    | self.shift_w(&group[l])
                    | self.shift_n(&group[l])
                    | self.shift_nw(&group[l]);
                let enemies = dep & self.levels[l + 1] & !color[l + 1];
                for pos in enemies.iter_set() {
                    seeds.push((pos % n, pos / n, l + 1));
                }
            }
        }
        seeds
    }

    /// Own-colored pieces that would connect to a stone placed at
    /// `(col, row, level)`. Uses the same stencils as [`flood`] but for a
    /// single seed position.
    pub fn connecting_own(
        &self,
        col: usize,
        row: usize,
        level: usize,
        color: &[LevelBoard],
    ) -> Vec<(usize, usize, usize)> {
        let n = self.n;
        let mut result = Vec::with_capacity(12);
        let pos = self.raster_index(col, row);
        let mut singleton = level_board(n);
        singleton.set_index(pos);

        // Same-level 4-way neighbors.
        let nbrs = self.shift_e(&singleton)
            | self.shift_w(&singleton)
            | self.shift_s(&singleton)
            | self.shift_n(&singleton);
        for np in (nbrs & self.levels[level] & color[level]).iter_set() {
            result.push((np % n, np / n, level));
        }

        // Supporters at level-1.
        if level > 0 {
            let sup = singleton
                | self.shift_e(&singleton)
                | self.shift_s(&singleton)
                | self.shift_se(&singleton);
            for sp in (sup & self.levels[level - 1] & color[level - 1]).iter_set() {
                result.push((sp % n, sp / n, level - 1));
            }
        }

        // Dependents at level+1.
        if level + 1 < n {
            let dep = singleton
                | self.shift_w(&singleton)
                | self.shift_n(&singleton)
                | self.shift_nw(&singleton);
            for dp in (dep & self.levels[level + 1] & color[level + 1]).iter_set() {
                result.push((dp % n, dp / n, level + 1));
            }
        }
        result
    }

    /// Partition all pieces of `color` into connected groups.
    /// Each group is a `Vec<LevelBoard>` (one bitboard per level).
    pub fn groups(&self, color: &[LevelBoard]) -> Vec<Vec<LevelBoard>> {
        let n = self.n;
        let mut seen: Vec<LevelBoard> = (0..n).map(|_| level_board(n)).collect();
        let mut groups = Vec::new();

        for l in 0..n {
            for pos in (self.levels[l] & color[l]).iter_set() {
                if seen[l].get_index(pos) {
                    continue;
                }
                let col = pos % n;
                let row = pos / n;
                let group = self.flood(col, row, l, color);
                for l2 in 0..n {
                    seen[l2] |= group[l2];
                }
                groups.push(group);
            }
        }
        groups
    }
}

// ── wall builder helpers (used outside `impl Raster` too) ──────────────

fn wall_col(n: usize, col: usize) -> LevelBoard {
    let mut w = level_board(n);
    for r in 0..n {
        w.set_index(r * n + col);
    }
    w
}

fn wall_row(n: usize, row: usize) -> LevelBoard {
    let mut w = level_board(n);
    for c in 0..n {
        w.set_index(row * n + c);
    }
    w
}
