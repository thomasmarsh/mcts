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
    surface: LevelBoard,
    n: usize,
    /// Wall masks for shift operations — prevent bit wrap-around.
    east_wall: LevelBoard, // rightmost column (col = n-1)
    west_wall: LevelBoard,  // leftmost column  (col = 0)
    south_wall: LevelBoard, // bottom row       (row = n-1)
    north_wall: LevelBoard, // top row          (row = 0)
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
        let surface = Self::compute_surface(n, &levels);
        Raster {
            levels,
            surface,
            n,
            east_wall: wall_col(n, n - 1),
            west_wall: wall_col(n, 0),
            south_wall: wall_row(n, n - 1),
            north_wall: wall_row(n, 0),
        }
    }

    pub fn surface(&self) -> &LevelBoard {
        &self.surface
    }
    pub fn n(&self) -> usize {
        self.n
    }

    // ── wall builders ──────────────────────────────────────────────────

    fn col_wall(n: usize, col: usize) -> LevelBoard {
        let mut w = level_board(n);
        for r in 0..n {
            w.set_index(r * n + col);
        }
        w
    }

    fn row_wall(n: usize, row: usize) -> LevelBoard {
        let mut w = level_board(n);
        for c in 0..n {
            w.set_index(row * n + c);
        }
        w
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

    // ── public API ─────────────────────────────────────────────────────

    pub fn raster_index(&self, col: usize, row: usize) -> usize {
        row * self.n + col
    }

    /// Place a piece at `(col, row, level)`, regenerating the surface.
    pub fn place(&mut self, col: usize, row: usize, level: usize) -> usize {
        let pos = self.raster_index(col, row);
        debug_assert!(level < self.n);
        self.levels[level].set_index(pos);
        self.surface = Self::compute_surface(self.n, &self.levels);
        pos
    }

    /// Remove a piece, regenerating the surface.
    pub fn remove(&mut self, col: usize, row: usize, level: usize) {
        let pos = self.raster_index(col, row);
        self.levels[level].clear_index(pos);
        self.surface = Self::compute_surface(self.n, &self.levels);
    }

    /// Flood connected same-color pieces from `(col, row, level)`.
    /// `color` provides per-level masks: `color[l]` has bits set for
    /// pieces of this group's colour at level `l`. Returns per-level
    /// bitboards of all group members.
    pub fn flood(
        &self,
        col: usize,
        row: usize,
        level: usize,
        color: &[LevelBoard],
    ) -> Vec<LevelBoard> {
        let n = self.n;
        let mut result: Vec<LevelBoard> = (0..n).map(|_| level_board(n)).collect();
        let mut seen: Vec<LevelBoard> = (0..n).map(|_| level_board(n)).collect();
        let mut queue: Vec<(usize, usize, usize)> = Vec::new();

        let sp = self.raster_index(col, row);
        result[level].set_index(sp);
        seen[level].set_index(sp);
        queue.push((col, row, level));

        while let Some((c, r, l)) = queue.pop() {
            // Same-level 4-way.
            for (dc, dr) in &[(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                let nc = c as isize + dc;
                let nr = r as isize + dr;
                if nc < 0 || nr < 0 || nc >= n as isize || nr >= n as isize {
                    continue;
                }
                let np = (nr as usize) * n + (nc as usize);
                if seen[l].get_index(np) {
                    continue;
                }
                if self.levels[l].get_index(np) && color[l].get_index(np) {
                    seen[l].set_index(np);
                    result[l].set_index(np);
                    queue.push((nc as usize, nr as usize, l));
                }
            }

            // Supporters (level-1): peel-back.
            if l > 0 {
                for (dc, dr) in &[(0isize, 0isize), (1, 0), (0, 1), (1, 1)] {
                    let sc = c as isize + dc;
                    let sr = r as isize + dr;
                    if sc < 0 || sr < 0 || sc >= n as isize || sr >= n as isize {
                        continue;
                    }
                    let sp2 = (sr as usize) * n + (sc as usize);
                    if seen[l - 1].get_index(sp2) {
                        continue;
                    }
                    if self.levels[l - 1].get_index(sp2) && color[l - 1].get_index(sp2) {
                        seen[l - 1].set_index(sp2);
                        result[l - 1].set_index(sp2);
                        queue.push((sc as usize, sr as usize, l - 1));
                    }
                }
            }

            // Dependents (level+1).
            if l + 1 < n {
                for (dc, dr) in &[(-1isize, -1isize), (0, -1), (-1, 0), (0, 0)] {
                    let dc2 = c as isize + dc;
                    let dr2 = r as isize + dr;
                    if dc2 < 0 || dr2 < 0 || dc2 >= n as isize || dr2 >= n as isize {
                        continue;
                    }
                    let dp = (dr2 as usize) * n + (dc2 as usize);
                    if seen[l + 1].get_index(dp) {
                        continue;
                    }
                    if self.levels[l + 1].get_index(dp) && color[l + 1].get_index(dp) {
                        seen[l + 1].set_index(dp);
                        result[l + 1].set_index(dp);
                        queue.push((dc2 as usize, dr2 as usize, l + 1));
                    }
                }
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

    /// Find enemy cells touching `group`. Returns `(col, row, level)` seeds
    /// for flooding enemy groups. `color` is the own-color mask.
    pub fn enemies_touching(
        &self,
        group: &[LevelBoard],
        color: &[LevelBoard],
    ) -> Vec<(usize, usize, usize)> {
        let n = self.n;
        let mut seeds = Vec::new();

        for l in 0..n {
            for pos in group[l].iter_set() {
                let col = pos % n;
                let row = pos / n;

                // Same-level 4-way.
                for (dc, dr) in &[(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                    let nc = col as isize + dc;
                    let nr = row as isize + dr;
                    if nc < 0 || nr < 0 || nc >= n as isize || nr >= n as isize {
                        continue;
                    }
                    let np = (nr as usize) * n + (nc as usize);
                    if self.levels[l].get_index(np) && !color[l].get_index(np) {
                        seeds.push((nc as usize, nr as usize, l));
                    }
                }

                // Above (resting on us).
                if l + 1 < n {
                    for (dc, dr) in &[(-1isize, -1isize), (0, -1), (-1, 0), (0, 0)] {
                        let nc = col as isize + dc;
                        let nr = row as isize + dr;
                        if nc < 0 || nr < 0 || nc >= n as isize || nr >= n as isize {
                            continue;
                        }
                        let np = (nr as usize) * n + (nc as usize);
                        if self.levels[l + 1].get_index(np) && !color[l + 1].get_index(np) {
                            seeds.push((nc as usize, nr as usize, l + 1));
                        }
                    }
                }

                // Below (supporting us).
                if l > 0 {
                    for (dc, dr) in &[(0isize, 0isize), (1, 0), (0, 1), (1, 1)] {
                        let nc = col as isize + dc;
                        let nr = row as isize + dr;
                        if nc < 0 || nr < 0 || nc >= n as isize || nr >= n as isize {
                            continue;
                        }
                        let np = (nr as usize) * n + (nc as usize);
                        if self.levels[l - 1].get_index(np) && !color[l - 1].get_index(np) {
                            seeds.push((nc as usize, nr as usize, l - 1));
                        }
                    }
                }
            }
        }
        seeds
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
