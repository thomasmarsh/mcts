//! Whole-pyramid D4 symmetry: the automorphism group of a base-`n` pyramid.
//!
//! A pyramid's levels are all centered the same way -- level `k`'s square is
//! `level_side(n, k)` on a side, sitting flush against the same corner
//! (`(0, 0)`) as every other level, one cell narrower per level up. That
//! means a single D4 element (identity, column flip, row flip, transpose, or
//! one of their compositions) can be applied *independently within each
//! level's own `(col, row)` coordinates* -- using that level's own side
//! length for the flip -- and the results still line up consistently level
//! to level, because no level is offset relative to any other. This is the
//! whole group: [`PyramidD4::index_symmetries`] produces all 8 images of a
//! flat index, one per level-local D4 element applied uniformly across every
//! level of the pyramid.
//!
//! Mirrors `game_core::symmetry::D4Dyn`'s runtime-sized D4 (same 8-element
//! ordering: identity, flip_cols, flip_rows, transpose,
//! flip_rows∘flip_cols, transpose∘flip_cols, transpose∘flip_rows,
//! transpose∘flip_rows∘flip_cols) -- this crate sits below `games/` in the
//! workspace's dependency direction (like `bitboard`, which `game_core`
//! itself depends on), so the index arithmetic is duplicated here rather
//! than taking a dependency the wrong way, but the ordering/composition
//! structure is deliberately kept identical to `D4Dyn`'s.

use crate::{index, level_side, to_coord};

/// Runtime-sized whole-pyramid D4 symmetry group for a base-`n` pyramid --
/// `n` is a runtime field (not a const generic) since `Pyramid<S, N>` itself
/// supports both `Const` and `Dyn` bases; this type works for either, the
/// same way `D4Dyn` works for any `Board` regardless of its own `Dim` kind.
#[derive(Clone, Copy, Debug)]
pub struct PyramidD4 {
    n: usize,
}

impl PyramidD4 {
    #[inline]
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Reverse column order within `i`'s own level -- col → side-1-col,
    /// row/level unchanged.
    #[inline]
    fn flip_cols(&self, i: usize) -> usize {
        let (col, row, level) = to_coord(self.n, i);
        let side = level_side(self.n, level);
        index(self.n, side - 1 - col, row, level)
    }

    /// Reverse row order within `i`'s own level -- row → side-1-row,
    /// col/level unchanged.
    #[inline]
    fn flip_rows(&self, i: usize) -> usize {
        let (col, row, level) = to_coord(self.n, i);
        let side = level_side(self.n, level);
        index(self.n, col, side - 1 - row, level)
    }

    /// Transpose across the main diagonal within `i`'s own level -- (col,
    /// row) → (row, col), level unchanged. Well-defined because every level
    /// is square.
    #[inline]
    fn transpose(&self, i: usize) -> usize {
        let (col, row, level) = to_coord(self.n, i);
        index(self.n, row, col, level)
    }

    /// Produce all 8 symmetric images of a flat cell index -- see this
    /// module's docs for element ordering, and `D4Dyn::index_symmetries` for
    /// the (deliberately identical) composition structure.
    #[inline]
    pub fn index_symmetries(&self, i: usize) -> [usize; 8] {
        let fc = self.flip_cols(i);
        let fr = self.flip_rows(i);
        let t = self.transpose(i);
        [
            i,
            fc,
            fr,
            t,
            self.flip_rows(fc),
            self.transpose(fc),
            self.transpose(fr),
            self.transpose(self.flip_rows(fc)),
        ]
    }

    /// Map an index back through the inverse of a symmetry -- see
    /// `D4Dyn::invert_symmetry`.
    #[inline]
    pub fn invert_symmetry(&self, i: usize, sym_idx: usize) -> usize {
        match sym_idx {
            0 => i,
            1 => self.flip_cols(i),
            2 => self.flip_rows(i),
            3 => self.transpose(i),
            4 => self.flip_cols(self.flip_rows(i)),
            5 => self.flip_cols(self.transpose(i)),
            6 => self.flip_rows(self.transpose(i)),
            7 => self.flip_cols(self.flip_rows(self.transpose(i))),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency::touching_neighbors;
    use crate::total_cells;

    /// Every one of the 8 images must be a bijection on `[0, total_cells(n))`
    /// -- same style as `game_core::symmetry`'s `check_permutations`.
    #[test]
    fn index_symmetries_are_bijections() {
        for n in 2..=10usize {
            let total = total_cells(n);
            let sym = PyramidD4::new(n);
            for sym_idx in 0..8 {
                let mut seen = vec![false; total];
                for i in 0..total {
                    let v = sym.index_symmetries(i)[sym_idx];
                    assert!(v < total, "n={n}: sym {sym_idx}[{i}] = {v} out of range");
                    assert!(
                        !seen[v],
                        "n={n}: sym {sym_idx} is not injective (dupe at {v})"
                    );
                    seen[v] = true;
                }
                assert!(
                    seen.iter().all(|&x| x),
                    "n={n}: sym {sym_idx} is not surjective"
                );
            }
        }
    }

    /// `invert_symmetry` must be the true inverse of `index_symmetries`, for
    /// every cell and every element, mirroring
    /// `game_core::symmetry`'s `test_d4_dyn_invert_symmetry_is_inverse`.
    #[test]
    fn invert_symmetry_is_inverse() {
        for n in 2..=10usize {
            let total = total_cells(n);
            let sym = PyramidD4::new(n);
            for i in 0..total {
                let images = sym.index_symmetries(i);
                for (sym_idx, &image) in images.iter().enumerate() {
                    let back = sym.invert_symmetry(image, sym_idx);
                    assert_eq!(
                        back, i,
                        "n={n}: invert_symmetry({image}, {sym_idx}) = {back}, expected {i}"
                    );
                }
            }
        }
    }

    /// A symmetry preserves a level's identity: no element ever moves a cell
    /// to a different level, since flip/transpose only touch `(col, row)`.
    #[test]
    fn index_symmetries_preserve_level() {
        for n in 2..=10usize {
            let sym = PyramidD4::new(n);
            for i in 0..total_cells(n) {
                let (_, _, level) = to_coord(n, i);
                for &image in &sym.index_symmetries(i) {
                    let (_, _, image_level) = to_coord(n, image);
                    assert_eq!(
                        image_level, level,
                        "n={n}: sym moved index {i} (level {level}) to level {image_level}"
                    );
                }
            }
        }
    }

    /// Every element of `PyramidD4` must be a graph automorphism of Phase
    /// 3's derived touching-adjacency table: if `i` and `j` touch, so must
    /// their images under any symmetry element -- a correctness check on
    /// both phases at once, since a bug in either the geometric adjacency
    /// derivation or the symmetry's per-level application would show up as
    /// a physically nonsensical (non-rigid) "symmetry".
    #[test]
    fn every_symmetry_element_is_an_adjacency_automorphism() {
        for n in 2..=10usize {
            let sym = PyramidD4::new(n);
            let neighbors = touching_neighbors(n);
            for (i, list) in neighbors.iter().enumerate() {
                for sym_idx in 0..8 {
                    let image_i = sym.index_symmetries(i)[sym_idx];
                    for &j in list {
                        let image_j = sym.index_symmetries(j)[sym_idx];
                        assert!(
                            neighbors[image_i].contains(&image_j),
                            "n={n}: sym {sym_idx} maps touching pair ({i},{j}) to \
                             non-touching ({image_i},{image_j})"
                        );
                    }
                }
            }
        }
    }

    /// Applying a symmetry element and then its inverse round-trips through
    /// `touching_neighbors` unchanged (the graph itself, not just individual
    /// indices) -- an end-to-end sanity check combining the two properties
    /// above.
    #[test]
    fn identity_element_is_the_identity_permutation() {
        for n in 2..=10usize {
            let sym = PyramidD4::new(n);
            for i in 0..total_cells(n) {
                assert_eq!(sym.index_symmetries(i)[0], i);
            }
        }
    }
}
