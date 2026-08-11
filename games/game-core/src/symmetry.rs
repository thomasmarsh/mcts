//! Dihedral symmetries (D4) for square boards of any size.
//!
//! The D4 group has 8 elements: identity, horizontal flip (H), vertical flip
//! (V), transpose across the main diagonal (D), and their compositions.
//!
//! Rather than pre-computing permutation tables (which would require
//! `[usize; S * S]` in a const-generic context, unstable on stable Rust),
//! all transformations are computed inline from the cell index using simple
//! arithmetic.  For any concrete `S` the compiler will constant-fold these
//! into the same machine code a table lookup would produce.

/// Board-side-length parameterised D4 symmetry group.
///
/// # Example
///
/// ```
/// use game_core::symmetry::D4Symmetry;
///
/// // 8×8 board (Othello)
/// type Sym8 = D4Symmetry<8>;
/// let syms = Sym8::index_symmetries(27);  // all 8 images of cell 27
/// assert_eq!(syms[0], 27);                // identity
/// assert_eq!(syms[1], 28);                // horizontal flip
/// ```
pub struct D4Symmetry<const S: usize>;

impl<const S: usize> D4Symmetry<S> {
    /// Horizontal mirror: reflect across the vertical axis — col → S-1-col.
    #[inline]
    fn h(i: usize) -> usize {
        let row = i / S;
        let col = i % S;
        row * S + (S - 1 - col)
    }

    /// Vertical mirror: reflect across the horizontal axis — row → S-1-row.
    #[inline]
    fn v(i: usize) -> usize {
        let row = i / S;
        let col = i % S;
        (S - 1 - row) * S + col
    }

    /// Transpose across the main diagonal — (row, col) → (col, row).
    #[inline]
    fn d(i: usize) -> usize {
        let row = i / S;
        let col = i % S;
        col * S + row
    }

    /// Produce all 8 symmetric images of a cell index.
    ///
    /// Order: identity, H, V, D, V∘H, D∘H, D∘V, D∘V∘H.
    #[inline]
    pub fn index_symmetries(i: usize) -> [usize; 8] {
        let h = Self::h(i);
        let v = Self::v(i);
        let d = Self::d(i);
        [
            i,
            h,
            v,
            d,
            Self::v(h),
            Self::d(h),
            Self::d(v),
            Self::d(Self::v(h)),
        ]
    }

    /// Map an index back through the inverse of a symmetry.
    ///
    /// For an involution (H, V, D) the inverse is the same permutation.
    /// For a composition the inverse is the reverse composition.
    #[inline]
    pub fn invert_symmetry(i: usize, sym_idx: usize) -> usize {
        match sym_idx {
            0 => i,
            1 => Self::h(i),
            2 => Self::v(i),
            3 => Self::d(i),
            4 => Self::h(Self::v(i)),    // (V∘H)⁻¹ = H∘V
            5 => Self::h(Self::d(i)),    // (D∘H)⁻¹ = H∘D
            6 => Self::v(Self::d(i)),    // (D∘V)⁻¹ = V∘D
            7 => Self::h(Self::v(Self::d(i))), // (D∘V∘H)⁻¹ = H∘V∘D
            _ => unreachable!(),
        }
    }

    /// Apply a symmetry permutation to a raw u64 bitboard.
    ///
    /// Iterates each set bit, maps it through the symmetry, and sets the
    /// result bit in the output.  Works for any board size ≤ 64 cells.
    #[inline]
    pub fn apply_to_bits(board: u64, sym_idx: usize) -> u64 {
        let mut result = 0u64;
        let mut bits = board;
        while bits != 0 {
            let lsb = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let dst = Self::index_symmetries(lsb)[sym_idx];
            result |= 1u64 << dst;
        }
        result
    }

    /// Compute all 8 symmetric images of a packed-`u32` board encoding (2 bits per cell).
    ///
    /// Each cell `i` occupies bits `(i * 2) .. (i * 2 + 1)`, with value `0b00` = empty,
    /// `0b01` = player 0, `0b10` = player 1.  The output `symmetries` array is filled
    /// with the 8 symmetric versions of the board.
    #[inline]
    pub fn packed_board_symmetries(board: u32, symmetries: &mut [u32; 8]) {
        debug_assert!(symmetries.iter().all(|x| *x == 0));
        for i in 0..(S * S) {
            let p = (board >> (i << 1)) & 0b11;
            for (s, &dst) in Self::index_symmetries(i).iter().enumerate() {
                symmetries[s] |= p << (dst << 1);
            }
        }
    }

    /// Index of the symmetry whose image of a packed-`u32` board is minimal.
    ///
    /// This is the canonical symmetry for the board: all 8 symmetric board
    /// values are compared and the smallest one's symmetry index is returned.
    #[inline]
    pub fn packed_canonical_symmetry(board: u32) -> usize {
        let mut sym = [0; 8];
        Self::packed_board_symmetries(board, &mut sym);
        sym.iter().enumerate().min_by_key(|(_, &v)| v).unwrap().0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that H, V, D are bijections on [0, S*S).
    fn check_permutations<const S: usize>() {
        let n = S * S;
        for (name, perm) in [
            ("H", D4Symmetry::<S>::h as fn(usize) -> usize),
            ("V", D4Symmetry::<S>::v as fn(usize) -> usize),
            ("D", D4Symmetry::<S>::d as fn(usize) -> usize),
        ] {
            let mut seen = vec![false; n];
            for i in 0..n {
                let v = perm(i);
                assert!(v < n, "{name}[{i}] = {v} out of range");
                assert!(!seen[v], "{name} is not injective (dupe at {v})");
                seen[v] = true;
            }
            assert!(seen.iter().all(|&x| x), "{name} is not surjective");
        }
    }

    #[test]
    fn test_permutations_3x3() {
        check_permutations::<3>();
    }

    #[test]
    fn test_permutations_8x8() {
        check_permutations::<8>();
    }

    /// Verify that index_symmetries produces valid images (within bounds).
    #[test]
    fn test_index_symmetries_in_range() {
        for i in 0..9 {
            let s = D4Symmetry::<3>::index_symmetries(i);
            for &v in &s {
                assert!(v < 9, "index_symmetries({i}) has out-of-range {v}");
            }
        }
        for i in 0..64 {
            let s = D4Symmetry::<8>::index_symmetries(i);
            for &v in &s {
                assert!(v < 64, "index_symmetries({i}) has out-of-range {v}");
            }
        }
    }

    /// Verify that invert_symmetry is the true inverse of index_symmetries.
    #[test]
    fn test_invert_symmetry_is_inverse() {
        for i in 0..9 {
            let s = D4Symmetry::<3>::index_symmetries(i);
            for (sym_idx, &s_i) in s.iter().enumerate() {
                let back = D4Symmetry::<3>::invert_symmetry(s_i, sym_idx);
                assert_eq!(
                    back, i,
                    "invert_symmetry({s_i}, {sym_idx}) = {back}, expected {i}"
                );
            }
        }
        for i in 0..64 {
            let s = D4Symmetry::<8>::index_symmetries(i);
            for (sym_idx, &s_i) in s.iter().enumerate() {
                let back = D4Symmetry::<8>::invert_symmetry(s_i, sym_idx);
                assert_eq!(
                    back, i,
                    "invert_symmetry({s_i}, {sym_idx}) = {back}, expected {i}"
                );
            }
        }
    }

    /// Applying then inverting a symmetry should yield the original board.
    #[test]
    fn test_apply_to_bits_inverse() {
        // Pre-computed inverse of each symmetry index.
        const INV: [usize; 8] = [0, 1, 2, 3, 4, 6, 5, 7];

        // 3×3 board: test a few representative bit patterns.
        for board in [0b_100_000_001u64, 0b_010_010_010, 0b_001_000_100] {
            for (sym_idx, &inv) in INV.iter().enumerate() {
                let transformed = D4Symmetry::<3>::apply_to_bits(board, sym_idx);
                let back = D4Symmetry::<3>::apply_to_bits(transformed, inv);
                assert_eq!(
                    back, board,
                    "sym {sym_idx} then inv {inv} on {board:#x} gave {back:#x}"
                );
            }
        }

        // 8×8 board: Othello's initial position.
        let board = (1 << 28) | (1 << 35);
        for (sym_idx, &inv) in INV.iter().enumerate() {
            let transformed = D4Symmetry::<8>::apply_to_bits(board, sym_idx);
            let back = D4Symmetry::<8>::apply_to_bits(transformed, inv);
            assert_eq!(
                back, board,
                "sym {sym_idx} then inv {inv} on {board:#x} gave {back:#x}"
            );
        }
    }
}