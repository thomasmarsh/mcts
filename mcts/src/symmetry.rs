//! 3x3 board symmetry utilities, extracted from the Tic-Tac-Toe game types
//! because traffic-lights reuses them. These operate on a packed-`u32` board
//! encoding where each cell occupies 2 bits (0b00 = empty, 0b01 = X/player0,
//! 0b10 = O/player1, 0b11 = reserved) and eight symmetries of the 3×3 grid
//! (identity, horizontal flip, vertical flip, diagonal flip, their three
//! two-way compositions, plus the full rotation+flip).

/// Number of symmetries (4 rotations × 2 reflections).
pub const NUM_SYMMETRIES: usize = 8;

/// Maps a cell index through each of the eight 3×3 symmetries.
pub mod sym {
    use super::NUM_SYMMETRIES;

    const H: [usize; 9] = [6, 7, 8, 3, 4, 5, 0, 1, 2];
    const V: [usize; 9] = [2, 1, 0, 5, 4, 3, 8, 7, 6];
    const D: [usize; 9] = [8, 5, 2, 7, 4, 1, 6, 3, 0];

    #[inline]
    pub fn index_symmetries(i: usize, symmetries: &mut [usize; NUM_SYMMETRIES]) {
        symmetries[0] = i;
        symmetries[1] = H[i];
        symmetries[2] = V[i];
        symmetries[3] = D[i];
        symmetries[4] = V[H[i]];
        symmetries[5] = D[H[i]];
        symmetries[6] = D[V[i]];
        symmetries[7] = D[V[H[i]]];
    }

    #[inline]
    pub fn invert_symmetry(i: usize, symmetry_index: usize) -> usize {
        match symmetry_index {
            0 => i,
            1 => H[i],
            2 => V[i],
            3 => D[i],
            4 => H[V[i]],
            5 => H[D[i]],
            6 => V[D[i]],
            7 => H[V[D[i]]],
            _ => unreachable!("Invalid symmetry index"),
        }
    }

    #[inline]
    pub fn board_symmetries(board: u32, symmetries: &mut [u32; NUM_SYMMETRIES]) {
        debug_assert!(symmetries.iter().all(|x| *x == 0));

        symmetries[0] = board;
        (0..9).for_each(|i| {
            let p = (board >> (i << 1)) & 0b11;
            symmetries[1] |= p << (H[i] * 2);
            symmetries[2] |= p << (V[i] * 2);
            symmetries[3] |= p << (D[i] * 2);
            symmetries[4] |= p << (V[H[i]] * 2);
            symmetries[5] |= p << (D[H[i]] * 2);
            symmetries[6] |= p << (D[V[i]] * 2);
            symmetries[7] |= p << (D[V[H[i]]] * 2);
        });
    }

    #[inline]
    pub fn canonical_symmetry(board: u32) -> usize {
        let mut sym = [0; 8];
        board_symmetries(board, &mut sym);
        sym.iter().enumerate().min_by_key(|(_, &v)| v).unwrap().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_idempotent_sym(original_index in 0..9usize, symmetry_used in 0..8usize) {
            let mut xs = [0; NUM_SYMMETRIES];
            sym::index_symmetries(original_index, &mut xs);
            let transformed_index = xs[symmetry_used];
            let inverted_index = sym::invert_symmetry(transformed_index, symmetry_used);
            prop_assert_eq!(inverted_index, original_index);
        }
    }
}