//! The dihedral group of the square (8 rotations and reflections) acting on a `size x size`
//! grid stored row-major.

pub const SYMMETRIES: usize = 8;

/// `map[cell]` is where `cell` lands under symmetry `sym`: the flip (if `sym >= 4`) is applied
/// first, then `sym % 4` quarter turns. Symmetry 0 is the identity.
pub fn cell_map(size: usize, sym: usize) -> Vec<usize> {
    assert!(sym < SYMMETRIES);
    (0..size * size)
        .map(|cell| {
            let (mut r, mut c) = (cell / size, cell % size);
            if sym >= 4 {
                c = size - 1 - c;
            }
            for _ in 0..sym % 4 {
                (r, c) = (c, size - 1 - r);
            }
            r * size + c
        })
        .collect()
}

/// Apply symmetry `sym` to one position's planes stored `(row, col, plane)`, i.e. the layout
/// [`crate::Net::forward`] takes. The plane at `cell` moves to `cell_map(size, sym)[cell]`.
pub fn transform_planes(planes: &[f32], size: usize, n_planes: usize, sym: usize) -> Vec<f32> {
    let map = cell_map(size, sym);
    let mut out = vec![0.0; planes.len()];
    for (cell, &dest) in map.iter().enumerate() {
        out[dest * n_planes..(dest + 1) * n_planes]
            .copy_from_slice(&planes[cell * n_planes..(cell + 1) * n_planes]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_symmetry_is_a_distinct_permutation() {
        let maps: Vec<Vec<usize>> = (0..SYMMETRIES).map(|s| cell_map(5, s)).collect();
        for m in &maps {
            let mut sorted = m.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..25).collect::<Vec<_>>());
        }
        for i in 0..SYMMETRIES {
            for j in i + 1..SYMMETRIES {
                assert_ne!(maps[i], maps[j]);
            }
        }
        assert_eq!(maps[0], (0..25).collect::<Vec<_>>());
    }

    #[test]
    fn the_eight_maps_are_closed_under_composition() {
        let maps: Vec<Vec<usize>> = (0..SYMMETRIES).map(|s| cell_map(7, s)).collect();
        for a in &maps {
            for b in &maps {
                let composed: Vec<usize> = (0..49).map(|c| b[a[c]]).collect();
                assert!(maps.contains(&composed));
            }
        }
    }

    #[test]
    fn a_quarter_turn_moves_the_corner_as_expected() {
        let m = cell_map(3, 1);
        assert_eq!(m[0], 2);
        assert_eq!(m[2], 8);
        assert_eq!(m[4], 4);
    }

    #[test]
    fn planes_follow_the_cell_map() {
        let planes: Vec<f32> = (0..9 * 2).map(|i| i as f32).collect();
        let out = transform_planes(&planes, 3, 2, 1);
        let m = cell_map(3, 1);
        for cell in 0..9 {
            assert_eq!(out[m[cell] * 2], planes[cell * 2]);
            assert_eq!(out[m[cell] * 2 + 1], planes[cell * 2 + 1]);
        }
    }
}
