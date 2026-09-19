//! N-tuple geometry: which cells each tuple reads, how orientations share
//! weights, and how a position's per-cell codes become weight indices.
//!
//! A tuple is an ordered list of cells. A position is described by one small
//! integer code per cell (the game adapter's job, see [`CellFeatures`]); a
//! tuple maps the codes of its cells to one base-`states_per_cell` feature
//! index, and each tuple owns a table of `states_per_cell^len` weights. All
//! orientations of the board (the game's symmetry group) share one table, so a
//! position selects one weight per (tuple, orientation) image.
//!
//! The geometry serialises to the same `model.toml` the Othello evaluator
//! reads (`[[tuple]] name/squares`, feature digit 0 empty, 1 side to move, 2
//! opponent), plus a top-level `states_per_cell` (absent means 3).

use std::fmt::Write as _;

use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::Rng;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The per-game adapter: everything the trainer needs to know about a board
/// game's cells. Everything else (actions, terminal tests, winners) comes from
/// [`Game`].
pub trait CellFeatures: Send + Sync {
    type G: Game;

    /// Number of board cells the tuples may read.
    fn num_cells(&self) -> usize;

    /// Cells adjacent to `cell`, the graph random-walk tuples grow along.
    fn neighbors(&self, cell: usize) -> Vec<usize>;

    /// One cell permutation per board orientation, `perms[k][c]` being the
    /// image of cell `c` under orientation `k`. Element 0 must be the
    /// identity. Tuples share weights across all of them.
    fn orientations(&self) -> Vec<Vec<u8>>;

    /// Largest `states_per_cell` the adapter can encode.
    fn max_states_per_cell(&self) -> usize;

    /// Write one code per cell into `out` (`out.len() == num_cells()`), from
    /// the side to move's point of view: 0 empty, 1 own piece, 2 opponent
    /// piece; with `states_per_cell == 4`, an empty cell the side to move can
    /// play on is 3 instead of 0. Every code is `< states_per_cell`.
    fn cell_codes(&self, state: &<Self::G as Game>::S, states_per_cell: usize, out: &mut [u8]);
}

/// One tuple: an ordered list of cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tuple {
    pub name: String,
    pub squares: Vec<u8>,
}

/// One (tuple, orientation) image: where the tuple's table starts in the flat
/// weight vector and where its permuted cells start in `Geometry::cells`.
#[derive(Clone, Copy, Debug)]
struct Image {
    offset: u32,
    start: u32,
    len: u32,
}

#[derive(Deserialize)]
struct ModelToml {
    #[serde(default = "default_states")]
    states_per_cell: usize,
    #[serde(default)]
    tuple: Vec<TupleToml>,
}

#[derive(Deserialize)]
struct TupleToml {
    name: String,
    squares: Vec<u8>,
}

fn default_states() -> usize {
    3
}

/// Parsed geometry: the tuples, the orientations, and the flat weight layout.
#[derive(Clone, Debug)]
pub struct Geometry {
    states_per_cell: usize,
    tuples: Vec<Tuple>,
    images: Vec<Image>,
    cells: Vec<u8>,
    n_weights: usize,
    toml_text: String,
    sha256_hex: String,
}

impl Geometry {
    /// Build a geometry from tuples, serialising it to `model.toml` text first
    /// so the in-memory geometry and the file on disk agree byte for byte.
    pub fn from_tuples(
        states_per_cell: usize,
        tuples: Vec<Tuple>,
        orientations: &[Vec<u8>],
    ) -> Geometry {
        let text = render_toml(states_per_cell, &tuples);
        Geometry::from_toml(text.as_bytes(), orientations)
    }

    /// Parse `model.toml` bytes. The SHA-256 covers the raw bytes, matching
    /// what `weights.meta.json` records.
    pub fn from_toml(bytes: &[u8], orientations: &[Vec<u8>]) -> Geometry {
        let text = std::str::from_utf8(bytes).expect("model.toml must be UTF-8");
        let parsed: ModelToml = toml::from_str(text).expect("model.toml must parse");
        assert!(!parsed.tuple.is_empty(), "model.toml defines no [[tuple]]");
        assert!(
            (2..=255).contains(&parsed.states_per_cell),
            "states_per_cell must be 2..=255"
        );
        assert!(!orientations.is_empty(), "need at least the identity orientation");
        let num_cells = orientations[0].len();
        assert!(
            orientations[0].iter().enumerate().all(|(i, &c)| c as usize == i),
            "orientation 0 must be the identity"
        );

        let m = parsed.states_per_cell;
        let mut tuples = Vec::with_capacity(parsed.tuple.len());
        let mut images = Vec::with_capacity(parsed.tuple.len() * orientations.len());
        let mut cells = Vec::new();
        let mut offset = 0usize;
        for t in parsed.tuple {
            assert!(
                !t.squares.is_empty() && t.squares.len() <= 12,
                "tuple {:?}: squares must be 1..=12 entries",
                t.name
            );
            for &s in &t.squares {
                assert!(
                    (s as usize) < num_cells,
                    "tuple {:?}: cell {s} out of range",
                    t.name
                );
            }
            let table = m.pow(t.squares.len() as u32);
            for perm in orientations {
                let start = cells.len() as u32;
                cells.extend(t.squares.iter().map(|&s| perm[s as usize]));
                images.push(Image {
                    offset: offset as u32,
                    start,
                    len: t.squares.len() as u32,
                });
            }
            offset += table;
            tuples.push(Tuple {
                name: t.name,
                squares: t.squares,
            });
        }
        assert!(offset <= u32::MAX as usize, "weight table too large for u32 indices");

        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Geometry {
            states_per_cell: m,
            tuples,
            images,
            cells,
            n_weights: offset,
            toml_text: text.to_string(),
            sha256_hex: hex_lower(&hasher.finalize()),
        }
    }

    pub fn states_per_cell(&self) -> usize {
        self.states_per_cell
    }

    pub fn tuples(&self) -> &[Tuple] {
        &self.tuples
    }

    pub fn n_tuples(&self) -> usize {
        self.tuples.len()
    }

    pub fn n_weights(&self) -> usize {
        self.n_weights
    }

    /// Weights a single position selects: one per (tuple, orientation) image.
    pub fn n_images(&self) -> usize {
        self.images.len()
    }

    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }

    /// The exact `model.toml` text this geometry was parsed from.
    pub fn toml_text(&self) -> &str {
        &self.toml_text
    }

    /// Global weight index selected by image `img` for a position's `codes`.
    #[inline]
    pub fn image_index(&self, img: usize, codes: &[u8]) -> usize {
        let im = self.images[img];
        let cells = &self.cells[im.start as usize..(im.start + im.len) as usize];
        let m = self.states_per_cell;
        let mut feat = 0usize;
        let mut place = 1usize;
        for &c in cells {
            feat += codes[c as usize] as usize * place;
            place *= m;
        }
        im.offset as usize + feat
    }

    /// Every global weight index a position selects, tuple-major then
    /// orientation. `out` is cleared first.
    pub fn active_indices(&self, codes: &[u8], out: &mut Vec<u32>) {
        out.clear();
        out.extend((0..self.images.len()).map(|i| self.image_index(i, codes) as u32));
    }
}

/// Generate `n_tuples` random-walk tuples of `len` cells: each tuple starts at
/// a uniformly random cell and repeatedly adds a uniformly random cell that is
/// adjacent to at least one cell already in the tuple and not yet in it.
pub fn random_walk_tuples(
    neighbors: &[Vec<usize>],
    n_tuples: usize,
    len: usize,
    rng: &mut SmallRng,
) -> Vec<Tuple> {
    let n = neighbors.len();
    assert!(len >= 1 && len <= n, "tuple length {len} must be 1..={n}");
    let mut out = Vec::with_capacity(n_tuples);
    for i in 0..n_tuples {
        let mut chosen: Vec<u8> = vec![rng.gen_range(0..n) as u8];
        while chosen.len() < len {
            let mut frontier: Vec<usize> = chosen
                .iter()
                .flat_map(|&c| neighbors[c as usize].iter().copied())
                .filter(|c| !chosen.contains(&(*c as u8)))
                .collect();
            frontier.sort_unstable();
            frontier.dedup();
            assert!(
                !frontier.is_empty(),
                "the cell graph has no room to grow a {len}-tuple"
            );
            chosen.push(frontier[rng.gen_range(0..frontier.len())] as u8);
        }
        out.push(Tuple {
            name: format!("rw{i}"),
            squares: chosen,
        });
    }
    out
}

fn render_toml(states_per_cell: usize, tuples: &[Tuple]) -> String {
    let mut s = String::new();
    writeln!(
        s,
        "# N-tuple geometry written by the ntuple trainer. Feature index per tuple is\n\
         # base-`states_per_cell`, least significant digit first; cell code 0 empty,\n\
         # 1 side-to-move piece, 2 opponent piece, 3 (4-state only) empty and playable."
    )
    .unwrap();
    writeln!(s, "states_per_cell = {states_per_cell}").unwrap();
    for t in tuples {
        writeln!(s, "\n[[tuple]]\nname = \"{}\"\nsquares = {:?}", t.name, t.squares).unwrap();
    }
    s
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").unwrap();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// 3x3 grid, identity plus the left-right mirror.
    fn grid3() -> (Vec<Vec<usize>>, Vec<Vec<u8>>) {
        let mut nb = vec![Vec::new(); 9];
        for r in 0..3i32 {
            for c in 0..3i32 {
                for (dr, dc) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                    let (rr, cc) = (r + dr, c + dc);
                    if (0..3).contains(&rr) && (0..3).contains(&cc) {
                        nb[(r * 3 + c) as usize].push((rr * 3 + cc) as usize);
                    }
                }
            }
        }
        let ident: Vec<u8> = (0..9).collect();
        let mirror: Vec<u8> = (0..9u8).map(|i| (i / 3) * 3 + (2 - i % 3)).collect();
        (nb, vec![ident, mirror])
    }

    #[test]
    fn random_walk_tuples_are_connected_distinct_and_seeded() {
        let (nb, _) = grid3();
        let a = random_walk_tuples(&nb, 20, 5, &mut SmallRng::seed_from_u64(7));
        let b = random_walk_tuples(&nb, 20, 5, &mut SmallRng::seed_from_u64(7));
        assert_eq!(a, b, "same seed, same tuples");
        for t in &a {
            let mut sorted = t.squares.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 5, "cells are distinct");
            for (k, &c) in t.squares.iter().enumerate().skip(1) {
                let touches = t.squares[..k]
                    .iter()
                    .any(|&p| nb[p as usize].contains(&(c as usize)));
                assert!(touches, "cell {c} of {:?} touches no earlier cell", t.squares);
            }
        }
    }

    #[test]
    fn toml_round_trips_and_indexes_by_hand() {
        let (_, perms) = grid3();
        let tuples = vec![
            Tuple {
                name: "a".into(),
                squares: vec![0, 1],
            },
            Tuple {
                name: "b".into(),
                squares: vec![4],
            },
        ];
        let g = Geometry::from_tuples(4, tuples.clone(), &perms);
        assert_eq!(g.n_weights(), 16 + 4);
        assert_eq!(g.n_images(), 4);
        let g2 = Geometry::from_toml(g.toml_text().as_bytes(), &perms);
        assert_eq!(g2.sha256_hex(), g.sha256_hex());
        assert_eq!(g2.tuples(), &tuples[..]);

        // codes: cell 0 -> 2, cell 1 -> 3, cell 2 -> 1, cell 4 -> 1, rest 0.
        let mut codes = [0u8; 9];
        codes[0] = 2;
        codes[1] = 3;
        codes[2] = 1;
        codes[4] = 1;
        let mut idx = Vec::new();
        g.active_indices(&codes, &mut idx);
        // Tuple a identity: 2 + 3*4 = 14. Mirrored cells (2, 1): 1 + 3*4 = 13.
        // Tuple b (cell 4 is its own mirror): offset 16 + 1, both orientations.
        assert_eq!(idx, vec![14, 13, 17, 17]);
    }

    #[test]
    fn a_missing_states_per_cell_means_three() {
        let (_, perms) = grid3();
        let g = Geometry::from_toml(b"[[tuple]]\nname = \"x\"\nsquares = [0, 1]\n", &perms);
        assert_eq!(g.states_per_cell(), 3);
        assert_eq!(g.n_weights(), 9);
    }
}
