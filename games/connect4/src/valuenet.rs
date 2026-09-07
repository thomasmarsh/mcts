//! N-tuple value heads for standard 6x7 Connect Four.
//!
//! The structured layout retains the original 69 winning windows and adds
//! 2x2 squares, all length-three/five/six lines, and seven column
//! height/base-parity tables.  Cells are row-major, row zero at the bottom.
//! Geometry and flat layout match `az_train.ntuple_c4` exactly.

use std::path::Path;

use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};

use crate::{Player, Standard, State};

const ROWS: usize = 6;
const COLS: usize = 7;
const MAX_TUPLES: usize = 30 + 98 + 69 + 44 + 23;

pub const N_WINDOWS: usize = 69;
/// The retained 69-window layout used by the existing policy sidecar.
pub const NT_WEIGHTS: usize = 1 + N_WINDOWS * 81;
/// Bias + 30 squares + 98/69/44/23 line tables + seven 13-state columns.
pub const STRUCTURED_WEIGHTS: usize = 38_216;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    Basic,
    Structured,
}

const fn pow3(n: usize) -> usize {
    let mut v = 1;
    let mut i = 0;
    while i < n {
        v *= 3;
        i += 1;
    }
    v
}

const fn add_lines(
    tuples: &mut [[usize; 6]; MAX_TUPLES],
    lens: &mut [usize; MAX_TUPLES],
    mut n: usize,
    len: usize,
) -> usize {
    let mut r = 0;
    while r < ROWS {
        let mut c = 0;
        while c + len <= COLS {
            let mut k = 0;
            while k < len {
                tuples[n][k] = r * COLS + c + k;
                k += 1;
            }
            lens[n] = len;
            n += 1;
            c += 1;
        }
        r += 1;
    }
    let mut c = 0;
    while c < COLS {
        let mut r = 0;
        while r + len <= ROWS {
            let mut k = 0;
            while k < len {
                tuples[n][k] = (r + k) * COLS + c;
                k += 1;
            }
            lens[n] = len;
            n += 1;
            r += 1;
        }
        c += 1;
    }
    let mut r = 0;
    while r + len <= ROWS {
        let mut c = 0;
        while c + len <= COLS {
            let mut k = 0;
            while k < len {
                tuples[n][k] = (r + k) * COLS + c + k;
                k += 1;
            }
            lens[n] = len;
            n += 1;
            c += 1;
        }
        r += 1;
    }
    let mut r = 0;
    while r + len <= ROWS {
        let mut c = len - 1;
        while c < COLS {
            let mut k = 0;
            while k < len {
                tuples[n][k] = (r + k) * COLS + c - k;
                k += 1;
            }
            lens[n] = len;
            n += 1;
            c += 1;
        }
        r += 1;
    }
    n
}

const fn build_geometry() -> ([[usize; 6]; MAX_TUPLES], [usize; MAX_TUPLES]) {
    let mut tuples = [[0; 6]; MAX_TUPLES];
    let mut lens = [0; MAX_TUPLES];
    let mut n = 0;
    let mut r = 0;
    while r + 1 < ROWS {
        let mut c = 0;
        while c + 1 < COLS {
            tuples[n][0] = r * COLS + c;
            tuples[n][1] = r * COLS + c + 1;
            tuples[n][2] = (r + 1) * COLS + c;
            tuples[n][3] = (r + 1) * COLS + c + 1;
            lens[n] = 4;
            n += 1;
            c += 1;
        }
        r += 1;
    }
    n = add_lines(&mut tuples, &mut lens, n, 3);
    n = add_lines(&mut tuples, &mut lens, n, 4);
    n = add_lines(&mut tuples, &mut lens, n, 5);
    n = add_lines(&mut tuples, &mut lens, n, 6);
    assert!(n == MAX_TUPLES);
    (tuples, lens)
}

const GEOMETRY: ([[usize; 6]; MAX_TUPLES], [usize; MAX_TUPLES]) = build_geometry();
pub(crate) const WINDOWS: [[usize; 4]; N_WINDOWS] = build_windows();
const fn build_windows() -> [[usize; 4]; N_WINDOWS] {
    let mut out = [[0; 4]; N_WINDOWS];
    let mut n = 0;
    let mut i = 30 + 98;
    while i < 30 + 98 + N_WINDOWS {
        let mut k = 0;
        while k < 4 {
            out[n][k] = GEOMETRY.0[i][k];
            k += 1;
        }
        n += 1;
        i += 1;
    }
    out
}

#[derive(Clone, Debug)]
pub struct NTupleValueNet {
    weights: Vec<f32>,
    head: Head,
}

impl Default for NTupleValueNet {
    fn default() -> Self {
        Self {
            weights: vec![0.0; STRUCTURED_WEIGHTS],
            head: Head::Structured,
        }
    }
}

impl NTupleValueNet {
    pub fn from_weights(weights: Vec<f32>) -> Self {
        let head = match weights.len() {
            NT_WEIGHTS => Head::Basic,
            STRUCTURED_WEIGHTS => Head::Structured,
            n => panic!("n-tuple head needs {NT_WEIGHTS} or {STRUCTURED_WEIGHTS} weights, got {n}"),
        };
        Self { weights, head }
    }
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if ![NT_WEIGHTS * 4, STRUCTURED_WEIGHTS * 4].contains(&bytes.len()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "expected {} or {} bytes, got {}",
                    NT_WEIGHTS * 4,
                    STRUCTURED_WEIGHTS * 4,
                    bytes.len()
                ),
            ));
        }
        Ok(Self::from_weights(
            bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
        ))
    }
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    #[inline]
    pub(crate) fn trit(state: &State<ROWS, COLS>, cell: usize) -> usize {
        let mover = state.turn();
        if state.black().get_index(cell) {
            if mover == Player::Black {
                1
            } else {
                2
            }
        } else if state.white().get_index(cell) {
            if mover == Player::White {
                1
            } else {
                2
            }
        } else {
            0
        }
    }
    /// Stable legacy indices used by the separately versioned policy sidecar.
    pub(crate) fn active_indices(
        state: &State<ROWS, COLS>,
        mirrored: bool,
    ) -> [usize; 1 + N_WINDOWS] {
        let mut out = [0; 1 + N_WINDOWS];
        let mut offset = 1;
        for (i, cells) in WINDOWS.iter().enumerate() {
            let mut feature = 0;
            let mut place = 1;
            for &cell in cells {
                let cell = if mirrored {
                    (cell / COLS) * COLS + COLS - 1 - cell % COLS
                } else {
                    cell
                };
                feature += Self::trit(state, cell) * place;
                place *= 3;
            }
            out[i + 1] = offset + feature;
            offset += 81;
        }
        out
    }
    fn raw_score(&self, state: &State<ROWS, COLS>) -> f32 {
        if self.head == Head::Basic {
            return Self::active_indices(state, false)
                .into_iter()
                .map(|i| self.weights[i])
                .sum();
        }
        let mut score = self.weights[0];
        let mut offset = 1;
        for i in 0..MAX_TUPLES {
            let len = GEOMETRY.1[i];
            let mut feature = 0;
            let mut place = 1;
            for k in 0..len {
                feature += Self::trit(state, GEOMETRY.0[i][k]) * place;
                place *= 3;
            }
            score += self.weights[offset + feature];
            offset += pow3(len);
        }
        for col in 0..COLS {
            let mut height = 0;
            while height < ROWS && Self::trit(state, height * COLS + col) != 0 {
                height += 1;
            }
            let feature = if height == 0 {
                0
            } else {
                1 + 2 * (height - 1) + Self::trit(state, col) - 1
            };
            score += self.weights[offset + feature];
            offset += 13;
        }
        debug_assert_eq!(offset, STRUCTURED_WEIGHTS);
        score
    }
    pub fn value(&self, state: &State<ROWS, COLS>) -> f32 {
        self.raw_score(state).tanh()
    }
}
impl Evaluator<Standard> for NTupleValueNet {
    fn evaluate(&self, state: &State<ROWS, COLS>) -> Score {
        (self.value(state) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_counts_bounds_uniqueness_and_order_are_frozen() {
        assert_eq!(STRUCTURED_WEIGHTS, 38_216);
        let mut at = 0;
        for (len, count) in [(4, 30), (3, 98), (4, 69), (5, 44), (6, 23)] {
            assert!(GEOMETRY.1[at..at + count]
                .iter()
                .all(|&actual| actual == len));
            at += count;
        }
        let mut seen = std::collections::HashSet::new();
        for i in 0..MAX_TUPLES {
            let tuple = &GEOMETRY.0[i][..GEOMETRY.1[i]];
            assert!(tuple.iter().all(|&cell| cell < ROWS * COLS));
            assert!(seen.insert(tuple.to_vec()), "duplicate tuple {tuple:?}");
        }
        assert_eq!(WINDOWS[0], [0, 1, 2, 3]);
        assert_eq!(WINDOWS[68], [20, 26, 32, 38]);
    }
    #[test]
    fn column_height_and_base_parity_have_deterministic_indices() {
        let mut weights = vec![0.0; STRUCTURED_WEIGHTS];
        let columns = STRUCTURED_WEIGHTS - COLS * 13;
        weights[columns + 5] = 2.0; // col 0: height 3, mover owns bottom.
        let net = NTupleValueNet::from_weights(weights);
        let mut black = crate::BitBoard::<ROWS, COLS>::EMPTY;
        black.set_index(0);
        black.set_index(14);
        let mut white = crate::BitBoard::<ROWS, COLS>::EMPTY;
        white.set_index(7);
        let state = State::from_parts(black, white, Player::Black, false);
        assert!((net.value(&state) - 2.0f32.tanh()).abs() < 1e-6);
    }
    #[test]
    fn basic_layout_stays_loadable_and_zero_nets_are_neutral() {
        assert_eq!(NTupleValueNet::default().value(&State::default()), 0.0);
        assert_eq!(
            NTupleValueNet::from_weights(vec![0.0; NT_WEIGHTS]).value(&State::default()),
            0.0
        );
    }
    #[test]
    fn value_matches_python_reference_prediction() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ntuple_weights_sample.bin"
        );
        let net = NTupleValueNet::load(path).unwrap();
        let mut black = crate::BitBoard::<ROWS, COLS>::EMPTY;
        black.set_index(0);
        black.set_index(2);
        let mut white = crate::BitBoard::<ROWS, COLS>::EMPTY;
        white.set_index(1);
        white.set_index(7);
        let state = State::from_parts(black, white, Player::Black, false);
        assert!((net.value(&state) + 0.801_430_9).abs() < 1e-5);
    }
    #[test]
    fn structured_value_matches_python_reference_prediction() {
        let weights = (0..STRUCTURED_WEIGHTS)
            .map(|i| (i as f64 - STRUCTURED_WEIGHTS as f64 / 2.0) as f32 * 0.0000002)
            .collect();
        let net = NTupleValueNet::from_weights(weights);
        let mut black = crate::BitBoard::<ROWS, COLS>::EMPTY;
        black.set_index(0);
        black.set_index(2);
        let mut white = crate::BitBoard::<ROWS, COLS>::EMPTY;
        white.set_index(1);
        white.set_index(7);
        let state = State::from_parts(black, white, Player::Black, false);
        assert!((net.value(&state) - (-0.479_704_23)).abs() < 1e-6);
    }
}
