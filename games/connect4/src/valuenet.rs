//! N-tuple value head for standard 6x7 Connect Four, for the self-play
//! training loop.
//!
//! The feature set is the 69 four-in-a-row windows of the board -- every
//! line that can win the game: 24 horizontal, 21 vertical, 12 up-right
//! diagonal, 12 up-left diagonal. Each window owns a table of `3^4`
//! weights, indexed by the base-3 code of its four cells from the side to
//! move's perspective (digit 0 empty, 1 own piece, 2 opponent piece,
//! least-significant digit first). The pre-`tanh` score is a bias term plus
//! the sum of the one selected weight per window; `value` squashes it into
//! `[-1, 1]`.
//!
//! Cells are row-major with row 0 at the bottom (`cell = row * 7 + col`),
//! matching [`State`]'s bit layout.
//!
//! Weights are a flat little-endian `f32` array in the order
//!
//! ```text
//! [bias, window[0][0..81], window[1][0..81], ..., window[68][0..81]]
//! ```
//!
//! `1 + 69*81 = 5590` floats -- byte-for-byte what
//! `research/az-train/`'s `az_train.ntuple_c4.write_weights` produces, so
//! the two sides need no schema beyond this comment. `Default` is the
//! all-zero net (every position scores as a draw) -- the generation-0
//! evaluator before any training has happened.

use std::path::Path;

use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};

use crate::{Player, Standard, State};

const ROWS: usize = 6;
const COLS: usize = 7;

/// Number of four-in-a-row windows on the standard board.
pub const N_WINDOWS: usize = 69;

/// `1` bias + `69 * 3^4` window-table weights.
pub const NT_WEIGHTS: usize = 1 + N_WINDOWS * 81;

pub(crate) const WINDOWS: [[usize; 4]; N_WINDOWS] = build_windows();

const fn build_windows() -> [[usize; 4]; N_WINDOWS] {
    let mut w = [[0usize; 4]; N_WINDOWS];
    let mut n = 0;

    // Horizontal: four consecutive columns on a row.
    let mut row = 0;
    while row < ROWS {
        let mut c0 = 0;
        while c0 + 3 < COLS {
            let mut k = 0;
            while k < 4 {
                w[n][k] = row * COLS + c0 + k;
                k += 1;
            }
            n += 1;
            c0 += 1;
        }
        row += 1;
    }

    // Vertical: four consecutive rows in a column.
    let mut col = 0;
    while col < COLS {
        let mut r0 = 0;
        while r0 + 3 < ROWS {
            let mut k = 0;
            while k < 4 {
                w[n][k] = (r0 + k) * COLS + col;
                k += 1;
            }
            n += 1;
            r0 += 1;
        }
        col += 1;
    }

    // Up-right diagonal.
    let mut r0 = 0;
    while r0 + 3 < ROWS {
        let mut c0 = 0;
        while c0 + 3 < COLS {
            let mut k = 0;
            while k < 4 {
                w[n][k] = (r0 + k) * COLS + (c0 + k);
                k += 1;
            }
            n += 1;
            c0 += 1;
        }
        r0 += 1;
    }

    // Up-left diagonal.
    let mut r0 = 0;
    while r0 + 3 < ROWS {
        let mut c0 = 3;
        while c0 < COLS {
            let mut k = 0;
            while k < 4 {
                w[n][k] = (r0 + k) * COLS + (c0 - k);
                k += 1;
            }
            n += 1;
            c0 += 1;
        }
        r0 += 1;
    }

    assert!(n == N_WINDOWS);
    w
}

/// N-tuple value head for `Connect4<6, 7>`. See the module doc comment for
/// the weight layout.
#[derive(Clone, Debug)]
pub struct NTupleValueNet {
    weights: Vec<f32>,
}

impl Default for NTupleValueNet {
    fn default() -> Self {
        Self {
            weights: vec![0.0; NT_WEIGHTS],
        }
    }
}

impl NTupleValueNet {
    pub fn from_weights(weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), NT_WEIGHTS, "n-tuple head needs {NT_WEIGHTS} weights");
        Self { weights }
    }

    /// Read the flat little-endian `f32` weight array written by
    /// `az-train --head ntuple` for the Connect Four geometry.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != NT_WEIGHTS * 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "expected {} bytes ({NT_WEIGHTS} f32), got {}",
                    NT_WEIGHTS * 4,
                    bytes.len()
                ),
            ));
        }
        let weights = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self { weights })
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Per-cell base-3 digit from the side-to-move perspective: 0 empty,
    /// 1 own piece, 2 opponent piece.
    #[inline]
    pub(crate) fn trit(state: &State<ROWS, COLS>, cell: usize) -> usize {
        let mover = state.turn();
        if state.black().get_index(cell) {
            if matches!(mover, Player::Black) {
                1
            } else {
                2
            }
        } else if state.white().get_index(cell) {
            if matches!(mover, Player::White) {
                1
            } else {
                2
            }
        } else {
            0
        }
    }

    /// Bias plus one table index per window. With `mirrored`, cells are read
    /// through the board's left-right symmetry before tuple lookup.
    pub(crate) fn active_indices(state: &State<ROWS, COLS>, mirrored: bool) -> [usize; 1 + N_WINDOWS] {
        let mut active = [0usize; 1 + N_WINDOWS];
        let mut offset = 1;
        for (i, win) in WINDOWS.iter().enumerate() {
            let mut feat = 0;
            let mut place = 1;
            for &cell in win {
                let cell = if mirrored { (cell / COLS) * COLS + (COLS - 1 - cell % COLS) } else { cell };
                feat += Self::trit(state, cell) * place;
                place *= 3;
            }
            active[i + 1] = offset + feat;
            offset += 81;
        }
        active
    }

    /// Pre-`tanh` linear score for `state`, side-to-move perspective.
    fn raw_score(&self, state: &State<ROWS, COLS>) -> f32 {
        let mut acc = self.weights[0];
        for index in Self::active_indices(state, false).into_iter().skip(1) { acc += self.weights[index]; }
        acc
    }

    /// Value estimate in `[-1, 1]`, side-to-move perspective.
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
    use mcts::game::Game;

    #[test]
    fn window_set_is_complete_and_well_formed() {
        assert_eq!(WINDOWS.len(), N_WINDOWS);
        assert_eq!(NT_WEIGHTS, 5590);
        for win in &WINDOWS {
            for &c in win {
                assert!(c < ROWS * COLS);
            }
            // Cells strictly increasing keeps the base-3 digit order fixed.
            for pair in win.windows(2) {
                assert!(pair[0] < pair[1]);
            }
        }
        // No duplicate windows.
        let mut seen = std::collections::HashSet::new();
        for win in &WINDOWS {
            assert!(seen.insert(*win), "duplicate window {win:?}");
        }
    }

    #[test]
    fn default_scores_every_position_as_a_draw() {
        let net = NTupleValueNet::default();
        assert_eq!(net.value(&State::<ROWS, COLS>::default()), 0.0);
        let state = Standard::apply(State::default(), &crate::Move(3));
        assert_eq!(net.value(&state), 0.0);
    }

    #[test]
    fn a_weight_isolates_one_completed_window() {
        // Bottom row, columns 0..4 is horizontal window 0. Its feature
        // index for "all four cells hold the mover" is 1 + 3 + 9 + 27 = 40.
        let mut w = vec![0.0f32; NT_WEIGHTS];
        w[1 + 40] = 3.0;
        let net = NTupleValueNet::from_weights(w);

        // Black owns (0,0),(0,1),(0,2),(0,3); Black to move.
        let black = State::<ROWS, COLS>::from_parts(
            {
                let mut b = crate::BitBoard::<ROWS, COLS>::EMPTY;
                for c in 0..4 {
                    b.set_index(c);
                }
                b
            },
            crate::BitBoard::<ROWS, COLS>::EMPTY,
            Player::Black,
            false,
        );
        assert!((net.value(&black) - 3.0f32.tanh()).abs() < 1e-6);

        // Only three of the four: window not completed, score stays 0.
        let three = State::<ROWS, COLS>::from_parts(
            {
                let mut b = crate::BitBoard::<ROWS, COLS>::EMPTY;
                for c in 0..3 {
                    b.set_index(c);
                }
                b
            },
            crate::BitBoard::<ROWS, COLS>::EMPTY,
            Player::Black,
            false,
        );
        assert_eq!(net.value(&three), 0.0);
    }

    #[test]
    fn load_rejects_a_wrong_length_file() {
        let path = std::env::temp_dir().join("game_connect4_valuenet_bad_len.bin");
        std::fs::write(&path, [0u8; 16]).unwrap();
        assert!(NTupleValueNet::load(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// Cross-language check against `research/az-train`'s `az_train.ntuple_c4`:
    /// the fixture is `write_weights((arange(5590) - 2795.0) * 0.0002)` and
    /// `predict` scores the board {Black@0, Black@2, White@1, White@7,
    /// Black to move} at `-0.8014309`.
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
        let state = State::<ROWS, COLS>::from_parts(black, white, Player::Black, false);

        assert!((net.value(&state) - (-0.801_430_9)).abs() < 1e-5);
    }
}
