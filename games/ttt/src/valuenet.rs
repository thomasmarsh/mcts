//! Linear value head for the self-play training loop.
//!
//! The feature vector for a position is, relative to the side to move, a
//! bias term plus a "my piece here" plane and an "opponent piece here"
//! plane over the 9 cells -- 19 weights. The value estimate is the dot
//! product squashed through `tanh` into `[-1, 1]`.
//!
//! Weights are a flat little-endian `f32` array in the order
//! `[bias, me[0..9], opp[0..9]]` -- byte-for-byte what
//! `research/az-train/`'s `az_train.model.write_weights` produces, so the
//! two sides need no schema beyond this comment.
//!
//! [`NTupleValueNet`] is the higher-capacity head the self-play loop
//! actually promotes; see its own doc comment for that layout.

use std::path::Path;

use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};

use crate::{HashedPosition, Piece, Position, TicTacToe};

/// `1` bias + `9` "me" cells + `9` "opp" cells.
pub const N_WEIGHTS: usize = 1 + 2 * 9;

/// A 19-weight linear value head. `Default` is the all-zero net (every
/// position scores as a draw) -- the generation-0 evaluator before any
/// training has happened.
#[derive(Clone, Debug)]
pub struct LinearValueNet {
    weights: [f32; N_WEIGHTS],
}

impl Default for LinearValueNet {
    fn default() -> Self {
        Self {
            weights: [0.0; N_WEIGHTS],
        }
    }
}

impl LinearValueNet {
    pub fn from_weights(weights: [f32; N_WEIGHTS]) -> Self {
        Self { weights }
    }

    /// Read the flat little-endian `f32` weight array written by `az-train`.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != N_WEIGHTS * 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "expected {} bytes ({N_WEIGHTS} f32), got {}",
                    N_WEIGHTS * 4,
                    bytes.len()
                ),
            ));
        }
        let mut weights = [0.0f32; N_WEIGHTS];
        for (w, chunk) in weights.iter_mut().zip(bytes.chunks_exact(4)) {
            *w = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        Ok(Self { weights })
    }

    pub fn weights(&self) -> &[f32; N_WEIGHTS] {
        &self.weights
    }

    /// Pre-`tanh` linear score for `pos`, side-to-move perspective.
    fn raw_score(&self, pos: &Position) -> f32 {
        let side_x = pos.turn == Piece::X;
        let mut acc = self.weights[0];
        for cell in 0..9 {
            let occ = (pos.board >> (cell * 2)) & 0b11;
            let (me, opp) = match occ {
                1 if side_x => (1.0f32, 0.0),
                2 if side_x => (0.0, 1.0f32),
                1 => (0.0, 1.0f32),
                2 => (1.0f32, 0.0),
                _ => (0.0, 0.0),
            };
            acc += self.weights[1 + cell] * me + self.weights[1 + 9 + cell] * opp;
        }
        acc
    }

    /// Value estimate in `[-1, 1]`, side-to-move perspective.
    pub fn value(&self, pos: &Position) -> f32 {
        self.raw_score(pos).tanh()
    }
}

impl Evaluator<TicTacToe> for LinearValueNet {
    fn evaluate(&self, state: &HashedPosition) -> Score {
        (self.value(&state.position) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score
    }
}

/// The 8 structural lines (3 rows, 3 columns, 2 diagonals) followed by the
/// 4 overlapping 2x2 squares. Cells are row-major (`row * 3 + col`); the
/// order within a tuple fixes the base-3 digit order (first cell is the
/// least-significant trit), matching `az_train.ntuple`.
const NT_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];
const NT_SQUARES: [[usize; 4]; 4] = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]];

/// `1` bias + `8 * 3^3` line weights + `4 * 3^4` square weights.
pub const NT_WEIGHTS: usize = 1 + 8 * 27 + 4 * 81;

/// N-tuple value head for tic-tac-toe: a bias term plus one weight table per
/// structural line and per 2x2 square, indexed by the base-3 code of the
/// tuple's cells (digit 0 empty, 1 side-to-move piece, 2 opponent piece,
/// least-significant digit first). The pre-`tanh` score is the sum of the
/// one selected weight per tuple; `value` squashes it into `[-1, 1]`.
///
/// Weights are a flat little-endian `f32` array in the order
///
/// ```text
/// [bias,
///  line[0][0..27], ..., line[7][0..27],
///  square[0][0..81], ..., square[3][0..81]]
/// ```
///
/// 1 + 8*27 + 4*81 = 541 floats -- byte-for-byte what
/// `research/az-train/`'s `az_train.ntuple.write_weights` produces.
/// `Default` is the all-zero net (every position scores as a draw).
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

    /// Read the flat little-endian `f32` weight array written by `az-train
    /// --head ntuple`.
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
    fn trit(pos: &Position, cell: usize) -> usize {
        let side_x = pos.turn == Piece::X;
        match (pos.board >> (cell * 2)) & 0b11 {
            0 => 0,
            1 => {
                if side_x {
                    1
                } else {
                    2
                }
            }
            2 => {
                if side_x {
                    2
                } else {
                    1
                }
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    fn table_score(pos: &Position, cells: &[usize], weights: &[f32], offset: usize) -> (f32, usize) {
        let mut feat = 0usize;
        let mut place = 1usize;
        for &c in cells {
            feat += Self::trit(pos, c) * place;
            place *= 3;
        }
        (weights[offset + feat], place)
    }

    /// Pre-`tanh` linear score for `pos`, side-to-move perspective.
    fn raw_score(&self, pos: &Position) -> f32 {
        let mut acc = self.weights[0];
        let mut offset = 1usize;
        for line in &NT_LINES {
            let (w, span) = Self::table_score(pos, line, &self.weights, offset);
            acc += w;
            offset += span;
        }
        for square in &NT_SQUARES {
            let (w, span) = Self::table_score(pos, square, &self.weights, offset);
            acc += w;
            offset += span;
        }
        acc
    }

    /// Value estimate in `[-1, 1]`, side-to-move perspective.
    pub fn value(&self, pos: &Position) -> f32 {
        self.raw_score(pos).tanh()
    }
}

impl Evaluator<TicTacToe> for NTupleValueNet {
    fn evaluate(&self, state: &HashedPosition) -> Score {
        (self.value(&state.position) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same array `research/az-train/tests/test_model.py::
    /// test_weights_round_trip` uses (`np.arange(19) * 0.5 - 3.0`), written
    /// with the identical little-endian `f32` layout, must load unchanged.
    #[test]
    fn weights_round_trip_matches_python_layout() {
        let expected: [f32; N_WEIGHTS] = std::array::from_fn(|i| i as f32 * 0.5 - 3.0);
        let mut bytes = Vec::new();
        for w in expected {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let path = std::env::temp_dir().join("game_ttt_valuenet_round_trip.bin");
        std::fs::write(&path, &bytes).unwrap();
        let net = LinearValueNet::load(&path).unwrap();
        assert_eq!(net.weights(), &expected);
        std::fs::remove_file(&path).ok();
    }

    /// Cross-language check against `research/az-train`: the fixture is
    /// `az_train.model.write_weights` output for `(arange(19) - 9) * 0.1`,
    /// and `az_train.model.predict` scores this board at `-0.8336546`.
    #[test]
    fn value_matches_python_reference_prediction() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/weights_sample.bin"
        );
        let net = LinearValueNet::load(path).unwrap();
        let mut pos = Position::new(); // X to move
        pos.set(0, Piece::X);
        pos.set(4, Piece::O);
        assert!((net.value(&pos) - (-0.833_654_6)).abs() < 1e-5);
    }

    /// Cross-language check for the n-tuple head against
    /// `research/az-train`'s `az_train.ntuple`: the fixture is
    /// `write_weights((arange(541) - 270.5) * 0.0005)` and
    /// `az_train.ntuple.predict` scores the board {X@0, O@4, X to move} at
    /// `-0.5684636`.
    #[test]
    fn ntuple_value_matches_python_reference_prediction() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ntuple_weights_sample.bin"
        );
        let net = NTupleValueNet::load(path).unwrap();
        let mut pos = Position::new(); // X to move
        pos.set(0, Piece::X);
        pos.set(4, Piece::O);
        assert!((net.value(&pos) - (-0.568_463_6)).abs() < 1e-5);
    }

    /// The n-tuple head represents a conjunction the linear head cannot: a
    /// weight placed only on the "mover owns all of cells 0,1,2" feature of
    /// the top-row tuple fires for that board and nothing else.
    #[test]
    fn ntuple_row_tuple_isolates_a_completed_line() {
        let mut w = vec![0.0f32; NT_WEIGHTS];
        // Top row is line 0, table starts at offset 1. Feature index for
        // "all three cells hold the mover" is 1 + 3 + 9 = 13.
        w[1 + 13] = 2.0;
        let net = NTupleValueNet::from_weights(w);

        let mut all_mine = Position::new();
        all_mine.set(0, Piece::X);
        all_mine.set(1, Piece::X);
        all_mine.set(2, Piece::X);
        assert!((net.value(&all_mine) - 2.0f32.tanh()).abs() < 1e-6);

        let mut two_of_three = Position::new();
        two_of_three.set(0, Piece::X);
        two_of_three.set(1, Piece::X);
        assert_eq!(net.value(&two_of_three), 0.0);
    }

    #[test]
    fn ntuple_default_scores_every_position_as_a_draw() {
        let net = NTupleValueNet::default();
        assert_eq!(net.value(&Position::new()), 0.0);
        let mut p = Position::new();
        p.apply(crate::Move(4));
        assert_eq!(net.value(&p), 0.0);
    }

    #[test]
    fn load_rejects_a_wrong_length_file() {
        let path = std::env::temp_dir().join("game_ttt_valuenet_bad_len.bin");
        std::fs::write(&path, [0u8; 12]).unwrap();
        assert!(LinearValueNet::load(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// A hand-checkable prediction: weights that give the mover +1 raw score
    /// per own centre stone. With the centre held, value is `tanh(1) > 0`;
    /// empty board is `tanh(0) == 0`.
    #[test]
    fn value_uses_side_to_move_planes() {
        let mut w = [0.0f32; N_WEIGHTS];
        w[1 + 4] = 1.0; // "me" plane, centre cell
        let net = LinearValueNet::from_weights(w);

        assert_eq!(net.value(&Position::new()), 0.0);

        let mut with_centre = Position::new();
        with_centre.apply(crate::Move(4)); // X takes centre, now O to move
        // Centre now holds the opponent's (X's) piece from O's perspective,
        // so the "me" plane is empty and the score is still 0.
        assert_eq!(net.value(&with_centre), 0.0);

        // From X's perspective right after, pretend it's X to move again:
        let mut x_to_move = with_centre;
        x_to_move.turn = Piece::X;
        assert!(net.value(&x_to_move) > 0.0);
        assert!((net.value(&x_to_move) - 1.0f32.tanh()).abs() < 1e-6);
    }
}
