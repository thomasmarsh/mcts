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
