//! Compact, versioned Othello convolutional value inference.
//!
//! `OTCNN001` is a two-plane 8x8 network with a 16-channel stem, two
//! residual blocks, and a value head -- the direct 8x8 generalization of
//! Connect Four's `C4CNN001` (`games/connect4/src/convnet.rs`). Value-only
//! for now: see `research/othello-eval/src/othello_eval/convnet.py`'s module
//! doc for why a policy head is deferred rather than built alongside it.
//!
//! Inference averages the value network's output over all 8 D4-transformed
//! copies of the input board, so the scalar is exactly D4-invariant
//! regardless of the learned weights -- generalizing `C4CNN001`'s
//! literal-plus-reflected averaging (Connect Four only has a left-right
//! mirror) to Othello's full 8-element D4 group. Weights are trained by the
//! Python counterpart (`othello_eval.convnet`), which fits on a single
//! (literal) orientation only and relies on this same D4-averaging at
//! evaluation time for the equivariance property, not a symmetrized
//! training loss.

use std::path::Path;

use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};

use crate::ntuple::D4;
use crate::{Othello, Player, State};

const BOARD: usize = 8;
const CHANNELS: usize = 16;
const BLOCKS: usize = 2;
const VALUE_HIDDEN: usize = 32;
const MAGIC: &[u8; 8] = b"OTCNN001";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 40;
pub const CNN_WEIGHTS: usize = CHANNELS * 2 * 9
    + CHANNELS
    + BLOCKS * 2 * (CHANNELS * CHANNELS * 9 + CHANNELS)
    + CHANNELS
    + 1
    + BOARD * BOARD * VALUE_HIDDEN
    + VALUE_HIDDEN
    + VALUE_HIDDEN
    + 1;

#[derive(Clone, Debug)]
pub struct CnnValueNet {
    weights: Vec<f32>,
}

impl Default for CnnValueNet {
    fn default() -> Self {
        Self::from_weights(vec![0.0; CNN_WEIGHTS])
    }
}

impl CnnValueNet {
    pub fn from_weights(weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), CNN_WEIGHTS, "OTCNN001 needs {CNN_WEIGHTS} weights");
        Self { weights }
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != HEADER_BYTES + CNN_WEIGHTS * 4 || !bytes.starts_with(MAGIC) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid OTCNN001 byte length or magic"));
        }
        let header: [u32; 8] = std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap())
        });
        if header != [VERSION, BOARD as u32, BOARD as u32, 2, CHANNELS as u32, BLOCKS as u32, VALUE_HIDDEN as u32, CNN_WEIGHTS as u32] {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported OTCNN001 layout"));
        }
        Ok(Self::from_weights(bytes[HEADER_BYTES..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
    }

    pub fn weights(&self) -> &[f32] { &self.weights }

    /// Two occupancy planes (own discs, opponent discs) for board square `j`
    /// read from square `D4[sym][j]` of the real board -- the same
    /// "read the transformed location" convention `crate::ntuple`'s
    /// `feature_indices`/`crate::policy`'s sidecar use, so this and the
    /// linear models' D4 averaging agree on what orientation `sym` means.
    fn input(state: &State, sym: usize) -> [f32; 2 * BOARD * BOARD] {
        std::array::from_fn(|i| {
            let plane = i / (BOARD * BOARD);
            let j = i % (BOARD * BOARD);
            let sq = D4[sym][j] as usize;
            let mover = state.turn;
            let own = (state.black.get_index(sq) && mover == Player::Black)
                || (state.white.get_index(sq) && mover == Player::White);
            if (plane == 0 && own) || (plane == 1 && !own && (state.black.get_index(sq) || state.white.get_index(sq))) { 1.0 } else { 0.0 }
        })
    }

    fn conv3(input: &[f32], out: &mut [f32], weights: &[f32], bias: &[f32], in_channels: usize) {
        for out_ch in 0..CHANNELS {
            for row in 0..BOARD {
                for col in 0..BOARD {
                    let mut sum = bias[out_ch];
                    for in_ch in 0..in_channels {
                        for kr in 0..3 {
                            for kc in 0..3 {
                                let rr = row as isize + kr as isize - 1;
                                let cc = col as isize + kc as isize - 1;
                                if (0..BOARD as isize).contains(&rr) && (0..BOARD as isize).contains(&cc) {
                                    sum += input[(in_ch * BOARD + rr as usize) * BOARD + cc as usize]
                                        * weights[((out_ch * in_channels + in_ch) * 3 + kr) * 3 + kc];
                                }
                            }
                        }
                    }
                    out[(out_ch * BOARD + row) * BOARD + col] = sum;
                }
            }
        }
    }

    /// Value from a single (already-transformed) orientation's input planes.
    fn raw(&self, input: &[f32; 2 * BOARD * BOARD]) -> f32 {
        let mut at = 0;
        let mut x = vec![0.0; CHANNELS * BOARD * BOARD];
        Self::conv3(input, &mut x, &self.weights[at..at + CHANNELS * 2 * 9], &self.weights[at + CHANNELS * 2 * 9..at + CHANNELS * 2 * 9 + CHANNELS], 2);
        x.iter_mut().for_each(|v| *v = v.max(0.0));
        at += CHANNELS * 2 * 9 + CHANNELS;
        for _ in 0..BLOCKS {
            let residual = x.clone();
            let mut y = vec![0.0; x.len()];
            Self::conv3(&x, &mut y, &self.weights[at..at + CHANNELS * CHANNELS * 9], &self.weights[at + CHANNELS * CHANNELS * 9..at + CHANNELS * CHANNELS * 9 + CHANNELS], CHANNELS);
            y.iter_mut().for_each(|v| *v = v.max(0.0));
            at += CHANNELS * CHANNELS * 9 + CHANNELS;
            Self::conv3(&y, &mut x, &self.weights[at..at + CHANNELS * CHANNELS * 9], &self.weights[at + CHANNELS * CHANNELS * 9..at + CHANNELS * CHANNELS * 9 + CHANNELS], CHANNELS);
            for (v, skip) in x.iter_mut().zip(residual) { *v = (*v + skip).max(0.0); }
            at += CHANNELS * CHANNELS * 9 + CHANNELS;
        }
        let value_conv = &self.weights[at..at + CHANNELS];
        let value_bias = self.weights[at + CHANNELS];
        at += CHANNELS + 1;
        let value_features: Vec<f32> = (0..BOARD * BOARD).map(|cell| (value_bias + (0..CHANNELS).map(|ch| x[ch * BOARD * BOARD + cell] * value_conv[ch]).sum::<f32>()).max(0.0)).collect();
        let value_w1 = &self.weights[at..at + BOARD * BOARD * VALUE_HIDDEN]; at += BOARD * BOARD * VALUE_HIDDEN;
        let value_b1 = &self.weights[at..at + VALUE_HIDDEN]; at += VALUE_HIDDEN;
        let hidden: Vec<f32> = (0..VALUE_HIDDEN).map(|unit| (value_b1[unit] + (0..BOARD * BOARD).map(|cell| value_features[cell] * value_w1[cell * VALUE_HIDDEN + unit]).sum::<f32>()).max(0.0)).collect();
        let value_w2 = &self.weights[at..at + VALUE_HIDDEN]; at += VALUE_HIDDEN;
        let value = (self.weights[at] + hidden.iter().zip(value_w2).map(|(a, b)| a * b).sum::<f32>()).tanh(); at += 1;
        debug_assert_eq!(at, CNN_WEIGHTS);
        value
    }

    /// D4-averaged value: the mean, over all 8 orientations, of the literal
    /// network's output on that orientation's transformed input.
    pub fn value(&self, state: &State) -> f32 {
        (0..8).map(|sym| self.raw(&Self::input(state, sym))).sum::<f32>() / 8.0
    }
}

impl Evaluator<Othello> for CnnValueNet {
    fn evaluate(&self, state: &State) -> Score { (self.value(state) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BB;

    fn state(black: u64, white: u64, turn: Player) -> State {
        State { black: BB::from_bits(black), white: BB::from_bits(white), turn, last_pass: false, hashes: [0u64; 8] }
    }

    #[test]
    fn layout_is_versioned_and_validated() {
        let path = std::env::temp_dir().join(format!("mcts-othello-cnn-{}", std::process::id()));
        let mut bytes = Vec::from(*MAGIC);
        for n in [VERSION, BOARD as u32, BOARD as u32, 2, CHANNELS as u32, BLOCKS as u32, VALUE_HIDDEN as u32, CNN_WEIGHTS as u32] { bytes.extend(n.to_le_bytes()); }
        bytes.extend(std::iter::repeat_n(0u8, CNN_WEIGHTS * 4));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(CnnValueNet::load(&path).unwrap().weights().len(), CNN_WEIGHTS);
        bytes[8] = 2; std::fs::write(&path, bytes).unwrap();
        assert!(CnnValueNet::load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn zero_weights_score_zero() {
        let net = CnnValueNet::default();
        assert_eq!(net.value(&State::default()), 0.0);
    }

    /// Rotate a raw bitboard by D4 element `k` (bit `i` moves to bit
    /// `D4[k][i]`), for building a transformed fixture state.
    fn transform_bits(bits: u64, k: usize) -> u64 {
        let mut out = 0u64;
        let mut r = bits;
        while r != 0 {
            let i = r.trailing_zeros() as usize;
            r &= r - 1;
            out |= 1u64 << D4[k][i];
        }
        out
    }

    #[test]
    fn value_is_d4_equivariant() {
        let weights: Vec<f32> = (0..CNN_WEIGHTS).map(|i| (i as f32 * 0.0007).sin()).collect();
        let net = CnnValueNet::from_weights(weights);
        let black = (1u64 << 0) | (1 << 9) | (1 << 20);
        let white = (1u64 << 27) | (1 << 36) | (1 << 45);
        let base = state(black, white, Player::Black);
        let base_value = net.value(&base);
        for k in 0..8 {
            let transformed = state(transform_bits(black, k), transform_bits(white, k), Player::Black);
            assert!((net.value(&transformed) - base_value).abs() < 1e-5, "orientation {k}");
        }
    }

    /// Cross-language fixture: same weights formula, geometry and state as
    /// `othello-eval/tests/test_convnet.py`'s
    /// `test_value_matches_the_rust_reference_fixture` -- pins that the Rust
    /// hot path and the numpy trainer agree bit-for-bit (within float
    /// tolerance) on the D4-averaged value, not just each independently
    /// passing its own tests.
    #[test]
    fn value_matches_python_reference_fixture() {
        let weights: Vec<f32> = (0..CNN_WEIGHTS).map(|i| ((i as f64 - CNN_WEIGHTS as f64 / 2.0) * 1e-6) as f32).collect();
        let net = CnnValueNet::from_weights(weights);
        let s = state((1 << 0) | (1 << 2) | (1 << 8), 1 << 1 | (1 << 7), Player::Black);
        let got = net.value(&s);
        assert!((got - 0.007_168_648).abs() < 1e-6);
    }
}
