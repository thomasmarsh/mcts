//! Compact, versioned Othello convolutional value+policy inference.
//!
//! `OTCNN001` (version 2) is a two-plane 8x8 network with a `CHANNELS`-wide
//! stem, `BLOCKS` residual blocks, and separate value and policy heads
//! sharing that trunk -- the direct 8x8 generalization of Connect Four's
//! `C4CNN001`
//! (`games/connect4/src/convnet.rs`), which also shares one trunk between
//! both heads. Version 1 was value-only; see
//! `research/othello-eval/src/othello_eval/convnet.py`'s module doc for why
//! the policy head was deferred rather than built alongside it, and for why
//! it's a version bump (not an additive extension) once built.
//!
//! Value inference averages the value network's output over all 8
//! D4-transformed copies of the input board, so the scalar is exactly
//! D4-invariant regardless of the learned weights -- generalizing
//! `C4CNN001`'s literal-plus-reflected averaging (Connect Four only has a
//! left-right mirror) to Othello's full 8-element D4 group. Policy inference
//! does the same for each of the 64 per-square logits, mapping each
//! orientation's canonical-frame output back to real board squares via
//! `crate::policy::INV` -- the same convention `NTuplePolicyNet` already
//! uses for the linear policy sidecar, so both models agree on what
//! orientation `sym` means and how PASS (mean of the 64 square logits) is
//! scored. Weights are trained by the Python counterpart
//! (`othello_eval.convnet`), which fits on a single (literal) orientation
//! only and relies on this same D4-averaging at evaluation time for the
//! equivariance property, not a symmetrized training loss.

use std::path::Path;

use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};

use crate::ntuple::D4;
use crate::policy::INV;
use crate::{Move, Othello, Player, State};

/// GPU-backed (MLX) reimplementation of this module's forward pass, gated
/// behind the `mlx` Cargo feature -- see `mlx.rs`'s module docs for the
/// layout conversions it does against this module's private weight-offset
/// constants and `input`/`trunk`-shaped architecture.
#[cfg(feature = "mlx")]
pub mod mlx;

const BOARD: usize = 8;
const CHANNELS: usize = 128;
const BLOCKS: usize = 6;
const VALUE_HIDDEN: usize = 32;
const POLICY_OUTPUTS: usize = 64;
const MAGIC: &[u8; 8] = b"OTCNN001";
const VERSION: u32 = 2;
const HEADER_BYTES: usize = 44;
const VALUE_HEAD_WEIGHTS: usize = CHANNELS
    + 1
    + BOARD * BOARD * VALUE_HIDDEN
    + VALUE_HIDDEN
    + VALUE_HIDDEN
    + 1;
const POLICY_HEAD_WEIGHTS: usize = CHANNELS + 1 + BOARD * BOARD * POLICY_OUTPUTS + POLICY_OUTPUTS;
pub const CNN_WEIGHTS: usize = CHANNELS * 2 * 9
    + CHANNELS
    + BLOCKS * 2 * (CHANNELS * CHANNELS * 9 + CHANNELS)
    + VALUE_HEAD_WEIGHTS
    + POLICY_HEAD_WEIGHTS;

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
        let header: [u32; 9] = std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap())
        });
        if header != [VERSION, BOARD as u32, BOARD as u32, 2, CHANNELS as u32, BLOCKS as u32, VALUE_HIDDEN as u32, POLICY_OUTPUTS as u32, CNN_WEIGHTS as u32] {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported OTCNN001 layout"));
        }
        Ok(Self::from_weights(bytes[HEADER_BYTES..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
    }

    pub fn weights(&self) -> &[f32] { &self.weights }

    /// Serialize to the exact `OTCNN001` byte layout `load` reads back --
    /// the counterpart test fixtures need to round-trip a checkpoint without
    /// depending on the Python trainer's own writer.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::from(*MAGIC);
        for n in [
            VERSION,
            BOARD as u32,
            BOARD as u32,
            2,
            CHANNELS as u32,
            BLOCKS as u32,
            VALUE_HIDDEN as u32,
            POLICY_OUTPUTS as u32,
            CNN_WEIGHTS as u32,
        ] {
            bytes.extend(n.to_le_bytes());
        }
        for w in &self.weights {
            bytes.extend(w.to_le_bytes());
        }
        bytes
    }

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

    /// Shared stem+residual trunk from a single (already-transformed)
    /// orientation's input planes, plus the weight offset just past it
    /// (where the value head's weights start).
    fn trunk(&self, input: &[f32; 2 * BOARD * BOARD]) -> (Vec<f32>, usize) {
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
        (x, at)
    }

    /// Value from a single (already-transformed) orientation's input planes.
    fn raw_value(&self, input: &[f32; 2 * BOARD * BOARD]) -> f32 {
        let (x, mut at) = self.trunk(input);
        let value_conv = &self.weights[at..at + CHANNELS];
        let value_bias = self.weights[at + CHANNELS];
        at += CHANNELS + 1;
        let value_features: Vec<f32> = (0..BOARD * BOARD).map(|cell| (value_bias + (0..CHANNELS).map(|ch| x[ch * BOARD * BOARD + cell] * value_conv[ch]).sum::<f32>()).max(0.0)).collect();
        let value_w1 = &self.weights[at..at + BOARD * BOARD * VALUE_HIDDEN]; at += BOARD * BOARD * VALUE_HIDDEN;
        let value_b1 = &self.weights[at..at + VALUE_HIDDEN]; at += VALUE_HIDDEN;
        let hidden: Vec<f32> = (0..VALUE_HIDDEN).map(|unit| (value_b1[unit] + (0..BOARD * BOARD).map(|cell| value_features[cell] * value_w1[cell * VALUE_HIDDEN + unit]).sum::<f32>()).max(0.0)).collect();
        let value_w2 = &self.weights[at..at + VALUE_HIDDEN]; at += VALUE_HIDDEN;
        let value = (self.weights[at] + hidden.iter().zip(value_w2).map(|(a, b)| a * b).sum::<f32>()).tanh(); at += 1;
        debug_assert_eq!(at, CNN_WEIGHTS - POLICY_HEAD_WEIGHTS);
        value
    }

    /// 64 canonical-frame (this orientation's own coordinate system) policy
    /// logits from a single (already-transformed) orientation's input
    /// planes -- no D4 averaging, no PASS.
    fn raw_policy(&self, input: &[f32; 2 * BOARD * BOARD]) -> [f32; POLICY_OUTPUTS] {
        let (x, mut at) = self.trunk(input);
        at += VALUE_HEAD_WEIGHTS;
        let policy_conv = &self.weights[at..at + CHANNELS];
        let policy_bias = self.weights[at + CHANNELS];
        at += CHANNELS + 1;
        let policy_features: Vec<f32> = (0..BOARD * BOARD).map(|cell| (policy_bias + (0..CHANNELS).map(|ch| x[ch * BOARD * BOARD + cell] * policy_conv[ch]).sum::<f32>()).max(0.0)).collect();
        let policy_w = &self.weights[at..at + BOARD * BOARD * POLICY_OUTPUTS]; at += BOARD * BOARD * POLICY_OUTPUTS;
        let policy_b = &self.weights[at..at + POLICY_OUTPUTS]; at += POLICY_OUTPUTS;
        let logits = std::array::from_fn(|out| {
            policy_b[out] + (0..BOARD * BOARD).map(|cell| policy_features[cell] * policy_w[cell * POLICY_OUTPUTS + out]).sum::<f32>()
        });
        debug_assert_eq!(at, CNN_WEIGHTS);
        logits
    }

    /// D4-averaged value: the mean, over all 8 orientations, of the literal
    /// network's output on that orientation's transformed input.
    pub fn value(&self, state: &State) -> f32 {
        (0..8).map(|sym| self.raw_value(&Self::input(state, sym))).sum::<f32>() / 8.0
    }

    /// D4-symmetrized policy logits over every board square, in the real
    /// board's coordinate frame: the average, over all 8 orientations, of
    /// that orientation's canonical-frame output mapped back via `INV` --
    /// the same convention `crate::policy::NTuplePolicyNet::all_logits`
    /// uses for the linear sidecar.
    pub fn all_policy_logits(&self, state: &State) -> [f64; POLICY_OUTPUTS] {
        let mut out = [0.0f64; POLICY_OUTPUTS];
        for sym in 0..8 {
            let raw = self.raw_policy(&Self::input(state, sym));
            for real_sq in 0..POLICY_OUTPUTS {
                out[real_sq] += raw[INV[sym][real_sq] as usize] as f64;
            }
        }
        for v in out.iter_mut() {
            *v /= 8.0;
        }
        out
    }
}

impl Evaluator<Othello> for CnnValueNet {
    fn evaluate(&self, state: &State) -> Score { (self.value(state) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score }
}

impl PolicyLogits<Othello> for CnnValueNet {
    fn logits(&mut self, state: &State, actions: &[Move]) -> Vec<f64> {
        let all = self.all_policy_logits(state);
        actions
            .iter()
            .map(|a| {
                if *a == Move::PASS {
                    // Same pass-as-mean-square-logit convention as
                    // `crate::policy::NTuplePolicyNet::logits`.
                    all.iter().sum::<f64>() / POLICY_OUTPUTS as f64
                } else {
                    all[a.0 as usize]
                }
            })
            .collect()
    }
}

/// Deterministic pseudo-random weights (splitmix64, uniform in
/// `[-amplitude, amplitude]`) for the tests here and in `mlx.rs`. The Python
/// tests (`othello-eval/tests/test_convnet.py`,
/// `az-train/tests/test_convnet_othello_torch.py`) implement the identical
/// generator, so a pinned fixture value means the same thing on both sides.
/// An `amplitude` near 0.07 keeps a production-geometry net's activations
/// near unit gain; larger values blow up through the residual stack and
/// leave nothing meaningful to compare.
#[cfg(test)]
pub(crate) fn splitmix_weights(amplitude: f64) -> Vec<f32> {
    (1..=CNN_WEIGHTS as u64)
        .map(|i| {
            let mut z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            ((u * 2.0 - 1.0) * amplitude) as f32
        })
        .collect()
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
        for n in [VERSION, BOARD as u32, BOARD as u32, 2, CHANNELS as u32, BLOCKS as u32, VALUE_HIDDEN as u32, POLICY_OUTPUTS as u32, CNN_WEIGHTS as u32] { bytes.extend(n.to_le_bytes()); }
        bytes.extend(std::iter::repeat_n(0u8, CNN_WEIGHTS * 4));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(CnnValueNet::load(&path).unwrap().weights().len(), CNN_WEIGHTS);
        bytes[8] = 3; std::fs::write(&path, bytes).unwrap();
        assert!(CnnValueNet::load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn zero_weights_score_zero_and_uniform_policy() {
        let net = CnnValueNet::default();
        assert_eq!(net.value(&State::default()), 0.0);
        assert_eq!(net.all_policy_logits(&State::default()), [0.0; POLICY_OUTPUTS]);
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

    /// D4-averaged value and first eight policy logits of `splitmix_weights(0.07)`
    /// on `state((1 << 0) | (1 << 2) | (1 << 8), 1 << 1 | (1 << 7), Black)`,
    /// computed by `othello_eval.convnet.predict_k` at this file's geometry.
    pub(crate) const REFERENCE_VALUE: f32 = 0.036_056_604;
    pub(crate) const REFERENCE_POLICY_HEAD: [f64; 8] = [
        0.008_687_069_639_563_56,
        0.001_992_151_839_658_618,
        -0.023_758_566_007_018_09,
        0.001_324_896_002_188_325,
        0.004_524_925_723_671_913,
        -0.027_064_848_691_225_05,
        0.004_268_915_392_458_439,
        0.011_740_943_416_953_087,
    ];

    #[test]
    fn value_is_d4_equivariant() {
        let net = CnnValueNet::from_weights(splitmix_weights(0.07));
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
        let net = CnnValueNet::from_weights(splitmix_weights(0.07));
        let s = state((1 << 0) | (1 << 2) | (1 << 8), 1 << 1 | (1 << 7), Player::Black);
        let got = net.value(&s);
        assert!((got - REFERENCE_VALUE).abs() < 1e-5, "{got}");
    }

    #[test]
    fn policy_is_d4_equivariant() {
        let net = CnnValueNet::from_weights(splitmix_weights(0.07));
        let black = (1u64 << 0) | (1 << 9) | (1 << 20);
        let white = (1u64 << 27) | (1 << 36) | (1 << 45);
        let base = state(black, white, Player::Black);
        let base_logits = net.all_policy_logits(&base);
        for k in 0..8 {
            let transformed = state(transform_bits(black, k), transform_bits(white, k), Player::Black);
            let got = net.all_policy_logits(&transformed);
            for sq in 0..POLICY_OUTPUTS {
                let want = base_logits[sq];
                let actual = got[D4[k][sq] as usize];
                assert!((want - actual).abs() < 1e-4, "orientation {k}, square {sq}: want {want}, got {actual}");
            }
        }
    }

    /// Cross-language fixture: same weights formula, geometry and state as
    /// `othello-eval/tests/test_convnet.py`'s
    /// `test_policy_matches_the_rust_reference_fixture` -- pins that the
    /// Rust hot path and the numpy trainer agree bit-for-bit (within float
    /// tolerance) on the D4-averaged policy logits, not just each
    /// independently passing its own tests.
    #[test]
    fn policy_matches_python_reference_fixture() {
        let net = CnnValueNet::from_weights(splitmix_weights(0.07));
        let s = state((1 << 0) | (1 << 2) | (1 << 8), 1 << 1 | (1 << 7), Player::Black);
        let got = net.all_policy_logits(&s);
        for (actual, expected) in got.iter().take(8).zip(REFERENCE_POLICY_HEAD) {
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }
    }

    /// A checkpoint written at a different geometry (here the previous
    /// 16-channel/4-block net's, valid header and all) must be refused
    /// loudly, not misread as this geometry's weights.
    #[test]
    fn load_rejects_a_checkpoint_from_a_different_geometry() {
        let (old_channels, old_blocks) = (16u32, 4u32);
        let old_weights = 25_171u32;
        let mut bytes = Vec::from(*MAGIC);
        for n in [VERSION, BOARD as u32, BOARD as u32, 2, old_channels, old_blocks, VALUE_HIDDEN as u32, POLICY_OUTPUTS as u32, old_weights] {
            bytes.extend(n.to_le_bytes());
        }
        bytes.extend(std::iter::repeat_n(0u8, old_weights as usize * 4));
        let path = std::env::temp_dir().join(format!("otcnn001-stale-geometry-{}.bin", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let err = CnnValueNet::load(&path).expect_err("a stale-geometry checkpoint must not load");
        std::fs::remove_file(&path).ok();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn pass_logit_is_the_mean_square_logit() {
        let mut net = CnnValueNet::from_weights(splitmix_weights(0.07));
        let s = state(1 << 0, 1 << 9, Player::Black);
        let all = net.all_policy_logits(&s);
        let want = all.iter().sum::<f64>() / POLICY_OUTPUTS as f64;
        let got = net.logits(&s, &[Move::PASS]);
        assert_eq!(got, vec![want]);
    }
}
