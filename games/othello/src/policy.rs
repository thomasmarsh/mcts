//! D4-equivariant n-tuple policy sidecar for Othello.
//!
//! Same active tuple features as the value head (`crate::ntuple`), sharing
//! `model.toml`'s geometry, but with one weight *row* of 64 columns (one per
//! board square) per feature instead of a single scalar weight. A row's
//! columns are indexed in the *canonical* (D4 identity) frame; evaluating a
//! real position averages the row over all 8 orientations, mapping each
//! orientation's canonical-frame output back to real board squares via the
//! inverse of that orientation's permutation. This generalizes the
//! two-orientation (identity + mirror) averaging Connect Four's sidecar
//! uses (`games/connect4/src/policynet.rs`) to Othello's full D4 group.
//!
//! Weights are trained by the Python counterpart (`othello_eval.policy`,
//! planned) and loaded the same way as the value head: a flat little-endian
//! `f32` array plus a `weights.meta.json` recording the geometry SHA-256, so
//! a policy/geometry mismatch is caught on load rather than silently
//! misreading the weight table.

use std::path::Path;
use std::sync::LazyLock;

use mcts::algorithms::mcts::policy::PolicyLogits;

use crate::ntuple::{ModelGeometry, D4};
use crate::{Move, Othello, State};

/// Number of board squares == number of policy output columns per feature
/// row. Pass (`Move::PASS`) has no square and is handled separately.
const SQUARES: usize = 64;

/// `INV[k]` is the inverse permutation of `D4[k]`: `INV[k][D4[k][i]] == i`.
/// Used to map a canonical-frame (orientation `k`) output column back to the
/// real board square it corresponds to. `pub(crate)` so `crate::convnet`'s
/// policy head can reuse the exact same D4-averaging convention rather than
/// duplicating this table.
pub(crate) static INV: LazyLock<[[u8; SQUARES]; 8]> = LazyLock::new(|| {
    std::array::from_fn(|k| {
        let mut inv = [0u8; SQUARES];
        for canon in 0..SQUARES {
            inv[D4[k][canon] as usize] = canon as u8;
        }
        inv
    })
});

#[derive(serde::Deserialize)]
struct WeightsMeta {
    model_toml_sha256: String,
    n_weights: usize,
}

/// A loaded policy sidecar: geometry (shared with the value head) plus
/// trained weights, `geom.n_weights() * 64` of them.
#[derive(Clone, Debug)]
pub struct NTuplePolicyNet {
    geom: ModelGeometry,
    weights: Vec<f32>,
}

impl NTuplePolicyNet {
    /// Build a zero-weight sidecar over `geom` -- logits are uniform (all
    /// zero) over every square until trained weights are loaded.
    pub fn zeros(geom: ModelGeometry) -> Self {
        let n = geom.n_weights() * SQUARES;
        Self {
            geom,
            weights: vec![0.0; n],
        }
    }

    /// Build from an explicit weight vector, asserting its length matches
    /// `geom.n_weights() * 64`.
    pub fn from_weights(geom: ModelGeometry, weights: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            geom.n_weights() * SQUARES,
            "policy sidecar needs {} weights ({} features * {SQUARES} squares), got {}",
            geom.n_weights() * SQUARES,
            geom.n_weights(),
            weights.len()
        );
        Self { geom, weights }
    }

    /// Load `<dir>/policy.bin` + `<dir>/policy.meta.json` against a
    /// `model.toml` already parsed into `geom` (typically the same geometry
    /// the sibling [`crate::ntuple::NTupleModel`] loaded from the same
    /// directory).
    pub fn from_dir(geom: ModelGeometry, dir: &Path) -> NTuplePolicyNet {
        let meta_path = dir.join("policy.meta.json");
        let meta_text = std::fs::read_to_string(&meta_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", meta_path.display()));
        let meta: WeightsMeta =
            serde_json::from_str(&meta_text).expect("policy.meta.json must parse");
        assert_eq!(
            meta.model_toml_sha256,
            geom.sha256_hex(),
            "policy.meta.json SHA-256 does not match the model.toml this geometry was parsed \
             from -- the policy sidecar was trained against a different geometry; retrain"
        );
        assert_eq!(
            meta.n_weights,
            geom.n_weights(),
            "policy.meta.json n_weights disagrees with parsed geometry"
        );

        let bin_path = dir.join("policy.bin");
        let raw = std::fs::read(&bin_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", bin_path.display()));
        assert_eq!(
            raw.len(),
            geom.n_weights() * SQUARES * 4,
            "{} holds {} bytes, expected {} (4 * n_weights * {SQUARES})",
            bin_path.display(),
            raw.len(),
            geom.n_weights() * SQUARES * 4
        );
        let weights: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        NTuplePolicyNet { geom, weights }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Raw per-square logits as seen through orientation `sym`: for each
    /// active tuple feature under that orientation, add its 64-column
    /// weight row. The output is indexed in the *canonical* (identity)
    /// frame, not yet mapped back to real board squares.
    fn raw_logits_for_orientation(&self, state: &State, sym: usize) -> [f64; SQUARES] {
        let flat = self.geom.feature_indices(state);
        let n_tuples = self.geom.n_tuples();
        debug_assert_eq!(flat.len(), n_tuples * 8);
        let mut out = [0.0f64; SQUARES];
        for t in 0..n_tuples {
            let feat = flat[t * 8 + sym] as usize;
            let row = &self.weights[feat * SQUARES..feat * SQUARES + SQUARES];
            for (col, w) in row.iter().enumerate() {
                out[col] += *w as f64;
            }
        }
        out
    }

    /// D4-symmetrized logits over every board square, in the real board's
    /// coordinate frame: the average, over all 8 orientations, of that
    /// orientation's canonical-frame output mapped back via [`INV`].
    pub fn all_logits(&self, state: &State) -> [f64; SQUARES] {
        let mut out = [0.0f64; SQUARES];
        for sym in 0..8 {
            let raw = self.raw_logits_for_orientation(state, sym);
            for real_sq in 0..SQUARES {
                out[real_sq] += raw[INV[sym][real_sq] as usize];
            }
        }
        for v in out.iter_mut() {
            *v /= 8.0;
        }
        out
    }
}

impl PolicyLogits<Othello> for NTuplePolicyNet {
    fn logits(&mut self, state: &State, actions: &[Move]) -> Vec<f64> {
        let all = self.all_logits(state);
        actions
            .iter()
            .map(|a| {
                if *a == Move::PASS {
                    // Pass has no board square; its logit doesn't compete
                    // against real squares for D4 equivariance, so it's
                    // scored as the mean square logit (a neutral prior)
                    // rather than a fixed 0.0, which would be systematically
                    // biased low/high whenever the average genuinely isn't 0.
                    all.iter().sum::<f64>() / SQUARES as f64
                } else {
                    all[a.0 as usize]
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Player, BB};
    use mcts::game::Game;

    fn tiny_geom() -> ModelGeometry {
        let bytes = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ntuple/tests/tiny.toml"),
        )
        .unwrap();
        ModelGeometry::parse(&bytes)
    }

    fn state(black: u64, white: u64, turn: Player) -> State {
        State {
            black: BB::from_bits(black),
            white: BB::from_bits(white),
            turn,
            last_pass: false,
            hashes: [0u64; 8],
        }
    }

    #[test]
    fn inv_is_the_inverse_permutation_of_d4() {
        for k in 0..8 {
            for i in 0..64 {
                assert_eq!(INV[k][D4[k][i] as usize], i as u8);
                assert_eq!(D4[k][INV[k][i] as usize], i as u8);
            }
        }
    }

    #[test]
    fn zero_sidecar_is_uniform_over_legal_actions() {
        let geom = tiny_geom();
        let mut net = NTuplePolicyNet::zeros(geom);
        let mut actions = Vec::new();
        Othello::generate_actions(&State::default(), &mut actions);
        let got = net.logits(&State::default(), &actions);
        assert_eq!(got, vec![0.0; actions.len()]);
    }

    /// Rotate a raw bitboard by D4 element `k` (i.e. bit `i` moves to bit
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
    fn all_logits_is_d4_equivariant() {
        let geom = tiny_geom();
        let n = geom.n_weights() * SQUARES;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0007).sin()).collect();
        let net = NTuplePolicyNet::from_weights(geom, weights);

        let black = (1u64 << 0) | (1 << 9) | (1 << 20);
        let white = (1u64 << 27) | (1 << 36) | (1 << 45);
        let base = state(black, white, Player::Black);
        let base_logits = net.all_logits(&base);

        for k in 0..8 {
            let transformed = state(
                transform_bits(black, k),
                transform_bits(white, k),
                Player::Black,
            );
            let got = net.all_logits(&transformed);
            for sq in 0..SQUARES {
                let want = base_logits[sq];
                let actual = got[D4[k][sq] as usize];
                assert!(
                    (want - actual).abs() < 1e-9,
                    "orientation {k}, square {sq}: want {want}, got {actual}"
                );
            }
        }
    }

    /// Cross-language fixture: same weights formula, geometry and state as
    /// `othello-eval/tests/test_policy.py`'s
    /// `test_policy_logits_matches_the_rust_reference_fixture` -- pins that
    /// the Rust hot path and the numpy trainer agree bit-for-bit (within
    /// float tolerance) on the D4-symmetrized averaging, not just each
    /// independently passing its own tests.
    #[test]
    fn logits_match_python_reference_fixture() {
        let geom = tiny_geom();
        let n = geom.n_weights() * SQUARES;
        let weights: Vec<f32> = (0..n)
            .map(|i| ((i as f64 - n as f64 / 2.0) * 0.00002) as f32)
            .collect();
        let net = NTuplePolicyNet::from_weights(geom, weights);
        let s = state((1 << 0) | (1 << 2), (1 << 1) | (1 << 7), Player::Black);
        let got = net.all_logits(&s);
        let expected = [
            -0.008340000036696438,
            -0.008339999985764734,
            -0.008339999963936862,
            -0.00834000005852431,
            -0.008340000051248353,
            -0.008339999956660904,
            -0.00833999997121282,
            -0.00834000005852431,
        ];
        for (actual, expected) in got.iter().take(8).zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        }
    }

    #[test]
    fn pass_logit_is_the_mean_square_logit() {
        let geom = tiny_geom();
        let n = geom.n_weights() * SQUARES;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0011).cos()).collect();
        let mut net = NTuplePolicyNet::from_weights(geom, weights);
        let s = state(1 << 0, 1 << 9, Player::Black);
        let all = net.all_logits(&s);
        let want = all.iter().sum::<f64>() / SQUARES as f64;
        let got = net.logits(&s, &[Move::PASS]);
        assert_eq!(got, vec![want]);
    }
}
