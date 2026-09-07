//! Seven-column n-tuple policy sidecar for Connect Four.

use std::path::Path;

use mcts::algorithms::mcts::policy::PolicyLogits;

use crate::valuenet::{NTupleValueNet, NT_WEIGHTS};
use crate::{Move, Standard, State};

pub const POLICY_WEIGHTS: usize = NT_WEIGHTS * 7;

#[derive(Clone, Debug)]
pub struct NTuplePolicyNet {
    weights: Vec<f32>,
}

impl Default for NTuplePolicyNet {
    fn default() -> Self {
        Self {
            weights: vec![0.0; POLICY_WEIGHTS],
        }
    }
}

impl NTuplePolicyNet {
    pub fn from_weights(weights: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            POLICY_WEIGHTS,
            "policy sidecar needs {POLICY_WEIGHTS} weights"
        );
        Self { weights }
    }
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != POLICY_WEIGHTS * 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "expected {} bytes ({POLICY_WEIGHTS} f32), got {}",
                    POLICY_WEIGHTS * 4,
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
    fn raw_logits(&self, state: &State<6, 7>, mirrored: bool) -> [f64; 7] {
        let mut out = [0.0; 7];
        for feat in NTupleValueNet::active_indices(state, mirrored) {
            let row = &self.weights[feat * 7..feat * 7 + 7];
            for col in 0..7 {
                out[col] += row[col] as f64;
            }
        }
        out
    }
    /// Symmetry-enforced absolute-column logits, before legal-action filtering.
    pub fn all_logits(&self, state: &State<6, 7>) -> [f64; 7] {
        let literal = self.raw_logits(state, false);
        let reflected = self.raw_logits(state, true);
        std::array::from_fn(|col| 0.5 * (literal[col] + reflected[6 - col]))
    }
}

impl PolicyLogits<Standard> for NTuplePolicyNet {
    fn logits(&mut self, state: &State<6, 7>, actions: &[Move]) -> Vec<f64> {
        let all = self.all_logits(state);
        actions
            .iter()
            .map(|action| all[action.0 as usize])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::algorithms::mcts::policy::PolicyLogits;
    use mcts::game::Game;

    #[test]
    fn zero_sidecar_is_uniform_over_legal_actions() {
        let state = State::default();
        let mut actions = Vec::new();
        Standard::generate_actions(&state, &mut actions);
        assert_eq!(
            NTuplePolicyNet::default().logits(&state, &actions),
            vec![0.0; 7]
        );
    }

    #[test]
    fn symmetry_average_is_mirror_equivariant() {
        let weights = (0..POLICY_WEIGHTS)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let net = NTuplePolicyNet::from_weights(weights);
        let mut state = State::default();
        let mut mirror = State::default();
        for col in [0, 3, 1, 5] {
            state = Standard::apply(state, &Move(col));
            mirror = Standard::apply(mirror, &Move(6 - col));
        }
        let a = net.all_logits(&state);
        let b = net.all_logits(&mirror);
        for col in 0..7 {
            assert!((a[col] - b[6 - col]).abs() < 1e-10);
        }
    }

    #[test]
    fn logits_match_python_reference_fixture() {
        let weights = (0..POLICY_WEIGHTS)
            .map(|i| (i as f64 - POLICY_WEIGHTS as f64 / 2.0) as f32 * 0.00002)
            .collect();
        let net = NTuplePolicyNet::from_weights(weights);
        let mut black = crate::BitBoard::<6, 7>::EMPTY;
        black.set_index(0);
        black.set_index(2);
        let mut white = crate::BitBoard::<6, 7>::EMPTY;
        white.set_index(1);
        white.set_index(7);
        let got = net.all_logits(&State::from_parts(
            black,
            white,
            crate::Player::Black,
            false,
        ));
        let expected = [
            -0.75586009,
            -0.75586015,
            -0.75585949,
            -0.75585961,
            -0.75585949,
            -0.75586015,
            -0.75586045,
        ];
        for (actual, expected) in got.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
