//! Compact, versioned Connect Four convolutional value-and-policy inference.

//! `C4CNN001` is a two-plane 6x7 network with a 16-channel stem, two residual
//! blocks, and separate value and absolute-column policy heads.  Inference
//! averages literal and reflected boards, making both outputs mirror-equivariant
//! independently of the learned parameters.

use std::path::Path;

use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};

use crate::{Move, Player, Standard, State};

const ROWS: usize = 6;
const COLS: usize = 7;
const CHANNELS: usize = 16;
const BLOCKS: usize = 2;
const VALUE_HIDDEN: usize = 32;
const POLICY_OUTPUTS: usize = 7;
const MAGIC: &[u8; 8] = b"C4CNN001";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 44;
pub const CNN_WEIGHTS: usize = 11_328;

#[derive(Clone, Debug)]
pub struct CnnValuePolicyNet {
    weights: Vec<f32>,
}

impl Default for CnnValuePolicyNet {
    fn default() -> Self {
        Self::from_weights(vec![0.0; CNN_WEIGHTS])
    }
}

impl CnnValuePolicyNet {
    pub fn from_weights(weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), CNN_WEIGHTS, "C4CNN001 needs {CNN_WEIGHTS} weights");
        Self { weights }
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != HEADER_BYTES + CNN_WEIGHTS * 4 || !bytes.starts_with(MAGIC) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid C4CNN001 byte length or magic"));
        }
        let header: [u32; 9] = std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap())
        });
        if header != [VERSION, ROWS as u32, COLS as u32, 2, CHANNELS as u32, BLOCKS as u32, VALUE_HIDDEN as u32, POLICY_OUTPUTS as u32, CNN_WEIGHTS as u32] {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported C4CNN001 layout"));
        }
        Ok(Self::from_weights(bytes[HEADER_BYTES..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()))
    }

    pub fn weights(&self) -> &[f32] { &self.weights }

    fn input(state: &State<ROWS, COLS>, mirrored: bool) -> [f32; 2 * ROWS * COLS] {
        std::array::from_fn(|i| {
            let plane = i / (ROWS * COLS);
            let literal = i % (ROWS * COLS);
            let cell = if mirrored { (literal / COLS) * COLS + COLS - 1 - literal % COLS } else { literal };
            let mover = state.turn();
            let own = (state.black().get_index(cell) && mover == Player::Black)
                || (state.white().get_index(cell) && mover == Player::White);
            if (plane == 0 && own) || (plane == 1 && !own && (state.black().get_index(cell) || state.white().get_index(cell))) { 1.0 } else { 0.0 }
        })
    }

    fn conv3(input: &[f32], out: &mut [f32], weights: &[f32], bias: &[f32], in_channels: usize) {
        for out_ch in 0..CHANNELS {
            for row in 0..ROWS {
                for col in 0..COLS {
                    let mut sum = bias[out_ch];
                    for in_ch in 0..in_channels {
                        for kr in 0..3 {
                            for kc in 0..3 {
                                let rr = row as isize + kr as isize - 1;
                                let cc = col as isize + kc as isize - 1;
                                if (0..ROWS as isize).contains(&rr) && (0..COLS as isize).contains(&cc) {
                                    sum += input[(in_ch * ROWS + rr as usize) * COLS + cc as usize]
                                        * weights[((out_ch * in_channels + in_ch) * 3 + kr) * 3 + kc];
                                }
                            }
                        }
                    }
                    out[(out_ch * ROWS + row) * COLS + col] = sum;
                }
            }
        }
    }

    fn raw(&self, input: &[f32; 84]) -> (f32, [f32; POLICY_OUTPUTS]) {
        let mut at = 0;
        let mut x = vec![0.0; CHANNELS * ROWS * COLS];
        self::CnnValuePolicyNet::conv3(input, &mut x, &self.weights[at..at + CHANNELS * 2 * 9], &self.weights[at + CHANNELS * 2 * 9..at + CHANNELS * 2 * 9 + CHANNELS], 2);
        x.iter_mut().for_each(|v| *v = v.max(0.0));
        at += CHANNELS * 2 * 9 + CHANNELS;
        for _ in 0..BLOCKS {
            let residual = x.clone();
            let mut y = vec![0.0; x.len()];
            self::CnnValuePolicyNet::conv3(&x, &mut y, &self.weights[at..at + CHANNELS * CHANNELS * 9], &self.weights[at + CHANNELS * CHANNELS * 9..at + CHANNELS * CHANNELS * 9 + CHANNELS], CHANNELS);
            y.iter_mut().for_each(|v| *v = v.max(0.0));
            at += CHANNELS * CHANNELS * 9 + CHANNELS;
            self::CnnValuePolicyNet::conv3(&y, &mut x, &self.weights[at..at + CHANNELS * CHANNELS * 9], &self.weights[at + CHANNELS * CHANNELS * 9..at + CHANNELS * CHANNELS * 9 + CHANNELS], CHANNELS);
            for (v, skip) in x.iter_mut().zip(residual) { *v = (*v + skip).max(0.0); }
            at += CHANNELS * CHANNELS * 9 + CHANNELS;
        }
        let value_conv = &self.weights[at..at + CHANNELS];
        let value_bias = self.weights[at + CHANNELS];
        at += CHANNELS + 1;
        let value_features: Vec<f32> = (0..ROWS * COLS).map(|cell| (value_bias + (0..CHANNELS).map(|ch| x[ch * ROWS * COLS + cell] * value_conv[ch]).sum::<f32>()).max(0.0)).collect();
        let value_w1 = &self.weights[at..at + ROWS * COLS * VALUE_HIDDEN]; at += ROWS * COLS * VALUE_HIDDEN;
        let value_b1 = &self.weights[at..at + VALUE_HIDDEN]; at += VALUE_HIDDEN;
        let hidden: Vec<f32> = (0..VALUE_HIDDEN).map(|unit| (value_b1[unit] + (0..ROWS * COLS).map(|cell| value_features[cell] * value_w1[cell * VALUE_HIDDEN + unit]).sum::<f32>()).max(0.0)).collect();
        let value_w2 = &self.weights[at..at + VALUE_HIDDEN]; at += VALUE_HIDDEN;
        let value = (self.weights[at] + hidden.iter().zip(value_w2).map(|(a,b)| a*b).sum::<f32>()).tanh(); at += 1;
        let policy_conv = &self.weights[at..at + CHANNELS];
        let policy_bias = self.weights[at + CHANNELS]; at += CHANNELS + 1;
        let policy_features: Vec<f32> = (0..ROWS * COLS).map(|cell| (policy_bias + (0..CHANNELS).map(|ch| x[ch * ROWS * COLS + cell] * policy_conv[ch]).sum::<f32>()).max(0.0)).collect();
        let policy_w = &self.weights[at..at + ROWS * COLS * POLICY_OUTPUTS]; at += ROWS * COLS * POLICY_OUTPUTS;
        let policy_b = &self.weights[at..at + POLICY_OUTPUTS]; at += POLICY_OUTPUTS;
        debug_assert_eq!(at, CNN_WEIGHTS);
        let logits = std::array::from_fn(|col| policy_b[col] + (0..ROWS * COLS).map(|cell| policy_features[cell] * policy_w[cell * POLICY_OUTPUTS + col]).sum::<f32>());
        (value, logits)
    }

    pub fn value_and_logits(&self, state: &State<ROWS, COLS>) -> (f32, [f64; POLICY_OUTPUTS]) {
        let (value, literal) = self.raw(&Self::input(state, false));
        let (mirror_value, reflected) = self.raw(&Self::input(state, true));
        (0.5 * (value + mirror_value), std::array::from_fn(|col| 0.5 * (literal[col] as f64 + reflected[COLS - 1 - col] as f64)))
    }

    pub fn value(&self, state: &State<ROWS, COLS>) -> f32 { self.value_and_logits(state).0 }
    pub fn all_logits(&self, state: &State<ROWS, COLS>) -> [f64; POLICY_OUTPUTS] { self.value_and_logits(state).1 }
}

impl Evaluator<Standard> for CnnValuePolicyNet {
    fn evaluate(&self, state: &State<ROWS, COLS>) -> Score { (self.value(state) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score }
}
impl PolicyLogits<Standard> for CnnValuePolicyNet {
    fn logits(&mut self, state: &State<ROWS, COLS>, actions: &[Move]) -> Vec<f64> {
        let all = self.all_logits(state);
        actions.iter().map(|action| all[action.0 as usize]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::game::Game;

    fn fixture() -> State<ROWS, COLS> {
        let mut state = State::default();
        for col in [0, 1, 2, 1, 3] { state = Standard::apply(state, &Move(col)); }
        state
    }
    #[test]
    fn layout_is_versioned_and_validated() {
        let path = std::env::temp_dir().join(format!("mcts-c4-cnn-{}", std::process::id()));
        let mut bytes = Vec::from(*MAGIC);
        for n in [VERSION, ROWS as u32, COLS as u32, 2, CHANNELS as u32, BLOCKS as u32, VALUE_HIDDEN as u32, POLICY_OUTPUTS as u32, CNN_WEIGHTS as u32] { bytes.extend(n.to_le_bytes()); }
        bytes.extend(std::iter::repeat_n(0u8, CNN_WEIGHTS * 4));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(CnnValuePolicyNet::load(&path).unwrap().weights().len(), CNN_WEIGHTS);
        bytes[8] = 2; std::fs::write(&path, bytes).unwrap();
        assert!(CnnValuePolicyNet::load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn side_to_move_planes_and_mirror_equivariance_hold() {
        let weights = (0..CNN_WEIGHTS).map(|i| (i as f32 * 0.001).sin()).collect();
        let net = CnnValuePolicyNet::from_weights(weights);
        let state = fixture();
        let mut mirrored = State::default();
        for col in [6, 5, 4, 5, 3] { mirrored = Standard::apply(mirrored, &Move(col)); }
        let (value, logits) = net.value_and_logits(&state);
        let (mirror_value, mirror_logits) = net.value_and_logits(&mirrored);
        assert!((value - mirror_value).abs() < 1e-6);
        for col in 0..COLS { assert!((logits[col] - mirror_logits[COLS - 1 - col]).abs() < 1e-6); }
        let reversed = State::from_parts(state.black(), state.white(), if state.turn() == Player::Black { Player::White } else { Player::Black }, false);
        assert_ne!(CnnValuePolicyNet::input(&state, false), CnnValuePolicyNet::input(&reversed, false));
    }
    #[test]
    fn value_and_policy_match_python_reference_fixture() {
        let weights = (0..CNN_WEIGHTS).map(|i| (i as f32 - CNN_WEIGHTS as f32 / 2.0) * 1e-6).collect();
        let net = CnnValuePolicyNet::from_weights(weights);
        let mut black = crate::BitBoard::<ROWS, COLS>::EMPTY;
        black.set_index(0); black.set_index(2); black.set_index(8);
        let mut white = crate::BitBoard::<ROWS, COLS>::EMPTY;
        white.set_index(1); white.set_index(7);
        let (value, logits) = net.value_and_logits(&State::from_parts(black, white, Player::Black, false));
        assert!((value - 0.006_387_100_6).abs() < 1e-7);
        let expected = [0.006_988_344_7, 0.006_988_343_8, 0.006_988_344_7, 0.006_988_344_2, 0.006_988_344_7, 0.006_988_343_8, 0.006_988_344_7];
        for (actual, expected) in logits.into_iter().zip(expected) { assert!((actual - expected).abs() < 1e-8); }
    }
    #[test]
    fn policy_returns_only_legal_actions_without_masked_columns() {
        let mut state = State::default();
        for _ in 0..6 { state = Standard::apply(state, &Move(0)); }
        let mut actions = Vec::new(); Standard::generate_actions(&state, &mut actions);
        let mut net = CnnValuePolicyNet::default();
        assert_eq!(net.logits(&state, &actions), vec![0.0; 6]);
        assert!(actions.iter().all(|action| action.0 != 0));
    }
}
