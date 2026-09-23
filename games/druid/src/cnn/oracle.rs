//! The batched search oracle: applies moves and scores the resulting positions with the CNN.
//! Every leaf of a simulation round is evaluated in GPU calls of at most `chunk_size` positions.
//!
//! Positions are always shown to the net in their real orientation: Druid is not D4-invariant
//! (Black joins top-bottom, White left-right, and a lintel's cell id is its anchor cell), so
//! Gonnect's random board orientations do not carry over.

use grid_cnn::Net;
use mcts::game::Game;
use mcts_batch::{EnvOracle, StepOutput, TransitionOutput};
use rayon::prelude::*;

use super::encode::{action_id, legal_moves, move_from_id, num_actions, planes, IN_PLANES};
use crate::{DruidSplit, HashedState, Move};

pub struct DruidOracle<const N: usize> {
    net: Net,
    chunk_size: usize,
}

/// The mover's outcome at a terminal position: `+1` if the side to move won, `-1` if it lost
/// (the usual case right after the winning placement, which passes the turn), `0` for a draw.
pub fn terminal_value(state: &HashedState) -> f32 {
    match DruidSplit::winner(state) {
        Some(w) if w == DruidSplit::player_to_move(state) => 1.0,
        Some(_) => -1.0,
        None => 0.0,
    }
}

impl<const N: usize> DruidOracle<N> {
    pub fn new(net: Net, chunk_size: usize) -> Self {
        let g = net.geometry();
        assert_eq!(
            (g.size, g.in_planes, g.policy_out),
            (N, IN_PLANES, num_actions(N)),
            "net geometry is not Druid {N}x{N}"
        );
        assert!(chunk_size > 0);
        DruidOracle { net, chunk_size }
    }

    fn evaluate(&self, states: &[HashedState]) -> StepOutput<HashedState> {
        let a = num_actions(N);
        let n = states.len();
        let mut valid_actions = vec![false; n * a];
        let mut policy_prior = vec![0.0f32; n * a];
        let mut value_prior = vec![0.0f32; n];

        let analysed: Vec<Option<(Vec<u16>, Vec<f32>)>> = states
            .par_iter()
            .map(|s| {
                (!DruidSplit::is_terminal(s)).then(|| (legal_moves(s).1, planes(s)))
            })
            .collect();
        let terminal: Vec<bool> = analysed.iter().map(Option::is_none).collect();
        for (i, s) in states.iter().enumerate() {
            if terminal[i] {
                value_prior[i] = terminal_value(s);
            }
        }
        let live: Vec<usize> = (0..n).filter(|&i| !terminal[i]).collect();

        for chunk in live.chunks(self.chunk_size) {
            let input: Vec<f32> = chunk
                .iter()
                .flat_map(|&i| analysed[i].as_ref().unwrap().1.iter().copied())
                .collect();
            let out = self.net.forward(&input, chunk.len());
            for (k, &i) in chunk.iter().enumerate() {
                let logits = &out.logits[k * a..(k + 1) * a];
                let ids = &analysed[i].as_ref().unwrap().0;
                let max = ids
                    .iter()
                    .map(|&id| logits[id as usize])
                    .fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = ids.iter().map(|&id| (logits[id as usize] - max).exp()).sum();
                for &id in ids {
                    valid_actions[i * a + id as usize] = true;
                    policy_prior[i * a + id as usize] =
                        (logits[id as usize] - max).exp() / total;
                }
                value_prior[i] = out.values[k];
            }
        }
        StepOutput { states: states.to_vec(), terminal, valid_actions, policy_prior, value_prior }
    }

    /// This state's value from one forward pass -- the single-tree counterpart of `evaluate`'s
    /// batched call, for `crates/mcts`'s `Evaluator` contract.
    pub(crate) fn single_value(&self, state: &HashedState) -> f32 {
        if DruidSplit::is_terminal(state) {
            return terminal_value(state);
        }
        self.net.forward(&planes(state), 1).values[0]
    }

    /// Raw per-action logits for `actions`, for `crates/mcts`'s `PolicyLogits` contract (which
    /// wants logits, not probabilities).
    pub(crate) fn single_logits(&self, state: &HashedState, actions: &[Move]) -> Vec<f64> {
        let out = self.net.forward(&planes(state), 1);
        actions
            .iter()
            .map(|m| f64::from(out.logits[action_id(m, N) as usize]))
            .collect()
    }
}

impl<const N: usize> EnvOracle<HashedState> for DruidOracle<N> {
    fn num_actions(&self) -> usize {
        num_actions(N)
    }

    fn init(&self, envs: &[HashedState]) -> StepOutput<HashedState> {
        self.evaluate(envs)
    }

    fn transition(&self, states: &[HashedState], actions: &[u16]) -> TransitionOutput<HashedState> {
        let next: Vec<HashedState> = states
            .par_iter()
            .zip(actions)
            .map(|(s, &id)| DruidSplit::apply(s.clone(), &move_from_id(id, N)))
            .collect();
        // A turn is several plies of the same player, so the search must not flip the value's
        // sign across those.
        let player_switched = states
            .iter()
            .zip(&next)
            .map(|(s, t)| s.state().player != t.state().player)
            .collect();
        let step = self.evaluate(&next);
        TransitionOutput { step, rewards: vec![0.0; states.len()], player_switched }
    }
}
