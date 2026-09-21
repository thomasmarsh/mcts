//! The batched search oracle: applies moves and scores the resulting positions with the CNN.
//! Every leaf of a simulation round is evaluated in GPU calls of at most `chunk_size` positions,
//! each in one random board orientation (one forward pass per leaf; averaging over the search
//! tree stands in for an orientation ensemble).

use std::sync::Mutex;

use grid_cnn::{cell_map, transform_planes, Net, SYMMETRIES};
use mcts::game::Game;
use mcts_batch::{EnvOracle, StepOutput, TransitionOutput};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::encode::{analyse, move_from_id, num_actions, IN_PLANES};
use crate::sized::{SizedGonnect, SizedState};

pub struct GonnectOracle<const N: usize> {
    net: Net,
    chunk_size: usize,
    maps: Vec<Vec<usize>>,
    orientation: Mutex<Option<SmallRng>>,
}

impl<const N: usize> GonnectOracle<N> {
    /// `orientation_seed`: `Some(seed)` draws one random board orientation per evaluated
    /// position, `None` always uses the identity orientation.
    pub fn new(net: Net, chunk_size: usize, orientation_seed: Option<u64>) -> Self {
        let g = net.geometry();
        assert_eq!(
            (g.size, g.in_planes, g.policy_out),
            (N, IN_PLANES, num_actions(N)),
            "net geometry is not Gonnect {N}x{N}"
        );
        assert!(chunk_size > 0);
        GonnectOracle {
            net,
            chunk_size,
            maps: (0..SYMMETRIES).map(|s| cell_map(N, s)).collect(),
            orientation: Mutex::new(orientation_seed.map(SmallRng::seed_from_u64)),
        }
    }

    /// Restart the orientation stream (a no-op for an identity-orientation oracle).
    pub fn reseed(&self, seed: u64) {
        if let Some(rng) = self.orientation.lock().unwrap().as_mut() {
            *rng = SmallRng::seed_from_u64(seed);
        }
    }

    fn draw_orientations(&self, n: usize) -> Vec<usize> {
        match self.orientation.lock().unwrap().as_mut() {
            Some(rng) => (0..n).map(|_| rng.gen_range(0..SYMMETRIES)).collect(),
            None => vec![0; n],
        }
    }

    /// Terminal positions are worth exactly `+1` to the side to move (the winner is always the
    /// side to move there); the rest are scored by the net.
    fn evaluate(&self, states: &[SizedState<N>]) -> StepOutput<SizedState<N>> {
        let a = num_actions(N);
        let n = states.len();
        let mut terminal = vec![false; n];
        let mut valid_actions = vec![false; n * a];
        let mut policy_prior = vec![0.0f32; n * a];
        let mut value_prior = vec![1.0f32; n];

        let analysed: Vec<Option<_>> = states
            .par_iter()
            .map(|s| {
                if SizedGonnect::<N>::is_terminal(s) {
                    None
                } else {
                    Some(analyse(&s.0).0)
                }
            })
            .collect();
        let live: Vec<usize> = (0..n).filter(|&i| analysed[i].is_some()).collect();
        for i in 0..n {
            terminal[i] = analysed[i].is_none();
        }
        let syms = self.draw_orientations(live.len());
        let planes: Vec<Vec<f32>> = live
            .par_iter()
            .zip(&syms)
            .map(|(&i, &sym)| transform_planes(&analysed[i].unwrap().planes(N), N, IN_PLANES, sym))
            .collect();

        for (chunk_no, chunk) in live.chunks(self.chunk_size).enumerate() {
            let first = chunk_no * self.chunk_size;
            let input: Vec<f32> = planes[first..first + chunk.len()].concat();
            let out = self.net.forward(&input, chunk.len());
            for (k, &i) in chunk.iter().enumerate() {
                let map = &self.maps[syms[first + k]];
                let logits = &out.logits[k * a..(k + 1) * a];
                let logit = |id: usize| {
                    if id < N * N {
                        logits[map[id]]
                    } else {
                        logits[id]
                    }
                };
                let ids = analysed[i].unwrap().legal_ids(N);
                let max = ids
                    .iter()
                    .map(|&id| logit(id as usize))
                    .fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = ids.iter().map(|&id| (logit(id as usize) - max).exp()).sum();
                for &id in &ids {
                    valid_actions[i * a + id as usize] = true;
                    policy_prior[i * a + id as usize] = (logit(id as usize) - max).exp() / total;
                }
                value_prior[i] = out.values[k];
            }
        }
        StepOutput {
            states: states.to_vec(),
            terminal,
            valid_actions,
            policy_prior,
            value_prior,
        }
    }
}

impl<const N: usize> EnvOracle<SizedState<N>> for GonnectOracle<N> {
    fn num_actions(&self) -> usize {
        num_actions(N)
    }

    fn init(&self, envs: &[SizedState<N>]) -> StepOutput<SizedState<N>> {
        self.evaluate(envs)
    }

    fn transition(
        &self,
        states: &[SizedState<N>],
        actions: &[u16],
    ) -> TransitionOutput<SizedState<N>> {
        let next: Vec<SizedState<N>> = states
            .par_iter()
            .zip(actions)
            .map(|(s, &id)| SizedGonnect::<N>::apply(s.clone(), &move_from_id(&s.0, id)))
            .collect();
        let player_switched = states
            .iter()
            .zip(&next)
            .map(|(s, t)| s.0.turn() != t.0.turn())
            .collect();
        let step = self.evaluate(&next);
        TransitionOutput {
            step,
            rewards: vec![0.0; states.len()],
            player_switched,
        }
    }
}
