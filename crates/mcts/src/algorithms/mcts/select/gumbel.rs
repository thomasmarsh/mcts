//! Deterministic interior completed-Q selection for Gumbel AlphaZero.

use rand::rngs::SmallRng;

use super::{SelectContext, SelectPolicy};
use crate::algorithms::mcts::config;
use crate::algorithms::mcts::gumbel::{completed_q, improved_policy, GumbelConfig};
use crate::algorithms::mcts::index::Id;
use crate::algorithms::mcts::node::ChildArray;
use crate::algorithms::mcts::BackpropFlags;
use crate::game::Game;

#[derive(Clone)]
pub struct GumbelCompletedQ {
    cfg: GumbelConfig,
}

impl GumbelCompletedQ {
    pub fn with_config(cfg: GumbelConfig) -> Self {
        Self { cfg }
    }

    /// Choose the increment that minimizes squared distance between the
    /// resulting empirical visit distribution and the improved policy.
    /// Exact ties retain action order, making the rule deterministic.
    pub fn visit_matching_index(
        logits: &[f64],
        visits: &[u32],
        q_values: &[f64],
        root_value: f64,
        cfg: &GumbelConfig,
    ) -> usize {
        assert_eq!(logits.len(), visits.len());
        assert_eq!(visits.len(), q_values.len());
        let completed = completed_q(root_value, visits, q_values);
        let target = improved_policy(logits, visits, &completed, cfg);
        let total = visits.iter().sum::<u32>() as f64 + 1.0;
        (0..visits.len())
            .min_by(|&a, &b| {
                let distance = |chosen: usize| {
                    visits
                        .iter()
                        .enumerate()
                        .map(|(i, &visits)| {
                            let empirical = (visits + u32::from(i == chosen)) as f64 / total;
                            let delta = empirical - target[i] as f64;
                            delta * delta
                        })
                        .sum::<f64>()
                };
                distance(a).partial_cmp(&distance(b)).unwrap()
            })
            .expect("completed-Q selection needs a legal action")
    }
}

impl Default for GumbelCompletedQ {
    fn default() -> Self {
        Self::with_config(GumbelConfig::default())
    }
}

impl<G: Game> SelectPolicy<G> for GumbelCompletedQ {
    type Score = usize;
    type Aux = ();

    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, _: &mut SmallRng) -> usize {
        let children = ctx.index.get(ctx.stack.current_id()).children();
        let visits = (0..children.len()).map(|i| children.num_visits(i)).collect::<Vec<_>>();
        let q_values = (0..children.len())
            .map(|i| children.expected_score(i, ctx.player))
            .collect::<Vec<_>>();
        // Interior nodes do not own an evaluator result. Their accumulated
        // child Q values are evidence; an unvisited node therefore starts
        // from the neutral game value rather than `QInit`, whose exploration
        // sentinel is not a value-model prediction.
        let root_value = 0.0;
        Self::visit_matching_index(children.policy_logits(), &visits, &q_values, root_value, &self.cfg)
    }

    fn score_child(
        &self,
        _: &SelectContext<'_, G>,
        _: Id,
        _: &ChildArray<G::A>,
        idx: usize,
        _: Self::Aux,
    ) -> Self::Score {
        usize::MAX - idx
    }

    fn unvisited_value(&self, _: &SelectContext<'_, G>, _: Self::Aux) -> Self::Score { 0 }

    fn backprop_flags(&self) -> BackpropFlags { BackpropFlags(0) }

    fn requirements(&self) -> config::Requirements { config::Requirements::default() }

    fn label(&self) -> String { "gumbel_completed_q".into() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visit_matching_is_deterministic_and_tracks_the_target() {
        let cfg = GumbelConfig { c_visit: 0.0, c_scale: 1.0, rescale_q: false, ..GumbelConfig::default() };
        assert_eq!(GumbelCompletedQ::visit_matching_index(&[0.0; 3], &[4, 1, 0], &[0.0; 3], 0.0, &cfg), 2);
        assert_eq!(GumbelCompletedQ::visit_matching_index(&[0.0; 3], &[0, 0, 0], &[0.0; 3], 0.0, &cfg), 0);
    }

    #[test]
    fn completion_and_action_order_follow_the_cached_logits() {
        let cfg = GumbelConfig { c_visit: 0.0, c_scale: 1.0, rescale_q: false, ..GumbelConfig::default() };
        let first = GumbelCompletedQ::visit_matching_index(&[3.0, -3.0], &[0, 0], &[9.0, 9.0], 0.25, &cfg);
        let mirrored = GumbelCompletedQ::visit_matching_index(&[-3.0, 3.0], &[0, 0], &[9.0, 9.0], 0.25, &cfg);
        assert_eq!(first, 0);
        assert_eq!(mirrored, 1);
    }
}
