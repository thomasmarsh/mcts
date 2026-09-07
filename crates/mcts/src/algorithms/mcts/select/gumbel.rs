//! Deterministic interior completed-Q selection for Gumbel AlphaZero.

use rand::rngs::SmallRng;

use super::{SelectContext, SelectPolicy};
use crate::algorithms::mcts::config;
use crate::algorithms::mcts::gumbel::{completed_q, improved_policy, GumbelConfig};
use crate::algorithms::mcts::index::Id;
use crate::algorithms::mcts::node::ChildArray;
use crate::algorithms::mcts::BackpropFlags;
use crate::game::{Game, PlayerIndex};

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
        let completed = completed_q(root_value, logits, visits, q_values);
        Self::visit_matching_completed(logits, visits, &completed, cfg)
    }

    fn visit_matching_completed(
        logits: &[f64],
        visits: &[u32],
        completed: &[f64],
        cfg: &GumbelConfig,
    ) -> usize {
        let target = improved_policy(logits, visits, completed, cfg);
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

    fn completed_q_for_node(
        raw_evaluator_value: Option<f64>,
        logits: &[f64],
        visits: &[u32],
        q_values: &[f64],
    ) -> Vec<f64> {
        // A missing value is intentionally distinct from an evaluated draw:
        // only profiles without an evaluator use neutral completion.
        completed_q(raw_evaluator_value.unwrap_or(0.0), logits, visits, q_values)
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
        debug_assert_eq!(G::player_to_move(ctx.state).to_index(), ctx.player);
        let visits = (0..children.len())
            .map(|i| children.num_visits(i))
            .collect::<Vec<_>>();
        let q_values = (0..children.len())
            .map(|i| children.expected_score(i, ctx.player))
            .collect::<Vec<_>>();
        // Both the cached evaluator value and edge Q values are in this
        // node's mover perspective. A profile without an evaluator is the
        // only case that deliberately falls back to a neutral completion.
        let completed = Self::completed_q_for_node(
            children.raw_evaluator_value(),
            children.policy_logits(),
            &visits,
            &q_values,
        );
        Self::visit_matching_completed(children.policy_logits(), &visits, &completed, &self.cfg)
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

    fn unvisited_value(&self, _: &SelectContext<'_, G>, _: Self::Aux) -> Self::Score {
        0
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(0)
    }

    fn requirements(&self) -> config::Requirements {
        config::Requirements::default()
    }

    fn label(&self) -> String {
        "gumbel_completed_q".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visit_matching_is_deterministic_and_tracks_the_target() {
        let cfg = GumbelConfig {
            c_visit: 0.0,
            c_scale: 1.0,
            rescale_q: false,
            ..GumbelConfig::default()
        };
        assert_eq!(
            GumbelCompletedQ::visit_matching_index(&[0.0; 3], &[4, 1, 0], &[0.0; 3], 0.0, &cfg),
            2
        );
        assert_eq!(
            GumbelCompletedQ::visit_matching_index(&[0.0; 3], &[0, 0, 0], &[0.0; 3], 0.0, &cfg),
            0
        );
    }

    #[test]
    fn completion_and_action_order_follow_the_cached_logits() {
        let cfg = GumbelConfig {
            c_visit: 0.0,
            c_scale: 1.0,
            rescale_q: false,
            ..GumbelConfig::default()
        };
        let first =
            GumbelCompletedQ::visit_matching_index(&[3.0, -3.0], &[0, 0], &[9.0, 9.0], 0.25, &cfg);
        let mirrored =
            GumbelCompletedQ::visit_matching_index(&[-3.0, 3.0], &[0, 0], &[9.0, 9.0], 0.25, &cfg);
        assert_eq!(first, 0);
        assert_eq!(mirrored, 1);
    }

    #[test]
    fn raw_node_value_overrides_neutral_completion_for_unvisited_children() {
        let logits = [0.0, 0.0, 0.0];
        let visits = [4, 0, 2];
        let q_values = [0.5, 99.0, -0.5];
        let from_evaluator =
            GumbelCompletedQ::completed_q_for_node(Some(0.75), &logits, &visits, &q_values);
        let unavailable = GumbelCompletedQ::completed_q_for_node(None, &logits, &visits, &q_values);
        assert_eq!(from_evaluator, vec![0.5, 0.75 / 7.0, -0.5]);
        assert_eq!(unavailable, vec![0.5, 0.0, -0.5]);
    }

    #[test]
    fn node_value_and_child_qs_keep_the_node_mover_sign() {
        let completed =
            GumbelCompletedQ::completed_q_for_node(Some(-0.6), &[0.0, 0.0], &[3, 0], &[-0.2, 99.0]);
        // The visited Q and the completion value are both negative for this
        // node's mover, so their mix stays negative rather than negating the
        // child Q a second time.
        assert_eq!(completed[0], -0.2);
        assert!((completed[1] + 0.3).abs() < 1e-12);
    }
}
