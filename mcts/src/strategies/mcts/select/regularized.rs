use super::super::index::Id;
use super::super::node::ChildArray;
use super::is_proven_loss;
use super::proven_exact_value;
use super::SelectContext;
use super::SelectStrategy;
use crate::game::Game;

use rand::rngs::SmallRng;

////////////////////////////////////////////////////////////////////////////////
// MENTS / E2W (Xiao, Huang, Weinman, Müller, "Maximum Entropy Monte-Carlo
// Planning", NeurIPS 2019). The regularised-policy family; B3's Grill ACT
// lands in this file too.

/// The E2W (Empirical Exponential Weight) stochastic tree policy: descent
/// samples a child from `softmax(Q_soft(a) / τ)` mixed with a uniform
/// exploration floor `ε`, rather than an argmax over an independent
/// per-child UCB score. `Q_soft` is the mellowmax soft-Bellman value the
/// paired [`backprop::SoftmaxBackprop`](super::super::backprop::SoftmaxBackprop)
/// writes back into each child; `Ments` sets
/// `Requirements::needs_softmax_value` so `SearchConfig::validate` rejects
/// the pairing with any other backup.
///
/// Deferred (noted so the limits are explicit):
/// - an explicit `+ ln π_prior(a)` logit term -- `prior::PriorStrategy`
///   seeds pseudo-visits, not a readable per-action probability, so this
///   needs a `node.rs` storage change. The prior still influences the draw
///   indirectly via the seeded child's `exploitation_score`. The
///   uniform-implicit-prior form here is the network-free target anyway.
/// - a decaying `ε` schedule (the paper's
///   `λ_s = ε·|A(s)| / log(Σn + 1)`) -- fixed `ε` is measured first.
/// - a MENTS-aware final action -- `strategy::Ments` keeps `RobustChild`.
#[derive(Clone)]
pub struct Ments {
    /// E2W softmax temperature -- MUST equal the paired
    /// `SoftmaxBackprop::tau` (the `ments` family catalog row wires both
    /// from one tuner field).
    pub tau: f64,
    /// E2W uniform exploration floor (the paper's fixed-`ε` form).
    pub epsilon: f64,
}

impl Default for Ments {
    fn default() -> Self {
        Self {
            tau: 1.0,
            epsilon: 0.1,
        }
    }
}

impl Ments {
    pub fn new(tau: f64, epsilon: f64) -> Self {
        Self { tau, epsilon }
    }
}

/// The E2W mixture `(1 - ε)·softmax(logits) + ε·uniform`, as normalised
/// probabilities. Numerically-stable softmax (max-shifted). No proven-loss
/// handling -- that is `best_child`'s job, since it needs a `SelectContext`.
/// Factored out for a pure unit test of the mixture arithmetic.
pub(crate) fn e2w_weights(logits: &[f64], epsilon: f64) -> Vec<f64> {
    let k = logits.len();
    debug_assert!(k > 0);
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    let inv_k = 1.0 / k as f64;
    exps.iter()
        .map(|&e| (1.0 - epsilon) * (e / z) + epsilon * inv_k)
        .collect()
}

impl<G: Game> SelectStrategy<G> for Ments {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    /// Only reached if some future caller bypasses `best_child`; kept total
    /// (the raw logit) rather than `unreachable!`.
    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        _: Self::Aux,
    ) -> f64 {
        ctx.child_snapshot(child_id, children, idx)
            .exploitation_score()
            / self.tau
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
            / self.tau
    }

    fn requirements(&self) -> super::super::config::Requirements {
        super::super::config::Requirements {
            needs_softmax_value: true,
            ..super::super::config::Requirements::none()
        }
    }

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.stack.current_id());
        let children = current.children();
        let k = children.len();

        let unvisited = <Self as SelectStrategy<G>>::unvisited_value(self, ctx, ());
        let mut logits = Vec::with_capacity(k);
        let mut proven_loss = Vec::with_capacity(k);
        for idx in 0..k {
            proven_loss.push(is_proven_loss(ctx, children, idx));
            let q_over_tau = match proven_exact_value(ctx, children, idx) {
                Some(v) => v / self.tau,
                None => match children.node_id(idx) {
                    Some(cid) => {
                        ctx.child_snapshot(cid, children, idx).exploitation_score() / self.tau
                    }
                    None if children.num_visits(idx) > 0 => {
                        ctx.child_snapshot(ctx.stack.current_id(), children, idx)
                            .exploitation_score()
                            / self.tau
                    }
                    None => unvisited,
                },
            };
            logits.push(q_over_tau);
        }

        let mix = e2w_weights(&logits, self.epsilon);
        let all_dead = proven_loss.iter().all(|&x| x);
        let weights: Vec<f32> = (0..k)
            .map(|idx| {
                if proven_loss[idx] && !all_dead {
                    f32::MIN_POSITIVE
                } else {
                    (mix[idx] as f32).max(f32::MIN_POSITIVE)
                }
            })
            .collect();

        use weighted_rand::builder::*;
        WalkerTableBuilder::new(&weights).build().next_rng(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn e2w_weights_matches_softmax() {
        let logits = [0.0, 2.0f64.ln(), 4.0f64.ln()];

        let w = e2w_weights(&logits, 0.0);
        for (got, exp) in w.iter().zip([1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0]) {
            assert!((got - exp).abs() < 1e-9, "{got} vs {exp}");
        }

        let w = e2w_weights(&logits, 1.0);
        for got in w {
            assert!((got - 1.0 / 3.0).abs() < 1e-9);
        }

        let w = e2w_weights(&logits, 0.5);
        let softmax = [1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0];
        for (got, s) in w.iter().zip(softmax) {
            let exp = 0.5 * s + 0.5 * (1.0 / 3.0);
            assert!((got - exp).abs() < 1e-9, "{got} vs {exp}");
        }
    }

    #[test]
    fn e2w_draw_histogram() {
        use weighted_rand::builder::*;

        let weights = e2w_weights(&[0.0, 2.0f64.ln(), 4.0f64.ln()], 0.1);
        let w32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        let table = WalkerTableBuilder::new(&w32).build();
        let mut rng = SmallRng::seed_from_u64(0xE2E2);

        let n = 20_000;
        let mut counts = [0u32; 3];
        for _ in 0..n {
            counts[table.next_rng(&mut rng)] += 1;
        }
        for (c, w) in counts.iter().zip(&weights) {
            let freq = *c as f64 / n as f64;
            assert!((freq - w).abs() < 0.02, "freq {freq} vs weight {w}");
        }
    }
}
