use super::super::index::Id;
use super::super::node::ChildArray;
use super::is_proven_loss;
use super::proven_exact_value;
use super::random_best_index_by;
use super::SelectContext;
use super::SelectPolicy;
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
/// - an explicit `+ ln π_prior(a)` logit term -- `prior::PriorPolicy`
///   seeds pseudo-visits, not a readable per-action probability, so this
///   needs a `node.rs` storage change. The prior still influences the draw
///   indirectly via the seeded child's `exploitation_score`. The
///   uniform-implicit-prior form here is the network-free target anyway.
/// - a decaying `ε` schedule (the paper's
///   `λ_s = ε·|A(s)| / log(Σn + 1)`) -- fixed `ε` is measured first.
/// - a MENTS-aware final action -- the MENTS pairing (`select::Ments` +
///   `backprop::SoftmaxBackprop`) keeps `RobustChild`.
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

impl<G: Game> SelectPolicy<G> for Ments {
    fn label(&self) -> String {
        "ments".into()
    }

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

        let unvisited = <Self as SelectPolicy<G>>::unvisited_value(self, ctx, ());
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

////////////////////////////////////////////////////////////////////////////////
// Grill, Valko, Munos et al., "Monte-Carlo Tree Search as Regularized Policy
// Optimization", ICML 2020 (arXiv 2007.12509).

const GRILL_LAMBDA_FLOOR: f64 = 1e-9;
const GRILL_ALPHA_EPS: f64 = 1e-12;
const GRILL_BISECTION_ITERS: usize = 48;

/// `λ_N = c · √N / (N + |A|)` (Grill et al. Eq. 4 constant). `n_total` is
/// `Σ_b n_b` (the parent visit count), `k` is `|A|`.
#[inline]
pub(crate) fn grill_lambda(c: f64, n_total: f64, k: usize) -> f64 {
    c * n_total.sqrt() / (n_total + k as f64)
}

/// The unique `α > max_a q` with `Σ_a λ·π_prior(a) / (α − q_a) = 1`, by
/// bracketed bisection. `priors` must be non-negative and sum to 1 (uniform
/// `1/k` this session). `lambda` must be `> 0` -- the caller shortcuts to
/// argmax-q when `lambda ≤ GRILL_LAMBDA_FLOOR`.
///
/// `g(α) = Σ λ·π_a / (α − q_a)` is strictly decreasing on `(max q, ∞)`,
/// `g(max q⁺) = +∞`, and `g(max q + λ) ≤ Σ π_a = 1` (each term ≤ `π_a`,
/// which is why the upper bracket needs `Σ π_a = 1`). So the root lies in
/// `(max q + ε, max q + λ]`; fixed-iteration bisection, no doubling.
#[inline]
pub(crate) fn grill_alpha(qs: &[f64], priors: &[f64], lambda: f64) -> f64 {
    debug_assert!(qs.len() == priors.len() && !qs.is_empty() && lambda > 0.0);
    debug_assert!((priors.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    let max_q = qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let g = |alpha: f64| -> f64 {
        qs.iter()
            .zip(priors)
            .map(|(&q, &p)| lambda * p / (alpha - q))
            .sum::<f64>()
    };
    let mut lo = max_q + GRILL_ALPHA_EPS;
    let mut hi = max_q + lambda + GRILL_ALPHA_EPS;
    for _ in 0..GRILL_BISECTION_ITERS {
        let mid = 0.5 * (lo + hi);
        if g(mid) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// `π̄(a) = λ·π_prior(a) / (α − q_a)`, renormalised so `Σ = 1` is exact for
/// the discrepancy rule (the bisection leaves a small residual).
#[inline]
pub(crate) fn grill_pi_bar(qs: &[f64], priors: &[f64], lambda: f64) -> Vec<f64> {
    let alpha = grill_alpha(qs, priors, lambda);
    let raw: Vec<f64> = qs
        .iter()
        .zip(priors)
        .map(|(&q, &p)| lambda * p / (alpha - q))
        .collect();
    let z: f64 = raw.iter().sum();
    raw.into_iter().map(|x| x / z).collect()
}

/// The paper's in-tree selection score `π̄(a) − n(a) / (1 + N)` per child --
/// the discrepancy between each child's `π̄` target and its current visit
/// fraction. Descent picks the argmax. Factored out for a pure unit test.
#[inline]
pub(crate) fn grill_discrepancy(pi_bar: &[f64], ns: &[f64]) -> Vec<f64> {
    let denom = 1.0 + ns.iter().sum::<f64>();
    pi_bar
        .iter()
        .zip(ns)
        .map(|(&p, &n)| p - n / denom)
        .collect()
}

/// Grill et al. ("MCTS as Regularized Policy Optimization", ICML 2020)
/// closed-form acting policy `π̄` used as the tree-descent selector.
/// `π̄(a) = λ_N · π_prior(a) / (α − Q(a))`, `λ_N = c·√N / (N + |A|)`, `α` the
/// unique scalar with `Σ_a π̄(a) = 1`. Descent picks
/// `argmax_a [π̄(a) − n(a)/(1+N)]` -- the child whose visit fraction most
/// undershoots its `π̄` target, so it explores even under the uniform
/// `π_prior` used here (see below).
///
/// A pure selection strategy: no backup change, no `Requirements`. Reads
/// `exploitation_score()` (Phase A's backup output if active, else the MC
/// mean), like every other selector -- so it inherits any active
/// `PowerMeanBackprop` / `TdBackprop` / `SoftmaxBackprop` for free, and
/// (unlike MENTS's backup half) has no MCGS caveat, since it writes nothing.
///
/// Deferred (noted so the limits are explicit):
/// - an explicit per-action `π_prior(a)` term -- `prior::PriorPolicy` seeds
///   pseudo-visits, not a readable probability, so this needs the `node.rs`
///   storage change `select::Ments` also waits on. `π_prior` is uniform
///   `1/|A|` here; the seeded prior still acts indirectly via `Q(a)`.
/// - a `π̄`-aware final action -- pairing `select::GrillAct` with the plain
///   `Classic` backup keeps `RobustChild`; the discrepancy rule already
///   drives visit counts toward
///   `π̄`. Measure a `π̄`-argmax final action first.
/// - "act" (sample `π̄`) vs "search" (this argmax-discrepancy rule) -- the
///   paper uses `π̄` for both; sampling `π̄` under a uniform prior collapses
///   to near-greedy at low `λ_N`, so descent uses the discrepancy argmax.
/// - the paper's richer `λ_N` form (`c_visit`/`c_scale` split with a
///   `log((N + c_base + 1)/c_base)` growth) -- one scalar `c ∈ [0, 3]` here,
///   matching every other selection family.
#[derive(Clone)]
pub struct GrillAct {
    pub exploration_constant: f64,
}

impl Default for GrillAct {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl GrillAct {
    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }
}

impl<G: Game> SelectPolicy<G> for GrillAct {
    fn label(&self) -> String {
        "grill_act".into()
    }

    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    /// Only reached if some future caller bypasses `best_child`; kept total
    /// (the raw `Q`) rather than `unreachable!`, same posture as `Ments`.
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
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.stack.current_id());
        let children = current.children();
        let k = children.len();

        let unvisited = <Self as SelectPolicy<G>>::unvisited_value(self, ctx, ());
        let mut qs = Vec::with_capacity(k);
        let mut ns = Vec::with_capacity(k);
        for idx in 0..k {
            let q = match proven_exact_value(ctx, children, idx) {
                Some(v) => v,
                None => match children.node_id(idx) {
                    Some(cid) => ctx.child_snapshot(cid, children, idx).exploitation_score(),
                    None if children.num_visits(idx) > 0 => ctx
                        .child_snapshot(ctx.stack.current_id(), children, idx)
                        .exploitation_score(),
                    None => unvisited,
                },
            };
            qs.push(q);
            ns.push(children.num_visits(idx) as f64);
        }

        let n_total: f64 = ns.iter().sum();
        let lambda = grill_lambda(self.exploration_constant, n_total, k);

        // λ_N → 0 (large N, or c = 0): π̄ degenerates to a point mass on
        // argmax Q -- skip the solve, argmax Q directly (tie-broken).
        if lambda <= GRILL_LAMBDA_FLOOR {
            return random_best_index_by(children, ctx, rng, |i| qs[i]);
        }

        let priors = vec![1.0 / k as f64; k];
        let pi_bar = grill_pi_bar(&qs, &priors, lambda);
        let discrepancy = grill_discrepancy(&pi_bar, &ns);
        random_best_index_by(children, ctx, rng, |i| discrepancy[i])
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

    #[test]
    fn grill_lambda_matches_hand_computed() {
        // 2 * sqrt(100) / (100 + 4) = 20 / 104
        assert!((grill_lambda(2.0, 100.0, 4) - 20.0 / 104.0).abs() < 1e-12);
    }

    #[test]
    fn grill_alpha_two_child_closed_form() {
        // qs = [0, 0.5], priors = [0.5, 0.5], lambda = 0.4.
        // 0.2/alpha + 0.2/(alpha - 0.5) = 1
        //   => alpha^2 - 0.9 alpha + 0.1 = 0
        //   => alpha = (0.9 + sqrt(0.81 - 0.4)) / 2 = (0.9 + sqrt(0.41)) / 2
        let expected = (0.9 + 0.41f64.sqrt()) / 2.0;
        let got = grill_alpha(&[0.0, 0.5], &[0.5, 0.5], 0.4);
        assert!((got - expected).abs() < 1e-9, "got {got} vs {expected}");
    }

    #[test]
    fn grill_alpha_stays_above_max_q() {
        let qs = [-0.3, 0.2, 1.0, 0.7];
        let priors = [0.25; 4];
        let max_q = qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for lambda in [0.1, 1.0, 3.0] {
            assert!(grill_alpha(&qs, &priors, lambda) > max_q);
        }
    }

    #[test]
    fn grill_pi_bar_sums_to_one() {
        let qs = [-0.2, 0.5, 0.1];
        let priors = [1.0 / 3.0; 3];
        for lambda in [0.05, 0.5, 2.0] {
            let pi = grill_pi_bar(&qs, &priors, lambda);
            assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(pi.iter().all(|&p| p > 0.0));
        }
    }

    #[test]
    fn grill_pi_bar_concentrates_on_argmax_q_as_lambda_shrinks() {
        let pi = grill_pi_bar(&[0.1, 0.9, 0.3], &[1.0 / 3.0; 3], 1e-6);
        assert!(pi[1] > 0.99, "{pi:?}");
    }

    #[test]
    fn grill_discrepancy_rule_explores_low_visit_child() {
        // pi_bar = [0.5, 0.3, 0.2], n = [10, 1, 1], N = 12.
        // index 0's visit share (10/13) already exceeds its target, so the
        // argmax discrepancy must land on an undershooting child.
        let d = grill_discrepancy(&[0.5, 0.3, 0.2], &[10.0, 1.0, 1.0]);
        let argmax = (0..3)
            .max_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap())
            .unwrap();
        assert_ne!(argmax, 0, "{d:?}");
    }
}
