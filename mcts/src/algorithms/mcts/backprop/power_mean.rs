use super::super::{node, TreeIndex};
use super::*;

/// Power-UCT (Dam et al., IJCAI 2020): a backpropagation policy that, after
/// the ordinary Monte-Carlo `update`, recomputes every ancestor's own value
/// as the visit-weighted power mean of its children -- see
/// `derive_power_mean_value`. One scalar `p`: `p = 1` recovers plain UCT (the
/// recompute pass is disabled outright, so the behavior is bit-identical to
/// `Classic`), `p -> inf` approaches a max / Full-Bellman backup. The value
/// sits between the negatively-biased mean and the positively-biased max;
/// convergence over the whole range (including for explicitly non-stationary
/// tree nodes) is Stochastic-Power-UCT (arXiv 2406.02235).
///
/// `alpha` blends that power mean with the plain max over children
/// (`(1 - alpha)·V_p + alpha·V_max`, per player) -- `alpha = 1` is the
/// Full-Bellman / max backup (Asai & Wissow, AAAI 2025) at any `p`. See
/// `derive_power_mean_value` for the EVT framing and the dead-child
/// exclusion that is always applied.
#[derive(Debug, Clone, Copy)]
pub struct PowerMeanBackprop {
    /// Power-mean exponent. `1.0` = plain visit-weighted mean (== `Classic`
    /// when `alpha == 0`, recompute pass disabled); larger = closer to max.
    /// Tuner bounds `[1.0, 50.0]`.
    pub p: f64,
    /// Mean<->max blend: final value = `(1-alpha)·power_mean + alpha·max`
    /// over children, per player. `0.0` = pure power-mean; `1.0` =
    /// Full-Bellman max backup (Asai & Wissow, AAAI 2025) at any `p`. Tuner
    /// bounds `[0.0, 1.0]`, default `0.0`.
    pub alpha: f64,
    /// How many plies of ancestors above the leaf get the power-mean backup.
    /// `None` = every ancestor; `Some(d)` = only the nearest `d` (parity with
    /// `MinimaxBackprop::depth`, for a future depth sweep).
    pub depth: Option<u32>,
}

impl Default for PowerMeanBackprop {
    fn default() -> Self {
        Self {
            p: 1.0,
            alpha: 0.0,
            depth: None,
        }
    }
}

impl PowerMeanBackprop {
    pub fn new(p: f64, depth: Option<u32>) -> Self {
        Self {
            p,
            alpha: 0.0,
            depth,
        }
    }

    pub fn new_mixed(p: f64, alpha: f64, depth: Option<u32>) -> Self {
        Self { p, alpha, depth }
    }
}

impl BackpropPolicy for PowerMeanBackprop {
    fn recompute_depth(&self) -> u32 {
        // p == 1, alpha == 0 is exactly the arithmetic mean; skip the pass
        // entirely so the strategy is bit-identical to `Classic` there (and
        // so the per-call overwrite-then-reaccumulate churn is avoided).
        // p == 1, alpha > 0 is a mean<->max blend -- the pass must run.
        if self.p == 1.0 && self.alpha == 0.0 {
            0
        } else {
            self.depth.unwrap_or(u32::MAX)
        }
    }

    fn recompute_value<A: crate::game::Action>(
        &self,
        node: &node::Node<A>,
        slot: &PosteriorSlot<A>,
        index: &TreeIndex<A>,
        num_players: usize,
    ) {
        derive_power_mean_value(node, slot, index, num_players, self.p, self.alpha);
    }
}
