use super::super::{node, TreeIndex};
use super::*;

/// MENTS's soft value backup (Xiao, Huang, Weinman, Müller, "Maximum
/// Entropy Monte-Carlo Planning", NeurIPS 2019): after the ordinary
/// Monte-Carlo `update`, recomputes every ancestor's own value as the
/// τ-mellowmax of its children (Asadi & Littman, ICML 2017 -- bounded,
/// unlike the paper's literal log-sum-exp; see `mellowmax` /
/// `derive_softmax_value`). `tau → 0` recovers the max backup, `tau → ∞`
/// approaches the `Classic` arithmetic mean. Pairs with `select::Ments`
/// (the E2W stochastic tree policy), which sets
/// `Requirements::needs_softmax_value` -- `SearchConfig::validate` rejects
/// `Ments` without this backprop. RENTS / TENTS (Dam et al., ICML 2021 --
/// Tsallis / Rényi regularisers) are a deferred follow-up; there is no
/// `mode` enum.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxBackprop {
    /// Entropy-regularisation temperature. `→ 0` recovers the max backup,
    /// `→ ∞` recovers the `Classic` arithmetic mean. Tuner bounds
    /// `[0.05, 5.0]`, default `1.0`.
    pub tau: f64,
    /// Plies of ancestors above the leaf that get the soft backup. `None` =
    /// every ancestor (the useful default); parity with
    /// `PowerMeanBackprop::depth`.
    pub depth: Option<u32>,
}

impl Default for SoftmaxBackprop {
    fn default() -> Self {
        Self {
            tau: 1.0,
            depth: None,
        }
    }
}

impl SoftmaxBackprop {
    pub fn new(tau: f64) -> Self {
        Self { tau, depth: None }
    }
}

impl BackpropStrategy for SoftmaxBackprop {
    fn provides_softmax_value(&self) -> bool {
        true
    }

    fn recompute_depth(&self) -> u32 {
        // No `tau` makes the recompute a literal no-op (the `τ → ∞` limit
        // only *approaches* the mean), so -- unlike `PowerMeanBackprop` --
        // there is no bit-identical-`Classic` shortcut.
        self.depth.unwrap_or(u32::MAX)
    }

    fn recompute_value<A: crate::game::Action>(
        &self,
        node: &node::Node<A>,
        slot: &PosteriorSlot<A>,
        index: &TreeIndex<A>,
        num_players: usize,
    ) {
        derive_softmax_value(node, slot, index, num_players, self.tau);
    }
}
