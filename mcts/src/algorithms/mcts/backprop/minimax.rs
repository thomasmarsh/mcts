use super::super::{node, TreeIndex};
use super::*;

/// MCTS-MB-n (Baier & Winands): backpropagation-phase hybrid -- within
/// `depth` plies of the just-backpropagated leaf, overwrite (not average
/// into) each ancestor's own per-player value with `derive_minimax_value`'s
/// backward-induction backup from its own already-updated children, instead
/// of leaving it as the plain Monte-Carlo average `BackpropPolicy::update`
/// otherwise produces. The paper's own Breakthrough numbers (2015,
/// domain-independent MR/MS/MB, no evaluation function) found MB-2 the
/// strongest of the three domain-independent techniques there, winning
/// 55.0% of 2000 games at equal time against an MCTS-Solver baseline.
#[derive(Debug, Clone, Copy)]
pub struct MinimaxBackprop {
    /// How many plies of ancestors, counting from (but not including) the
    /// backpropagated leaf, get their value overwritten. `0` disables the
    /// backup entirely (every node keeps the ordinary Monte-Carlo average),
    /// equivalent to `Classic`.
    pub depth: u32,
}

impl Default for MinimaxBackprop {
    fn default() -> Self {
        Self {
            // MB-2 is the literature's own best-performing depth on
            // Breakthrough (Baier & Winands 2015), matching
            // `prior::NegamaxPrior`'s default `depth` for the same reason.
            depth: 2,
        }
    }
}

impl MinimaxBackprop {
    pub fn new(depth: u32) -> Self {
        Self { depth }
    }
}

impl BackpropPolicy for MinimaxBackprop {
    fn recompute_depth(&self) -> u32 {
        self.depth
    }

    fn recompute_value<A: crate::game::Action>(
        &self,
        node: &node::Node<A>,
        slot: &PosteriorSlot<A>,
        _index: &TreeIndex<A>,
        num_players: usize,
    ) {
        derive_minimax_value(node, slot, num_players);
    }
}
