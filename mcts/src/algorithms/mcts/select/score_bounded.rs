use super::super::index::Id;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectPolicy;
use super::random_best_index_by;
use super::score_child_or_prior;
use super::ucb::Ucb1;
use crate::game::Game;

use rand::rngs::SmallRng;

/// Score-Bounded UCT (Cazenave & Saffidine, *Score Bounded Monte-Carlo Tree
/// Search*, CG 2010): UCB1 augmented with the graded-score interval
/// `[pess, opti]` that `backprop::derive_score_bounds` maintains per node.
/// Two things use it:
///
///  - **Alpha-beta-style pruning** (paper §3.3). At a Max node (mover is
///    player 0), a child `s` is provably not worth exploring once
///    `opti(s) <= pess(n)`: some sibling already guarantees at least
///    `pess(n)`, and `s` can never beat it. At a Min node the mirror rule,
///    `pess(s) >= opti(n)`. A pruned child gets `-inf`, so it's chosen only
///    if *every* sibling is pruned (which only happens once the node itself
///    is score-solved).
///  - **Bound-induced value bias** (paper §3.4). Cazenave's `Q'` term: a
///    child's normalized `pess`/`opti` are added to its UCB1 score, weighted
///    by `gamma`/`delta`, oriented so that a Max mover prefers higher bounds
///    and a Min mover prefers lower ones.
///
/// Only meaningful when `use_mcts_solver` is on *and* the game overrides
/// `Game::score_bounds()` (two-player, graded terminal score). With either
/// missing, every node stays on its `i32::MIN`/`i32::MAX` seed: no child is
/// ever pruned and the bias term is a constant `gamma + delta` (Max) or
/// `-(gamma + delta)` (Min) added to every sibling alike -- i.e. plain
/// UCB1, not a configuration error. `requirements()` still reports
/// `max_players: Some(2)`, since the interval is a single Max-vs-Min scalar.
#[derive(Clone)]
pub struct ScoreBoundedUct {
    pub ucb1: Ucb1,
    /// Weight on the pessimistic bound in the §3.4 value bias.
    pub gamma: f64,
    /// Weight on the optimistic bound in the §3.4 value bias.
    pub delta: f64,
}

impl ScoreBoundedUct {
    pub fn with_c(exploration_constant: f64, gamma: f64, delta: f64) -> Self {
        Self {
            ucb1: Ucb1::with_c(exploration_constant),
            gamma,
            delta,
        }
    }
}

impl Default for ScoreBoundedUct {
    fn default() -> Self {
        Self {
            ucb1: Ucb1::default(),
            gamma: 0.1,
            delta: 0.1,
        }
    }
}

impl<G: Game> SelectPolicy<G> for ScoreBoundedUct {
    fn label(&self) -> String {
        "score_bounded_uct".into()
    }

    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        SelectPolicy::<G>::setup(&mut self.ucb1, ctx)
    }

    /// Plain UCB1 with no bound term -- like `UctPn::score_child`, a lone
    /// child can't see the parent interval it prunes against or the sibling
    /// range it normalizes into, so this is only correct in isolation.
    /// `best_child` computes the real combined score directly.
    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        aux: f64,
    ) -> f64 {
        SelectPolicy::<G>::score_child(&self.ucb1, ctx, child_id, children, idx, aux)
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, aux: f64) -> f64 {
        SelectPolicy::<G>::unvisited_value(&self.ucb1, ctx, aux)
    }

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.stack.current_id());
        let children = current.children();
        let maximizing = current.player_idx == 0;

        let (score_min, score_max) = G::score_bounds().unwrap_or((0, 0));
        let range = (score_max - score_min).max(1) as f64;
        let node_pess = current.pess();
        let node_opti = current.opti();
        // Only prune while the node's own interval is still open and
        // actually informative -- once it's collapsed (score-solved) every
        // child would prune and the choice would be arbitrary.
        let pruning_active = node_pess < node_opti
            && node_pess > score_min.saturating_sub(1)
            && node_opti < score_max.saturating_add(1);

        let parent_log = SelectPolicy::<G>::setup(&mut self.ucb1, ctx);
        let unvisited_ucb1 = SelectPolicy::<G>::unvisited_value(&self.ucb1, ctx, parent_log);

        let child_bounds = |idx: usize| match children.node_id(idx) {
            Some(child_id) => {
                let child = ctx.index.get(child_id);
                (child.pess(), child.opti())
            }
            None => (score_min, score_max),
        };

        random_best_index_by(children, ctx, rng, |idx| {
            let (child_pess, child_opti) = child_bounds(idx);
            if pruning_active {
                let pruned = if maximizing {
                    child_opti <= node_pess
                } else {
                    child_pess >= node_opti
                };
                if pruned {
                    return f64::NEG_INFINITY;
                }
            }

            let ucb1_score =
                score_child_or_prior(ctx, &self.ucb1, children, idx, parent_log, unvisited_ucb1);

            let pess_n = (child_pess.clamp(score_min, score_max) - score_min) as f64 / range;
            let opti_n = (child_opti.clamp(score_min, score_max) - score_min) as f64 / range;
            let bias = if maximizing {
                self.gamma * pess_n + self.delta * opti_n
            } else {
                -self.gamma * opti_n - self.delta * pess_n
            };
            ucb1_score + bias
        })
    }

    fn requirements(&self) -> super::config::Requirements {
        super::config::Requirements {
            solver: true,
            max_players: Some(2),
            ..super::config::Requirements::from_backprop_flags(
                <Self as SelectPolicy<G>>::backprop_flags(self),
            )
        }
    }
}
