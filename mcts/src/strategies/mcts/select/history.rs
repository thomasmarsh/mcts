use super::super::config::BackpropFlags;
use super::super::config::GLOBAL;
use super::super::index::Id;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectStrategy;
use super::ucb::Ucb1;
use crate::game::Game;

// Nijssen, J.A.M., Winands, M.H.M., 2011. Enhancements for Multi-Player
// Monte-Carlo Tree Search, in: van den Herik, H.J., Iida, H., Plaat, A.
// (Eds.), Computers and Games. Springer, Berlin, Heidelberg, pp. 238-249.
// https://doi.org/10.1007/978-3-642-17928-0_22

/// Progressive History: UCB1 plus a decaying bonus drawn from the
/// GLOBAL/MAST per-action history table (`TreeStats::player_actions`) --
/// the same running `action -> mean score` table `Mast`/`Nst` already
/// maintain for the simulation policy, reused here to bias in-tree
/// selection. The bonus is strongest while a child has few visits of its
/// own (where its local UCB statistics are still noisy) and decays toward
/// zero as `n(s,a)` grows, so it warm-starts exploration without
/// permanently distorting the UCB estimate once real evidence accumulates.
///
/// Deliberately does not extend the history bonus to the "no tree node
/// yet" case (`unvisited_value`): that hook is computed once per
/// `best_child` call and applied uniformly to every not-yet-created child
/// (see `random_best_index`), so it has no way to look up a specific
/// action's history score without recomputing it per-candidate on every
/// tree descent -- the same tradeoff `Rave`'s `unvisited_value` already
/// makes for AMAF/GRAVE data.
#[derive(Clone)]
pub struct ProgressiveHistory {
    pub ucb: Ucb1,
    /// Weight on the history bonus (`W` in Nijssen & Winands).
    pub weight: f64,
}

impl Default for ProgressiveHistory {
    fn default() -> Self {
        Self {
            ucb: Ucb1::default(),
            weight: 1.,
        }
    }
}

impl ProgressiveHistory {
    pub fn new(ucb: Ucb1, weight: f64) -> Self {
        Self { ucb, weight }
    }

    pub fn ucb(mut self, ucb: Ucb1) -> Self {
        self.ucb = ucb;
        self
    }

    pub fn weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

impl<G: Game> SelectStrategy<G> for ProgressiveHistory {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        self.ucb.setup(ctx)
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        parent_log: f64,
    ) -> f64 {
        let ucb = self
            .ucb
            .score_child(ctx, child_id, children, idx, parent_log);

        let action = children.action(idx);
        let player_actions = ctx.global.player_actions[ctx.player].read().unwrap();
        let Some(stats) = player_actions.get(action).filter(|s| s.num_visits > 0) else {
            return ucb;
        };
        let history_score = stats.score / stats.num_visits as f64;
        let n = children.snapshot(idx, ctx.player).total_visits();

        ucb + self.weight * history_score / (n as f64 + 1.)
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, parent_log: f64) -> f64 {
        self.ucb.unvisited_value(ctx, parent_log)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GLOBAL)
    }
}
