use super::super::index::Id;
use super::super::node::Edge;
use super::super::node::Proven;
use super::super::select::SelectContext;
use super::super::select::SelectStrategy;
use crate::game::Game;

use rand::rngs::SmallRng;

fn is_proven_loss<G: Game>(ctx: &SelectContext<'_, G>, edge: &Edge<G::A>) -> bool {
    edge.node_id().is_some_and(|child_id| {
        matches!(ctx.index.get(child_id).proven(), Proven::Win(w) if w != ctx.player)
    })
}

////////////////////////////////////////////////////////////////////////////////

/// Select the most visited root child.
#[derive(Default, Clone)]
pub struct RobustChild;

impl<G: Game> SelectStrategy<G> for RobustChild {
    type Score = (i64, f64);
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        _: Self::Aux,
    ) -> (i64, f64) {
        (
            edge.stats.num_visits() as i64,
            edge.stats.expected_score(ctx.player),
        )
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> (i64, f64) {
        let q = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        (0, q)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Select the root child with the highest reward.
#[derive(Default, Clone)]
pub struct MaxAvgScore;

impl<G: Game> SelectStrategy<G> for MaxAvgScore {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        _: Self::Aux,
    ) -> f64 {
        edge.stats.expected_score(ctx.player)
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// The secure child is the child that maximizes a lower confidence bound.
#[derive(Clone)]
pub struct SecureChild {
    pub a: f64,
}

impl Default for SecureChild {
    fn default() -> Self {
        // This quantity comes from the Chaslot, Winands progressive strategies paper
        Self { a: 4. }
    }
}

impl<G: Game> SelectStrategy<G> for SecureChild {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        _: Self::Aux,
    ) -> f64 {
        let q = edge.stats.expected_score(ctx.player);
        let n = edge.stats.total_visits();

        q + self.a / (n as f64).sqrt()
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
pub struct ThompsonSampling;

impl<G: Game> SelectStrategy<G> for ThompsonSampling {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline]
    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.stack.current_id());
        // This is just a weighted sampling. Need to implement some stuff for thompson sampling.
        let weights = current
            .edges()
            .iter()
            .map(|edge| {
                if is_proven_loss(ctx, edge) {
                    // MCTS-Solver: force a proven-loss child's weight toward
                    // (not exactly -- `WalkerTableBuilder` wants strictly
                    // positive weights) zero rather than excluding it, so
                    // this stays a normal weighted sampling even when every
                    // sibling happens to be a proven loss.
                    f32::MIN_POSITIVE
                } else {
                    edge.node_id()
                        .map(|child_id| self.score_child(ctx, child_id, edge, ()))
                        .unwrap_or(self.unvisited_value(ctx, ())) as f32
                }
            })
            .collect::<Vec<_>>();

        use weighted_rand::builder::*;
        let builder = WalkerTableBuilder::new(&weights);
        let wa_table = builder.build();
        wa_table.next_rng(rng)
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        edge: &Edge<G::A>,
        _: Self::Aux,
    ) -> f64 {
        let q = edge.stats.expected_score(ctx.player);
        let n = edge.stats.total_visits();

        q / (n as f64).sqrt()
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }
}