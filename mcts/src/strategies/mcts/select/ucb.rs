use super::super::index::Id;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectStrategy;
use crate::game::Game;

/// Upper Confidence Bounds (UCB1)
#[derive(Clone)]
pub struct Ucb1 {
    pub exploration_constant: f64,
}

impl Ucb1 {
    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }
}

impl Default for Ucb1 {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl<G: Game> SelectStrategy<G> for Ucb1 {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        let stats = ctx.current_stats();
        ((stats.num_visits() as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        parent_log: f64,
    ) -> f64 {
        let snap = ctx.child_snapshot(_child_id, children, idx);
        let exploit = snap.exploitation_score();
        let explore = (parent_log / snap.total_visits() as f64).sqrt();
        exploit + self.exploration_constant * explore
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, parent_log: f64) -> f64 {
        let unvisited_value = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);

        unvisited_value + self.exploration_constant * parent_log.sqrt()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct Ucb1Tuned {
    pub exploration_constant: f64,
}

impl Default for Ucb1Tuned {
    fn default() -> Self {
        Self {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl Ucb1Tuned {
    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
        }
    }
}

pub(crate) const VARIANCE_UPPER_BOUND: f64 = 1.;

#[inline(always)]
pub(crate) fn ucb1_tuned(
    exploration_constant: f64,
    exploit: f64,
    sample_variance: f64,
    visits_fraction: f64,
) -> f64 {
    exploit
        + (visits_fraction * VARIANCE_UPPER_BOUND.min(sample_variance)
            + exploration_constant * visits_fraction.sqrt())
}

impl<G: Game> SelectStrategy<G> for Ucb1Tuned {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        ((ctx.current_stats().num_visits() as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        parent_log: f64,
    ) -> f64 {
        let snap = ctx.child_snapshot(_child_id, children, idx);
        let exploit = snap.exploitation_score();
        let num_visits = snap.total_visits();
        let sample_variance =
            0f64.max(snap.sum_squared_score / num_visits as f64 - exploit * exploit);
        let visits_fraction = parent_log / num_visits as f64;

        ucb1_tuned(
            self.exploration_constant,
            exploit,
            sample_variance,
            visits_fraction,
        )
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, parent_log: f64) -> Self::Score {
        let unvisited_value = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        ucb1_tuned(
            self.exploration_constant,
            unvisited_value,
            VARIANCE_UPPER_BOUND,
            parent_log,
        )
    }
}
