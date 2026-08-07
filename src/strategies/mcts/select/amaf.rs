use super::super::config::BackpropFlags;
use super::super::config::AMAF;
use super::super::index::Id;
use super::super::node::Edge;
use super::super::select::SelectContext;
use super::super::select::SelectStrategy;
use crate::game::Game;

#[derive(Clone)]
pub struct Amaf {
    pub alpha: f64,
    pub exploration_constant: f64,
}

impl Amaf {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_c(exploration_constant: f64) -> Self {
        Self {
            exploration_constant,
            ..Default::default()
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn exploration_constant(mut self, exploration_constant: f64) -> Self {
        self.exploration_constant = exploration_constant;
        self
    }
}

impl Default for Amaf {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl<G: Game> SelectStrategy<G> for Amaf {
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
        edge: &Edge<G::A>,
        parent_log: f64,
    ) -> f64 {
        let amaf_stats = edge.stats.amaf(ctx.player);
        let amaf_n = 1.max(amaf_stats.num_visits) as f64;
        let amaf_q = amaf_stats.score;
        let amaf = amaf_q / amaf_n;

        let exploit = edge.stats.exploitation_score(ctx.player);
        let num_visits = edge.stats.total_visits();
        let explore = (parent_log / num_visits as f64).sqrt();

        // alpha = 1 is standard AMAF
        // alpha = 0 is standard UCT
        let ucb1 = exploit + self.exploration_constant * explore;
        self.alpha * amaf + (1. - self.alpha) * ucb1
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: f64) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(AMAF)
    }
}