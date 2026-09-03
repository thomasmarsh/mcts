use super::super::config::GraphStats;
use super::super::config::McgsCorrection;
use super::super::correction::rave_blend_correction;
use super::super::index::Id;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectPolicy;
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

impl<G: Game> SelectPolicy<G> for Ucb1 {
    type Score = f64;
    type Aux = f64;

    fn supports_ismcts() -> bool {
        true
    }

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        let stats = ctx.current_stats();
        ((stats.num_visits() as f64).max(1.)).ln()
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
        let snap = ctx.child_snapshot(child_id, children, idx);
        let mut exploit = snap.exploitation_score();
        // `McgsCorrection::RaveBlend` (see its doc comment): unlike
        // `Residual`, which intercepts descent *after* selection has
        // already chosen `best_idx` (`search/shared.rs::
        // mcgs_correction_at_edge`), this blends a DAG-merged target's
        // pooled estimate directly into the score a child is chosen *by* --
        // never gating or skipping the traversal that follows, so the edge
        // keeps accumulating its own direct samples regardless of how the
        // blend leans this iteration. Only meaningful under `GraphStats::
        // Both` (the only mode that keeps both an edge-local and a
        // node-pooled estimate to blend between) and only once this child
        // has actually been reached by more than one parent
        // (`Node::is_transposition`) -- an unshared child has no pooled
        // estimate that differs from its own.
        if let McgsCorrection::RaveBlend { schedule } = ctx.mcgs_correction {
            if matches!(ctx.graph_stats, Some(GraphStats::Both)) {
                let target = ctx.index.get(child_id);
                if target.is_transposition() {
                    exploit = rave_blend_correction(
                        schedule,
                        exploit,
                        snap.total_visits(),
                        target.stats.expected_score(ctx.player),
                        target.stats.total_visits(),
                    );
                }
            }
        }
        // ISMCTS (Cowling, Powley & Whitehouse 2012): a `growable` array's
        // children aren't all legal on every iteration (see `search/
        // shared.rs::select_step`'s `ismcts_legal`), so the ordinary UCB
        // denominator -- every child sharing the *node's* total visit count
        // in its `ln` term -- overstates how much exploration a
        // rarely-legal action still needs relative to one that's legal
        // every time. Cowling et al.'s fix: track each child's own
        // "availability" (how many iterations it was legal at all,
        // `ChildArray::availability`) and use that in place of the shared
        // parent count. A non-growable array's availability is never
        // written, so this only ever takes the ordinary branch.
        let log_n = if children.is_growable() {
            ((children.availability(idx) as f64).max(1.)).ln()
        } else {
            parent_log
        };
        let explore = (log_n / snap.total_visits() as f64).sqrt();
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
