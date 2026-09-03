use super::super::index::Id;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectPolicy;
use super::is_proven_loss;
use super::score_child_or_prior;
use crate::game::Game;

use rand::rngs::SmallRng;

////////////////////////////////////////////////////////////////////////////////

/// Select the most visited root child.
#[derive(Default, Clone)]
pub struct RobustChild;

impl<G: Game> SelectPolicy<G> for RobustChild {
    type Score = (i64, f64);
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        _: Self::Aux,
    ) -> (i64, f64) {
        let snap = ctx.child_snapshot(_child_id, children, idx);
        (snap.num_visits as i64, snap.expected_score())
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

impl<G: Game> SelectPolicy<G> for MaxAvgScore {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        _: Self::Aux,
    ) -> f64 {
        ctx.child_snapshot(_child_id, children, idx)
            .expected_score()
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }
}

////////////////////////////////////////////////////////////////////////////////

// Chaslot, G.M.J-B., Winands, M.H.M., van den Herik, H.J., Uiterwijk, J.W.H.M.,
// Bouzy, B., 2008. Progressive Strategies for Monte-Carlo Tree Search, in:
// New Mathematics and Natural Computation.

/// The max-robust child is the child that has both the highest visit count
/// (`RobustChild`) and the highest average score (`MaxAvgScore`) at once --
/// the two criteria agreeing is meant to be read as extra confidence in the
/// pick. When no single child dominates both, Chaslot et al. suggest either
/// falling back to the max child or the robust child; this picks the max
/// child (highest average score). That fallback choice is what keeps this
/// distinct from plain `RobustChild`: `RobustChild` always sorts on visits
/// first, so falling back to a visits-first tie-break here (the other
/// option in the paper) would make the dominance check a no-op -- the
/// dominant child, by construction, already has the most visits, so a
/// visits-first fallback would pick it anyway whether or not it's flagged
/// dominant, and every other case is decided by the fallback alone.
#[derive(Default, Clone)]
pub struct MaxRobustChild;

impl MaxRobustChild {
    /// The child index that is simultaneously the most-visited and the
    /// highest-scoring, if one exists. `None` when the two criteria pick
    /// different children (or when no children are visited yet).
    fn dominant_child<G: Game>(ctx: &SelectContext<'_, G>) -> Option<usize> {
        let current = ctx.index.get(ctx.stack.current_id());
        let children = current.children();

        let mut most_visited: Option<(usize, u32)> = None;
        let mut highest_scoring: Option<(usize, f64)> = None;
        for idx in 0..children.len() {
            if children.node_id(idx).is_none() {
                continue;
            }
            let child_id = children.node_id(idx).unwrap();
            let snap = ctx.child_snapshot(child_id, children, idx);
            let visits = snap.total_visits();
            let score = snap.expected_score();
            if most_visited.is_none_or(|(_, v)| visits > v) {
                most_visited = Some((idx, visits));
            }
            if highest_scoring.is_none_or(|(_, s)| score > s) {
                highest_scoring = Some((idx, score));
            }
        }

        match (most_visited, highest_scoring) {
            (Some((v_idx, _)), Some((s_idx, _))) if v_idx == s_idx => Some(v_idx),
            _ => None,
        }
    }
}

impl<G: Game> SelectPolicy<G> for MaxRobustChild {
    type Score = (bool, f64);
    type Aux = Option<usize>;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> Self::Aux {
        Self::dominant_child(ctx)
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        dominant: Self::Aux,
    ) -> (bool, f64) {
        let score = ctx
            .child_snapshot(_child_id, children, idx)
            .expected_score();
        (dominant == Some(idx), score)
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> (bool, f64) {
        let q = ctx
            .current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init);
        (false, q)
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

impl<G: Game> SelectPolicy<G> for SecureChild {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        _child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        _: Self::Aux,
    ) -> f64 {
        let snap = ctx.child_snapshot(_child_id, children, idx);
        let q = snap.expected_score();
        let n = snap.total_visits();

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

impl<G: Game> SelectPolicy<G> for ThompsonSampling {
    type Score = f64;
    type Aux = ();

    #[inline(always)]
    fn setup(&mut self, _: &SelectContext<'_, G>) -> Self::Aux {}

    #[inline]
    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.stack.current_id());
        let children = current.children();
        // This is just a weighted sampling. Need to implement some stuff for thompson sampling.
        let weights = (0..children.len())
            .map(|idx| {
                if is_proven_loss(ctx, children, idx) {
                    // MCTS-Solver: force a proven-loss child's weight toward
                    // (not exactly -- `WalkerTableBuilder` wants strictly
                    // positive weights) zero rather than excluding it, so
                    // this stays a normal weighted sampling even when every
                    // sibling happens to be a proven loss.
                    f32::MIN_POSITIVE
                } else {
                    score_child_or_prior(
                        ctx,
                        self,
                        children,
                        idx,
                        (),
                        self.unvisited_value(ctx, ()),
                    ) as f32
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
        children: &ChildArray<G::A>,
        idx: usize,
        _: Self::Aux,
    ) -> f64 {
        let snap = ctx.child_snapshot(_child_id, children, idx);
        let q = snap.expected_score();
        let n = snap.total_visits();

        q / (n as f64).sqrt()
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: Self::Aux) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }
}
