use super::super::index::Id;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectStrategy;
use super::random_best_index_by;
use super::ucb::Ucb1;
use crate::game::Game;

use rand::rngs::SmallRng;

/// UCT-PN (Kowalski, Doe, Winands, Górski & Soemers, "Proof Number Based
/// Monte-Carlo Tree Search", 2023): UCB1 augmented with a rank-based bonus
/// from proof/disproof numbers (`Node::pn`/`Node::dpn`, maintained by
/// `derive_pn_dpn` in backprop.rs). The paper found raw (dis)proof-number
/// *magnitudes* aren't meaningful in a formula like this -- a `dpn` of 100
/// isn't "ten times worse" than 10, it may just be more explored -- but
/// their *order* reliably identifies which child PNS itself would pick
/// next, so that order (normalized to `[0, 1]`) is what gets added in.
///
/// Only meaningful when `use_mcts_solver` is on: with it off, every node's
/// `pn`/`dpn` stay at their seed value of `1` (see `Node::pn`'s doc
/// comment), so every child ties for rank 1 and the bonus term degenerates
/// to a harmless constant `c_pn * (1 - 1/1) = 0` added to every child --
/// i.e. plain UCB1, not a configuration error.
#[derive(Clone)]
pub struct UctPn {
    pub ucb1: Ucb1,
    /// `C_pn` in the paper's Eq. 4 -- controls how much the PNS ranking
    /// influences selection relative to UCB1's own exploit/explore terms.
    /// The paper's best-performing values were domain-dependent but
    /// clustered around 1.0-2.0.
    pub c_pn: f64,
}

impl UctPn {
    pub fn with_c(exploration_constant: f64, c_pn: f64) -> Self {
        Self {
            ucb1: Ucb1::with_c(exploration_constant),
            c_pn,
        }
    }
}

impl Default for UctPn {
    fn default() -> Self {
        Self {
            ucb1: Ucb1::default(),
            c_pn: 1.0,
        }
    }
}

impl<G: Game> SelectStrategy<G> for UctPn {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        SelectStrategy::<G>::setup(&mut self.ucb1, ctx)
    }

    /// Plain UCB1, with no PN-rank bonus -- unlike `best_child` below, a
    /// single child's score can't see its siblings' (dis)proof numbers to
    /// rank against, so this is only correct in isolation. `best_child`
    /// computes the true combined score directly rather than routing
    /// through this method; it's implemented anyway because
    /// `SelectStrategy` requires it, and in case anything else (tests,
    /// introspection) calls it directly against a lone child.
    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        aux: f64,
    ) -> f64 {
        SelectStrategy::<G>::score_child(&self.ucb1, ctx, child_id, children, idx, aux)
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, aux: f64) -> f64 {
        SelectStrategy::<G>::unvisited_value(&self.ucb1, ctx, aux)
    }

    fn best_child(&mut self, ctx: &SelectContext<'_, G>, rng: &mut SmallRng) -> usize {
        let current = ctx.index.get(ctx.stack.current_id());
        let children = current.children();
        let n = children.len();

        // `pn(parent) = min` over children of the child's `dpn` (see
        // `derive_pn_dpn`'s doc comment) -- so the child achieving that
        // minimum is the one PNS itself would descend into next to prove
        // `ctx.player`'s win here. Rank children by ascending `dpn`, tying
        // equal values to the same rank, exactly as PNS's own child
        // ordering would.
        let dpn_of = |idx: usize| match children.node_id(idx) {
            Some(child_id) => ctx.index.get(child_id).dpn(),
            None => 1,
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&idx| dpn_of(idx));

        let mut ranks = vec![0u32; n];
        let mut rank = 1u32;
        for (pos, &idx) in order.iter().enumerate() {
            if pos > 0 && dpn_of(idx) != dpn_of(order[pos - 1]) {
                rank = pos as u32 + 1;
            }
            ranks[idx] = rank;
        }
        // Always >= 1: `n >= 1` (a node with zero legal actions is never
        // expanded -- see `expand`'s `debug_assert!(!actions.is_empty())`),
        // so `ranks` always has at least one entry.
        let max_rank = ranks.iter().copied().max().unwrap() as f64;

        let parent_log = SelectStrategy::<G>::setup(&mut self.ucb1, ctx);
        let unvisited_ucb1 = SelectStrategy::<G>::unvisited_value(&self.ucb1, ctx, parent_log);
        let c_pn = self.c_pn;

        random_best_index_by(children, ctx, rng, |idx| {
            let ucb1_score = match children.node_id(idx) {
                Some(child_id) => SelectStrategy::<G>::score_child(
                    &self.ucb1, ctx, child_id, children, idx, parent_log,
                ),
                None => unvisited_ucb1,
            };
            ucb1_score + c_pn * (1.0 - ranks[idx] as f64 / max_rank)
        })
    }
}
