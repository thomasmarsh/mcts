use super::super::index::Id;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectPolicy;
use super::random_best_index_by;
use super::score_child_or_prior;
use super::ucb::Ucb1;
use crate::game::Game;

use rand::rngs::SmallRng;

/// Which proof-number bias formula `GpnUct` adds to the UCB1 score
/// (Kowalski, Soemers, Kosakowski & Winands, *Generalized Proof-Number
/// Monte-Carlo Tree Search*, arXiv:2506.13249, §3.2). All three take a
/// child and its sibling set and return a value in `[0, 1]` -- larger means
/// a stronger pull toward that child -- and all three return `0` for a
/// child whose proof number is infinite (provably not a win for the mover).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpnBias {
    /// PNMax (paper Eq. 4): scale by the range of finite sibling proof
    /// numbers, `1 - (pn(i) - minf) / (1 + maxf - minf)`. Satisfies all four
    /// of the paper's stated conditions and needs only a min/max scan, not a
    /// sort. The default.
    #[default]
    Max,
    /// PNSum (paper Eq. 5): scale by the sum of finite sibling proof
    /// numbers, `1 - pn(i) / (1 + Σ finite pn(j))`. More strongly coupled to
    /// the parent's branching factor than the other two.
    Sum,
    /// PNRank (Kowalski et al. 2023, Eq. 3): the original rank-based bonus,
    /// `1 - rank(i) / max rank`, ties sharing a rank. Kept for parity with
    /// `UctPn`; needs a per-update sort of the siblings.
    Rank,
}

/// GPN-MCTS's selection strategy (Kowalski et al., arXiv:2506.13249): UCB1
/// plus `c_pn * bias(i, siblings)`, where `bias` is one of `GpnBias` applied
/// to the per-player proof number `Node::player_pn` of the *node's own
/// mover* (`backprop::derive_player_pn` maintains those). Unlike `UctPn`,
/// which ranks by the per-mover disproof number and is framed around
/// alternating AND/OR layers, this reads one proof number per player and so
/// works unchanged at any player count -- the multi-player generalization
/// the paper's contribution (1) buys.
///
/// Only meaningful with `use_mcts_solver` on: with it off every node's
/// `player_pn` stays at its seed `1`, so PNMax/PNRank give a constant `1`
/// bonus to every child and PNSum a constant `1 / (1 + n)` -- i.e. plain
/// UCB1 shifted by a constant, not a configuration error.
#[derive(Clone)]
pub struct GpnUct {
    pub ucb1: Ucb1,
    /// `C_pn` in the paper's Eq. 2 -- weight of the proof-number bias
    /// relative to UCB1's own terms. The paper found the best value is
    /// strongly game- *and* formula-dependent (anywhere in `0.1..=5.0`), so
    /// this wants tuning per game.
    pub c_pn: f64,
    pub bias: GpnBias,
}

impl GpnUct {
    pub fn with_c(exploration_constant: f64, c_pn: f64) -> Self {
        Self {
            ucb1: Ucb1::with_c(exploration_constant),
            c_pn,
            bias: GpnBias::default(),
        }
    }

    pub fn bias(mut self, bias: GpnBias) -> Self {
        self.bias = bias;
        self
    }
}

impl Default for GpnUct {
    fn default() -> Self {
        Self {
            ucb1: Ucb1::default(),
            c_pn: 1.0,
            bias: GpnBias::default(),
        }
    }
}

/// PNS-style "competition ranking" (1, 2, 2, 4, ...) over `pns` by ascending
/// value -- `u32::MAX` (infinite proof number) sorts last, as the worst.
/// Pure function so the ranking arithmetic can be unit tested directly.
fn competition_rank(pns: &[u32]) -> Vec<u32> {
    let mut order: Vec<usize> = (0..pns.len()).collect();
    order.sort_by_key(|&i| pns[i]);
    let mut ranks = vec![1u32; pns.len()];
    let mut rank = 1u32;
    for (pos, &i) in order.iter().enumerate() {
        if pos > 0 && pns[i] != pns[order[pos - 1]] {
            rank = pos as u32 + 1;
        }
        ranks[i] = rank;
    }
    ranks
}

/// PNMax, paper Eq. 4. `min_f`/`max_f` are over finite sibling proof numbers
/// only; `None` when every sibling is infinite (then every child returns 0).
pub(crate) fn pnmax(pn: u32, min_f: Option<u32>, max_f: Option<u32>) -> f64 {
    if pn == u32::MAX {
        return 0.0;
    }
    let (min_f, max_f) = (min_f.unwrap() as f64, max_f.unwrap() as f64);
    1.0 - (pn as f64 - min_f) / (1.0 + max_f - min_f)
}

/// PNSum, paper Eq. 5. `sum_f` is the sum of finite sibling proof numbers.
pub(crate) fn pnsum(pn: u32, sum_f: u64) -> f64 {
    if pn == u32::MAX {
        return 0.0;
    }
    1.0 - (pn as f64) / (1.0 + sum_f as f64)
}

impl<G: Game> SelectPolicy<G> for GpnUct {
    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        SelectPolicy::<G>::setup(&mut self.ucb1, ctx)
    }

    /// Plain UCB1 -- a lone child can't see the siblings its bias normalizes
    /// against, so this is only correct in isolation. `best_child` computes
    /// the real combined score directly, exactly like `UctPn`.
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
        let mover = current.player_idx;
        let n = children.len();

        // `pn(i)` in the paper's formulas is child `i`'s proof number for
        // the player choosing among these children -- this node's own mover.
        let pns: Vec<u32> = (0..n)
            .map(|idx| match children.node_id(idx) {
                Some(child_id) => ctx.index.get(child_id).player_pn(mover),
                None => 1,
            })
            .collect();

        let finite = || pns.iter().copied().filter(|&x| x != u32::MAX);
        let min_f = finite().min();
        let max_f = finite().max();
        let sum_f: u64 = finite().map(|x| x as u64).sum();
        let ranks = (self.bias == GpnBias::Rank).then(|| competition_rank(&pns));
        let max_rank = ranks.as_ref().map(|r| r.iter().copied().max().unwrap_or(1));

        let bias_value = |idx: usize| -> f64 {
            match self.bias {
                GpnBias::Max => pnmax(pns[idx], min_f, max_f),
                GpnBias::Sum => pnsum(pns[idx], sum_f),
                GpnBias::Rank => {
                    let ranks = ranks.as_ref().unwrap();
                    1.0 - ranks[idx] as f64 / max_rank.unwrap().max(1) as f64
                }
            }
        };

        let parent_log = SelectPolicy::<G>::setup(&mut self.ucb1, ctx);
        let unvisited_ucb1 = SelectPolicy::<G>::unvisited_value(&self.ucb1, ctx, parent_log);
        let c_pn = self.c_pn;

        random_best_index_by(children, ctx, rng, |idx| {
            let ucb1_score =
                score_child_or_prior(ctx, &self.ucb1, children, idx, parent_log, unvisited_ucb1);
            ucb1_score + c_pn * bias_value(idx)
        })
    }

    /// Needs MCTS-Solver's proof bookkeeping on (`solver: true`, advisory --
    /// same as `UctPn`). No `max_players` cap: per-player proof numbers are
    /// exactly what makes this sound beyond two players.
    fn requirements(&self) -> super::config::Requirements {
        super::config::Requirements {
            solver: true,
            ..super::config::Requirements::from_backprop_flags(
                <Self as SelectPolicy<G>>::backprop_flags(self),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pnmax_is_one_for_the_lowest_finite_and_zero_for_infinite() {
        // siblings: pn = [2, 4, MAX]; finite min 2, max 4.
        assert!((pnmax(2, Some(2), Some(4)) - 1.0).abs() < 1e-12);
        assert!((pnmax(4, Some(2), Some(4)) - (1.0 - 2.0 / 3.0)).abs() < 1e-12);
        assert_eq!(pnmax(u32::MAX, Some(2), Some(4)), 0.0);
    }

    #[test]
    fn pnsum_spreads_proportionally_to_the_finite_sum() {
        // siblings: pn = [2, 4, MAX]; finite sum 6.
        assert!((pnsum(2, 6) - (1.0 - 2.0 / 7.0)).abs() < 1e-12);
        assert!((pnsum(4, 6) - (1.0 - 4.0 / 7.0)).abs() < 1e-12);
        assert_eq!(pnsum(u32::MAX, 6), 0.0);
    }

    #[test]
    fn competition_rank_ties_share_and_skip_and_infinite_sorts_last() {
        assert_eq!(competition_rank(&[3, 3, 7]), vec![1, 1, 3]);
        assert_eq!(competition_rank(&[5, 1, u32::MAX]), vec![2, 1, 3]);
    }
}
