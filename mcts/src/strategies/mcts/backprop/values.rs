use super::super::node::{self, ChildArray, NodeState, NodeStats, Proven};
use super::super::*;
use super::BAYES_GRID_SIZE;

/// Where a node's Bayesian posterior is read from and written to during
/// `BackpropStrategy::update_posterior` -- mirrors the root/edge split
/// `BackpropStrategy::update`'s score accumulation already has (`NodeStats`
/// for the root, a row of the parent's `ChildArray` otherwise), since
/// that's the same representation `SelectContext::child_snapshot`/
/// `current_stats` read from -- a node's posterior needs to live wherever
/// its *parent* looks it up as a candidate child, not on the node's own
/// `NodeStats` field (which, outside MCGS's graph-search mode, isn't what
/// `select` reads for a non-root node).
pub enum PosteriorSlot<'a, A: crate::game::Action> {
    Root(&'a NodeStats),
    Edge(&'a ChildArray<A>, usize),
}

impl<A: crate::game::Action> PosteriorSlot<'_, A> {
    pub(crate) fn own_observation(&self, player: usize) -> (f64, u32) {
        match self {
            PosteriorSlot::Root(stats) => (stats.score(player), stats.num_visits()),
            PosteriorSlot::Edge(children, idx) => {
                (children.score(*idx, player), children.num_visits(*idx))
            }
        }
    }

    pub(crate) fn set(&self, player: usize, mean: f64, variance: f64) {
        match self {
            PosteriorSlot::Root(stats) => stats.set_posterior(player, mean, variance),
            PosteriorSlot::Edge(children, idx) => {
                children.set_posterior(*idx, player, mean, variance)
            }
        }
    }

    pub(crate) fn set_grid(&self, player: usize, grid: [f64; BAYES_GRID_SIZE]) {
        match self {
            PosteriorSlot::Root(stats) => stats.set_posterior_grid(player, grid),
            PosteriorSlot::Edge(children, idx) => children.set_posterior_grid(*idx, player, grid),
        }
    }

    /// See `NodeStats::overwrite_score`'s doc comment -- `MinimaxBackprop`'s
    /// (MCTS-MB-n) own write path, root/edge-dispatched the same way `set`
    /// above is.
    pub(crate) fn overwrite_score(&self, player: usize, mean: f64) {
        match self {
            PosteriorSlot::Root(stats) => stats.overwrite_score(player, mean),
            PosteriorSlot::Edge(children, idx) => children.overwrite_score(*idx, player, mean),
        }
    }
}

/// MCTS-MB-n's per-ancestor minimax backup (Baier & Winands): recomputes
/// `node`'s own per-player value as the outcome of its own mover's best
/// (currently highest-expected-value, among already-searched) child,
/// propagated to *every* player's row via that same child -- not each
/// player's own independent max, since only the mover actually gets to
/// choose. This reads entirely from `node`'s own `ChildArray` edges (which
/// already carry a full per-player utility vector per action, not a single
/// mover-relative value the way `negamax::Score` does), so unlike the
/// paper's own single-value formulation this needs no negamax sign flip
/// between plies and isn't restricted to two-player zero-sum games --
/// ordinary backward induction generalizes to any player count as long as
/// each player is assumed to maximize their own payoff.
///
/// A child counts as "already searched" (participates in the max) only if
/// it has a real tree node (`ChildArray::node_id` is `Some`) -- an
/// unresolved slot or a `prior::PriorStrategy`-seeded-but-unvisited one
/// (pseudo-visits only, no subtree of its own yet) contributes nothing to
/// back up from, the same "unknown leaf, skip it" treatment
/// `derive_pn_dpn` gives an unresolved child. No-ops (leaving the node's
/// Monte-Carlo average from this call's earlier `update` untouched) when
/// no child qualifies, or when `node` isn't `Expanded` at all.
pub(crate) fn derive_minimax_value<A: crate::game::Action>(
    node: &node::Node<A>,
    slot: &PosteriorSlot<A>,
    num_players: usize,
) {
    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };
    let mover = node.player_idx;

    let mut best_idx: Option<usize> = None;
    let mut best_value = f64::NEG_INFINITY;
    for i in 0..children.len() {
        if children.node_id(i).is_none() {
            continue;
        }
        let value = children.expected_score(i, mover);
        if value > best_value {
            best_value = value;
            best_idx = Some(i);
        }
    }
    let Some(best_idx) = best_idx else {
        return;
    };

    for player in 0..num_players {
        slot.overwrite_score(player, children.expected_score(best_idx, player));
    }
}

/// One backward step of the truncated λ-return recursion
/// `G_t ← (1 − λ)·v_boot + λ·G_t` (Sarsa-UCT(λ), Vodopivec, Samothrakis,
/// Šter, "On Monte Carlo Tree Search and Reinforcement Learning", JAIR 2017;
/// γ = 1, zero intermediate reward). `g` enters holding `G_{t+1}` and leaves
/// holding `G_t`, elementwise over the per-player utility vector.
#[inline]
pub(crate) fn td_lambda_step(g: &mut [f64], v_boot: &[f64], lambda: f64) {
    debug_assert_eq!(g.len(), v_boot.len());
    for p in 0..g.len() {
        g[p] = (1.0 - lambda) * v_boot[p] + lambda * g[p];
    }
}

/// Whole-path truncated λ-return, built from the same `td_lambda_step`
/// primitive `update` walks incrementally -- kept as a test-only entry point
/// so the two forms can't drift (AGENTS.md: the instrumentation logic itself
/// gets a fast deterministic test on a hand-verifiable input). `v_boot[i]` is
/// the bootstrap value `V(s_{i+1})` at path node `i`, ordered root-first,
/// `len == returned.len() - 1`; the deepest target is `z`. Returns `G_i` for
/// every path node `i`.
#[cfg(test)]
pub(crate) fn td_lambda_returns(z: &[f64], v_boot: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let mut targets = vec![z.to_vec(); v_boot.len() + 1];
    let mut g = z.to_vec();
    for i in (0..v_boot.len()).rev() {
        td_lambda_step(&mut g, &v_boot[i], lambda);
        targets[i] = g.clone();
    }
    targets
}

/// MaxMCTS(λ)'s off-policy bootstrap (Khandelwal, Liebman, Niekum, Stone,
/// "On the Analysis of Complex Backup Strategies in Monte Carlo Tree
/// Search", ICML 2016): `max` over `node`'s explored children of each
/// player's `expected_score`, or `None` if `node` isn't expanded / has no
/// explored child (the caller then falls back to the on-path Sarsa value).
/// Same "skip the unknown/unvisited-leaf slot" treatment as
/// `derive_power_mean_value`.
pub(crate) fn max_child_bootstrap<A: crate::game::Action>(
    node: &node::Node<A>,
    num_players: usize,
) -> Option<Vec<f64>> {
    let Some(NodeState::Expanded(children)) = node.status() else {
        return None;
    };
    let mut out = vec![f64::NEG_INFINITY; num_players];
    let mut any = false;
    for i in 0..children.len() {
        if children.node_id(i).is_none() || children.num_visits(i) == 0 {
            continue;
        }
        any = true;
        for (p, slot) in out.iter_mut().enumerate() {
            *slot = slot.max(children.expected_score(i, p));
        }
    }
    any.then_some(out)
}

/// Power-UCT's per-ancestor value backup (Dam et al., "Generalized Mean
/// Estimation in Monte-Carlo Tree Search", IJCAI 2020): recomputes `node`'s
/// own per-player value as the visit-weighted Hölder power mean
/// `V_p = (Σ_i (n_i / N) · q_i^p)^{1/p}` over its already-searched children,
/// and overwrites it in place via `slot` (the same "recompute from children,
/// overwrite the score sum" mechanism `derive_minimax_value` uses for
/// MCTS-MB-n). `p = 1` is the plain visit-weighted mean; `p -> inf` is the
/// max over children. The convergence proof for explicitly non-stationary
/// tree nodes is Stochastic-Power-UCT (Dam et al., arXiv 2406.02235).
///
/// UNLIKE `derive_minimax_value`, this is per-player *independent*: player
/// `p`'s row is the power mean of the children's `expected_score(_, p)`, not
/// the mover-selected child's whole utility vector. Consequence: at `p > 1`
/// every player's row is biased toward that player's own best child, so in a
/// 2-player zero-sum game both sides read slightly winning -- the same
/// over-optimism `derive_minimax_value`'s doc comment notes, and the reason
/// `BayesGaussian` combines MAX for the mover but MIN for everyone else.
/// Resolving that adversarial direction for a tunable `p` is deferred to the
/// max/mixed backup work.
///
/// Numerics: utilities live in `[-1, 1]`; `q^p` for non-integer `p` is
/// undefined on negatives, so `q` is shifted to `[0, 1]` before
/// exponentiating and shifted back after. `p == 1` skips `powf` entirely
/// (exact), and `p >= MAX_P` falls back to the plain max to dodge `qs^p`
/// underflowing to `0` (then `0^{1/p} = 0`).
///
/// A child that is `Proven` contributes its exact value (`+1` / `0` / `-1`
/// for this player), never a `powf`-smoothed one, regardless of `p`. A child
/// with no tree node yet (`ChildArray::node_id(i).is_none()`) or zero visits
/// is skipped -- PNS's "unknown leaf" treatment, same as `derive_pn_dpn` and
/// `derive_minimax_value`. No-op (leaving the node's Monte-Carlo average
/// untouched) when no child qualifies or `node` isn't `Expanded`.
///
/// `alpha` mixes the power mean with the plain max over the same per-player
/// child values: `V = (1 - alpha)·V_p + alpha·V_max`, per player,
/// independently. `alpha = 0` is pure Power-UCT (above); `alpha = 1` is the
/// "Full Bellman" / max backup (Schulte & Keller 2014; Asai & Wissow,
/// "Extreme Value Monte Carlo Tree Search", AAAI 2025 / arXiv 2405.18248) at
/// *any* `p` -- that paper frames a max-ward backup as fitting a
/// short-tailed (Uniform / generalized-Pareto) reward distribution rather
/// than a Gaussian, i.e. a modelling choice, not a hack. Its empirical
/// caveat (their §6): pure max helps a *non-performant* base algorithm and
/// tends to hurt an already-strong one, so the useful regime is the interior
/// blend, and `alpha` defaults to `0`. Khandelwal et al. (ICML 2016) report
/// the same domain dependence for `MaxMCTS(λ)`'s max backup: it helps most
/// with sparse/delayed reward and low branching.
///
/// Dead-child exclusion (always on): a child whose `proven()` is
/// `Proven::Win(w)` with `w != node.player_idx` is a proven loss for the
/// mover, who will never enter it, so it contributes nothing to *any*
/// player's backed-up value (not its exact `-1`/`+1`, not `V_max`). This
/// mirrors the EVT paper's dead-end removal (their §3-4): the fit is
/// conditioned on `x > θ` -- you are modelling the tail -- so discarding
/// known-bad samples is correct rather than cheating. A `Proven::Win(mover)`
/// child is left to the MCTS-Solver pass; a `Proven::Draw` child is not dead
/// (a draw is not a mover loss) and contributes its exact `0`.
/// TODO(phase-c): score-bound-dominated (non-`Proven`) subtrees could also
/// be excluded once AmEx/NCES bound reasoning lands.
pub(crate) fn derive_power_mean_value<A: crate::game::Action>(
    node: &node::Node<A>,
    slot: &PosteriorSlot<A>,
    index: &TreeIndex<A>,
    num_players: usize,
    p: f64,
    alpha: f64,
) {
    const POWF_FLOOR: f64 = 1e-12;
    const MAX_P: f64 = 30.0;

    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };

    let mover = node.player_idx;

    for player in 0..num_players {
        let mut n_sum = 0.0_f64;
        let mut acc = 0.0_f64;
        let mut max_q = f64::NEG_INFINITY;
        let mut any = false;

        for i in 0..children.len() {
            let Some(child_id) = children.node_id(i) else {
                continue;
            };
            let n = children.num_visits(i) as f64;
            if n <= 0.0 {
                continue;
            }
            if matches!(index.get(child_id).proven(), Proven::Win(w) if w != mover) {
                continue;
            }
            let q = match index.get(child_id).proven() {
                Proven::Win(w) if w == player => 1.0,
                Proven::Win(_) => -1.0,
                Proven::Draw => 0.0,
                Proven::Unproven => children.expected_score(i, player),
            }
            .clamp(-1.0, 1.0);

            any = true;
            n_sum += n;
            max_q = max_q.max(q);
            let qs = (q + 1.0) / 2.0;
            if p == 1.0 {
                acc += n * qs;
            } else {
                acc += n * qs.max(POWF_FLOOR).powf(p);
            }
        }

        if !any {
            continue;
        }

        let v_p = if p == 1.0 {
            (acc / n_sum) * 2.0 - 1.0
        } else if p >= MAX_P {
            max_q
        } else {
            (acc / n_sum).powf(1.0 / p) * 2.0 - 1.0
        };
        let value = if alpha == 0.0 {
            v_p
        } else if alpha == 1.0 {
            max_q
        } else {
            (1.0 - alpha) * v_p + alpha * max_q
        };
        slot.overwrite_score(player, value);
    }
}

/// Below this temperature `(q - m) / tau` overflows to `±inf` for every
/// non-max entry, so `mellowmax` is exactly the plain max there anyway.
pub(crate) const MELLOW_TAU_FLOOR: f64 = 1e-6;

/// τ-mellowmax of `qs` (Asadi & Littman, "An Alternative Softmax Operator
/// for Reinforcement Learning", ICML 2017):
/// `V = τ · ln( (1/K) · Σ exp(q_k / τ) )`. Unlike the literal log-sum-exp
/// `τ · ln Σ exp(q/τ)` MENTS's paper writes (Xiao et al., NeurIPS 2019),
/// this subtracts the `τ·ln K` constant, so `min qs ≤ V ≤ max qs` at every
/// node instead of `V ≥ max qs` compounding `τ·ln K` per ply up the tree
/// and diverging outside this codebase's `[-1, 1]` utility range. `V → max`
/// as `τ → 0`, `V → mean` as `τ → ∞` (the `Classic` arithmetic backup).
/// Max-shifted for numerical stability; `qs` must be non-empty, `tau > 0`.
#[inline]
pub(crate) fn mellowmax(qs: &[f64], tau: f64) -> f64 {
    debug_assert!(!qs.is_empty() && tau > 0.0);
    let k = qs.len() as f64;
    let m = qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if tau <= MELLOW_TAU_FLOOR {
        return m;
    }
    let sum_exp: f64 = qs.iter().map(|&q| ((q - m) / tau).exp()).sum();
    m + tau * (sum_exp / k).ln()
}

/// MENTS's soft value backup (Xiao et al., NeurIPS 2019): recomputes
/// `node`'s own per-player value as the τ-`mellowmax` of its
/// already-searched children instead of their arithmetic mean, and
/// overwrites it in place via `slot` -- structurally identical to
/// `derive_power_mean_value`, only the aggregation differs. Per-player
/// independent (same 2-player over-optimism caveat, and the same deferral
/// of a mellowmin-for-non-movers refinement). A `Proven` child contributes
/// its exact `±1`/`0` value, never a smoothed one; a child proven a loss
/// for the mover (`Proven::Win(w != node.player_idx)`) is dead and
/// contributes to no player's aggregate; a child with no tree node yet or
/// zero visits is skipped. No-op when no child qualifies or `node` isn't
/// `Expanded`. `tau` is floored at `MELLOW_TAU_FLOOR` (defence in depth --
/// the tuner floor is higher, but a hand-written config could pass 0).
pub(crate) fn derive_softmax_value<A: crate::game::Action>(
    node: &node::Node<A>,
    slot: &PosteriorSlot<A>,
    index: &TreeIndex<A>,
    num_players: usize,
    tau: f64,
) {
    let tau = tau.max(MELLOW_TAU_FLOOR);

    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };

    let mover = node.player_idx;

    for player in 0..num_players {
        let mut qs = Vec::with_capacity(children.len());

        for i in 0..children.len() {
            let Some(child_id) = children.node_id(i) else {
                continue;
            };
            if children.num_visits(i) == 0 {
                continue;
            }
            if matches!(index.get(child_id).proven(), Proven::Win(w) if w != mover) {
                continue;
            }
            let q = match index.get(child_id).proven() {
                Proven::Win(w) if w == player => 1.0,
                Proven::Win(_) => -1.0,
                Proven::Draw => 0.0,
                Proven::Unproven => children.expected_score(i, player),
            }
            .clamp(-1.0, 1.0);
            qs.push(q);
        }

        if qs.is_empty() {
            continue;
        }

        slot.overwrite_score(player, mellowmax(&qs, tau).clamp(-1.0, 1.0));
    }
}

/// Normal-normal conjugate update of a node's *own* observations (its
/// accumulated `score`/`num_visits`, ignoring any children) into a
/// posterior `(mean, variance)` -- Tesauro/Rajan/Segal 2010's leaf-node
/// prior/posterior step, generalized from their 0/1-reward Beta example to
/// this codebase's real-valued utilities. `prior_variance`/`obs_variance`
/// are `BayesGaussian`/`BayesNumeric`'s own hyperparameters; the prior mean
/// is fixed at `0.0`, matching this codebase's symmetric `[-1, 1]` utility
/// convention. Used both as the posterior for a true leaf (no expanded
/// children yet) and, inside `numeric_max_of_pdfs`'s caller, as the
/// per-grid-point PDF an interior node's MAX-of-children combination starts
/// from for any not-yet-visited child.

#[cfg(test)]
mod softmax_tests {
    use super::*;
    use crate::strategies::mcts::node::{ChildArray, Node, NodeState, Proven};
    use crate::strategies::mcts::search::TreeIndex;

    #[test]
    fn mellowmax_matches_hand_computed() {
        // m = 1; sum_exp = e^-1 + 1 = 1.3678794; 1 + ln(1.3678794 / 2).
        let got = mellowmax(&[0.0, 1.0], 1.0);
        let expected = 1.0 + ((1.0f64.exp().recip() + 1.0) / 2.0).ln();
        assert!((got - expected).abs() < 1e-9, "got {got}");
        assert!((got - 0.620_114).abs() < 1e-5, "got {got}");
    }

    #[test]
    fn mellowmax_tau_to_zero_is_max() {
        assert!((mellowmax(&[-0.3, 0.7, 0.1], 1e-9) - 0.7).abs() < 1e-9);
        // The `tau <= MELLOW_TAU_FLOOR` early-return path.
        assert_eq!(mellowmax(&[-0.3, 0.7, 0.1], MELLOW_TAU_FLOOR / 2.0), 0.7);
    }

    #[test]
    fn mellowmax_tau_large_is_mean() {
        let got = mellowmax(&[-0.5, 0.0, 0.5, 1.0], 1e6);
        assert!((got - 0.25).abs() < 1e-3, "got {got}");
    }

    #[test]
    fn mellowmax_shift_invariant() {
        for c in [-0.4, 0.0, 0.3, 1.7] {
            let base = mellowmax(&[0.2, -0.1], 0.7);
            let shifted = mellowmax(&[0.2 + c, -0.1 + c], 0.7);
            assert!((shifted - (base + c)).abs() < 1e-9, "c = {c}");
        }
    }

    #[test]
    fn mellowmax_bracketed_by_min_and_max() {
        let qs = [-0.8, -0.2, 0.1, 0.55, 0.9];
        let lo = qs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for tau in [0.1, 1.0, 3.0] {
            let v = mellowmax(&qs, tau);
            assert!(lo - 1e-12 <= v && v <= hi + 1e-12, "tau = {tau}, v = {v}");
        }
    }

    // Arena check for `derive_softmax_value`, mirroring the
    // `derive_power_mean_value` arena tests in `strategies/tests.rs`.
    #[test]
    fn derive_softmax_value_hand_verified_and_proven_handling() {
        let build = |prove_idx0: Option<Proven>, visit_idx2: bool| {
            let index = TreeIndex::<u32>::new();
            let c0 = index.insert(Node::new(1, 0));
            let c1 = index.insert(Node::new(1, 0));
            let c2 = index.insert(Node::new(1, 0));
            if let Some(p) = prove_idx0 {
                index.get(c0).try_prove(p);
            }

            let children = ChildArray::<u32>::new(vec![10, 11, 12], 2, false, false);
            children.get_or_create_child(0, || c0);
            children.update(0, &[0.2, -0.2]);
            children.get_or_create_child(1, || c1);
            children.update(1, &[0.6, -0.6]);
            children.get_or_create_child(2, || c2);
            if visit_idx2 {
                children.update(2, &[-0.4, 0.4]);
            }

            let root = Node::<u32>::new(0, 0);
            root.expand(|| NodeState::Expanded(children));
            root.stats.update(&[0.0, 0.0]);

            let slot = PosteriorSlot::Root(&root.stats);
            derive_softmax_value(&root, &slot, &index, 2, 0.5);
            let n = root.stats.num_visits().max(1) as f64;
            (root.stats.score(0) / n, root.stats.score(1) / n)
        };

        // Unproven, idx2 unvisited -> mellowmax over the two visited children.
        let (p0, _) = build(None, false);
        assert!((p0 - mellowmax(&[0.2, 0.6], 0.5)).abs() < 1e-9, "p0 = {p0}");

        // idx0 proven Draw contributes exact 0.0, not its MC 0.2.
        let (p0, _) = build(Some(Proven::Draw), false);
        assert!((p0 - mellowmax(&[0.0, 0.6], 0.5)).abs() < 1e-9, "p0 = {p0}");

        // idx0 proven a loss for the mover (player 1 wins) is dead: excluded
        // from every player's aggregate.
        let (p0, p1) = build(Some(Proven::Win(1)), true);
        assert!(
            (p0 - mellowmax(&[0.6, -0.4], 0.5)).abs() < 1e-9,
            "p0 = {p0}"
        );
        assert!(
            (p1 - mellowmax(&[-0.6, 0.4], 0.5)).abs() < 1e-9,
            "p1 = {p1}"
        );
    }

    #[test]
    fn derive_softmax_value_noop_on_bare_leaf() {
        let index = TreeIndex::<u32>::new();
        let leaf = Node::<u32>::new(0, 0);
        leaf.stats.update(&[0.3, -0.3]);
        let slot = PosteriorSlot::Root(&leaf.stats);
        derive_softmax_value(&leaf, &slot, &index, 2, 1.0);
        let n = leaf.stats.num_visits().max(1) as f64;
        assert!((leaf.stats.score(0) / n - 0.3).abs() < 1e-12);
    }
}
