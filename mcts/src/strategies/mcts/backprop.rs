use super::config::GraphStats;
use super::node::{self, ChildArray, NodeState, NodeStats, Proven};
use super::stack::NodeStack;
use super::*;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::TerminalStatus;
use crate::game::Transform;

use rustc_hash::FxHashMap;

/// Converts a playout's terminal check into the `Proven` value it directly
/// witnesses -- used for the zero-length-`Trial` case: a leaf that was
/// terminal before `expand()` ever ran on it (only
/// possible with `expand_threshold > 1`) has no other proof source, since
/// the tree node itself was never resolved to `NodeState::Terminal`.
fn proven_from_terminal<P: PlayerIndex>(status: &TerminalStatus<P>) -> Option<Proven> {
    match status {
        TerminalStatus::NotTerminal => None,
        TerminalStatus::Draw => Some(Proven::Draw),
        TerminalStatus::Winner(w) => Some(Proven::Win(w.to_index())),
    }
}

/// Re-derives `node_id`'s `Proven` status from its (already up to date)
/// children and, if resolved, writes it -- the per-ancestor step of MCTS-
/// Solver's backprop pass. No-ops on a node
/// that isn't `Expanded` (a `Terminal` node's `Proven` was already set once
/// at `expand()`-time and isn't re-derived here).
///
/// Implements the "Standard" update rule (Nijssen & Winands, *Enhancements
/// for Multi-Player Monte-Carlo Tree Search*, CG 2010) for backing up a
/// proof through a node with more than one possible opponent: a win only
/// propagates when *every* fully-resolved child agrees on the same winner
/// (see the `win_q`/`win_q_consistent` tracking below); if children disagree
/// on which opponent wins, the node is left `Unproven` rather than guessing.
/// This is also exactly right at `num_players() == 2`, where "the other
/// player" is the only possible `win_q` and the ambiguous case can't arise.
/// The paper found this the only one of its three proposed rules (the
/// others being Paranoid and First-Winner) that actually improved over no
/// solver at all for a sudden-death game like Focus.
///
/// Deliberately stricter than a literal reading of the Draw
/// clause: `Draw` is only written once *every* explored child is itself
/// already proven (`Proven != Unproven`), not merely explored. Concluding
/// `Draw` while a sibling is still `Unproven` would risk permanently
/// cementing the wrong value -- `Node::try_prove` only ever writes once
/// (compare-exchange-from-`Unproven`), so if that sibling later resolves to
/// `Win(p)` (a real forced win for the mover), there would be no way to
/// correct an already-committed `Draw`. Requiring every child to be proven
/// first costs nothing this rule needs: it can only delay a proof, never
/// weaken one, so it wouldn't observably narrow what the plan asks for
/// (every child *is* fully resolved once search actually finishes proving
/// this subtree). It's also why a Draw takes priority over a `win_q`
/// resolution when both are present among the mover's children: the mover
/// (an OR node, choosing its own best available outcome) always prefers a
/// proven draw over letting any opponent win, so a drawn escape being
/// available proves this node a draw regardless of what its other,
/// worse-for-the-mover children individually prove.
///
/// `pub(crate)` and generic over the action type `A` directly, matching
/// `derive_pn_dpn` below -- it never actually calls a `Game` method either,
/// and this is what lets `tests` (below) exercise the `win_q`/`any_draw`
/// recurrence directly against a hand-built arena instead of only through a
/// full game-playing search.
pub(crate) fn derive_proven<A: crate::game::Action>(node: &node::Node<A>, index: &TreeIndex<A>) {
    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };
    let p = node.player_idx;

    let mut all_children_proven = true;
    let mut win_q: Option<usize> = None;
    let mut win_q_consistent = true;
    let mut any_draw = false;

    for i in 0..children.len() {
        let Some(child_id) = children.node_id(i) else {
            all_children_proven = false;
            continue;
        };
        match index.get(child_id).proven() {
            // Fires on the first winning child found, independent of every
            // other sibling's status (including unexplored ones) -- this is
            // what lets a node become decided the moment *one* winning line
            // is found.
            Proven::Win(w) if w == p => {
                node.try_prove(Proven::Win(p));
                return;
            }
            Proven::Win(w) => match win_q {
                None => win_q = Some(w),
                Some(q) if q == w => {}
                // Only reachable if `num_players() > 2`: two children prove
                // wins for two different opponents. The Standard rule
                // (see this function's doc comment) deliberately doesn't
                // guess which one actually wins from the parent's
                // perspective -- `win_q_consistent` going `false` here
                // suppresses the `Win(q)` branch below, leaving the node
                // `Unproven` (unless `any_draw` already resolves it to
                // `Draw`).
                Some(_) => win_q_consistent = false,
            },
            Proven::Draw => any_draw = true,
            Proven::Unproven => all_children_proven = false,
        }
    }

    if !all_children_proven {
        return;
    }
    if any_draw {
        node.try_prove(Proven::Draw);
    } else if let (Some(q), true) = (win_q, win_q_consistent) {
        node.try_prove(Proven::Win(q));
    }
}

/// Re-derives `node_id`'s proof/disproof numbers from its (already up to
/// date) children -- the magnitude counterpart to `derive_proven`, and the
/// substrate `select::UctPn` ranks children by. Every node is scored
/// "OR-style" relative to its own mover (matching `Proven`'s per-mover
/// framing above), which collapses PNS's usual separate AND/OR recurrences
/// into one uniform negamax pair: `pn(n)` is the minimum, over children, of
/// the child's `dpn` (this node's mover needs only one child the opponent
/// can't escape), and `dpn(n)` is the sum, over children, of the child's
/// `pn` (refuting this node's mover requires every child to fail). An
/// unexplored child slot (a legal action with no tree node yet, already
/// known from `ChildArray`'s fixed action list -- see its doc comment)
/// counts as PNS's "unknown leaf" case, `pn = dpn = 1`, without needing to
/// force it into existence. Saturates at `u32::MAX` rather than
/// overflowing. No-ops on a non-`Expanded` node, for the same reason as
/// `derive_proven`.
///
/// `pub(crate)`, unlike `derive_proven` above, so `strategies::tests` can
/// exercise this recurrence directly against a hand-built arena -- an
/// integer min/sum/saturate recurrence is easy to get subtly wrong (e.g.
/// swapping which side feeds `pn` vs `dpn`) in a way a purely behavioral
/// test wouldn't necessarily catch. Generic over the action type `A`
/// directly (unlike `derive_proven`'s `G: Game` bound) since, like
/// `derive_proven`, it never actually calls a `Game` method -- keeping it
/// to the bound it really needs is what lets a test build an arena without
/// a real `Game` impl to hand.
pub(crate) fn derive_pn_dpn<A: crate::game::Action>(node: &node::Node<A>, index: &TreeIndex<A>) {
    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };

    let mut pn: u32 = u32::MAX;
    let mut dpn: u32 = 0;
    for i in 0..children.len() {
        let (child_pn, child_dpn) = match children.node_id(i) {
            Some(child_id) => {
                let child = index.get(child_id);
                (child.pn(), child.dpn())
            }
            None => (1, 1),
        };
        pn = pn.min(child_dpn);
        dpn = dpn.saturating_add(child_pn);
    }
    node.set_pn_dpn(pn, dpn);
}

/// Second-layer counterpart to `derive_pn_dpn` above (Kowalski et al. 2023,
/// Section VII "Double-Layer PN-MCTS"): the identical negamax recurrence,
/// but reading/writing `pn2`/`dpn2` (goal "not lost") instead of `pn`/`dpn`
/// (goal "won"). Kept as a separate pass rather than folded into
/// `derive_pn_dpn` so the two magnitudes -- which diverge exactly when a
/// `Proven::Draw` leaf is reachable -- stay independently correct and
/// independently testable. Always safe to run alongside the first layer,
/// even for a game that never draws: with no `Proven::Draw` node ever
/// produced, `pn2()`/`dpn2()` collapse to the same values as `pn()`/`dpn()`
/// (see their doc comments), so this is a no-op in cost only, not a
/// conditional feature.
pub(crate) fn derive_pn_dpn2<A: crate::game::Action>(node: &node::Node<A>, index: &TreeIndex<A>) {
    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };

    let mut pn2: u32 = u32::MAX;
    let mut dpn2: u32 = 0;
    for i in 0..children.len() {
        let (child_pn2, child_dpn2) = match children.node_id(i) {
            Some(child_id) => {
                let child = index.get(child_id);
                (child.pn2(), child.dpn2())
            }
            None => (1, 1),
        };
        pn2 = pn2.min(child_dpn2);
        dpn2 = dpn2.saturating_add(child_pn2);
    }
    node.set_pn_dpn2(pn2, dpn2);
}

/// Re-derives every player's per-player proof number on `node` from its
/// children's already-updated numbers -- Generalized Proof-Number MCTS
/// (Kowalski, Soemers, Kosakowski & Winands, arXiv:2506.13249, §3.1,
/// Alg. 1 `UPDATEPROOFNUMBER`). For each player `p`: the node is an OR node
/// in `p`'s proof tree when `p` is the node's own mover (`min` over
/// children -- one forcing move suffices) and an AND node otherwise (`sum`
/// -- every opponent reply must fail). An unexpanded child slot counts as
/// PNS's unknown leaf, `1`, matching `derive_pn_dpn`'s treatment of the
/// same case. A proven child contributes `0`/`u32::MAX` directly via
/// `Node::player_pn`. Saturates rather than overflowing.
///
/// Unlike `derive_pn_dpn`'s single per-mover pair, this keeps `P` numbers
/// per node, which is what makes the technique sound for more than two
/// players and removes the need for a separate disproof-number recurrence.
/// `A`-generic and `pub(crate)` for the same reason as `derive_pn_dpn`: a
/// hand-built-arena test shouldn't need a real `Game`.
pub(crate) fn derive_player_pn<A: crate::game::Action>(
    node: &node::Node<A>,
    index: &TreeIndex<A>,
    num_players: usize,
) {
    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };
    let mover = node.player_idx;

    for p in 0..num_players {
        let is_or = p == mover;
        let mut acc: u32 = if is_or { u32::MAX } else { 0 };
        for i in 0..children.len() {
            let child_pn = match children.node_id(i) {
                Some(child_id) => index.get(child_id).player_pn(p),
                None => 1,
            };
            if is_or {
                acc = acc.min(child_pn);
            } else {
                acc = acc.saturating_add(child_pn);
            }
        }
        node.set_player_pn(p, acc);
    }
}

/// Re-derives `node`'s Score-Bounded MCTS interval `[pess, opti]` (Cazenave
/// & Saffidine, *Score Bounded Monte-Carlo Tree Search*, CG 2010, §3.1-3.2)
/// from its children's already-updated intervals -- the graded-score
/// counterpart to `derive_proven`. Bounds are always from player 0's
/// ("Max's") perspective, so a node combines its children by `max` when its
/// own mover is player 0 (`maximizing`) and by `min` otherwise, matching the
/// paper's Max-node/Min-node rules. An unexpanded child slot counts as the
/// paper's dummy child `[score_min, score_max]` (no information yet); a
/// resolved child that no `derive_score_bounds` pass has reached is still on
/// its `i32::MIN`/`i32::MAX` seed, which is the identity element for both
/// `max` and `min` and lands on the same clamped result, so the two cases
/// need no distinction here.
///
/// Two-player only -- the interval is a single Max-vs-Min scalar. Gated by
/// the caller on `Game::score_bounds()` being `Some` and
/// `Game::num_players() == 2`. `A`-generic and `pub(crate)` for the same
/// reason as `derive_pn_dpn`: an integer min/max/clamp recurrence is easy to
/// get subtly wrong, and a hand-built-arena test shouldn't need a real
/// `Game`.
pub(crate) fn derive_score_bounds<A: crate::game::Action>(
    node: &node::Node<A>,
    index: &TreeIndex<A>,
    score_min: i32,
    score_max: i32,
    maximizing: bool,
) {
    let Some(NodeState::Expanded(children)) = node.status() else {
        return;
    };

    let mut pess = if maximizing { i32::MIN } else { i32::MAX };
    let mut opti = if maximizing { i32::MIN } else { i32::MAX };
    for i in 0..children.len() {
        let (child_pess, child_opti) = match children.node_id(i) {
            Some(child_id) => {
                let child = index.get(child_id);
                (child.pess(), child.opti())
            }
            None => (score_min, score_max),
        };
        if maximizing {
            pess = pess.max(child_pess);
            opti = opti.max(child_opti);
        } else {
            pess = pess.min(child_pess);
            opti = opti.min(child_opti);
        }
    }
    node.set_score_bounds(
        pess.clamp(score_min, score_max),
        opti.clamp(score_min, score_max),
    );
}

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
    fn own_observation(&self, player: usize) -> (f64, u32) {
        match self {
            PosteriorSlot::Root(stats) => (stats.score(player), stats.num_visits()),
            PosteriorSlot::Edge(children, idx) => {
                (children.score(*idx, player), children.num_visits(*idx))
            }
        }
    }

    fn set(&self, player: usize, mean: f64, variance: f64) {
        match self {
            PosteriorSlot::Root(stats) => stats.set_posterior(player, mean, variance),
            PosteriorSlot::Edge(children, idx) => {
                children.set_posterior(*idx, player, mean, variance)
            }
        }
    }

    fn set_grid(&self, player: usize, grid: [f64; BAYES_GRID_SIZE]) {
        match self {
            PosteriorSlot::Root(stats) => stats.set_posterior_grid(player, grid),
            PosteriorSlot::Edge(children, idx) => children.set_posterior_grid(*idx, player, grid),
        }
    }

    /// See `NodeStats::overwrite_score`'s doc comment -- `MinimaxBackprop`'s
    /// (MCTS-MB-n) own write path, root/edge-dispatched the same way `set`
    /// above is.
    fn overwrite_score(&self, player: usize, mean: f64) {
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
fn conjugate_leaf_posterior<A: crate::game::Action>(
    slot: &PosteriorSlot<A>,
    player: usize,
    prior_variance: f64,
    obs_variance: f64,
) -> (f64, f64) {
    let (score, num_visits) = slot.own_observation(player);
    if num_visits == 0 {
        return (0.0, prior_variance);
    }
    let n = num_visits as f64;
    let sample_mean = score / n;
    let posterior_precision = 1.0 / prior_variance + n / obs_variance;
    let posterior_variance = 1.0 / posterior_precision;
    let posterior_mean = posterior_variance * (n * sample_mean / obs_variance);
    (posterior_mean, posterior_variance)
}

fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// Abramowitz & Stegun 7.1.26, max error ~1.5e-7 -- plenty for a UCB-style
// exploration bound, and avoids pulling in a stats crate for one function.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Clark (1961)'s closed-form mean/variance of `max(X1, X2)` for two
/// independent Gaussians -- Tesauro/Rajan/Segal 2010 section 4.1, equations
/// following "we reduce the computation time by restructuring the equations
/// above to yield...". `BayesGaussian` folds this pairwise, left to right,
/// over however many children a node has (the paper's simpler O(K)
/// alternative to its O(K^2 log K) min-error pairing scheme).
pub(crate) fn clark_max_of_gaussians(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> (f64, f64) {
    let sigma_m = (sigma1 * sigma1 + sigma2 * sigma2).sqrt();
    if sigma_m == 0.0 {
        return (mu1.max(mu2), 0.0);
    }
    let alpha = (mu1 - mu2) / sigma_m;
    let phi = standard_normal_pdf(alpha);
    let big_phi = standard_normal_cdf(alpha);
    let f1 = alpha * big_phi + phi;
    let f2 =
        alpha * alpha * big_phi * (1.0 - big_phi) + (1.0 - 2.0 * big_phi) * alpha * phi - phi * phi;
    let mean = mu2 + sigma_m * f1;
    let variance =
        sigma2 * sigma2 + (sigma1 * sigma1 - sigma2 * sigma2) * big_phi + sigma_m * sigma_m * f2;
    (mean, variance.max(0.0))
}

/// `min(X1, X2)`'s mean/variance, via `min(X1, X2) = -max(-X1, -X2)` --
/// negating both means leaves the (order-independent) variance unchanged.
/// Needed alongside `clark_max_of_gaussians` because an interior node's
/// combined posterior for a player who is *not* that node's own mover must
/// track the mover's adversarial choice (the mover picks the child that
/// minimizes a non-mover's value, in this codebase's zero-sum two-player
/// convention -- see `update_posterior`'s `mover` parameter).
pub(crate) fn clark_min_of_gaussians(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> (f64, f64) {
    let (neg_mean, variance) = clark_max_of_gaussians(-mu1, sigma1, -mu2, sigma2);
    (-neg_mean, variance)
}

/// Left-folds `combine` (`clark_max_of_gaussians`/`clark_min_of_gaussians`)
/// pairwise over `posteriors`' `(mean, variance)` pairs, returning the
/// combined `(mean, variance)` -- or `None` for an empty input. `combine`
/// takes/returns `(mean, variance)`, but its own two-Gaussian formula is
/// written in terms of *standard deviation* (`sigma1`/`sigma2`), so each
/// fold step must convert the running variance to sigma before feeding it
/// back in, not thread it straight through as if it already were one: doing
/// that squares an already-squared quantity every step, collapsing the
/// combined variance toward zero well before the true combined spread is
/// reached once there are more than two inputs.
fn fold_gaussian_extremum(
    posteriors: impl Iterator<Item = (f64, f64)>,
    combine: fn(f64, f64, f64, f64) -> (f64, f64),
) -> Option<(f64, f64)> {
    posteriors
        .map(|(mu, var)| (mu, var.max(0.0).sqrt()))
        .reduce(|(mu0, sigma0), (mu1, sigma1)| {
            let (mean, variance) = combine(mu0, sigma0, mu1, sigma1);
            (mean, variance.max(0.0).sqrt())
        })
        .map(|(mean, sigma)| (mean, sigma * sigma))
}

/// Grid resolution `BayesNumeric` discretizes each node's posterior PDF
/// over -- also the fixed-size backing array for `PlayerStats::posterior_grid`.
pub const BAYES_GRID_SIZE: usize = 64;

fn grid_points(lo: f64, hi: f64) -> [f64; BAYES_GRID_SIZE] {
    let mut points = [0.0; BAYES_GRID_SIZE];
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    for (i, p) in points.iter_mut().enumerate() {
        *p = lo + step * i as f64;
    }
    points
}

fn trapezoid_integral(values: &[f64; BAYES_GRID_SIZE], step: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..BAYES_GRID_SIZE - 1 {
        sum += (values[i] + values[i + 1]) * 0.5 * step;
    }
    sum
}

fn normal_pdf_grid(mean: f64, variance: f64, lo: f64, hi: f64) -> [f64; BAYES_GRID_SIZE] {
    let sigma = variance.max(1e-12).sqrt();
    let points = grid_points(lo, hi);
    let mut pdf = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        let z = (points[i] - mean) / sigma;
        pdf[i] = (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    }
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    let mass = trapezoid_integral(&pdf, step);
    if mass > 0.0 {
        for v in pdf.iter_mut() {
            *v /= mass;
        }
    }
    pdf
}

fn cdf_from_pdf(pdf: &[f64; BAYES_GRID_SIZE], step: f64) -> [f64; BAYES_GRID_SIZE] {
    let mut cdf = [0.0; BAYES_GRID_SIZE];
    let mut acc = 0.0;
    for i in 0..BAYES_GRID_SIZE {
        if i > 0 {
            acc += (pdf[i - 1] + pdf[i]) * 0.5 * step;
        }
        cdf[i] = acc;
    }
    cdf
}

fn pdf_from_cdf(cdf: &[f64; BAYES_GRID_SIZE], step: f64) -> [f64; BAYES_GRID_SIZE] {
    let mut pdf = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        pdf[i] = if i == 0 {
            (cdf[1] - cdf[0]) / step
        } else if i == BAYES_GRID_SIZE - 1 {
            (cdf[i] - cdf[i - 1]) / step
        } else {
            (cdf[i + 1] - cdf[i - 1]) / (2.0 * step)
        }
        .max(0.0);
    }
    pdf
}

fn mean_variance_from_pdf(pdf: &[f64; BAYES_GRID_SIZE], lo: f64, hi: f64) -> (f64, f64) {
    let points = grid_points(lo, hi);
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    let mut weighted = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        weighted[i] = pdf[i] * points[i];
    }
    let mean = trapezoid_integral(&weighted, step);
    let mut sq = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        sq[i] = pdf[i] * (points[i] - mean) * (points[i] - mean);
    }
    let variance = trapezoid_integral(&sq, step).max(0.0);
    (mean, variance)
}

/// Numeric MAX distribution of `k` independent grid-discretized PDFs: a
/// value `v` is `<= max(X1..Xk)` iff every `Xi <= v`, so the parent's CDF is
/// the elementwise product of the children's CDFs; differentiating that
/// product back to a PDF (via central differences) gives the parent's
/// distribution (Tesauro/Rajan/Segal 2010 section 4, "a more convenient
/// calculation is to first compute the parent CDF... as the product of
/// child CDFs"). `BayesNumeric`'s exact (non-Gaussian-approximated)
/// counterpart to `clark_max_of_gaussians`.
pub(crate) fn numeric_max_of_pdfs(
    pdfs: &[[f64; BAYES_GRID_SIZE]],
    lo: f64,
    hi: f64,
) -> [f64; BAYES_GRID_SIZE] {
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    let mut cdf_product = [1.0; BAYES_GRID_SIZE];
    for pdf in pdfs {
        let cdf = cdf_from_pdf(pdf, step);
        for i in 0..BAYES_GRID_SIZE {
            cdf_product[i] *= cdf[i];
        }
    }
    pdf_from_cdf(&cdf_product, step)
}

/// Numeric MIN distribution of `k` independent grid-discretized PDFs: a
/// value `v` is `>= min(X1..Xk)`'s complement iff every `Xi > v`, so
/// `P(min <= v) = 1 - product(1 - CDF_i(v))` (the survival-function dual of
/// `numeric_max_of_pdfs`'s CDF product). Needed for the same reason
/// `clark_min_of_gaussians` is: an interior node's posterior must be
/// combined via MIN, not MAX, for every player who isn't that node's own
/// mover (see `update_posterior`'s `mover` parameter).
pub(crate) fn numeric_min_of_pdfs(
    pdfs: &[[f64; BAYES_GRID_SIZE]],
    lo: f64,
    hi: f64,
) -> [f64; BAYES_GRID_SIZE] {
    let step = (hi - lo) / (BAYES_GRID_SIZE - 1) as f64;
    let mut survival_product = [1.0; BAYES_GRID_SIZE];
    for pdf in pdfs {
        let cdf = cdf_from_pdf(pdf, step);
        for i in 0..BAYES_GRID_SIZE {
            survival_product[i] *= 1.0 - cdf[i];
        }
    }
    let mut cdf_min = [0.0; BAYES_GRID_SIZE];
    for i in 0..BAYES_GRID_SIZE {
        cdf_min[i] = 1.0 - survival_product[i];
    }
    pdf_from_cdf(&cdf_min, step)
}

pub trait BackpropStrategy: Clone + Sync + Send + Default {
    /// Whether this strategy populates `PlayerStats::posterior_mean`/
    /// `posterior_variance` (and, for `BayesNumeric`, `posterior_grid`) via
    /// `update_posterior` below. `BayesUct1`/`BayesUct2`
    /// (`select/bayes.rs`) set `Requirements::needs_posterior`, checked
    /// against this at `SearchConfig::validate()`-time so that pairing is
    /// caught as a config error rather than silently reading zeroed fields.
    fn provides_posterior(&self) -> bool {
        false
    }

    /// Recomputes and writes one node's Bayesian posterior for `player`:
    /// the normal-normal conjugate posterior from the node's own
    /// observations if it has no expanded children yet, otherwise the
    /// extremum distribution over its (already-updated-this-call, or
    /// stale-from-a-previous-call for off-path siblings -- tolerated the
    /// same way `derive_pn_dpn`'s partial recomputation already is)
    /// children's posteriors. The extremum direction depends on `mover`
    /// (this node's own `Node::player_idx`, i.e. which player actually
    /// chooses among these children): MAX when `player == mover` (that
    /// player picks their own best child), MIN otherwise (the mover's
    /// choice is adversarial to every other player, in this codebase's
    /// zero-sum two-player convention -- picking a MAX unconditionally
    /// here previously gave every non-mover player a systematically
    /// over-optimistic posterior, since it credited them with a choice
    /// only the actual mover gets to make). Called once per player, per
    /// ancestor node, from the default `update` body below -- default
    /// no-op, only overridden by `BayesGaussian`/`BayesNumeric`.
    fn update_posterior<A: crate::game::Action>(
        &self,
        _player: usize,
        _mover: usize,
        _slot: &PosteriorSlot<A>,
        _own_children: Option<&ChildArray<A>>,
    ) {
    }

    /// How many plies of ancestors, counting from (but not including) the
    /// just-backpropagated leaf, get their own per-player value *recomputed
    /// from their children and overwritten in place* by `recompute_value`
    /// below, instead of being left as the ordinary Monte-Carlo average
    /// `update`'s per-node block already wrote earlier this same call. `0`
    /// (the default) disables the pass entirely -- every ancestor keeps the
    /// plain averaging behavior. `u32::MAX` means "every ancestor".
    /// Overridden by `MinimaxBackprop` (MCTS-MB-n) and `PowerMeanBackprop`
    /// (Power-UCT).
    fn recompute_depth(&self) -> u32 {
        0
    }

    /// Recompute `node`'s own per-player value from its (already-updated-
    /// this-call) children and overwrite it in place via `slot`. Called once
    /// per ancestor within `recompute_depth()` plies of the leaf, leaf-to-
    /// root, from `update` below. Default no-op. `index` is passed for
    /// operators that need a child's proven status (`PowerMeanBackprop`);
    /// operators that don't (`MinimaxBackprop`) ignore it.
    fn recompute_value<A: crate::game::Action>(
        &self,
        _node: &node::Node<A>,
        _slot: &PosteriorSlot<A>,
        _index: &TreeIndex<A>,
        _num_players: usize,
    ) {
    }

    #[allow(clippy::too_many_arguments)]
    fn update_amaf<G: Game>(
        &self,
        parent_id_opt: Option<index::Id>,
        parent_incoming_sym: Transform,
        trace: &[(G::A, usize)],
        index: &TreeIndex<G::A>,
        node_id: index::Id,
        utilities: &[f64],
    ) {
        // NOTE: O(n) here, but amaf could be calculated top down
        let node = index.get(node_id);
        if !node.is_root() {
            // parent_id must come from the caller's (parent, node) pair for
            // *this* node, not `stack.parent_id()` (always the leaf's
            // parent) — using the latter silently attributed AMAF updates
            // to the wrong parent for every non-leaf node in a multi-level
            // stack.
            let parent_id = parent_id_opt.unwrap();
            debug_assert_ne!(parent_id, node_id);
            debug_assert!(index.get(parent_id).is_expanded());
            let parent = index.get(parent_id);
            let children = parent.children();
            // Maps directly to the sibling's index in `children`, rather
            // than to its arena `Id`, so a match below can call
            // `add_amaf` without a second (previously O(n)) reverse lookup
            // from `Id` back to array position. `parent_incoming_sym` (not
            // `children.sym(i)`, a different value per `real_action`'s doc
            // comment) is `parent`'s own incoming symmetry.
            let sibling_actions: FxHashMap<_, usize> = (0..children.len())
                .filter_map(|i| {
                    children
                        .node_id(i)
                        .map(|_| (node::real_action::<G>(children, i, parent_incoming_sym), i))
                })
                .collect();

            // The player who could have chosen any of `parent_id`'s sibling
            // actions is `parent_id`'s own mover -- not the resulting
            // child's `player_idx`, which names the mover of the *next* ply
            // (the opposite player in an alternating game). Matching against
            // the child's `player_idx` here inverted the check.
            let mover = parent.player_idx;
            for (action, p) in trace {
                if *p == mover {
                    if let Some(&idx) = sibling_actions.get(action) {
                        (0..G::num_players()).for_each(|i| {
                            children.add_amaf(idx, i, utilities[i]);
                        })
                    }
                }
            }
        }
    }

    fn update_grave<G: Game>(
        &self,
        trace: &[(G::A, usize)],
        index: &TreeIndex<G::A>,
        global: &TreeStats<G>,
        node_id: index::Id,
        utilities: &[f64],
    ) {
        let node = index.get(node_id);
        if !node.is_root() {
            let mut grave = global.grave.write().unwrap();
            for (action, p) in trace {
                let players = grave
                    .entry(node.hash)
                    .or_insert_with(|| vec![Default::default(); G::num_players()]);
                let player = players.get_mut(*p).unwrap();
                let grave_stats = player.entry(action.clone()).or_default();
                grave_stats.num_visits += 1;
                grave_stats.score += utilities[*p];
            }
        }
    }

    // TODO: cleanup the arguments to this, or just move it to TreeSearch
    #[allow(clippy::too_many_arguments)]
    fn update<G>(
        &self,
        stack: &NodeStack<G::A>,
        global: &TreeStats<G>,
        index: &TreeIndex<G::A>,
        root_stats: &NodeStats,
        root_state: &G::S,
        canonicalizes: bool,
        trial: simulate::Trial<G>,
        flags: BackpropFlags,
        use_mcts_solver: bool,
        graph_stats: Option<GraphStats>,
    ) where
        G: Game,
    {
        // init_amaf: AMAF | GRAVE | GLOBAL | LGR | LGR2
        let needs_actions =
            flags.amaf() || flags.grave() || flags.global() || flags.lgr() || flags.lgr2();
        let mut amaf_actions = if needs_actions {
            trial.actions.clone()
        } else {
            vec![]
        };
        // Every stack node's own incoming symmetry, replayed from
        // `root_state` -- see `crate::symmetry::incoming_sym`'s doc comment for why
        // this can't be a value cached on the edge. Computed once, up
        // front, only when `needs_actions` actually reads a tree action
        // below.
        let incoming_syms = if needs_actions {
            stack.incoming_syms::<G>(index, root_state, canonicalizes).0
        } else {
            Default::default()
        };

        // `trial.terminal` already carries the winner if `playout` ended
        // naturally (rather than hitting the depth cutoff) -- reuse it
        // instead of re-deriving utilities from `trial.state` via
        // `Game::compute_utilities`, which for games like Druid would redo
        // the same connectivity scan `playout` just paid for. A depth-cutoff
        // trial falls back to `cutoff_utilities` next (MCTS-IC-E/-M's hook,
        // `simulate::EvaluatedCutoff`) before finally defaulting to
        // `compute_utilities`'s `winner`-based (draw-for-non-terminal) score.
        let utilities = trial
            .terminal
            .utilities(G::num_players())
            .or_else(|| trial.cutoff_utilities.clone())
            .unwrap_or_else(|| G::compute_utilities(&trial.state));
        let mut is_leaf = true;
        let recompute_depth = self.recompute_depth();
        // Ply distance of the entry currently being processed from the
        // backpropagated leaf, `0` for the leaf's own entry. Only
        // meaningful (and only read) when `recompute_depth > 0`.
        for (ply_from_leaf, (parent_entry_opt, (node_id, node_idx))) in
            (0u32..).zip(stack.reverse_pairs2())
        {
            let parent_id_opt = parent_entry_opt.map(|(id, _)| *id);
            debug_assert!(
                (parent_id_opt.is_some() && !index.get(*node_id).is_root())
                    || (parent_id_opt.is_none() && index.get(*node_id).is_root())
            );
            if index.get(*node_id).is_root() {
                if graph_stats.is_some_and(GraphStats::uses_nodes) {
                    index.get(*node_id).stats.update(&utilities);
                } else {
                    root_stats.update(&utilities);
                }
            } else {
                let parent_id = parent_id_opt.unwrap();
                debug_assert_ne!(parent_id, *node_id);
                let parent = index.get(parent_id);
                // `node_idx` is this entry's own stored slot in `parent`'s
                // `ChildArray` -- see `stack::StackEntry`'s doc comment for
                // why this must come from the traversed path itself, never a
                // `ChildArray` reverse lookup by `Id` (unsound once several
                // of a parent's actions canonicalize to the same shared
                // child under symmetry-aware merging).
                let idx = *node_idx;
                let children = parent.children();
                if graph_stats.is_none_or(GraphStats::uses_edges) {
                    children.update(idx, &utilities);
                    children.remove_virtual_loss(idx);
                }
                if graph_stats.is_some_and(GraphStats::uses_nodes) {
                    let node = index.get(*node_id);
                    node.stats.update(&utilities);
                    node.stats.remove_virtual_loss();
                }
            }

            // Bayesian posterior: recompute this node's (mean, variance) for
            // every player, same leaf-to-root timing as the MCTS-Solver pass
            // below -- by the time an ancestor is processed here, its
            // on-path child already has this call's updated posterior.
            // No-op (the trait default) for every non-Bayesian backprop
            // strategy, so this costs nothing when unused.
            if self.provides_posterior() {
                let node = index.get(*node_id);
                let own_children = node.is_expanded().then(|| node.children());
                let mover = node.player_idx;
                if node.is_root() {
                    let slot = PosteriorSlot::Root(root_stats);
                    for player in 0..G::num_players() {
                        self.update_posterior(player, mover, &slot, own_children);
                    }
                } else {
                    let parent_id = parent_id_opt.unwrap();
                    let parent = index.get(parent_id);
                    let idx = *node_idx;
                    let slot = PosteriorSlot::Edge(parent.children(), idx);
                    for player in 0..G::num_players() {
                        self.update_posterior(player, mover, &slot, own_children);
                    }
                }
            }

            // Configured backup operator: recompute this node's own per-
            // player value from its own children and overwrite it in place,
            // for ancestors strictly within `recompute_depth` plies of the
            // backpropagated leaf. `ply_from_leaf == 0` (the leaf's own
            // entry) is deliberately excluded -- its value *is* this trial's
            // rollout/evaluator result, not something to re-derive from
            // children. No-op (the default `recompute_depth() == 0`) for
            // `Classic` and every other strategy that doesn't override the
            // pair, so this costs nothing when unused.
            if recompute_depth > 0 && ply_from_leaf >= 1 && ply_from_leaf <= recompute_depth {
                let node = index.get(*node_id);
                let slot = if node.is_root() {
                    PosteriorSlot::Root(root_stats)
                } else {
                    let parent_id = parent_id_opt.unwrap();
                    PosteriorSlot::Edge(index.get(parent_id).children(), *node_idx)
                };
                self.recompute_value(node, &slot, index, G::num_players());
            }

            // MCTS-Solver: derive/propagate proven status for this node.
            // Runs unconditionally on every backprop call for every node on
            // the visited path (not gated on the trial having ended at a
            // terminal state) -- a node can newly become provable on any
            // call once its last unproven child resolves, and the walk
            // already visits every ancestor every time regardless of trial
            // outcome, so no extra triggering logic is needed.
            if use_mcts_solver {
                let node = index.get(*node_id);
                // Zero-length trial: `playout`'s very first terminal check
                // already found this exact leaf state terminal, which is
                // the one case a rollout's endpoint *is* the tree leaf. The
                // only leaf this can apply to is the one `expand()` hasn't
                // resolved yet (below `expand_threshold`, still a bare
                // "leaf" node, not `Expanded` or `Terminal`) -- an already-
                // `Expanded` node's state is never terminal, since
                // `expand()` would have marked it `Terminal` instead.
                if is_leaf && trial.depth == 0 {
                    if let Some(proven) = proven_from_terminal(&trial.terminal) {
                        node.try_prove(proven);
                    }
                    if G::score_bounds().is_some() {
                        if let Some(score) = G::terminal_score(&trial.state) {
                            node.set_terminal_score(score);
                        }
                    }
                }
                derive_proven(node, index);
                derive_pn_dpn(node, index);
                derive_pn_dpn2(node, index);
                // GPN-MCTS: per-player proof numbers alongside the per-mover
                // pair above. A no-op in cost only for a search whose select
                // strategy never reads them (same posture as `derive_pn_dpn2`).
                derive_player_pn(node, index, G::num_players());
                // Score-Bounded MCTS: propagate the graded-score interval
                // alongside the win/draw/loss proof. Two-player only, and a
                // no-op unless the game overrides `Game::score_bounds()`.
                if G::num_players() == 2 {
                    if let Some((score_min, score_max)) = G::score_bounds() {
                        derive_score_bounds(
                            node,
                            index,
                            score_min,
                            score_max,
                            node.player_idx == 0,
                        );
                    }
                }
            }
            is_leaf = false;

            // update: AMAF
            //
            // `amaf_actions` (not the fixed `trial.actions`) so that
            // ancestors above the immediate parent of the playout's leaf
            // see actions played across the whole rest of the simulation --
            // both the remaining tree-path descent and the playout -- not
            // just the playout suffix. Mirrors GRAVE's use of the same
            // accumulator below.
            if flags.amaf() {
                let parent_incoming_sym = parent_id_opt
                    .and_then(|id| incoming_syms.get(&id))
                    .copied()
                    .unwrap_or(Transform::IDENTITY);
                self.update_amaf::<G>(
                    parent_id_opt,
                    parent_incoming_sym,
                    &amaf_actions,
                    index,
                    *node_id,
                    &utilities,
                );
            } else if flags.grave() {
                self.update_grave::<G>(&amaf_actions, index, global, *node_id, &utilities);
            }

            // push_action: AMAF | GRAVE | GLOBAL | LGR | LGR2
            if flags.amaf() || flags.grave() || flags.global() || flags.lgr() || flags.lgr2() {
                let node = index.get(*node_id);
                if !node.is_root() {
                    let parent_id = parent_id_opt.unwrap();
                    let idx = *node_idx;
                    let parent_incoming_sym = incoming_syms
                        .get(&parent_id)
                        .copied()
                        .unwrap_or(Transform::IDENTITY);
                    let action = node::real_action::<G>(
                        index.get(parent_id).children(),
                        idx,
                        parent_incoming_sym,
                    );
                    // The edge from `parent_id` to `node_id` was played by
                    // whoever was to move *at* `parent_id` -- `node_id`'s own
                    // `player_idx` is the mover of `node_id`'s own outgoing
                    // edges (i.e. the *next* ply), the opposite player in an
                    // alternating game.
                    amaf_actions.push((action, index.get(parent_id).player_idx));
                };
            }
        }

        // update: GLOBAL
        if flags.global() {
            let mut actions = global.actions.write().unwrap();
            for (action, p) in &amaf_actions {
                let action_stats = actions.entry(action.clone()).or_default();
                action_stats.num_visits += 1;
                action_stats.score += utilities[*p];

                let mut player_actions = global.player_actions[*p].write().unwrap();
                let player_action_stats = player_actions.entry(action.clone()).or_default();
                player_action_stats.num_visits += 1;
                player_action_stats.score += utilities[*p];
            }
        }

        // update: NST -- bigram extension of GLOBAL/MAST, keyed by
        // (prev_action, action) instead of just `action`.
        //
        // `amaf_actions` is *not* in chronological play order: it's built
        // as [playout suffix (already chronological)] followed by [tree-path
        // actions appended leaf-to-root, i.e. reverse chronological] -- fine
        // for GLOBAL/GRAVE/AMAF above, which only ever treat it as an
        // unordered bag of (action, player) pairs, but NST's bigram context
        // needs true consecutive-ply order. Reconstruct it by reversing the
        // tree-path segment (the tail past `trial.actions.len()`) back to
        // root-to-leaf order and prepending it to the (already-chronological)
        // playout suffix -- the tree-path is always played before the
        // playout continues from its leaf.
        if flags.nst() {
            let mut chronological = amaf_actions[trial.actions.len()..].to_vec();
            chronological.reverse();
            chronological.extend(trial.actions.iter().cloned());

            for pair in chronological.windows(2) {
                let (prev_action, _) = &pair[0];
                let (action, p) = &pair[1];
                let mut bigram_actions = global.player_bigram_actions[*p].write().unwrap();
                let bigram_stats = bigram_actions
                    .entry((prev_action.clone(), action.clone()))
                    .or_default();
                bigram_stats.num_visits += 1;
                bigram_stats.score += utilities[*p];
            }
        }

        // update: LGR -- last-write-wins reply table, keyed by (mover,
        // opponent's preceding move). Only the winning player(s) of this
        // trial teach their table anything -- a losing player's replies
        // aren't "good replies" by definition, so recording them would just
        // add noise a plain last-write-wins map (no visit counting to drown
        // it back out) can't recover from. "Won" is "this player's utility
        // is (tied for) the max of the trial" rather than a hardcoded
        // `== 1.0`, so this stays correct under any utility normalization.
        if flags.lgr() {
            let max_utility = utilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut chronological = amaf_actions[trial.actions.len()..].to_vec();
            chronological.reverse();
            chronological.extend(trial.actions.iter().cloned());

            for pair in chronological.windows(2) {
                let (prev_action, _) = &pair[0];
                let (action, p) = &pair[1];
                if utilities[*p] >= max_utility {
                    let mut replies = global.player_replies[*p].write().unwrap();
                    replies.insert(prev_action.clone(), action.clone());
                }
            }
        }

        // update: LGR2 -- LGRF-2's own 2-ply table, keyed by (this
        // player's own previous move, the opponent's reply to it). Same
        // windowed chronological reconstruction as LGR above, but over
        // triples: `window[0]` is this player's own earlier move,
        // `window[1]` is the opponent's reply, `window[2]` is this
        // player's next move -- the "reply to a reply" LGRF-2 uses as
        // context. `window[0]` and `window[2]` must belong to the same
        // player for the triple to have a well-defined own-move context;
        // it's skipped otherwise (only possible in non-alternating
        // turn orders).
        //
        // Unlike LGR's plain table, this one *forgets*: a losing player's
        // move is removed from the table when it's still the entry
        // recorded for that context, so a reply that stops winning stops
        // being played instead of lingering until some later winning
        // trial happens to overwrite it.
        if flags.lgr2() {
            let max_utility = utilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut chronological = amaf_actions[trial.actions.len()..].to_vec();
            chronological.reverse();
            chronological.extend(trial.actions.iter().cloned());

            for window in chronological.windows(3) {
                let (own_prev_action, p0) = &window[0];
                let (opp_action, _p1) = &window[1];
                let (action, p2) = &window[2];
                if p0 != p2 {
                    continue;
                }
                let context = (own_prev_action.clone(), opp_action.clone());
                let mut replies2 = global.player_replies2[*p2].write().unwrap();
                if utilities[*p2] >= max_utility {
                    replies2.insert(context, action.clone());
                } else if replies2.get(&context) == Some(action) {
                    replies2.remove(&context);
                }
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct Classic;

impl BackpropStrategy for Classic {}

////////////////////////////////////////////////////////////////////////////////

/// MCTS-MB-n (Baier & Winands): backpropagation-phase hybrid -- within
/// `depth` plies of the just-backpropagated leaf, overwrite (not average
/// into) each ancestor's own per-player value with `derive_minimax_value`'s
/// backward-induction backup from its own already-updated children, instead
/// of leaving it as the plain Monte-Carlo average `BackpropStrategy::update`
/// otherwise produces. The paper's own Breakthrough numbers (2015,
/// domain-independent MR/MS/MB, no evaluation function) found MB-2 the
/// strongest of the three domain-independent techniques there, winning
/// 55.0% of 2000 games at equal time against an MCTS-Solver baseline.
#[derive(Debug, Clone, Copy)]
pub struct MinimaxBackprop {
    /// How many plies of ancestors, counting from (but not including) the
    /// backpropagated leaf, get their value overwritten. `0` disables the
    /// backup entirely (every node keeps the ordinary Monte-Carlo average),
    /// equivalent to `Classic`.
    pub depth: u32,
}

impl Default for MinimaxBackprop {
    fn default() -> Self {
        Self {
            // MB-2 is the literature's own best-performing depth on
            // Breakthrough (Baier & Winands 2015), matching
            // `prior::NegamaxPrior`'s default `depth` for the same reason.
            depth: 2,
        }
    }
}

impl MinimaxBackprop {
    pub fn new(depth: u32) -> Self {
        Self { depth }
    }
}

impl BackpropStrategy for MinimaxBackprop {
    fn recompute_depth(&self) -> u32 {
        self.depth
    }

    fn recompute_value<A: crate::game::Action>(
        &self,
        node: &node::Node<A>,
        slot: &PosteriorSlot<A>,
        _index: &TreeIndex<A>,
        num_players: usize,
    ) {
        derive_minimax_value(node, slot, num_players);
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Power-UCT (Dam et al., IJCAI 2020): a backpropagation strategy that, after
/// the ordinary Monte-Carlo `update`, recomputes every ancestor's own value
/// as the visit-weighted power mean of its children -- see
/// `derive_power_mean_value`. One scalar `p`: `p = 1` recovers plain UCT (the
/// recompute pass is disabled outright, so the behavior is bit-identical to
/// `Classic`), `p -> inf` approaches a max / Full-Bellman backup. The value
/// sits between the negatively-biased mean and the positively-biased max;
/// convergence over the whole range (including for explicitly non-stationary
/// tree nodes) is Stochastic-Power-UCT (arXiv 2406.02235).
///
/// `alpha` blends that power mean with the plain max over children
/// (`(1 - alpha)·V_p + alpha·V_max`, per player) -- `alpha = 1` is the
/// Full-Bellman / max backup (Asai & Wissow, AAAI 2025) at any `p`. See
/// `derive_power_mean_value` for the EVT framing and the dead-child
/// exclusion that is always applied.
#[derive(Debug, Clone, Copy)]
pub struct PowerMeanBackprop {
    /// Power-mean exponent. `1.0` = plain visit-weighted mean (== `Classic`
    /// when `alpha == 0`, recompute pass disabled); larger = closer to max.
    /// Tuner bounds `[1.0, 50.0]`.
    pub p: f64,
    /// Mean<->max blend: final value = `(1-alpha)·power_mean + alpha·max`
    /// over children, per player. `0.0` = pure power-mean; `1.0` =
    /// Full-Bellman max backup (Asai & Wissow, AAAI 2025) at any `p`. Tuner
    /// bounds `[0.0, 1.0]`, default `0.0`.
    pub alpha: f64,
    /// How many plies of ancestors above the leaf get the power-mean backup.
    /// `None` = every ancestor; `Some(d)` = only the nearest `d` (parity with
    /// `MinimaxBackprop::depth`, for a future depth sweep).
    pub depth: Option<u32>,
}

impl Default for PowerMeanBackprop {
    fn default() -> Self {
        Self {
            p: 1.0,
            alpha: 0.0,
            depth: None,
        }
    }
}

impl PowerMeanBackprop {
    pub fn new(p: f64, depth: Option<u32>) -> Self {
        Self {
            p,
            alpha: 0.0,
            depth,
        }
    }

    pub fn new_mixed(p: f64, alpha: f64, depth: Option<u32>) -> Self {
        Self { p, alpha, depth }
    }
}

impl BackpropStrategy for PowerMeanBackprop {
    fn recompute_depth(&self) -> u32 {
        // p == 1, alpha == 0 is exactly the arithmetic mean; skip the pass
        // entirely so the strategy is bit-identical to `Classic` there (and
        // so the per-call overwrite-then-reaccumulate churn is avoided).
        // p == 1, alpha > 0 is a mean<->max blend -- the pass must run.
        if self.p == 1.0 && self.alpha == 0.0 {
            0
        } else {
            self.depth.unwrap_or(u32::MAX)
        }
    }

    fn recompute_value<A: crate::game::Action>(
        &self,
        node: &node::Node<A>,
        slot: &PosteriorSlot<A>,
        index: &TreeIndex<A>,
        num_players: usize,
    ) {
        derive_power_mean_value(node, slot, index, num_players, self.p, self.alpha);
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Tesauro/Rajan/Segal 2010's Bayesian backprop, Gaussian-approximation
/// variant ("_g" in the paper): each node's posterior is a Gaussian, and an
/// interior node's posterior is the MAX-of-children distribution computed
/// via `clark_max_of_gaussians`. Paired with `select::BayesUct1`/
/// `BayesUct2` (`select/bayes.rs`), which read the `posterior_mean`/
/// `posterior_variance` this writes.
#[derive(Debug, Clone, Copy)]
pub struct BayesGaussian {
    /// Prior variance on a node's true value before any observations.
    pub prior_variance: f64,
    /// Assumed variance of a single rollout's return, i.e. the conjugate
    /// update's observation noise.
    pub obs_variance: f64,
}

impl Default for BayesGaussian {
    fn default() -> Self {
        Self {
            prior_variance: 1.0,
            obs_variance: 1.0,
        }
    }
}

impl BayesGaussian {
    pub fn new(prior_variance: f64, obs_variance: f64) -> Self {
        Self {
            prior_variance,
            obs_variance,
        }
    }
}

impl BackpropStrategy for BayesGaussian {
    fn provides_posterior(&self) -> bool {
        true
    }

    fn update_posterior<A: crate::game::Action>(
        &self,
        player: usize,
        mover: usize,
        slot: &PosteriorSlot<A>,
        own_children: Option<&ChildArray<A>>,
    ) {
        let combine = if player == mover {
            clark_max_of_gaussians
        } else {
            clark_min_of_gaussians
        };
        let combined = own_children.and_then(|children| {
            fold_gaussian_extremum(
                (0..children.len())
                    .filter(|&i| children.is_explored(i))
                    .map(|i| children.posterior(i, player)),
                combine,
            )
        });
        let (mean, variance) = combined.unwrap_or_else(|| {
            conjugate_leaf_posterior(slot, player, self.prior_variance, self.obs_variance)
        });
        slot.set(player, mean, variance);
    }
}

/// Tesauro/Rajan/Segal 2010's Bayesian backprop, numeric-integration
/// variant ("_n" in the paper): each node's posterior is a discretized PDF
/// over `BAYES_GRID_SIZE` grid points spanning `[value_lo, value_hi]`, and
/// an interior node's posterior is the exact (to grid resolution) MAX-of-
/// children distribution computed via `numeric_max_of_pdfs`.
#[derive(Debug, Clone, Copy)]
pub struct BayesNumeric {
    pub prior_variance: f64,
    pub obs_variance: f64,
    /// Grid lower/upper bounds -- must cover the game's real utility range;
    /// defaults to this codebase's symmetric `[-1, 1]` convention.
    pub value_lo: f64,
    pub value_hi: f64,
}

impl Default for BayesNumeric {
    fn default() -> Self {
        Self {
            prior_variance: 1.0,
            obs_variance: 1.0,
            value_lo: -1.0,
            value_hi: 1.0,
        }
    }
}

impl BayesNumeric {
    pub fn new(prior_variance: f64, obs_variance: f64, value_lo: f64, value_hi: f64) -> Self {
        Self {
            prior_variance,
            obs_variance,
            value_lo,
            value_hi,
        }
    }

    fn leaf_pdf<A: crate::game::Action>(
        &self,
        slot: &PosteriorSlot<A>,
        player: usize,
    ) -> [f64; BAYES_GRID_SIZE] {
        let (mean, variance) =
            conjugate_leaf_posterior(slot, player, self.prior_variance, self.obs_variance);
        normal_pdf_grid(mean, variance, self.value_lo, self.value_hi)
    }
}

impl BackpropStrategy for BayesNumeric {
    fn provides_posterior(&self) -> bool {
        true
    }

    fn update_posterior<A: crate::game::Action>(
        &self,
        player: usize,
        mover: usize,
        slot: &PosteriorSlot<A>,
        own_children: Option<&ChildArray<A>>,
    ) {
        let child_pdfs: Vec<[f64; BAYES_GRID_SIZE]> = own_children
            .map(|children| {
                (0..children.len())
                    .filter(|&i| children.is_explored(i))
                    .map(|i| {
                        children.posterior_grid(i, player).unwrap_or_else(|| {
                            let (mu, var) = children.posterior(i, player);
                            normal_pdf_grid(mu, var, self.value_lo, self.value_hi)
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let pdf = if child_pdfs.is_empty() {
            self.leaf_pdf(slot, player)
        } else if player == mover {
            numeric_max_of_pdfs(&child_pdfs, self.value_lo, self.value_hi)
        } else {
            numeric_min_of_pdfs(&child_pdfs, self.value_lo, self.value_hi)
        };
        let (mean, variance) = mean_variance_from_pdf(&pdf, self.value_lo, self.value_hi);
        slot.set(player, mean, variance);
        slot.set_grid(player, pdf);
    }
}

#[cfg(test)]
mod bayes_tests {
    use super::*;

    #[test]
    fn conjugate_posterior_matches_prior_when_unvisited() {
        let stats = NodeStats::new(1, false);
        let slot: PosteriorSlot<u8> = PosteriorSlot::Root(&stats);
        let (mean, variance) = conjugate_leaf_posterior(&slot, 0, 2.0, 1.0);
        assert_eq!(mean, 0.0);
        assert_eq!(variance, 2.0);
    }

    #[test]
    fn conjugate_posterior_shrinks_toward_sample_mean_with_visits() {
        let stats = NodeStats::new(1, false);
        // 10 observations averaging 0.5.
        for _ in 0..10 {
            stats.update(&[0.5]);
        }
        let slot: PosteriorSlot<u8> = PosteriorSlot::Root(&stats);
        let (mean, variance) = conjugate_leaf_posterior(&slot, 0, 1.0, 1.0);
        // precision = 1/1 + 10/1 = 11; mean = (10*0.5/1) / 11 = 5/11
        assert!((mean - 5.0 / 11.0).abs() < 1e-9);
        assert!((variance - 1.0 / 11.0).abs() < 1e-9);
        // More visits should pull the posterior mean closer to the sample
        // mean than a single observation would.
        assert!(mean > 0.4 && mean < 0.5);
    }

    #[test]
    fn clark_max_of_identical_gaussians_is_symmetric() {
        let (mean, variance) = clark_max_of_gaussians(0.0, 1.0, 0.0, 1.0);
        // max of two i.i.d. N(0,1): known closed form mean = 1/sqrt(pi).
        assert!(
            (mean - std::f64::consts::FRAC_1_SQRT_2 * (2.0 / std::f64::consts::PI).sqrt()).abs()
                < 1e-6
        );
        assert!(variance > 0.0 && variance < 1.0);
    }

    #[test]
    fn clark_max_dominated_by_higher_mean_when_variance_tiny() {
        // With near-zero variance, max(X1, X2) collapses to whichever has
        // the higher mean.
        let (mean, variance) = clark_max_of_gaussians(1.0, 1e-6, -1.0, 1e-6);
        assert!((mean - 1.0).abs() < 1e-3);
        assert!(variance < 1e-3);
    }

    #[test]
    fn numeric_max_of_pdfs_matches_clark_approximately() {
        let lo = -6.0;
        let hi = 6.0;
        let pdf1 = normal_pdf_grid(0.5, 1.0, lo, hi);
        let pdf2 = normal_pdf_grid(-0.5, 1.5, lo, hi);
        let combined = numeric_max_of_pdfs(&[pdf1, pdf2], lo, hi);
        let (numeric_mean, numeric_variance) = mean_variance_from_pdf(&combined, lo, hi);
        let (clark_mean, clark_variance) = clark_max_of_gaussians(0.5, 1.0, -0.5, 1.5f64.sqrt());
        assert!(
            (numeric_mean - clark_mean).abs() < 0.05,
            "numeric={numeric_mean} clark={clark_mean}"
        );
        assert!(
            (numeric_variance - clark_variance).abs() < 0.1,
            "numeric={numeric_variance} clark={clark_variance}"
        );
    }

    #[test]
    fn bayes_gaussian_leaf_then_max_combine() {
        let strategy = BayesGaussian::new(1.0, 1.0);
        assert!(strategy.provides_posterior());

        // Leaf node: no children, posterior comes from own observations.
        let leaf_stats = NodeStats::new(1, false);
        leaf_stats.update(&[1.0]);
        let leaf_slot = PosteriorSlot::Root(&leaf_stats);
        strategy.update_posterior::<u8>(0, 0, &leaf_slot, None);
        let (leaf_mean, _) = leaf_stats.posterior(0);
        assert!(leaf_mean > 0.0);
    }

    /// Regression test for a fold-wiring bug found via
    /// `examples/bayes_uct_bandit_tree.rs`: combining more than two
    /// children's posteriors collapsed the running variance toward zero
    /// (each fold step squared an already-squared quantity), which made
    /// `BayesUct1`/`BayesUct2`'s exploration term vanish and left search
    /// unable to correct an early wrong guess. Five children -- reproducing
    /// the exact case that surfaced it -- should combine to a variance in
    /// the same order of magnitude as its inputs, not many orders smaller.
    #[test]
    fn fold_gaussian_extremum_does_not_collapse_variance_past_two_children() {
        let arms: [(f64, f64); 5] = [
            (-0.020548, 0.006849),
            (0.171875, 0.015625),
            (0.076190, 0.009524),
            (0.750000, 0.083333),
            (0.343750, 0.031250),
        ];
        let (_, variance) =
            fold_gaussian_extremum(arms.into_iter(), clark_min_of_gaussians).unwrap();
        let min_input_variance = arms.iter().map(|&(_, v)| v).fold(f64::INFINITY, f64::min);
        assert!(
            variance > min_input_variance * 0.1,
            "variance {variance:e} collapsed far below the smallest input variance {min_input_variance:e}"
        );
    }
}
