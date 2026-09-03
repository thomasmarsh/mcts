use super::super::node::{self, NodeState, Proven};
use super::super::*;
use crate::game::{PlayerIndex, TerminalStatus};

/// Converts a playout's terminal check into the `Proven` value it directly
/// witnesses -- used for the zero-length-`Trial` case: a leaf that was
/// terminal before `expand()` ever ran on it (only
/// possible with `expand_threshold > 1`) has no other proof source, since
/// the tree node itself was never resolved to `NodeState::Terminal`.
pub(crate) fn proven_from_terminal<P: PlayerIndex>(status: &TerminalStatus<P>) -> Option<Proven> {
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
/// `pub(crate)`, unlike `derive_proven` above, so `algorithms::tests` can
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
