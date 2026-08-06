use super::node::{self, NodeStats, NodeState, Proven};
use super::stack::NodeStack;
use super::*;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::TerminalStatus;

use rustc_hash::FxHashMap;

/// Converts a playout's terminal check into the `Proven` value it directly
/// witnesses -- used for the zero-length-`Trial` case (design note session 3
/// point 1): a leaf that was terminal before `expand()` ever ran on it (only
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
/// Solver's backprop pass (design note session 3 point 3). No-ops on a node
/// that isn't `Expanded` (a `Terminal` node's `Proven` was already set once
/// at `expand()`-time and isn't re-derived here).
///
/// Deliberately stricter than a literal reading of the design note's Draw
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
/// this subtree).
fn derive_proven<G: Game>(node: &node::Node<G::A>, index: &TreeIndex<G::A>) {
    let Some(NodeState::Expanded(edges)) = node.status() else {
        return;
    };
    let p = node.player_idx;

    let mut all_children_proven = true;
    let mut win_q: Option<usize> = None;
    let mut win_q_consistent = true;
    let mut any_draw = false;

    for edge in edges {
        let Some(child_id) = edge.node_id() else {
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
                // Only reachable if `num_players() > 2`, which the solver is
                // not scoped to (see the `debug_assert!`s at its call
                // sites) -- guarded rather than assumed away.
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

pub trait BackpropStrategy: Clone + Sync + Send + Default {
    fn update_amaf<G: Game>(
        &self,
        parent_id_opt: Option<index::Id>,
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
            let sibling_actions: FxHashMap<_, _> = index
                .get(parent_id)
                .edges()
                .iter()
                .filter_map(|edge| edge.node_id().map(|node_id| (edge.action.clone(), node_id)))
                .collect();

            for (action, p) in trace {
                if let Some(child_id) = sibling_actions.get(action) {
                    let child = index.get(*child_id);
                    if child.player_idx == *p {
                        let stats = &index.get(parent_id).child_edge(*child_id).stats;
                        (0..G::num_players()).for_each(|i| {
                            stats.add_amaf(i, utilities[i]);
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
        trial: simulate::Trial<G>,
        flags: BackpropFlags,
        use_mcts_solver: bool,
    ) where
        G: Game,
    {
        // init_amaf: GRAVE | GLOBAL
        let mut amaf_actions = if flags.grave() || flags.global() {
            trial.actions.clone()
        } else {
            vec![]
        };

        // `trial.terminal` already carries the winner if `playout` ended
        // naturally (rather than hitting the depth cutoff) -- reuse it
        // instead of re-deriving utilities from `trial.state` via
        // `Game::compute_utilities`, which for games like Druid would redo
        // the same connectivity scan `playout` just paid for.
        let utilities = trial
            .terminal
            .utilities(G::num_players())
            .unwrap_or_else(|| G::compute_utilities(&trial.state));
        let mut is_leaf = true;
        for (parent_id_opt, node_id) in stack.reverse_pairs2() {
            debug_assert!(
                (parent_id_opt.is_some() && !index.get(*node_id).is_root())
                    || (parent_id_opt.is_none() && index.get(*node_id).is_root())
            );
            if index.get(*node_id).is_root() {
                root_stats.update(&utilities);
            } else {
                let parent_id = parent_id_opt.cloned().unwrap();
                debug_assert_ne!(parent_id, *node_id);
                let edge_stats = &index.get(parent_id).child_edge(*node_id).stats;
                edge_stats.update(&utilities);
                edge_stats.remove_virtual_loss();
            }

            // MCTS-Solver: derive/propagate proven status for this node.
            // Runs unconditionally on every backprop call for every node on
            // the visited path (not gated on the trial having ended at a
            // terminal state) -- a node can newly become provable on any
            // call once its last unproven child resolves, and the walk
            // already visits every ancestor every time regardless of trial
            // outcome, so no extra triggering logic is needed.
            if use_mcts_solver {
                debug_assert!(G::num_players() <= 2);
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
                }
                derive_proven::<G>(node, index);
            }
            is_leaf = false;

            // update: AMAF
            if flags.amaf() {
                self.update_amaf::<G>(
                    parent_id_opt.cloned(),
                    &trial.actions,
                    index,
                    *node_id,
                    &utilities,
                );
            } else if flags.grave() {
                self.update_grave::<G>(&amaf_actions, index, global, *node_id, &utilities);
            }

            // push_action: GRAVE | GLOBAL
            if flags.grave() || flags.global() {
                let node = index.get(*node_id);
                if !node.is_root() {
                    let parent_id = parent_id_opt.cloned().unwrap();
                    let action = stack.edge(index, parent_id, *node_id).action.clone();
                    amaf_actions.push((action, node.player_idx));
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
    }
}

#[derive(Default, Clone)]
pub struct Classic;

impl BackpropStrategy for Classic {}
