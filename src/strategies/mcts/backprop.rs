use super::node::NodeStats;
use super::stack::NodeStack;
use super::*;
use crate::game::Game;

use rustc_hash::FxHashMap;

pub trait BackpropStrategy: Clone + Sync + Send + Default {
    fn update_amaf<G: Game>(
        &self,
        parent_id_opt: Option<index::Id>,
        trace: &[(G::A, usize)],
        index: &mut TreeIndex<G::A>,
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
                .filter_map(|edge| edge.node_id.map(|node_id| (edge.action.clone(), node_id)))
                .collect();

            for (action, p) in trace {
                if let Some(child_id) = sibling_actions.get(action) {
                    let child = index.get_mut(*child_id);
                    if child.player_idx == *p {
                        (0..G::num_players()).for_each(|i| {
                            let parent = index.get_mut(parent_id);
                            // NOTE: O(n) lookup
                            let stats = &mut parent.child_edge_mut(*child_id).stats;
                            stats.player[i].amaf.num_visits += 1;
                            stats.player[i].amaf.score += utilities[i];
                        })
                    }
                }
            }
        }
    }

    fn update_grave<G: Game>(
        &self,
        trace: &[(G::A, usize)],
        index: &mut TreeIndex<G::A>,
        global: &mut TreeStats<G>,
        node_id: index::Id,
        utilities: &[f64],
    ) {
        let node = index.get_mut(node_id);
        if !node.is_root() {
            for (action, p) in trace {
                let players = global
                    .grave
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
        global: &mut TreeStats<G>,
        index: &mut TreeIndex<G::A>,
        root_stats: &mut NodeStats,
        trial: simulate::Trial<G>,
        flags: BackpropFlags,
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
                let parent = index.get_mut(parent_id);
                let edge_stats = &mut parent.child_edge_mut(*node_id).stats;
                edge_stats.update(&utilities);
                edge_stats.remove_virtual_loss();
            }

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
            for (action, p) in &amaf_actions {
                let action_stats = global.actions.entry(action.clone()).or_default();
                action_stats.num_visits += 1;
                action_stats.score += utilities[*p];

                let player_action_stats = global.player_actions[*p]
                    .entry(action.clone())
                    .or_default();
                player_action_stats.num_visits += 1;
                player_action_stats.score += utilities[*p];
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct Classic;

impl BackpropStrategy for Classic {}
