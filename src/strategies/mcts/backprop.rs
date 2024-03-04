use super::index::Id;
use super::node::NodeStats;
use super::*;
use crate::game::Game;

use rustc_hash::FxHashMap;

pub trait BackpropStrategy: Clone + Sync + Send + Default {
    fn update_amaf<G: Game>(
        &self,
        stack: &[Id],
        trace: &[(G::A, usize)],
        index: &mut TreeIndex<G::A>,
        node_id: index::Id,
        utilities: &[f64],
    ) {
        // NOTE: O(n) here, but amaf could be calculated top down
        let node = index.get(node_id);
        if !node.is_root() {
            assert!(!stack.is_empty());
            let parent_id = *stack.last().unwrap();
            let sibling_actions: FxHashMap<_, _> = index
                .get(parent_id)
                .edges()
                .iter()
                .filter_map(|edge| edge.node_id.map(|node_id| (edge.action.clone(), node_id)))
                .collect();

            for (action, p) in trace {
                if let Some(child_id) = sibling_actions.get(action) {
                    let child = index.get_mut(*child_id);
                    let action_idx = child.action_idx;
                    if child.player_idx == *p {
                        (0..G::num_players()).for_each(|i| {
                            let stats = &mut index.get_mut(parent_id).edges_mut()[action_idx].stats;
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
        mut stack: Vec<Id>,
        global: &mut TreeStats<G>,
        index: &mut TreeIndex<G::A>,
        root_stats: &mut NodeStats,
        trial: simulate::Trial<G>,
        player: usize,
        flags: BackpropFlags,
    ) where
        G: Game,
    {
        // TODO: this could be split up a bit. I've marked which logic is
        // relevant to which strategy and reorganized to show the shape more
        // clearly.

        // init_amaf: GRAVE | GLOBAL
        let mut amaf_actions = if flags.grave() || flags.global() {
            trial.actions.clone()
        } else {
            vec![]
        };

        let utilities = G::compute_utilities(&trial.state);
        while let Some(node_id) = stack.pop() {
            // Standard update
            if index.get(node_id).is_root() {
                root_stats.update(&utilities);
            } else {
                let parent_id = stack.last().cloned().unwrap();
                let action_idx = index.get(node_id).action_idx;
                index.get_mut(parent_id).edges_mut()[action_idx]
                    .stats
                    .update(&utilities);
            }

            // update: AMAF
            if flags.amaf() {
                self.update_amaf::<G>(&stack, &trial.actions, index, node_id, &utilities);
            } else if flags.grave() {
                self.update_grave::<G>(&amaf_actions, index, global, node_id, &utilities);
            }

            // push_action: GRAVE | GLOBAL
            if flags.grave() || flags.global() {
                let node = index.get(node_id);
                if !node.is_root() {
                    let parent_id = stack.last().cloned().unwrap();
                    let action = index.get(parent_id).edges()[node.action_idx].action.clone();
                    amaf_actions.push((action, node.player_idx));
                };
            }
        }

        // update: GLOBAL
        if flags.global() {
            for (action, _) in &amaf_actions {
                // let player = G::player_to_move(&ctx.state).to_index();
                let action_stats = global.actions.entry(action.clone()).or_default();
                action_stats.num_visits += 1;
                action_stats.score += utilities[player];

                let player_action_stats = global.player_actions[player]
                    .entry(action.clone())
                    .or_default();
                player_action_stats.num_visits += 1;
                for u in utilities.iter().take(G::num_players()) {
                    player_action_stats.score += u;
                }
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct Classic;

impl BackpropStrategy for Classic {}
