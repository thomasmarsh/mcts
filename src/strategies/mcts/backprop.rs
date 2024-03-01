use super::index::Id;
use super::*;
use crate::game::Game;

use rustc_hash::FxHashMap;

pub trait BackpropStrategy: Clone + Sync + Send + Default {
    // TODO: cleanup the arguments to this, or just move it to TreeSearch
    #[allow(clippy::too_many_arguments)]
    fn update<G>(
        &self,
        mut stack: Vec<Id>,
        global: &mut TreeStats<G>,
        index: &mut TreeIndex<G::A>,
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
            index.get_mut(node_id).update(&utilities);

            // update: AMAF
            if flags.amaf() {
                let node = index.get(node_id);
                if !node.is_root() {
                    let parent_id = node.parent_id;
                    assert!(!stack.is_empty());
                    assert_eq!(parent_id, *stack.last().unwrap());
                    let parent = index.get(parent_id);
                    let sibling_actions: FxHashMap<_, _> = parent
                        .actions()
                        .iter()
                        .cloned()
                        .zip(parent.children().iter().cloned().flatten())
                        .collect();

                    for (action, p) in &trial.actions {
                        if let Some(child_id) = sibling_actions.get(action) {
                            let child = index.get_mut(*child_id);
                            if child.player_idx == *p {
                                (0..G::num_players()).for_each(|i| {
                                    child.stats.player[i].amaf.num_visits += 1;
                                    child.stats.player[i].amaf.score += utilities[i];
                                })
                            }
                        }
                    }
                }
            }

            // update: GRAVE
            if flags.grave() {
                let node = index.get_mut(node_id);
                for (action, _) in &amaf_actions {
                    let grave_stats = node.stats.grave_stats.entry(action.clone()).or_default();
                    grave_stats.num_visits += 1;
                    // TODO: what about other players utilities?
                    grave_stats.score += utilities[player];
                }
            }

            // push_action: GRAVE | GLOBAL
            if flags.grave() || flags.global() {
                let node = index.get(node_id);
                if !node.is_root() {
                    amaf_actions.push((node.action(index), node.player_idx));
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
