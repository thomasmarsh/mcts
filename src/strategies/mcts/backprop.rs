use super::index::Id;
use super::*;
use crate::game::{Game, PlayerIndex};

use rustc_hash::FxHashMap as HashMap;

pub trait BackpropStrategy: Clone + Sync + Send {
    // TODO: cleanup the arguments to this, or just move it to TreeSearch
    #[allow(clippy::too_many_arguments)]
    fn update<G>(
        &self,
        ctx: &mut SearchContext<G>,
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

                if node.is_expanded() {
                    let child_actions: HashMap<_, _> = node
                        .actions()
                        .iter()
                        .cloned()
                        .zip(node.children().iter().cloned().flatten())
                        .collect();

                    for action in &trial.actions {
                        if let Some(child_id) = child_actions.get(action) {
                            let child = index.get_mut(*child_id);
                            child.stats.scalar_amaf.num_visits += 1;

                            // TODO: I'm not convinced which is the right update strategy for this one
                            // child.stats.scalar_amaf.score += utilities[player];
                            child.stats.scalar_amaf.score +=
                                utilities[G::player_to_move(&ctx.state).to_index()];
                        }
                    }
                }
            }

            // update: GRAVE
            if flags.grave() {
                let node = index.get_mut(node_id);
                for action in &amaf_actions {
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
                    amaf_actions.push(node.action(index));
                };
            }
        }

        // update: GLOBAL
        if flags.global() {
            for action in &amaf_actions {
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
