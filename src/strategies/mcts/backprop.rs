use super::*;

use crate::game::{Game, PlayerIndex};

pub trait BackpropStrategy: Clone + Sync + Send {
    fn update<G>(
        &self,
        ctx: &mut SearchContext<G>,
        global: &mut TreeStats<G>,
        index: &mut TreeIndex<G::A>,
        trial: simulate::Trial<G>,
        player: usize,
        flags: BackpropFlags,
    ) where
        G: Game,
    {
        let utilities = G::compute_utilities(&trial.state);

        let mut amaf_actions = if flags.grave() || flags.global() {
            trial.actions.clone()
        } else {
            vec![]
        };

        while let Some(node_id) = ctx.stack.pop() {
            let node = index.get(node_id);
            let next_action = if !node.is_root() {
                Some(node.action(index))
            } else {
                None
            };

            // Needed for scalar amaf
            if flags.amaf() && node.is_expanded() {
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

            let node = index.get_mut(node_id);

            // Standard update
            node.update(&utilities);

            // GRAVE update
            if flags.grave() {
                for action in &amaf_actions {
                    let grave_stats = node.stats.grave_stats.entry(action.clone()).or_default();
                    grave_stats.num_visits += 1;
                    // TODO: what about other players utilities?
                    grave_stats.score += utilities[player];
                }
            }

            if flags.grave() || flags.global() {
                if let Some(action) = next_action {
                    amaf_actions.push(action);
                }
            }
        }

        // GlobalActionStats
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
