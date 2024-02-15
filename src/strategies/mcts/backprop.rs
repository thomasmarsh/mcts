use super::*;

use crate::game::{Game, PlayerIndex};

pub trait BackpropStrategy {
    fn update<G>(
        &self,
        ctx: &mut SearchContext<G>,
        global: &mut TreeStats<G>,
        index: &mut TreeIndex<G::A>,
        trial: simulate::Trial<G>,
    ) where
        G: Game,
    {
        let current_player = G::player_to_move(&ctx.state).to_index();
        let utilities = compute_utilities::<G>(&trial.state);

        let mut amaf_actions = trial.actions.clone();

        while let Some(node_id) = ctx.stack.pop() {
            let node = index.get(node_id);
            let next_action = if !node.is_root() {
                Some(node.action(index))
            } else {
                None
            };

            // Needed for scalar amaf
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
                        child.stats.scalar_amaf.score += utilities[current_player];
                    }
                }
            }

            let node = index.get_mut(node_id);

            // Standard update
            node.update(&utilities);

            for action in &amaf_actions {
                let grave_stats = node.stats.grave_stats.entry(action.clone()).or_default();
                grave_stats.num_visits += 1;
                grave_stats.score += utilities[current_player];
            }

            if let Some(action) = next_action {
                amaf_actions.push(action);
            }
        }

        // GlobalActionStats
        for action in &amaf_actions {
            let action_stats = global.actions.entry(action.clone()).or_default();
            action_stats.num_visits += 1;
            action_stats.score += utilities[G::player_to_move(&ctx.state).to_index()];
        }
    }
}

#[derive(Default)]
pub struct Classic;

impl BackpropStrategy for Classic {}

#[inline]
fn rank_to_util(rank: f64, num_players: usize) -> f64 {
    let n = num_players as f64;

    if n == 1. {
        2. * rank - 1.
    } else {
        1. - ((rank - 1.) * (2. / (n - 1.)))
    }
}

#[inline]
pub fn compute_utilities<G>(state: &G::S) -> Vec<f64>
where
    G: Game,
{
    let winner = G::winner(state).map(|p| p.to_index());
    (0..G::num_players())
        .map(|i| match winner {
            None => 0.,
            Some(w) if w == i => 1.,
            _ => -1.,
        })
        .collect()

    // TODO: think about the best way to handle ranking
    //
    // (0..G::num_players())
    //     .map(|i| {
    //         let n = G::num_players();
    //         let rank = G::rank(state, i);
    //         rank_to_util(rank, n)
    //     })
    //     .collect()
}
