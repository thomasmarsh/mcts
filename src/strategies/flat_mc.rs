use crate::game::Game;
use crate::strategies::rollout::{NaiveRolloutPolicy, RolloutPolicy};
use crate::strategies::Strategy;
use crate::util::random_best;

use super::mcts::MCTSOptions;

use std::marker::PhantomData;

pub struct FlatMonteCarloStrategy<G: Game> {
    samples_per_move: u32, // TODO: also suppose samples per state
    verbose: bool,
    game_type: PhantomData<G>,
}

impl<G: Game> FlatMonteCarloStrategy<G> {
    pub fn new() -> Self {
        Self {
            samples_per_move: 100,
            verbose: false,
            game_type: PhantomData,
        }
    }

    pub fn set_samples_per_move(&self, samples_per_move: u32) -> Self {
        Self {
            samples_per_move,
            ..*self
        }
    }

    pub fn verbose(&self) -> Self {
        Self {
            verbose: true,
            ..*self
        }
    }
}

impl<G: Game> Default for FlatMonteCarloStrategy<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Game> Strategy<G> for FlatMonteCarloStrategy<G> {
    fn choose_move(&mut self, state: &<G as Game>::S) -> Option<<G as Game>::M>
    where
        <G as Game>::S: Clone,
    {
        if G::get_winner(state).is_some() {
            return None;
        }

        // TODO: need to narrow rollout policy options
        let options = MCTSOptions {
            verbose: false,
            max_rollout_depth: 100,
            rollouts_before_expanding: 0,
        };

        // TODO: this is so complicated
        let policy = NaiveRolloutPolicy::<G> {
            game_type: PhantomData,
        };
        let mut moves = Vec::new();
        G::generate_moves(state, &mut moves);
        let wins = moves
            .iter()
            .map(|m| {
                let mut tmp = state.clone();
                if let Some(new_state) = G::apply(&mut tmp, m.clone()) {
                    tmp = new_state;
                }
                let mut n = 0;
                for _ in 0..self.samples_per_move {
                    let result = policy.rollout(&options, &tmp);
                    if result > 0 {
                        n += 1;
                    }
                }
                (n, m.clone())
            })
            .collect::<Vec<(i32, G::M)>>();

        if self.verbose {
            let mut w = wins.clone();
            w.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprintln!("Flat MC:");
            for (n, m) in w {
                let pct = 100. * (n as f32 / self.samples_per_move as f32);
                let notation = G::notation(state, m).unwrap_or("??".to_string());
                eprintln!(
                    "- {:0.2}% {} ({}/{} wins)",
                    pct, notation, n, self.samples_per_move
                );
            }
        }

        random_best(wins.as_slice(), |x| x.0 as f32).map(|x| x.1.clone())
    }
}
