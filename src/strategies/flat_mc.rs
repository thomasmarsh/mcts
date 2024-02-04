use rand::seq::SliceRandom;
use rand_core::SeedableRng;
use rand_xorshift::XorShiftRng;

use crate::game::Game;
use crate::strategies::Strategy;
use crate::util::random_best;

use std::marker::PhantomData;

// Maybe rename NaiveMonteCarlo
pub struct FlatMonteCarloStrategy<G: Game> {
    samples_per_move: u32, // TODO: also suppose samples per state
    max_rollout_depth: u32,
    max_rollouts: u32,
    verbose: bool,
    game_type: PhantomData<G>,
}

impl<G: Game> FlatMonteCarloStrategy<G> {
    pub fn new() -> Self {
        Self {
            samples_per_move: 100,
            max_rollout_depth: 100,
            max_rollouts: u32::MAX,
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

fn rollout<G: Game>(
    max_rollout_depth: u32,
    init_state: &G::S,
    rng: &mut rand_xorshift::XorShiftRng,
) -> i32
where
    G::S: Clone,
{
    let mut state = init_state.clone();
    for _ in 0..max_rollout_depth {
        if G::is_terminal(&state) {
            return G::get_reward(init_state, &state);
        }
        let moves = G::gen_moves(&state);
        if let Some(m) = moves.choose(rng) {
            state = G::apply(&state, m.clone())
        } else {
            return 0;
        }
    }
    0
}

impl<G: Game> Strategy<G> for FlatMonteCarloStrategy<G> {
    fn set_max_depth(&mut self, depth: u32) {
        self.max_rollout_depth = depth;
    }

    fn set_max_rollouts(&mut self, max_rollouts: u32) {
        self.max_rollouts = max_rollouts;
    }

    fn choose_move(&mut self, state: &<G as Game>::S) -> Option<<G as Game>::M>
    where
        <G as Game>::S: Clone,
    {
        if G::is_terminal(state) {
            return None;
        }

        let mut rng = XorShiftRng::from_entropy();

        let moves = G::gen_moves(state);
        let wins = moves
            .iter()
            .map(|m| {
                let mut tmp = state.clone();
                let new_state = G::apply(&tmp, m.clone());
                tmp = new_state;
                let mut n = 0;
                for _ in 0..self.samples_per_move {
                    let result = rollout::<G>(self.max_rollout_depth, &tmp, &mut rng);
                    if result > 0 {
                        n += 1;
                    }
                }
                (n, m.clone())
            })
            .collect::<Vec<_>>();

        if self.verbose {
            let mut w = wins.clone();
            w.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprintln!("Flat MC:");
            for (n, m) in w.into_iter().take(10) {
                let pct = 100. * (n as f64 / self.samples_per_move as f64);
                let notation = G::notation(state, &m);
                eprintln!(
                    "- {:0.2}% {} ({}/{} wins)",
                    pct, notation, n, self.samples_per_move
                );
            }
        }

        random_best(wins.as_slice(), &mut rng, |x| x.0 as f64).map(|x| x.1.clone())
    }
}
