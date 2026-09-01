//! A strategy that randomly chooses a move, for use in tests.

use super::super::game::Game;
use super::Search;
use rand::Rng;
use rand_core::SeedableRng;
use std::marker::PhantomData;

pub struct Random<G: Game> {
    rng: rand::rngs::SmallRng,
    game_type: PhantomData<G>,
}

impl<G: Game> Random<G> {
    pub fn new() -> Self {
        Self {
            rng: rand::rngs::SmallRng::seed_from_u64(0),
            game_type: PhantomData,
        }
    }

    /// Reseeds the move RNG. Two `Random` strategies built with the same
    /// seed play the same sequence of moves.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = rand::rngs::SmallRng::seed_from_u64(seed);
        self
    }
}

impl<G: Game> Default for Random<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Game + Sync + Send> Search for Random<G> {
    type G = G;

    fn friendly_name(&self) -> String {
        "random".into()
    }

    fn set_friendly_name(&mut self, _name: &str) {}

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A {
        let mut actions = Vec::new();
        G::generate_actions(state, &mut actions);
        actions[self.rng.gen_range(0..actions.len())].clone()
    }
}
