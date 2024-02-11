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
            rng: rand::rngs::SmallRng::from_entropy(),
            game_type: PhantomData,
        }
    }
}

impl<G: Game> Default for Random<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Game> Search for Random<G> {
    type G = G;

    fn friendly_name(&self) -> String {
        "random".into()
    }

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A {
        let mut actions = Vec::new();
        G::generate_actions(state, &mut actions);
        actions[self.rng.gen_range(0..actions.len())].clone()
    }
}
