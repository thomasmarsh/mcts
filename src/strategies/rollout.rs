use crate::game::{Game, Winner};
use crate::strategies::mcts::{MCTSOptions, LOSS, WIN};

use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::marker::PhantomData;

// TODO: this needs to be better parameterized for rng, rather than calling thread_rng.
pub trait RolloutPolicy {
    type G: Game;

    fn random_move(
        &self,
        state: &mut <Self::G as Game>::S,
        move_scratch: &mut Vec<<Self::G as Game>::M>,
        rng: &mut ThreadRng,
    ) -> <Self::G as Game>::M;

    fn rollout(
        &self,
        options: &MCTSOptions,
        state: &<Self::G as Game>::S,
        // amaf: &mut Vec<<Self::G as Game>::M>,
    ) -> i32
    where
        <Self::G as Game>::S: Clone,
    {
        let mut rng = rand::thread_rng();
        let mut depth = options.max_rollout_depth;
        let mut state = state.clone();
        let mut moves = Vec::new();
        let mut sign = 1;
        loop {
            if let Some(winner) = Self::G::get_winner(&state) {
                let first = depth == options.max_rollout_depth;
                return match winner {
                    Winner::PlayerJustMoved => {
                        if first {
                            WIN
                        } else {
                            1
                        }
                    }
                    Winner::PlayerToMove => {
                        if first {
                            LOSS
                        } else {
                            -1
                        }
                    }
                    Winner::Draw => 0,
                } * sign;
            }

            if depth == 0 {
                return 0;
            }

            moves.clear();
            let m = self.random_move(&mut state, &mut moves, &mut rng);
            //amaf.push(m);
            if let Some(new_state) = Self::G::apply(&mut state, m) {
                state = new_state;
            } else {
                unimplemented!();
            }
            sign = -sign;
            depth -= 1;
        }
    }
}

pub struct NaiveRolloutPolicy<G: Game> {
    pub game_type: PhantomData<G>,
}

impl<G: Game> RolloutPolicy for NaiveRolloutPolicy<G> {
    type G = G;
    fn random_move(
        &self,
        state: &mut <Self::G as Game>::S,
        moves: &mut Vec<<Self::G as Game>::M>,
        rng: &mut ThreadRng,
    ) -> <Self::G as Game>::M {
        G::generate_moves(state, moves);
        moves.choose(rng).unwrap().clone()
    }
}
