use super::*;

use crate::game::Game;

#[derive(Debug)]
pub enum EndType {
    NaturalEnd,
    // MoveLimit,
    TurnLimit,
}

#[derive(Debug)]
pub struct Status {
    pub end_type: Option<EndType>,
}

#[derive(Debug)]
pub struct Trial<G: Game> {
    pub actions: Vec<G::A>,
    pub state: G::S,
    pub status: Status,
}

pub trait SimulateStrategy {
    fn playout<G: Game>(
        &self,
        state: G::S,
        max_playout_depth: usize,
        rng: &mut FastRng,
    ) -> Trial<G>;
}

#[derive(Default)]
pub struct Uniform;

impl SimulateStrategy for Uniform {
    fn playout<G: Game>(
        &self,
        mut state: G::S,
        max_playout_depth: usize,
        rng: &mut FastRng,
    ) -> Trial<G> {
        let mut actions = Vec::new();
        let mut available = Vec::new();
        let mut depth = 0;
        let end_type;
        loop {
            if G::is_terminal(&state) {
                end_type = Some(EndType::NaturalEnd);
                break;
            }
            if depth >= max_playout_depth {
                end_type = Some(EndType::TurnLimit);
                break;
            }
            available.clear();
            G::generate_actions(&state, &mut available);
            if available.is_empty() {
                end_type = Some(EndType::NaturalEnd);
                break;
            }
            let action = &available[rng.gen_range(0..available.len())];
            actions.push(action.clone());
            state = G::apply(state, action);
            depth += 1;
        }

        Trial {
            actions,
            state,
            status: Status { end_type },
        }
    }
}

// pub struct EpsilonGreedy {
// epsilon: f64,
// }

/*
impl SimulateStrategy for EpsilonGreedy {
    fn playout<G: game::Game>(
        &self,
        state: &G::S,
        max_playout_depth: u32,
        rng: &mut FastRng,
    ) -> Trial<G> {
    }
}

pub struct Mast {
    epsilon: f64,
}

impl Default for Mast {
    fn default() -> Self {
        Self { epsilon: 0.1 }
    }
}

impl SimulateStrategy for Mast {
    fn playout<G: game::Game>(
        &self,
        state: &G::S,
        max_playout_depth: u32,
        rng: &mut FastRng,
    ) -> Trial<G> {
    }
}
*/
