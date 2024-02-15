use std::marker::PhantomData;

use rand::rngs::SmallRng;

use super::*;

use crate::{game::Game, util::random_best};

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

pub trait SimulateStrategy<G>
where
    G: Game,
{
    // The default implementation is a uniform selection
    #[allow(unused_variables)]
    fn select_move<'a>(
        &self,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        &available[rng.gen_range(0..available.len())]
    }

    fn playout(
        &self,
        mut state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
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
            let action: &G::A = self.select_move(&available, stats, rng);
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

#[derive(Default)]
pub struct Uniform;

impl<G: Game> SimulateStrategy<G> for Uniform {}

pub struct EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G>,
{
    pub epsilon: f64,
    pub inner: S,
    pub marker: PhantomData<G>,
}

impl<G, S> Default for EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            inner: Default::default(),
            marker: PhantomData,
        }
    }
}

impl<G, S> SimulateStrategy<G> for EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G>,
{
    fn select_move<'a>(
        &self,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        if rng.gen::<f64>() < self.epsilon {
            <Uniform as SimulateStrategy<G>>::select_move(&Uniform, available, stats, rng)
        } else {
            self.inner.select_move(available, stats, rng)
        }
    }

    fn playout(
        &self,
        state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        self.inner.playout(state, max_playout_depth, stats, rng)
    }
}

pub struct Mast<A: Action> {
    pub global_actions: HashMap<A, node::ActionStats>,
}

impl<A: Action> Default for Mast<A> {
    fn default() -> Self {
        Self {
            global_actions: Default::default(),
        }
    }
}

impl<G> SimulateStrategy<G> for Mast<G::A>
where
    G: Game,
{
    fn select_move<'a>(
        &self,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        let action_scores = available
            .iter()
            .map(|action| {
                let score = stats.actions.get(action).map_or(1., |stats| {
                    if stats.num_visits > 0 {
                        stats.score / stats.num_visits as f64
                    } else {
                        1.
                    }
                });

                (score, action)
            })
            .collect::<Vec<_>>();

        random_best(&action_scores, rng, |(score, _)| *score)
            .unwrap()
            .1
    }
}
