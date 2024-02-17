use std::marker::PhantomData;

use rand::rngs::SmallRng;

use super::*;

use crate::game::Game;
use crate::strategies::Search;
use crate::util::random_best;

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
    pub depth: usize,
}

pub trait SimulateStrategy<G>: Clone + Sync + Send
where
    G: Game,
{
    // The default implementation is a uniform selection
    #[allow(unused_variables)]
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        &available[rng.gen_range(0..available.len())]
    }

    fn playout(
        &mut self,
        mut state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        player: usize,
        rng: &mut SmallRng,
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
            let action: &G::A = self.select_move(&state, &available, stats, player, rng);
            actions.push(action.clone());
            state = G::apply(state, action);
            depth += 1;
        }

        Trial {
            actions,
            state,
            status: Status { end_type },
            depth,
        }
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(0)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Default, Clone)]
pub struct Uniform;

impl<G: Game> SimulateStrategy<G> for Uniform {}

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G>,
{
    pub epsilon: f64,
    pub inner: S,
    pub marker: PhantomData<G>,
}
impl<G, S> EpsilonGreedy<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self {
            epsilon,
            ..Default::default()
        }
    }
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
        &mut self,
        state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        if rng.gen::<f64>() < self.epsilon {
            <Uniform as SimulateStrategy<G>>::select_move(
                &mut Uniform,
                state,
                available,
                stats,
                player,
                rng,
            )
        } else {
            self.inner.select_move(state, available, stats, player, rng)
        }
    }

    fn playout(
        &mut self,
        state: G::S,
        max_playout_depth: usize,
        stats: &TreeStats<G>,
        player: usize,
        rng: &mut SmallRng,
    ) -> Trial<G> {
        self.inner
            .playout(state, max_playout_depth, stats, player, rng)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Default, Clone)]
pub struct Mast;

impl<G> SimulateStrategy<G> for Mast
where
    G: Game,
{
    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GLOBAL)
    }

    fn select_move<'a>(
        &mut self,
        _state: &G::S,
        available: &'a [G::A],
        stats: &TreeStats<G>,
        player: usize,
        rng: &mut SmallRng,
    ) -> &'a G::A {
        let action_scores = available
            .iter()
            .map(|action| {
                let score = stats.player_actions[player]
                    .get(action)
                    .map_or(1., |stats| {
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

////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct MetaMcts<G: Game, S: Strategy<G>> {
    inner: TreeSearch<G, S>,
}

impl<G, S> SimulateStrategy<G> for MetaMcts<G, S>
where
    G: Game,
    S: Strategy<G>,
    MctsStrategy<G, S>: Default,
{
    fn select_move<'a>(
        &mut self,
        state: &G::S,
        available: &'a [<G as Game>::A],
        _stats: &TreeStats<G>,
        _player: usize,
        _rng: &mut SmallRng,
    ) -> &'a <G as Game>::A {
        let action = self.inner.choose_action(state);
        let index = available.iter().position(|p| *p == action).unwrap();
        &available[index]
    }
}
