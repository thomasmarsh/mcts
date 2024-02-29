use super::*;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::strategies::Search;
use crate::util::random_best;

use rand::rngs::SmallRng;
use rand::Rng;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub enum EndType {
    NaturalEnd,
    // MoveLimit,
    TurnLimit,
}

#[derive(Debug, Clone)]
pub struct Status {
    pub end_type: Option<EndType>,
}

#[derive(Debug, Clone)]
pub struct Trial<G: Game> {
    pub actions: Vec<G::A>,
    pub state: G::S,
    pub status: Status,
    pub depth: usize,
}

pub trait SimulateStrategy<G>: Clone + Sync + Send + Default
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
    inner: S,
    marker: PhantomData<G>,
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

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn inner(mut self, inner: S) -> Self {
        self.inner = inner;
        self
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
}

/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
pub enum DecisiveMoveMode {
    #[default]
    Win, // Decisive move
    WinLoss,     // Decisive move + anti-decisive move
    WinLossDraw, // Any terminal state
}

#[derive(Clone)]
pub struct DecisiveMove<G, S = Uniform>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    mode: DecisiveMoveMode,
    inner: S,
    marker: PhantomData<G>,
}

impl<G, S> DecisiveMove<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mode(mut self, mode: DecisiveMoveMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn inner(mut self, inner: S) -> Self {
        self.inner = inner;
        self
    }

    fn choose<'a>(
        &self,
        state: &<G as Game>::S,
        available: &'a [<G as Game>::A],
        player: usize,
    ) -> Option<&'a <G as Game>::A> {
        use DecisiveMoveMode::*;

        let mut draw = None;
        let mut loser = None;
        match self.mode {
            WinLossDraw => {
                for action in available {
                    let child_state = G::apply(state.clone(), action);
                    if G::is_terminal(&child_state) {
                        return Some(action);
                    }
                }
                None
            }

            WinLoss => {
                for action in available {
                    let child_state = G::apply(state.clone(), action);
                    if G::is_terminal(&child_state) {
                        if G::winner(&child_state).is_some() {
                            return Some(action);
                        }
                        draw = Some(action);
                    }
                }
                draw
            }

            Win => {
                for action in available {
                    let child_state = G::apply(state.clone(), action);
                    if G::is_terminal(&child_state) {
                        if let Some(winner) = G::winner(&child_state) {
                            if winner.to_index() == player {
                                return Some(action);
                            }
                            loser = Some(action);
                        } else {
                            draw = Some(action);
                        }
                    }
                }
                loser.or(draw)
            }
        }
    }
}

impl<G, S> Default for DecisiveMove<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn default() -> Self {
        Self {
            mode: DecisiveMoveMode::default(),
            inner: S::default(),
            marker: PhantomData,
        }
    }
}

impl<G, S> SimulateStrategy<G> for DecisiveMove<G, S>
where
    G: Game,
    S: SimulateStrategy<G> + Default,
{
    fn select_move<'a>(
        &mut self,
        state: &<G as Game>::S,
        available: &'a [<G as Game>::A],
        stats: &TreeStats<G>,
        player: usize,
        rng: &mut SmallRng,
    ) -> &'a <G as Game>::A {
        self.choose(state, available, player)
            .unwrap_or_else(|| self.inner.select_move(state, available, stats, player, rng))
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
pub struct MetaMcts<G: Game, S: Strategy<G>> {
    pub inner: TreeSearch<G, S>,
}

impl<G, S> Default for MetaMcts<G, S>
where
    G: Game,
    S: Strategy<G>,
{
    fn default() -> Self {
        Self {
            inner: TreeSearch::default(),
        }
    }
}

impl<G, S> SimulateStrategy<G> for MetaMcts<G, S>
where
    G: Game,
    S: Strategy<G>,
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
