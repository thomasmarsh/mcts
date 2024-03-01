use super::*;

use crate::game::Game;
use node::QInit;
use rand::rngs::SmallRng;
use rand_core::SeedableRng;

////////////////////////////////////////////////////////////////////////////////

pub const GRAVE: usize = 0b001;
pub const GLOBAL: usize = 0b010;
pub const AMAF: usize = 0b100;

pub struct BackpropFlags(pub usize);

impl BackpropFlags {
    pub fn grave(&self) -> bool {
        self.0 & GRAVE == GRAVE
    }

    pub fn global(&self) -> bool {
        self.0 & GLOBAL == GLOBAL
    }

    pub fn amaf(&self) -> bool {
        self.0 & AMAF == AMAF
    }
}

impl std::ops::BitOr for BackpropFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait Strategy<G: Game>: Clone + Sync + Send + Default {
    type Select: select::SelectStrategy<G>;
    type Simulate: simulate::SimulateStrategy<G>;
    type Backprop: backprop::BackpropStrategy;
    type FinalAction: select::SelectStrategy<G>;

    fn friendly_name() -> String {
        "unknown".into()
    }

    // Override new to provide strategy specific defaults
    fn config() -> SearchConfig<G, Self> {
        SearchConfig::default()
    }
}

#[derive(Clone)]
pub struct SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G> + Default,
{
    pub select: S::Select,
    pub simulate: S::Simulate,
    pub backprop: S::Backprop,
    pub final_action: S::FinalAction,
    pub q_init: QInit,
    pub expand_threshold: u32,
    pub max_playout_depth: usize,
    pub max_iterations: usize,
    pub max_time: std::time::Duration,
    pub use_transpositions: bool,
    pub rng: SmallRng,
    pub verbose: bool,
    pub name: String,
}

impl<G, S> Default for SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G> + Default,
{
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: QInit::default(),
            expand_threshold: 1,
            max_playout_depth: usize::MAX,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
            rng: SmallRng::from_entropy(),
            verbose: false,
            name: format!("mcts[{}]", S::friendly_name()),
        }
    }
}

impl<G, S> SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G> + Default,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn select(mut self, select: S::Select) -> Self {
        self.select = select;
        self
    }

    pub fn simulate(mut self, simulate: S::Simulate) -> Self {
        self.simulate = simulate;
        self
    }

    pub fn backprop(mut self, backprop: S::Backprop) -> Self {
        self.backprop = backprop;
        self
    }

    pub fn final_action(mut self, final_action: S::FinalAction) -> Self {
        self.final_action = final_action;
        self
    }

    pub fn q_init(mut self, q_init: QInit) -> Self {
        self.q_init = q_init;
        self
    }

    pub fn expand_threshold(mut self, expand_threshold: u32) -> Self {
        self.expand_threshold = expand_threshold;
        self
    }

    pub fn max_playout_depth(mut self, max_playout_depth: usize) -> Self {
        self.max_playout_depth = max_playout_depth;
        self
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    // NOTE: special logic here
    pub fn max_time(mut self, max_time: std::time::Duration) -> Self {
        self.max_time = max_time;
        if self.max_time != std::time::Duration::default() {
            self.max_iterations(usize::MAX)
        } else {
            self
        }
    }

    pub fn use_transpositions(mut self, use_transpositions: bool) -> Self {
        self.use_transpositions = use_transpositions;
        self
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn rng(mut self, rng: SmallRng) -> Self {
        self.rng = rng;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}
