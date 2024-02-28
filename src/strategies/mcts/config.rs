use super::*;

use crate::game::Game;
use node::UnvisitedValueEstimate;

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

pub trait Strategy<G: Game>: Clone + Sync + Send {
    type Select: select::SelectStrategy<G>;
    type Simulate: simulate::SimulateStrategy<G>;
    type Backprop: backprop::BackpropStrategy;
    type FinalAction: select::SelectStrategy<G>;

    fn friendly_name() -> String;
}

#[derive(Clone)]
pub struct SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G>,
{
    pub select: S::Select,
    pub simulate: S::Simulate,
    pub backprop: S::Backprop,
    pub final_action: S::FinalAction,
    pub q_init: UnvisitedValueEstimate,
    pub expand_threshold: u32,
    pub max_playout_depth: usize,
    pub max_iterations: usize,
    pub max_time: std::time::Duration,
    pub use_transpositions: bool,
}

impl<G, S> SearchConfig<G, S>
where
    G: Game,
    S: Strategy<G>,
{
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

    pub fn q_init(mut self, q_init: UnvisitedValueEstimate) -> Self {
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
}
