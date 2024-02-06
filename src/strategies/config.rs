use rand_core::SeedableRng;
use std::time::Duration;

type Rng = rand_xorshift::XorShiftRng;

#[derive(Clone, Copy)]
pub enum SelectionStrategy {
    /// Select the root child with the highest reward.
    Max,

    /// Select the most visited root child.
    Robust,

    // theoretically c = sqrt(2), rave_param = 3000.0
    UCT(f64, f64),
    // Select the child which has both the highest visit count and the highest
    // value. If there is no max-robust child at the moment, it is better to
    // continue the search until a max-robust child is found rather than
    // returning a child with a low visit count
    // MaxRobust,

    // Select the child which maximizes a lower confidence bound.
    // SecureChild(f64)
}

#[derive(Copy, Clone, PartialEq)]
pub enum ExpansionStrategy {
    Single,
    Full,
}

impl ExpansionStrategy {
    pub fn is_single(self) -> bool {
        self == Self::Single
    }
}

pub struct Config {
    pub rng: Rng,
    pub max_time: Duration,
    pub use_rave: bool,
    pub use_mast: bool,
    pub action_selection_strategy: SelectionStrategy,
    pub tree_selection_strategy: SelectionStrategy,
    pub expansion_strategy: ExpansionStrategy,
    pub rollouts_before_expanding: u32,
    pub max_rollouts: u32,
    pub verbose: bool,
    pub max_simulate_depth: u32,
}

impl Config {
    pub fn new() -> Self {
        Self {
            rng: Rng::from_entropy(),
            max_time: Duration::from_secs(5),
            use_rave: true,
            use_mast: true,
            action_selection_strategy: SelectionStrategy::UCT(2.0_f64.sqrt(), 3000.0),
            tree_selection_strategy: SelectionStrategy::UCT(2.0_f64.sqrt(), 3000.0),
            expansion_strategy: ExpansionStrategy::Single,
            rollouts_before_expanding: 5,
            max_rollouts: u32::MAX,
            verbose: false,
            max_simulate_depth: 20000,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}
