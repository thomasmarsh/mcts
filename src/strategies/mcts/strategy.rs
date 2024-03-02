use super::SearchConfig;

use super::node::QInit;
use super::*;
use crate::game::Game;

// Vanilla UCT
#[derive(Clone, Default)]
pub struct Ucb1;

impl<G: Game> Strategy<G> for Ucb1 {
    type Select = select::Ucb1;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1".into()
    }
}

// Vanilla UCT + decisive move
#[derive(Clone, Default)]
pub struct Ucb1DM;

impl<G: Game> Strategy<G> for Ucb1DM {
    type Select = select::Ucb1;
    type Simulate = simulate::DecisiveMove<G>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1".into()
    }
}

// Vanilla UCT + Mast
#[derive(Clone, Default)]
pub struct Ucb1Mast;

impl<G: Game> Strategy<G> for Ucb1Mast {
    type Select = select::Ucb1;
    type Simulate = simulate::EpsilonGreedy<G, simulate::Mast>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_mast".into()
    }
}

#[derive(Clone, Default)]
pub struct Amaf;

impl<G: Game> Strategy<G> for Amaf {
    type Select = select::Amaf;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "amaf".into()
    }
}

#[derive(Clone, Default)]
pub struct AmafMast;

impl<G: Game> Strategy<G> for AmafMast {
    type Select = select::Amaf;
    type Simulate = simulate::EpsilonGreedy<G, simulate::Mast>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "amaf+mast".into()
    }
}

#[derive(Clone, Default)]
pub struct Ucb1Tuned;

impl<G: Game> Strategy<G> for Ucb1Tuned {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }

    fn config() -> SearchConfig<G, Self> {
        SearchConfig::new().q_init(QInit::Infinity)
    }
}

#[derive(Clone, Default)]
pub struct Ucb1TunedMast;

impl<G: Game> Strategy<G> for Ucb1TunedMast {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::Mast;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }

    fn config() -> SearchConfig<G, Self> {
        SearchConfig::new().q_init(QInit::Infinity)
    }
}

#[derive(Clone, Default)]
pub struct Ucb1TunedDM;

impl<G: Game> Strategy<G> for Ucb1TunedDM {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::DecisiveMove<G>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }

    fn config() -> SearchConfig<G, Self> {
        SearchConfig::new().q_init(QInit::Infinity)
    }
}

#[derive(Clone, Default)]
pub struct Ucb1TunedDMMast;

impl<G: Game> Strategy<G> for Ucb1TunedDMMast {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Mast>>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }

    fn config() -> SearchConfig<G, Self> {
        SearchConfig::new().q_init(QInit::Infinity)
    }
}

#[derive(Clone, Default)]
pub struct MetaMcts;

impl<G: Game> Strategy<G> for MetaMcts {
    type Select = select::Ucb1;
    type Simulate = simulate::MetaMcts<G, strategy::Ucb1>;
    type Backprop = backprop::Classic;
    type FinalAction = select::MaxAvgScore;

    fn friendly_name() -> String {
        "meta-mcts".into()
    }
}

#[derive(Clone, Default)]
pub struct QuasiBestFirst;

impl<G: Game> Strategy<G> for QuasiBestFirst {
    type Select = select::EpsilonGreedy<G, select::QuasiBestFirst<G, Ucb1Mast>>;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::MaxAvgScore;

    fn friendly_name() -> String {
        "qbf/ucb1+mast".into()
    }

    fn config() -> SearchConfig<G, Self> {
        SearchConfig::new().select(select::EpsilonGreedy::new().epsilon(0.3))
    }
}

#[derive(Clone, Copy, Default)]
pub struct RaveMastDm;

impl<G: Game> Strategy<G> for RaveMastDm {
    type Select = select::Rave;
    type Simulate = simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Mast>>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;
}
