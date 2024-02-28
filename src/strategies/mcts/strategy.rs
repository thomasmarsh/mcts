use super::SearchConfig;

use super::*;
use crate::game::Game;

// TODO: this whole module should be removed. It is not possible to use custom
// strategies outside the crate since you cannot define a Default implementation
// for SearchConfig<T> in another crate. Also, there is clearly a combinatorial
// explosion happening here.

// Vanilla UCT
#[derive(Clone)]
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

impl<G: Game> Default for SearchConfig<G, Ucb1> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Parent,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

// Vanilla UCT + decisive move
#[derive(Clone)]
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

impl<G: Game> Default for SearchConfig<G, Ucb1DM> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::DecisiveMove::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Parent,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

// Vanilla UCT + Mast
#[derive(Clone)]
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

impl<G: Game> Default for SearchConfig<G, Ucb1Mast> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Parent,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct ScalarAmaf;

impl<G: Game> Strategy<G> for ScalarAmaf {
    type Select = select::ScalarAmaf;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "scalar_amaf".into()
    }
}

impl<G: Game> Default for SearchConfig<G, ScalarAmaf> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Infinity,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct ScalarAmafMast;

impl<G: Game> Strategy<G> for ScalarAmafMast {
    type Select = select::ScalarAmaf;
    type Simulate = simulate::EpsilonGreedy<G, simulate::Mast>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "scalar_amaf+mast".into()
    }
}

impl<G: Game> Default for SearchConfig<G, ScalarAmafMast> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Infinity,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct Ucb1Tuned;

impl<G: Game> Strategy<G> for Ucb1Tuned {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }
}

impl<G: Game> Default for SearchConfig<G, Ucb1Tuned> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Uniform,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct Ucb1TunedMast;

impl<G: Game> Strategy<G> for Ucb1TunedMast {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::Mast;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }
}

impl<G: Game> Default for SearchConfig<G, Ucb1TunedMast> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Mast,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct Ucb1TunedDM;

impl<G: Game> Strategy<G> for Ucb1TunedDM {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::DecisiveMove<G>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }
}

impl<G: Game> Default for SearchConfig<G, Ucb1TunedDM> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::DecisiveMove::default(),
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct Ucb1TunedDMMast;

impl<G: Game> Strategy<G> for Ucb1TunedDMMast {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::DecisiveMove<G, simulate::Mast>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }
}

impl<G: Game> Default for SearchConfig<G, Ucb1TunedDMMast> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::DecisiveMove::default(),
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}
#[derive(Clone)]
pub struct McGrave;

impl<G: Game> Strategy<G> for McGrave {
    type Select = select::McGrave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "mc-grave".into()
    }
}

impl<G: Game> Default for SearchConfig<G, McGrave> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Uniform,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct McBrave;

impl<G: Game> Strategy<G> for McBrave {
    type Select = select::McBrave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "mc-brave".into()
    }
}

impl<G: Game> Default for SearchConfig<G, McBrave> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Uniform,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct Ucb1Grave;

impl<G: Game> Strategy<G> for Ucb1Grave {
    type Select = select::Ucb1Grave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1-grave".into()
    }
}

impl<G: Game> Default for SearchConfig<G, Ucb1Grave> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Uniform,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Parent,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct Ucb1GraveMast;

impl<G: Game> Strategy<G> for Ucb1GraveMast {
    type Select = select::Ucb1Grave;
    type Simulate = simulate::EpsilonGreedy<G, simulate::Mast>;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1-grave+mast".into()
    }
}

impl<G: Game> Default for SearchConfig<G, Ucb1GraveMast> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Parent,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

#[derive(Clone)]
pub struct MetaMcts;

impl<G: Game> Default for SearchConfig<G, MetaMcts> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::MetaMcts {
                inner: TreeSearch::default(),
            },
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Parent,
            expand_threshold: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}

impl<G: Game> Strategy<G> for MetaMcts {
    type Select = select::Ucb1;
    type Simulate = simulate::MetaMcts<G, strategy::Ucb1>;
    type Backprop = backprop::Classic;
    type FinalAction = select::MaxAvgScore;

    fn friendly_name() -> String {
        "meta-mcts".into()
    }
}

#[derive(Clone)]
pub struct QuasiBestFirst;

impl<G: Game> Strategy<G> for QuasiBestFirst {
    type Select = select::EpsilonGreedy<G, select::QuasiBestFirst<G, Ucb1Mast>>;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::MaxAvgScore;

    fn friendly_name() -> String {
        "qbf/ucb1+mast".into()
    }
}

impl<G: Game> Default for SearchConfig<G, QuasiBestFirst> {
    fn default() -> Self {
        Self {
            select: select::EpsilonGreedy {
                epsilon: 0.3,
                ..Default::default()
            },
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Parent,
            expand_threshold: 0,
            max_playout_depth: 200,
            max_iterations: 1,
            max_time: Default::default(),
            use_transpositions: false,
        }
    }
}
