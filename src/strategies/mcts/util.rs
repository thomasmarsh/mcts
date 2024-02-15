use super::MctsStrategy;

use super::*;

// Vanilla UCT
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

impl<G: Game> Default for MctsStrategy<G, Ucb1> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Parent,
            playouts_before_expanding: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
        }
    }
}

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

impl<G: Game> Default for MctsStrategy<G, ScalarAmaf> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Infinity,
            playouts_before_expanding: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
        }
    }
}

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

impl<G: Game> Default for MctsStrategy<G, ScalarAmafMast> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: Default::default(),
            backprop: Default::default(),
            final_action: Default::default(),
            q_init: node::UnvisitedValueEstimate::Infinity,
            playouts_before_expanding: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
        }
    }
}

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

impl<G: Game> Default for MctsStrategy<G, Ucb1Tuned> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Uniform,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            playouts_before_expanding: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
        }
    }
}

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

impl<G: Game> Default for MctsStrategy<G, McGrave> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Uniform,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            playouts_before_expanding: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
        }
    }
}

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

impl<G: Game> Default for MctsStrategy<G, McBrave> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Uniform,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Infinity,
            playouts_before_expanding: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
        }
    }
}

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

impl<G: Game> Default for MctsStrategy<G, Ucb1Grave> {
    fn default() -> Self {
        Self {
            select: Default::default(),
            simulate: simulate::Uniform,
            backprop: backprop::Classic,
            final_action: select::RobustChild,
            q_init: node::UnvisitedValueEstimate::Parent,
            playouts_before_expanding: 5,
            max_playout_depth: 200,
            max_iterations: usize::MAX,
            max_time: Default::default(),
        }
    }
}
