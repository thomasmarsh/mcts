use super::MctsStrategy;

use super::*;

// Vanilla UCT
pub struct Ucb1;

impl<A: Action> Strategy<A> for Ucb1 {
    type Select = select::Ucb1;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1".into()
    }
}

impl<A: Action> Default for MctsStrategy<Ucb1, A> {
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

impl<A: Action> Strategy<A> for ScalarAmaf {
    type Select = select::ScalarAmaf;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "scalar_amaf".into()
    }
}

impl<A: Action> Default for MctsStrategy<ScalarAmaf, A> {
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

impl<A: Action> Strategy<A> for Ucb1Tuned {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }
}

impl<A: Action> Default for MctsStrategy<Ucb1Tuned, A> {
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

impl<A: Action> Strategy<A> for McGrave {
    type Select = select::McGrave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "mc-grave".into()
    }
}

impl<A: Action> Default for MctsStrategy<McGrave, A> {
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

impl<A: Action> Strategy<A> for McBrave {
    type Select = select::McBrave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "mc-brave".into()
    }
}

impl<A: Action> Default for MctsStrategy<McBrave, A> {
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

impl<A: Action> Strategy<A> for Ucb1Grave {
    type Select = select::Ucb1Grave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1-grave".into()
    }
}

impl<A: Action> Default for MctsStrategy<Ucb1Grave, A> {
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
