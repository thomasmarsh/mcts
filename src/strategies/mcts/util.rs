use super::MctsStrategy;

use super::*;

// Vanilla UCT
pub struct Ucb1;

impl Strategy for Ucb1 {
    type Select = select::Ucb1;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1".into()
    }
}

impl Default for MctsStrategy<Ucb1> {
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

impl Strategy for ScalarAmaf {
    type Select = select::ScalarAmaf;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "scalar_amaf".into()
    }
}

impl Default for MctsStrategy<ScalarAmaf> {
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

impl Strategy for Ucb1Tuned {
    type Select = select::Ucb1Tuned;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1_tuned".into()
    }
}

impl Default for MctsStrategy<Ucb1Tuned> {
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

impl Strategy for McGrave {
    type Select = select::McGrave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "mc-grave".into()
    }
}

impl Default for MctsStrategy<McGrave> {
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

impl Strategy for McBrave {
    type Select = select::McBrave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "mc-brave".into()
    }
}

impl Default for MctsStrategy<McBrave> {
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

impl Strategy for Ucb1Grave {
    type Select = select::Ucb1Grave;
    type Simulate = simulate::Uniform;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "ucb1-grave".into()
    }
}

impl Default for MctsStrategy<Ucb1Grave> {
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
