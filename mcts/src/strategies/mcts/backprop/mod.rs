//! Backpropagation strategies and shared backup machinery.

mod bayes;
mod classic;
mod engine;
mod heuristics;
mod minimax;
mod power_mean;
mod softmax;
mod solver;
mod td;
mod values;

pub use bayes::{BayesGaussian, BayesNumeric, BAYES_GRID_SIZE};
pub use classic::Classic;
pub use engine::BackpropStrategy;
pub use minimax::MinimaxBackprop;
pub use power_mean::PowerMeanBackprop;
pub use softmax::SoftmaxBackprop;
pub use td::TdBackprop;

pub use heuristics::update_amaf;
pub(crate) use heuristics::*;
pub(crate) use solver::*;
pub(crate) use values::*;
