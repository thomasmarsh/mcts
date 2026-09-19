//! Game-agnostic n-tuple value networks trained by self-play TD(lambda) with
//! temporal coherence learning and a final-adaptation step, no tree search.
//!
//! A game plugs in through [`CellFeatures`] (cells, adjacency, symmetry
//! permutations, per-cell codes); everything else comes from `mcts::game::Game`.

pub mod config;
pub mod geometry;
pub mod selfplay;
pub mod tcl;
pub mod td;
pub mod weights;

pub use config::TrainConfig;
pub use geometry::{random_walk_tuples, CellFeatures, Geometry, Tuple};
pub use selfplay::{
    play_match, terminal_value, uniform_random, EpisodeStats, GreedyPlayer, MatchRecord, Trainer,
};
pub use td::{Learner, TdParams};
pub use weights::Model;
