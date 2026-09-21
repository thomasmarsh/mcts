//! The run configuration (`games/gonnect/cnn/az-7x7.toml`). Rust reads the sections it needs
//! (`net`, `selfplay`, `play`); the trainer and coordinator read `train`, `gate` and `loop`.

use grid_cnn::Geometry;
use serde::Deserialize;

use super::encode::{num_actions, IN_PLANES};

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub net: NetConfig,
    pub selfplay: SelfPlayConfig,
    pub play: PlayConfig,
}

#[derive(Deserialize, Clone, Debug)]
pub struct NetConfig {
    pub size: usize,
    pub channels: usize,
    pub blocks: usize,
    pub policy_planes: usize,
    pub value_planes: usize,
    pub value_hidden: usize,
}

impl NetConfig {
    pub fn geometry(&self) -> Geometry {
        Geometry {
            size: self.size,
            in_planes: IN_PLANES,
            channels: self.channels,
            blocks: self.blocks,
            policy_planes: self.policy_planes,
            policy_out: num_actions(self.size),
            value_planes: self.value_planes,
            value_hidden: self.value_hidden,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct SearchSettings {
    pub simulations: usize,
    pub considered_actions: usize,
    pub value_scale: f32,
    pub max_visit_init: i32,
}

impl SearchSettings {
    pub fn batch_config(&self) -> mcts_batch::Config {
        mcts_batch::Config {
            num_simulations: self.simulations,
            num_considered_actions: self.considered_actions,
            value_scale: self.value_scale,
            max_visit_init: self.max_visit_init,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct SelfPlayConfig {
    pub games: usize,
    #[serde(flatten)]
    pub search: SearchSettings,
    /// Plies at the start of a game whose move is sampled from the improved policy; after that
    /// the search's own (Sequential Halving) choice is played.
    pub temp_moves: usize,
    /// Games still running at this many plies are dropped (positional ko can cycle).
    pub max_plies: usize,
    /// States per GPU forward call.
    pub chunk_size: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct PlayConfig {
    #[serde(flatten)]
    pub search: SearchSettings,
    pub chunk_size: usize,
}

pub fn load(path: &str) -> Config {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    toml::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}
