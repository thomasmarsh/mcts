//! Training configuration. Every field is required: a run's hyperparameters
//! live in its TOML file (`games/<game>/ntuple/*.toml`), never in code.

use serde::Deserialize;

use crate::td::TdParams;

#[derive(Deserialize, Clone, Debug)]
pub struct TrainConfig {
    /// Seeds tuple generation and every random choice in training.
    pub seed: u64,
    /// Self-play episodes to train for.
    pub episodes: u64,
    /// Number of random-walk tuples.
    pub n_tuples: usize,
    /// Cells per tuple.
    pub tuple_len: usize,
    /// Cell codes per cell: 3 (empty, own, opponent) or 4 (adds "empty and
    /// playable" where the adapter supports it).
    pub states_per_cell: usize,
    /// Step size in value space (divided across the selected weights).
    pub alpha: f32,
    /// Eligibility-trace decay.
    pub lambda: f32,
    /// Discount on the next state's value.
    pub gamma: f32,
    /// Probability of a uniformly random move, falling linearly from
    /// `epsilon_start` at episode 0 to `epsilon_end` at the last episode.
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    /// Temporal coherence learning on or off.
    pub tcl: bool,
    /// Clear the eligibility trace of the chain that just took an exploratory
    /// move, so a random move does not pass credit to earlier states.
    pub reset_traces_on_explore: bool,
    /// Eligibility traces below this magnitude are dropped.
    pub trace_cutoff: f32,
    /// Log a JSONL row and write a checkpoint every this many episodes.
    pub log_every: u64,
    /// Games per opponent in each periodic evaluation (half as each colour).
    pub eval_games: u32,
    /// Where the run writes `train.jsonl`, `model.toml`, `weights.bin` and
    /// `weights.meta.json`.
    pub out_dir: String,
}

impl TrainConfig {
    pub fn td_params(&self) -> TdParams {
        TdParams {
            alpha: self.alpha,
            lambda: self.lambda,
            gamma: self.gamma,
            tcl: self.tcl,
            trace_cutoff: self.trace_cutoff,
        }
    }

    /// Exploration rate for `episode` (0-based).
    pub fn epsilon_at(&self, episode: u64) -> f32 {
        let frac = if self.episodes > 1 {
            episode as f32 / (self.episodes - 1) as f32
        } else {
            0.0
        };
        self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac.min(1.0)
    }
}
