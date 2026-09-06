//! The Gumbel self-play player for tic-tac-toe -- the search configuration
//! shared by `dump --label gumbel` and the generation gate.

use mcts::algorithms::mcts::gumbel::{gumbel_search, GumbelConfig, GumbelOutcome};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::algorithms::Search;

use crate::valuenet::LinearValueNet;
use crate::{HashedPosition, Move, TicTacToe};

/// Completed-Q interior selection (a PUCT stub today) over a linear value
/// head consulted at every leaf (`max_playout_depth == 0`).
pub type GumbelProfile = Mcts<GumbelCompletedQ, EvaluatedCutoff<TicTacToe, LinearValueNet>>;

/// One generation's player: a persistent `TreeSearch` re-rooted per move by
/// the Gumbel schedule.
pub struct GumbelPlayer {
    search: TreeSearch<TicTacToe, GumbelProfile>,
    cfg: GumbelConfig,
    name: String,
}

impl GumbelPlayer {
    pub fn new(net: LinearValueNet, cfg: GumbelConfig, seed: u64) -> Self {
        let search = TreeSearch::default().config(
            SearchConfig::default()
                .expand_threshold(1)
                .max_playout_depth(0)
                .q_init(QInit::Loss)
                .simulate(EvaluatedCutoff::new().evaluator(net))
                .seed(seed),
        );
        Self {
            search,
            cfg,
            name: "gumbel".to_string(),
        }
    }

    /// The full Gumbel outcome -- action plus the visit-distribution policy
    /// target -- for the self-play dump path.
    pub fn choose(&mut self, state: &HashedPosition) -> GumbelOutcome<Move> {
        gumbel_search(&mut self.search, state, &self.cfg)
    }
}

impl Search for GumbelPlayer {
    type G = TicTacToe;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    fn choose_action(&mut self, state: &HashedPosition) -> Move {
        gumbel_search(&mut self.search, state, &self.cfg).action
    }
}
