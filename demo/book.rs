use std::time::Duration;

use mcts::strategies::mcts::meta::QuasiBestFirst;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::util;
use mcts::strategies::mcts::MctsStrategy;
use mcts::strategies::mcts::TreeSearch;

use mcts::games::druid::Druid;
use mcts::games::druid::State;

use rand::rngs::SmallRng;
use rand_core::SeedableRng;

const PLAYOUT_DEPTH: usize = 200;
const C_TUNED: f64 = 1.625;
const MAX_ITER: usize = 100;
const EXPAND_THRESHOLD: u32 = 1;
const VERBOSE: bool = false;
const MAX_TIME_SECS: u64 = 0; // 0 = infinite

fn main() {
    color_backtrace::install();

    let search = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::Ucb1 {
                    exploration_constant: C_TUNED,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
        )
        .verbose(VERBOSE);
    let mut qbf: QuasiBestFirst<Druid, util::Ucb1Mast> =
        QuasiBestFirst::new(search, SmallRng::from_entropy());

    for _ in 0..10000 {
        qbf.search(&State::new());
    }
}
