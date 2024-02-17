/// This benchmark utility is intended to be invoke by
/// [SMAC3](https://github.com/automl/SMAC3) for hyperparameter optimization. We
/// Could do a grid search, but we'll try to do something smarter to save time.
use std::time::Duration;

use clap::Parser;

use mcts::games::druid::Druid;
use mcts::games::druid::State;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::util;
use mcts::strategies::mcts::MctsStrategy;
use mcts::strategies::mcts::TreeSearch;
use mcts::util::round_robin_multiple;
use mcts::util::AnySearch;
use mcts::util::Verbosity;
use rand::rngs::SmallRng;
use rand_core::SeedableRng;

const ROUNDS: usize = 20;
const PLAYOUT_DEPTH: usize = 200;
const C_TUNED: f64 = 1.625;
const MAX_ITER: usize = 10000;
const EXPAND_THRESHOLD: u32 = 5;
const VERBOSE: bool = false;
const MAX_TIME_SECS: u64 = 0;

type TS<S> = TreeSearch<Druid, S>;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    seed: u64,

    #[arg(long)]
    threshold: u32,

    #[arg(long)]
    bias: f64,

    #[arg(long)]
    c: f64,

    #[arg(long)]
    epsilon: f64,
}

fn main() {
    let args = Args::parse();
    let opponent = make_opponent(args.seed);
    let candidate = make_candidate(Args::parse());

    let mut strategies = vec![AnySearch::new(opponent), AnySearch::new(candidate)];
    let results = round_robin_multiple::<Druid, AnySearch<'_, Druid>>(
        &mut strategies,
        ROUNDS,
        &State::new(),
        Verbosity::Silent,
    );
    let cost = calc_cost(results);
    println!("cost={}", cost);
}

fn calc_cost(results: Vec<mcts::util::Result>) -> f64 {
    let w = results[1].wins as f64;
    1.0 - w / (ROUNDS * 2) as f64
}

fn make_opponent(seed: u64) -> TS<util::Ucb1> {
    TS::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::Ucb1 {
                    exploration_constant: C_TUNED,
                }),
        )
        .verbose(VERBOSE)
        .rng(SmallRng::seed_from_u64(seed))
}

fn make_candidate(args: Args) -> TS<util::Ucb1GraveMast> {
    TS::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::Ucb1Grave {
                    exploration_constant: args.c,
                    threshold: args.threshold,
                    bias: args.bias,
                    current_ref_id: None,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(args.epsilon)),
        )
        .verbose(VERBOSE)
        .rng(SmallRng::seed_from_u64(args.seed))
}
