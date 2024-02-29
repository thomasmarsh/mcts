/// This benchmark utility is intended to be invoke by
/// [SMAC3](https://github.com/automl/SMAC3) for hyperparameter optimization. We
/// Could do a grid search, but we'll try to do something smarter to save time.
use std::time::Duration;

use clap::Parser;

use mcts::game::Game;
use mcts::strategies::mcts::node::QInit;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::util::round_robin_multiple;
use mcts::util::AnySearch;
use mcts::util::Verbosity;
use rand::rngs::SmallRng;
use rand_core::SeedableRng;

const ROUNDS: usize = 20;
const PLAYOUT_DEPTH: usize = 200;
const C_TUNED: f64 = 1.625;
const MAX_ITER: usize = 10_000;
const EXPAND_THRESHOLD: u32 = 1;
const VERBOSE: bool = false;
const MAX_TIME_SECS: u64 = 0;

use mcts::games::traffic_lights;

type G = traffic_lights::TrafficLights;

type TS<S> = TreeSearch<G, S>;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    seed: u64,

    // #[arg(long)]
    // threshold: u32,

    // #[arg(long)]
    // bias: f64,
    #[arg(long)]
    c: f64,

    // #[arg(long)]
    // epsilon: f64,
    #[arg(long)]
    q_init: String,
}

fn main() {
    let args = Args::parse();
    let opponent = make_opponent(args.seed);
    let candidate = make_candidate(Args::parse());

    let mut strategies = vec![AnySearch::new(opponent), AnySearch::new(candidate)];
    let results = round_robin_multiple::<G, AnySearch<'_, G>>(
        &mut strategies,
        ROUNDS,
        &<G as Game>::S::default(),
        Verbosity::Silent,
    );
    let cost = calc_cost(results);
    println!("cost={}", cost);
}

fn calc_cost(results: Vec<mcts::util::Result>) -> f64 {
    let w = results[1].wins as f64;
    1.0 - w / (ROUNDS * 2) as f64
}

fn make_opponent(seed: u64) -> TS<strategy::Ucb1GraveMast> {
    TS::default()
        .config(
            SearchConfig::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .expand_threshold(EXPAND_THRESHOLD)
                .q_init(QInit::Parent)
                .select(select::Ucb1Grave {
                    exploration_constant: 0.69535,
                    threshold: 285,
                    bias: 628.,
                    current_ref_id: None,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.0015)),
        )
        .verbose(VERBOSE)
        .rng(SmallRng::seed_from_u64(seed))
}

fn _make_opponent(seed: u64) -> TS<strategy::Ucb1> {
    TS::default()
        .config(
            SearchConfig::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .expand_threshold(EXPAND_THRESHOLD)
                .select(select::Ucb1 {
                    exploration_constant: C_TUNED,
                }),
        )
        .verbose(VERBOSE)
        .rng(SmallRng::seed_from_u64(seed))
}

fn parse_q_init(s: &str) -> Option<QInit> {
    match s {
        "Draw" => Some(QInit::Draw),
        "Infinity" => Some(QInit::Infinity),
        "Loss" => Some(QInit::Loss),
        "Parent" => Some(QInit::Parent),
        "Win" => Some(QInit::Win),
        _ => None,
    }
}

fn make_candidate(args: Args) -> TS<strategy::Ucb1> {
    TS::default()
        .config(
            SearchConfig::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .expand_threshold(EXPAND_THRESHOLD)
                .q_init(parse_q_init(args.q_init.as_str()).unwrap())
                .use_transpositions(true)
                .select(select::Ucb1 {
                    exploration_constant: args.c,
                }),
        )
        .verbose(VERBOSE)
        .rng(SmallRng::seed_from_u64(args.seed))
}
