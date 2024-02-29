/// This benchmark utility is intended to be invoke by
/// [SMAC3](https://github.com/automl/SMAC3) for hyperparameter optimization. We
/// Could do a grid search, but we'll try to do something smarter to save time.
use clap::Parser;
use std::str::FromStr;
use std::time::Duration;

use mcts::game::Game;
use mcts::strategies::mcts::backprop;
use mcts::strategies::mcts::node::QInit;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::Strategy;
use mcts::strategies::mcts::TreeSearch;
use mcts::util::round_robin_multiple;
use mcts::util::AnySearch;
use mcts::util::Verbosity;
use rand::rngs::SmallRng;
use rand_core::SeedableRng;

use mcts::games::traffic_lights;

type G = traffic_lights::TrafficLights;

type TS<S> = TreeSearch<G, S>;

////////////////////////////////////////////////////////////////////////////////////////

const ROUNDS: usize = 20;
const PLAYOUT_DEPTH: usize = 200;
const MAX_ITER: usize = 10_000;
const EXPAND_THRESHOLD: u32 = 1;
const MAX_TIME_SECS: u64 = 0;

fn base_config<G: Game, S: Strategy<G>>() -> SearchConfig<G, S> {
    SearchConfig::new()
        .max_iterations(MAX_ITER)
        .max_playout_depth(PLAYOUT_DEPTH)
        .max_time(Duration::from_secs(MAX_TIME_SECS))
        .expand_threshold(EXPAND_THRESHOLD)
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    seed: u64,

    #[arg(long)]
    c: f64,

    #[arg(long)]
    epsilon: f64,

    #[arg(long)]
    q_init: String,
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Default)]
struct CandidateStrategy;

impl CandidateStrategy {
    fn config_with_args(args: &Args) -> SearchConfig<G, CandidateStrategy> {
        Self::config()
            .q_init(QInit::from_str(args.q_init.as_str()).unwrap())
            .use_transpositions(true)
            .select(select::Amaf::with_c(args.c))
            .simulate(
                simulate::DecisiveMove::new()
                    .mode(simulate::DecisiveMoveMode::WinLoss)
                    .inner(simulate::EpsilonGreedy::with_epsilon(args.epsilon)),
            )
            .rng(SmallRng::seed_from_u64(args.seed))
    }
}

impl Strategy<G> for CandidateStrategy {
    type Select = select::Amaf;
    type Simulate = simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Mast>>;
    type Backprop = backprop::Classic;
    type FinalAction = select::MaxAvgScore;

    fn friendly_name() -> String {
        "candidate".into()
    }

    fn config() -> SearchConfig<G, Self> {
        base_config()
    }
}

fn make_candidate(args: Args) -> TreeSearch<G, CandidateStrategy> {
    TS::default().config(CandidateStrategy::config_with_args(&args))
}

////////////////////////////////////////////////////////////////////////////////////////

fn make_opponent(seed: u64) -> TS<strategy::Ucb1GraveMast> {
    TS::new().config(
        base_config()
            .q_init(QInit::Parent)
            .select(
                select::Ucb1Grave::new()
                    .exploration_constant(0.69535)
                    .threshold(285)
                    .bias(628.),
            )
            .simulate(simulate::EpsilonGreedy::with_epsilon(0.0015))
            .rng(SmallRng::seed_from_u64(seed)),
    )
}

fn _make_opponent(seed: u64) -> TS<strategy::Ucb1> {
    TS::new().config(
        base_config()
            .select(select::Ucb1::with_c(1.625))
            .rng(SmallRng::seed_from_u64(seed)),
    )
}

////////////////////////////////////////////////////////////////////////////////////////

fn calc_cost(results: Vec<mcts::util::Result>) -> f64 {
    let w = results[1].wins as f64;
    1.0 - w / (ROUNDS * 2) as f64
}

fn optimize() {
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

fn main() {
    optimize();
}
