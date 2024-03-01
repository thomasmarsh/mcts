/// This benchmark utility is intended to be invoke by
/// [SMAC3](https://github.com/automl/SMAC3) for hyperparameter optimization. We
/// Could do a grid search, but we'll try to do something smarter to save time.
use clap::Parser;
use mcts::strategies::mcts::select::SelectStrategy;
use std::marker::PhantomData;
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

    #[arg(long)]
    final_action: String,

    #[arg(long)]
    alpha: f64,

    #[arg(long)]
    a: Option<f64>,
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Default)]
struct CandidateStrategy<FinalAction: SelectStrategy<G>>(PhantomData<FinalAction>);

impl<FinalAction: SelectStrategy<G>> CandidateStrategy<FinalAction> {
    fn config_with_args(args: &Args) -> SearchConfig<G, Self> {
        Self::config()
            .q_init(QInit::from_str(args.q_init.as_str()).unwrap())
            .use_transpositions(true)
            .select(select::Amaf::with_c(args.c).alpha(args.alpha))
            .simulate(
                simulate::DecisiveMove::new()
                    .mode(simulate::DecisiveMoveMode::WinLoss)
                    .inner(simulate::EpsilonGreedy::with_epsilon(args.epsilon)),
            )
            .rng(SmallRng::seed_from_u64(args.seed))
    }
}

impl<FinalAction: SelectStrategy<G>> Strategy<G> for CandidateStrategy<FinalAction> {
    type Select = select::Amaf;
    type Simulate = simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Mast>>;
    type Backprop = backprop::Classic;
    type FinalAction = FinalAction;

    fn friendly_name() -> String {
        "candidate".into()
    }

    fn config() -> SearchConfig<G, Self> {
        base_config()
    }
}

fn make_candidate<FinalAction: SelectStrategy<G>>(
    args: &Args,
) -> TreeSearch<G, CandidateStrategy<FinalAction>> {
    TS::default().config(CandidateStrategy::config_with_args(args))
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
    let candidate = match args.final_action.as_str() {
        "max_avg" => AnySearch::new(make_candidate::<select::SecureChild>(&args)),
        "secure_child" => {
            let mut ts = make_candidate::<select::SecureChild>(&args);
            ts.config.final_action.a = args.a.unwrap();
            AnySearch::new(ts)
        }
        "robust_child" => AnySearch::new(make_candidate::<select::RobustChild>(&args)),
        _ => unreachable!(),
    };

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
