/// This benchmark utility is intended to be invoke by
/// [SMAC3](https://github.com/automl/SMAC3) for hyperparameter optimization. We
/// Could do a grid search, but we'll try to do something smarter to save time.
use clap::Parser;
use mcts::strategies::mcts::select::RaveSchedule;
use mcts::strategies::mcts::select::RaveUcb;
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

type G = mcts::games::druid::Druid;

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
    threshold: Option<u32>,

    #[arg(long)]
    c: Option<f64>,

    #[arg(long)]
    epsilon: f64,

    #[arg(long)]
    q_init: String,

    #[arg(long)]
    final_action: String,

    // #[arg(long)]
    alpha: Option<f64>,

    #[arg(long)]
    a: Option<f64>,

    #[arg(long)]
    bias: Option<f64>,

    #[arg(long)]
    schedule: Option<String>,

    #[arg(long)]
    k: Option<u32>,

    #[arg(long)]
    rave: Option<u32>,

    #[arg(long)]
    rave_ucb: Option<String>,
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Default)]
struct CandidateStrategy<FinalAction: SelectStrategy<G>>(PhantomData<FinalAction>);

impl<FinalAction: SelectStrategy<G>> Strategy<G> for CandidateStrategy<FinalAction> {
    type Select = select::Rave;
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

impl<FinalAction: SelectStrategy<G>> CandidateStrategy<FinalAction> {
    fn config_with_args(args: &Args) -> SearchConfig<G, Self> {
        let schedule = match args.schedule.clone().unwrap().as_str() {
            "hand_selected" => RaveSchedule::HandSelected { k: args.k.unwrap() },
            "min_mse" => RaveSchedule::MinMSE {
                bias: args.bias.unwrap(),
            },
            "threshold" => RaveSchedule::Threshold {
                rave: args.rave.unwrap(),
            },
            _ => unreachable!(),
        };

        let ucb = match args.rave_ucb.clone().unwrap().as_str() {
            "none" => RaveUcb::None,
            "ucb1" => RaveUcb::Ucb1 {
                exploration_constant: args.c.unwrap(),
            },
            "tuned" => RaveUcb::Ucb1Tuned {
                exploration_constant: args.c.unwrap(),
            },
            _ => unreachable!(),
        };
        Self::config()
            .q_init(QInit::from_str(args.q_init.as_str()).unwrap())
            .use_transpositions(false)
            .select(select::Rave::new(args.threshold.unwrap(), schedule, ucb))
            .simulate(
                simulate::DecisiveMove::new()
                    .mode(simulate::DecisiveMoveMode::WinLoss)
                    .inner(simulate::EpsilonGreedy::with_epsilon(args.epsilon)),
            )
            .seed(args.seed)
    }
}

fn make_candidate<FinalAction: SelectStrategy<G>>(
    args: &Args,
) -> TreeSearch<G, CandidateStrategy<FinalAction>> {
    TS::default().config(CandidateStrategy::config_with_args(args))
}

////////////////////////////////////////////////////////////////////////////////////////

fn make_baseline(seed: u64) -> TS<strategy::Ucb1DM> {
    type UcdDm = TreeSearch<G, strategy::Ucb1DM>;
    UcdDm::new().config(
        SearchConfig::new()
            .name("mcts[ucb1]+ucd+dm")
            .max_iterations(10_000)
            .expand_threshold(1)
            .use_transpositions(false)
            .q_init(QInit::Infinity)
            .select(select::Ucb1::with_c(0.01f64.sqrt()))
            .seed(seed),
    )
    // TS::new().config(
    //     base_config()
    //         .q_init(QInit::Parent)
    //         .select(
    //             select::Ucb1Grave::new()
    //                 .exploration_constant(0.69535)
    //                 .threshold(285)
    //                 .bias(628.),
    //         )
    //         .simulate(simulate::EpsilonGreedy::with_epsilon(0.0015))
    //         .rng(SmallRng::seed_from_u64(seed)),
    // )
}

fn _make_baseline(seed: u64) -> TS<strategy::Ucb1> {
    TS::new().config(base_config().select(select::Ucb1::with_c(1.625)).seed(seed))
}

////////////////////////////////////////////////////////////////////////////////////////

fn calc_cost(results: Vec<mcts::util::Result>) -> f64 {
    let w = results[1].wins as f64;
    1.0 - w / (ROUNDS * 2) as f64
}

fn optimize() {
    let args = Args::parse();
    let baseline = make_baseline(args.seed);
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

    let mut strategies = vec![AnySearch::new(baseline), AnySearch::new(candidate)];
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
    color_backtrace::install();
    optimize();
}
