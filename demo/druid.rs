#![allow(unused)]
use std::time::Duration;

use mcts::game::Game;
use mcts::games::druid::{Druid, State};
use mcts::strategies::mcts::node::QInit;
use mcts::strategies::mcts::select::SelectStrategy;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::strategies::mcts::{backprop, select, Strategy};
use mcts::util::{round_robin_multiple, AnySearch};
use mcts::util::{self_play, Verbosity};

const NUM_ROUNDS: usize = 10;
const PLAYOUT_DEPTH: usize = 200;
const C_LOW: f64 = 0.1;
const C_TUNED: f64 = 1.625;
const C_STD: f64 = 1.414;
const MAX_ITER: usize = 10000; //usize::MAX;
const BIAS: f64 = 700.0;
const EXPAND_THRESHOLD: u32 = 1;
const VERBOSE: bool = false;
const MAX_TIME_SECS: u64 = 0; // 0 = infinite

fn base_config<S: Strategy<Druid>>() -> SearchConfig<Druid, S>
where
    SearchConfig<Druid, S>: Default,
{
    SearchConfig::new()
        .max_iterations(MAX_ITER)
        .max_playout_depth(PLAYOUT_DEPTH)
        .max_time(Duration::from_secs(MAX_TIME_SECS))
        .expand_threshold(EXPAND_THRESHOLD)
        .verbose(VERBOSE)
}

fn main() {
    assert_eq!(Duration::default(), Duration::from_secs(0));

    // SMAC3 found:
    //
    // Configuration(values={
    //   'bias': 266.8785210698843,
    //   'c': 1.86169408634305,
    //   'epsilon': 0.10750788170844316,
    //   'threshold': 211,
    // })

    // hash=64f84b cost=0.22499999999999998 dict={'epsilon': 0.2460151296108961, 'final-action': 'robust_child', 'q-init': 'Win', 'rave-ucb': 'ucb1', 'schedule': 'threshold', 'threshold': 1456, 'c': 2.138511646538973, 'rave': 746}
    // hash=840351 cost=0.275 dict={'epsilon': 0.7775134909898043, 'final-action': 'robust_child', 'q-init': 'Infinity', 'rave-ucb': 'tuned', 'schedule': 'min_mse', 'threshold': 204, 'bias': 5.286671416833997, 'c': 0.28941824845969677}
    let rave_mast_ucd: TreeSearch<Druid, strategy::RaveMastDm> = TreeSearch::new().config(
        SearchConfig::new()
            .name("mcts[rave]+mast+ucd")
            .expand_threshold(1)
            .max_iterations(10_000)
            .use_transpositions(true)
            .q_init(QInit::Infinity)
            .select(
                select::Rave::default()
                    .ucb(select::RaveUcb::Ucb1Tuned {
                        exploration_constant: 0.2894182,
                    })
                    .threshold(204)
                    .schedule(select::RaveSchedule::MinMSE { bias: 5.2866714 }),
            )
            .simulate(
                simulate::DecisiveMove::new()
                    .inner(simulate::EpsilonGreedy::with_epsilon(0.7775134)),
            ),
    );
    self_play(rave_mast_ucd.clone());

    let mut amaf: TreeSearch<Druid, strategy::Amaf> =
        TreeSearch::new().config(base_config().select(select::Amaf::with_c(C_TUNED)));

    let mut amaf_mast: TreeSearch<Druid, strategy::AmafMast> = TreeSearch::new().config(
        base_config()
            .select(select::Amaf::with_c(C_TUNED))
            .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
    );

    let mut uct: TreeSearch<Druid, strategy::Ucb1> =
        TreeSearch::new().config(base_config().select(select::Ucb1::with_c(C_TUNED)));

    let mut uct_mast_low: TreeSearch<Druid, strategy::Ucb1Mast> = TreeSearch::new().config(
        base_config()
            .select(select::Ucb1::with_c(C_TUNED))
            .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
    );

    let mut uct_mast_high: TreeSearch<Druid, strategy::Ucb1Mast> = TreeSearch::new().config(
        base_config()
            .select(select::Ucb1::with_c(C_TUNED))
            .simulate(simulate::EpsilonGreedy::with_epsilon(0.9)),
    );

    let mut tuned: TreeSearch<Druid, strategy::Ucb1Tuned> =
        TreeSearch::new().config(base_config().select(select::Ucb1Tuned::with_c(C_TUNED)));

    let meta: TreeSearch<Druid, strategy::MetaMcts> = TreeSearch::new().config(
        base_config()
            .select(select::Ucb1::with_c(C_TUNED))
            .simulate(simulate::MetaMcts {
                inner: TreeSearch::new().config(
                    SearchConfig::new()
                        .max_iterations(3)
                        .max_playout_depth(PLAYOUT_DEPTH)
                        .max_time(Duration::default())
                        .expand_threshold(1)
                        .select(select::Ucb1::with_c(C_TUNED)),
                ),
            }),
    );

    let mut strategies: Vec<AnySearch<'_, Druid>> = vec![
        AnySearch::new(amaf),
        AnySearch::new(amaf_mast),
        AnySearch::new(tuned),
        AnySearch::new(uct),
        AnySearch::new(uct_mast_high),
        AnySearch::new(rave_mast_ucd),
        // AnySearch::new(meta),
    ];

    // Convert the vector of trait objects into a vector of mutable references

    round_robin_multiple::<Druid, AnySearch<'_, Druid>>(
        &mut strategies,
        NUM_ROUNDS,
        &<Druid as Game>::S::default(),
        Verbosity::Verbose,
    );
}
