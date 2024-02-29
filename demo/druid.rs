#![allow(unused)]
use std::time::Duration;

use mcts::game::Game;
use mcts::games::druid::{Druid, State};
use mcts::strategies::mcts::select::SelectStrategy;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::strategies::mcts::{backprop, select, Strategy};
use mcts::util::Verbosity;
use mcts::util::{round_robin_multiple, AnySearch};

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
    SearchConfig::default()
        .max_iterations(MAX_ITER)
        .max_playout_depth(PLAYOUT_DEPTH)
        .max_time(Duration::from_secs(MAX_TIME_SECS))
        .expand_threshold(EXPAND_THRESHOLD)
}

fn main() {
    assert_eq!(Duration::default(), Duration::from_secs(0));

    let grave: TreeSearch<Druid, strategy::McGrave> = TreeSearch::default()
        .config(base_config().select(select::McGrave {
            threshold: 40,
            bias: 5.,
            ..Default::default()
        }))
        .verbose(VERBOSE);

    let mut brave: TreeSearch<Druid, strategy::McBrave> = TreeSearch::default()
        .config(base_config().select(select::McBrave { bias: BIAS }))
        .verbose(VERBOSE);

    let mut ucb1_grave: TreeSearch<Druid, strategy::Ucb1Grave> = TreeSearch::default()
        .config(base_config().select(select::Ucb1Grave {
            bias: BIAS,
            exploration_constant: C_LOW,
            ..Default::default()
        }))
        .verbose(VERBOSE);

    // SMAC3 found:
    //
    // Configuration(values={
    //   'bias': 266.8785210698843,
    //   'c': 1.86169408634305,
    //   'epsilon': 0.10750788170844316,
    //   'threshold': 211,
    // })
    let mut ucb1_grave_mast: TreeSearch<Druid, strategy::Ucb1GraveMast> = TreeSearch::default()
        .config(
            base_config()
                .select(select::Ucb1Grave {
                    bias: 266.8785210698843,
                    exploration_constant: 1.86169408634305,
                    threshold: 211,
                    current_ref_id: None,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.10750788170844316)),
        )
        .verbose(VERBOSE);

    let mut amaf: TreeSearch<Druid, strategy::Amaf> = TreeSearch::default()
        .config(base_config().select(select::Amaf {
            exploration_constant: C_TUNED,
        }))
        .verbose(VERBOSE);

    let mut amaf_mast: TreeSearch<Druid, strategy::AmafMast> = TreeSearch::default()
        .config(
            base_config()
                .select(select::Amaf {
                    exploration_constant: C_TUNED,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
        )
        .verbose(VERBOSE);

    let mut uct: TreeSearch<Druid, strategy::Ucb1> = TreeSearch::default()
        .config(base_config().select(select::Ucb1 {
            exploration_constant: C_TUNED,
        }))
        .verbose(VERBOSE);

    let mut uct_mast_low: TreeSearch<Druid, strategy::Ucb1Mast> = TreeSearch::default()
        .config(
            base_config()
                .select(select::Ucb1 {
                    exploration_constant: C_TUNED,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
        )
        .verbose(VERBOSE);

    let mut uct_mast_high: TreeSearch<Druid, strategy::Ucb1Mast> = TreeSearch::default()
        .config(
            base_config()
                .select(select::Ucb1 {
                    exploration_constant: C_TUNED,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.9)),
        )
        .verbose(VERBOSE);

    let mut tuned: TreeSearch<Druid, strategy::Ucb1Tuned> = TreeSearch::default()
        .config(base_config().select(select::Ucb1Tuned {
            exploration_constant: C_TUNED,
        }))
        .verbose(VERBOSE);

    let meta: TreeSearch<Druid, strategy::MetaMcts> = TreeSearch::default()
        .config(
            base_config()
                .select(select::Ucb1 {
                    exploration_constant: C_TUNED,
                })
                .simulate(simulate::MetaMcts {
                    inner: TreeSearch::default().config(
                        SearchConfig::default()
                            .max_iterations(3)
                            .max_playout_depth(PLAYOUT_DEPTH)
                            .max_time(Duration::default())
                            .expand_threshold(1)
                            .select(select::Ucb1 {
                                exploration_constant: C_TUNED,
                            }),
                    ),
                }),
        )
        .verbose(VERBOSE);

    let mut strategies: Vec<AnySearch<'_, Druid>> = vec![
        AnySearch::new(amaf),
        AnySearch::new(amaf_mast),
        AnySearch::new(tuned),
        AnySearch::new(uct),
        AnySearch::new(uct_mast_high),
        AnySearch::new(grave),
        AnySearch::new(ucb1_grave_mast),
        // AnySearch::new(meta),
    ];

    // Convert the vector of trait objects into a vector of mutable references

    round_robin_multiple::<Druid, AnySearch<'_, Druid>>(
        &mut strategies,
        NUM_ROUNDS,
        &State::new(),
        Verbosity::Verbose,
    );
}
