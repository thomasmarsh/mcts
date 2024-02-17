#![allow(unused)]
use std::time::Duration;

use mcts::games::druid::{Druid, State};
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::util;
use mcts::strategies::mcts::MctsStrategy;
use mcts::strategies::mcts::TreeSearch;
use mcts::util::Verbosity;
use mcts::util::{round_robin_multiple, AnySearch};

fn main() {
    const PLAYOUT_DEPTH: usize = 200;
    const C_LOW: f64 = 0.1;
    const C_TUNED: f64 = 1.625;
    const C_STD: f64 = 1.414;
    const MAX_ITER: usize = 10000; //usize::MAX; // 10000;
    const BIAS: f64 = 700.0;
    const EXPAND_THRESHOLD: u32 = 1;
    const VERBOSE: bool = false;
    const MAX_TIME_SECS: u64 = 0; // 0 = infinite

    assert_eq!(Duration::default(), Duration::from_secs(0));

    let grave: TreeSearch<Druid, util::McGrave> = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::McGrave {
                    threshold: 40,
                    bias: 5.,
                    ..Default::default()
                }),
        )
        .verbose(VERBOSE);

    let mut brave: TreeSearch<Druid, util::McBrave> = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::McBrave { bias: BIAS }),
        )
        .verbose(VERBOSE);

    let mut ucb1_grave: TreeSearch<Druid, util::Ucb1Grave> = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::Ucb1Grave {
                    bias: BIAS,
                    exploration_constant: C_LOW,
                    ..Default::default()
                }),
        )
        .verbose(VERBOSE);

    // SMAC3 found:
    //
    // Configuration(values={
    //   'bias': 266.8785210698843,
    //   'c': 1.86169408634305,
    //   'epsilon': 0.10750788170844316,
    //   'threshold': 211,
    // })
    let mut ucb1_grave_mast: TreeSearch<Druid, util::Ucb1GraveMast> = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::Ucb1Grave {
                    bias: 266.8785210698843,
                    exploration_constant: 1.86169408634305,
                    threshold: 211,
                    current_ref_id: None,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.10750788170844316)),
        )
        .verbose(VERBOSE);

    let mut amaf: TreeSearch<Druid, util::ScalarAmaf> = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::ScalarAmaf {
                    bias: BIAS,
                    exploration_constant: C_LOW,
                }),
        )
        .verbose(VERBOSE);

    let mut amaf_mast: TreeSearch<Druid, util::ScalarAmafMast> = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::ScalarAmaf {
                    bias: BIAS,
                    exploration_constant: C_LOW,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
        )
        .verbose(VERBOSE);

    let mut uct: TreeSearch<Druid, util::Ucb1> = TreeSearch::default()
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
        .verbose(VERBOSE);

    let mut uct_mast_low: TreeSearch<Druid, util::Ucb1Mast> = TreeSearch::default()
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

    let mut uct_mast_high: TreeSearch<Druid, util::Ucb1Mast> = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::Ucb1 {
                    exploration_constant: C_TUNED,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.9)),
        )
        .verbose(VERBOSE);

    let mut tuned: TreeSearch<Druid, util::Ucb1Tuned> = TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::Ucb1Tuned {
                    exploration_constant: C_TUNED,
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
    ];

    // Convert the vector of trait objects into a vector of mutable references

    round_robin_multiple::<Druid, AnySearch<'_, Druid>>(
        &mut strategies,
        10,
        &State::new(),
        Verbosity::Verbose,
    );
}
