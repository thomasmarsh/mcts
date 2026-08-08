//! `BenchGame` implementation for Druid.

use std::time::Duration;

use crate::game::{Game, PlayerIndex};
use crate::games::druid::{Druid, HashedState};
use crate::strategies::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use crate::strategies::Search;

use super::{BenchGame, MatchOutcome, StrategyInfo};

/// Default Druid presets for benchmarking, matching the server's preset
/// definitions (easy/medium/strong/master) but without engine caching
/// (each match creates fresh search instances).
pub struct DruidBenchGame;

impl DruidBenchGame {
    fn build_strategy(&self, strategy_id: &str) -> Box<dyn Search<G = Druid>> {
        match strategy_id {
            "1s-ucb1-nosolver" => Box::new(
                TreeSearch::<Druid, strategy::Ucb1>::new().config(
                    SearchConfig::new()
                        .name("1s-ucb1-nosolver")
                        .expand_threshold(1)
                        .use_transpositions(true)
                        .use_mcts_solver(false)
                        .q_init(QInit::Infinity)
                        .max_time(Duration::from_secs(1))
                        .select(select::Ucb1::with_c(1.414)),
                ),
            ),
            "1s-ucb1-solver" => Box::new(
                TreeSearch::<Druid, strategy::Ucb1>::new().config(
                    SearchConfig::new()
                        .name("1s-ucb1-solver")
                        .expand_threshold(1)
                        .use_transpositions(true)
                        .use_mcts_solver(true)
                        .q_init(QInit::Infinity)
                        .max_time(Duration::from_secs(1))
                        .select(select::Ucb1::with_c(1.414)),
                ),
            ),
            "2s-ucb1mast-nosolver" => Box::new(
                TreeSearch::<Druid, strategy::Ucb1Mast>::new().config(
                    SearchConfig::new()
                        .name("2s-ucb1mast-nosolver")
                        .expand_threshold(1)
                        .use_transpositions(true)
                        .use_mcts_solver(false)
                        .q_init(QInit::Infinity)
                        .max_time(Duration::from_secs(2))
                        .select(select::Ucb1::with_c(1.625))
                        .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
                ),
            ),
            "2s-ucb1mast-solver" => Box::new(
                TreeSearch::<Druid, strategy::Ucb1Mast>::new().config(
                    SearchConfig::new()
                        .name("2s-ucb1mast-solver")
                        .expand_threshold(1)
                        .use_transpositions(true)
                        .use_mcts_solver(true)
                        .q_init(QInit::Infinity)
                        .max_time(Duration::from_secs(2))
                        .select(select::Ucb1::with_c(1.625))
                        .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
                ),
            ),
            // Strong (3s) and Master (8s) use the NST-based composition.
            "strong" => {
                type S = strategy::Compose<
                    select::Ucb1,
                    simulate::DecisiveMove<Druid, simulate::EpsilonGreedy<Druid, simulate::Nst>>,
                >;
                Box::new(
                    TreeSearch::<Druid, S>::new().config(
                        SearchConfig::new()
                            .name("strong")
                            .expand_threshold(1)
                            .use_transpositions(true)
                            .use_mcts_solver(true)
                            .q_init(QInit::Infinity)
                            .max_time(Duration::from_secs(3))
                            .num_tree_threads(1)
                            .simulate(simulate::DecisiveMove::new().inner(
                                simulate::EpsilonGreedy::default()
                                    .epsilon(0.3)
                                    .inner(simulate::Nst::new().backoff_threshold(5)),
                            )),
                    ),
                )
            }
            "master" => {
                type S = strategy::Compose<
                    select::Ucb1,
                    simulate::DecisiveMove<Druid, simulate::EpsilonGreedy<Druid, simulate::Nst>>,
                >;
                Box::new(
                    TreeSearch::<Druid, S>::new().config(
                        SearchConfig::new()
                            .name("master")
                            .expand_threshold(1)
                            .use_transpositions(true)
                            .use_mcts_solver(true)
                            .q_init(QInit::Infinity)
                            .max_time(Duration::from_secs(8))
                            .num_tree_threads(1)
                            .simulate(simulate::DecisiveMove::new().inner(
                                simulate::EpsilonGreedy::default()
                                    .epsilon(0.3)
                                    .inner(simulate::Nst::new().backoff_threshold(5)),
                            )),
                    ),
                )
            }
            _ => {
                // Default: 1s UCB1 with solver.
                Box::new(
                    TreeSearch::<Druid, strategy::Ucb1>::new().config(
                        SearchConfig::new()
                            .name(strategy_id)
                            .expand_threshold(1)
                            .use_transpositions(true)
                            .use_mcts_solver(true)
                            .q_init(QInit::Infinity)
                            .max_time(Duration::from_secs(1))
                            .select(select::Ucb1::with_c(1.414)),
                    ),
                )
            }
        }
    }
}

impl BenchGame for DruidBenchGame {
    fn kind(&self) -> &'static str {
        "druid"
    }

    fn strategies(&self) -> Vec<StrategyInfo> {
        vec![
            StrategyInfo {
                id: "1s-ucb1-nosolver".into(),
                label: "1s UCB1 (no solver)".into(),
                description: "Plain UCB1 with random playouts, no MCTS-Solver, 1s per move."
                    .into(),
            },
            StrategyInfo {
                id: "1s-ucb1-solver".into(),
                label: "1s UCB1 (solver)".into(),
                description: "Plain UCB1 with random playouts and MCTS-Solver, 1s per move."
                    .into(),
            },
            StrategyInfo {
                id: "2s-ucb1mast-nosolver".into(),
                label: "2s UCB1+MAST (no solver)".into(),
                description: "UCB1 with MAST-biased playouts, no MCTS-Solver, 2s per move."
                    .into(),
            },
            StrategyInfo {
                id: "2s-ucb1mast-solver".into(),
                label: "2s UCB1+MAST (solver)".into(),
                description: "UCB1 with MAST-biased playouts and MCTS-Solver, 2s per move."
                    .into(),
            },
            StrategyInfo {
                id: "strong".into(),
                label: "Strong".into(),
                description: "N-gram-guided (NST) decisive-move search with MCTS-Solver, ~3s per move."
                    .into(),
            },
            StrategyInfo {
                id: "master".into(),
                label: "Master".into(),
                description: "Same search as Strong, ~8s thinking budget."
                    .into(),
            },
        ]
    }

    fn play_match(&self, strategy_a: &str, strategy_b: &str) -> MatchOutcome {
        let mut a = self.build_strategy(strategy_a);
        let mut b = self.build_strategy(strategy_b);

        let mut state = HashedState::default();
        let mut strategies: [&mut dyn Search<G = Druid>; 2] = [&mut *a, &mut *b];
        let mut s = 0;

        loop {
            if Druid::is_terminal(&state) {
                let winner = Druid::winner(&state);
                return MatchOutcome {
                    winner: winner.map(|p| {
                        if p.to_index() == s { 0 } else { 1 }
                    }),
                    extra: None,
                };
            }
            let strategy = &mut strategies[s];
            let action = strategy.choose_action(&state);
            state = Druid::apply(state, &action);
            s = 1 - s;
        }
    }
}