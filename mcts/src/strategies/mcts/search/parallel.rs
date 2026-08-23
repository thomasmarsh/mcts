use crate::game::Game;
use crate::game::PlayerIndex;
use crate::strategies::mcts::config::{GraphSearch, GraphStats};
use crate::strategies::mcts::node::Proven;
use crate::strategies::mcts::search::shared::SearchContext;
use crate::strategies::mcts::search::shared::{
    add_path_virtual_loss, backprop_correction_step, backprop_step, last_tree_action, select_step,
    simulate_step,
};
use crate::strategies::mcts::search::shared::{ActionTotal, Shared};
use crate::strategies::mcts::search::TreeSearch;
use crate::strategies::mcts::select::SelectStrategy;
use crate::strategies::mcts::simulate::SimulateStrategy;
use crate::strategies::mcts::stack::NodeStack;
use crate::strategies::mcts::table::TranspositionKey;
use crate::strategies::Search;
use crate::util::random_best;

use rand::rngs::SmallRng;
use rand::Rng;
use rand_core::SeedableRng;
use rustc_hash::FxHashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    fn root_action_totals(&self) -> Vec<ActionTotal<G::A>> {
        let node = self.index.get(self.root_id);
        let children = node.children();
        (0..children.len())
            .filter(|&i| children.is_explored(i))
            .map(|i| {
                let child_id = children.node_id(i).unwrap();
                if matches!(self.config.graph_stats(), Some(GraphStats::Nodes)) {
                    let child = self.index.get(child_id);
                    let scores = (0..G::num_players())
                        .map(|p| child.stats.score(p))
                        .collect();
                    (children.action(i).clone(), child.stats.num_visits(), scores)
                } else {
                    let scores = (0..G::num_players())
                        .map(|p| children.score(i, p))
                        .collect();
                    (children.action(i).clone(), children.num_visits(i), scores)
                }
            })
            .collect()
    }

    pub(crate) fn choose_action_root_parallel(&mut self, state: &G::S) -> G::A {
        let num_threads = self.config.num_threads.max(1);
        debug_assert!(num_threads > 1);
        debug_assert!(
            !self.config.use_mcts_solver,
            "root parallelism's visit-sum merge doesn't account for trees that stop \
             early on a solver proof -- combining num_threads > 1 with use_mcts_solver \
             is not supported yet"
        );

        let seeds: Vec<u64> = (0..num_threads).map(|_| self.config.rng.gen()).collect();
        let mut workers: Vec<Self> = (0..num_threads - 1).map(|_| self.clone()).collect();

        let totals = std::thread::scope(|scope| {
            let handles: Vec<_> = workers
                .iter_mut()
                .zip(&seeds)
                .map(|(worker, &seed)| {
                    worker.config.num_threads = 1;
                    worker.config.rng = SmallRng::seed_from_u64(seed);
                    scope.spawn(move || {
                        worker.choose_action(state);
                        worker.root_action_totals()
                    })
                })
                .collect();

            self.config.num_threads = 1;
            self.config.rng = SmallRng::seed_from_u64(seeds[num_threads - 1]);
            self.choose_action(state);

            let mut totals = vec![self.root_action_totals()];
            totals.extend(handles.into_iter().map(|h| h.join().unwrap()));
            totals
        });

        self.config.num_threads = num_threads;

        let mut merged: FxHashMap<G::A, (u32, Vec<f64>)> = FxHashMap::default();
        for worker_totals in totals {
            for (action, visits, scores) in worker_totals {
                let entry = merged
                    .entry(action)
                    .or_insert_with(|| (0, vec![0.; scores.len()]));
                entry.0 += visits;
                for (i, s) in scores.into_iter().enumerate() {
                    entry.1[i] += s;
                }
            }
        }

        let merged: Vec<ActionTotal<G::A>> = merged
            .into_iter()
            .map(|(action, (visits, scores))| (action, visits, scores))
            .collect();
        random_best(&merged, &mut self.config.rng, |(_, visits, _)| {
            *visits as f64
        })
        .map(|(action, _, _)| action.clone())
        .unwrap()
    }

    pub(crate) fn choose_action_tree_parallel(
        &mut self,
        state: &G::S,
        report_start: super::SearchReportStart,
    ) -> G::A {
        let num_threads = self.config.num_tree_threads.max(1);
        debug_assert!(num_threads > 1);
        debug_assert_eq!(self.config.num_threads, 1);

        let explicit_dag = matches!(self.config.graph_search, GraphSearch::Dag(_));
        let hash = G::zobrist_hash(state);
        let root_id = if explicit_dag {
            self.reuse_or_reset_graph(G::player_to_move(state).to_index(), state)
        } else {
            self.reuse_or_reset(G::player_to_move(state).to_index(), state)
        };
        if explicit_dag {
            self.table.insert_graph(
                TranspositionKey {
                    position_hash: hash,
                    ply: 0,
                },
                root_id,
            );
        } else if self.config.use_transpositions {
            self.table.insert(hash, root_id);
        }

        self.timer.start(self.config.max_time);

        let shared = Shared {
            index: &self.index,
            root_state: state,
            root_stats: &self.root_stats,
            table: &self.table,
            global: &self.stats,
            expand_threshold: self.config.expand_threshold,
            q_init: self.config.q_init,
            use_transpositions: self.config.uses_transpositions(),
            graph_stats: self.config.graph_stats(),
            explicit_dag: matches!(self.config.graph_search, GraphSearch::Dag(_)),
            use_mcts_solver: self.config.use_mcts_solver,
            max_playout_depth: self.config.max_playout_depth,
            solver_loss_threshold: self.config.solver_loss_threshold,
            has_amaf: self.config.requirements().amaf,
            mcgs_correction: self.config.mcgs_correction,
        };
        let iterations_remaining = AtomicUsize::new(self.config.max_iterations);
        let k = self.config.num_rollouts_per_leaf.max(1);
        let timer = &self.timer;
        let backprop_strategy = &self.config.backprop;

        let seeds: Vec<u64> = (0..num_threads).map(|_| self.config.rng.gen()).collect();
        let mut select_strategies: Vec<S::Select> = (0..num_threads)
            .map(|_| self.config.select.clone())
            .collect();
        let mut simulate_strategies: Vec<S::Simulate> = (0..num_threads)
            .map(|_| self.config.simulate.clone())
            .collect();
        let mut prior_strategies: Vec<
            Option<Box<dyn crate::strategies::mcts::prior::PriorStrategyDyn<G>>>,
        > = (0..num_threads)
            .map(|_| self.config.prior.clone())
            .collect();

        std::thread::scope(|scope| {
            for (((seed, select_strategy), simulate_strategy), prior_strategy) in seeds
                .into_iter()
                .zip(select_strategies.iter_mut())
                .zip(simulate_strategies.iter_mut())
                .zip(prior_strategies.iter_mut())
            {
                let shared = &shared;
                let iterations_remaining = &iterations_remaining;
                scope.spawn(move || {
                    let mut rng = SmallRng::seed_from_u64(seed);
                    loop {
                        if timer.done() {
                            break;
                        }
                        if shared.use_mcts_solver
                            && shared.index.get(root_id).proven() != Proven::Unproven
                        {
                            break;
                        }
                        if iterations_remaining
                            .fetch_update(Relaxed, Relaxed, |n| n.checked_sub(1))
                            .is_err()
                        {
                            break;
                        }

                        let mut stack = Vec::new();
                        let mut ctx = SearchContext::new(root_id, state.clone());
                        let correction = select_step(
                            shared,
                            &mut ctx,
                            &mut stack,
                            select_strategy,
                            &mut rng,
                            prior_strategy.as_deref_mut(),
                        );

                        if let Some(utilities) = correction {
                            backprop_correction_step(shared, &stack, &utilities);
                            continue;
                        }

                        let node_stack = NodeStack::<G::A>::new(stack.clone());
                        if k > 1 {
                            add_path_virtual_loss(
                                shared.index,
                                &node_stack,
                                k - 1,
                                shared.graph_stats,
                            );
                        }
                        let prev_action = last_tree_action::<G>(
                            shared.index,
                            &stack,
                            state,
                            shared.use_transpositions,
                        );
                        for _ in 0..k {
                            let trial = simulate_step(
                                shared.max_playout_depth,
                                shared.global,
                                simulate_strategy,
                                &ctx.state,
                                prev_action.clone(),
                                &mut rng,
                            );
                            let flags = select_strategy.backprop_flags()
                                | simulate_strategy.backprop_flags();
                            backprop_step(shared, &stack, backprop_strategy, trial, flags);
                        }
                    }
                });
            }
        });

        let solved =
            self.config.use_mcts_solver && self.index.get(root_id).proven() != Proven::Unproven;
        let time_expired = self.timer.done();

        let selected = self.select_final_action(state);
        self.compute_pv(state, Some(&selected));
        self.verbose_summary(state, num_threads);
        self.finish_search_report(report_start, time_expired, solved, false);
        selected
    }
}
