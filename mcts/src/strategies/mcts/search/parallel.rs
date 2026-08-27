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

/// One completed independent worker's reportable totals. This is collected
/// only after a worker has stopped, so root-parallel reporting adds no work or
/// synchronization to the simulation hot path.
#[derive(Debug, Clone)]
pub(crate) struct RootParallelWorker<A> {
    pub action_totals: Vec<ActionTotal<A>>,
    pub completed_iterations: usize,
    pub tree_nodes: usize,
    pub accum_depth: usize,
    pub max_depth: usize,
    pub tt_reads: usize,
    pub tt_writes: usize,
    pub tt_hits: usize,
}

/// The numerical part of a root-parallel final report. Independent trees have
/// no globally mergeable structure, but their root totals and per-search work
/// counters do compose exactly.
#[derive(Debug, Clone)]
pub(crate) struct RootParallelTotals<A> {
    pub action_totals: Vec<ActionTotal<A>>,
    pub completed_iterations: usize,
    pub tree_nodes: usize,
    pub accum_depth: usize,
    pub max_depth: usize,
    pub tt_reads: usize,
    pub tt_writes: usize,
    pub tt_hits: usize,
}

/// Merges finished worker snapshots without treating the coordinator as an
/// extra tree. The caller supplies it once alongside the spawned workers.
pub(crate) fn merge_root_parallel_workers<A>(
    workers: impl IntoIterator<Item = RootParallelWorker<A>>,
) -> RootParallelTotals<A>
where
    A: Eq + std::hash::Hash,
{
    let mut actions: FxHashMap<A, (u32, Vec<f64>)> = FxHashMap::default();
    let mut totals = RootParallelTotals {
        action_totals: vec![],
        completed_iterations: 0,
        tree_nodes: 0,
        accum_depth: 0,
        max_depth: 0,
        tt_reads: 0,
        tt_writes: 0,
        tt_hits: 0,
    };
    for worker in workers {
        totals.completed_iterations = totals
            .completed_iterations
            .saturating_add(worker.completed_iterations);
        totals.tree_nodes = totals.tree_nodes.saturating_add(worker.tree_nodes);
        totals.accum_depth = totals.accum_depth.saturating_add(worker.accum_depth);
        totals.max_depth = totals.max_depth.max(worker.max_depth);
        totals.tt_reads = totals.tt_reads.saturating_add(worker.tt_reads);
        totals.tt_writes = totals.tt_writes.saturating_add(worker.tt_writes);
        totals.tt_hits = totals.tt_hits.saturating_add(worker.tt_hits);
        for (action, visits, scores) in worker.action_totals {
            let entry = actions
                .entry(action)
                .or_insert_with(|| (0, vec![0.; scores.len()]));
            entry.0 = entry.0.saturating_add(visits);
            for (i, score) in scores.into_iter().enumerate() {
                entry.1[i] += score;
            }
        }
    }
    totals.action_totals = actions
        .into_iter()
        .map(|(action, (visits, scores))| (action, visits, scores))
        .collect();
    totals
}

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

    fn root_parallel_worker_report(&self) -> RootParallelWorker<G::A> {
        let run = self
            .last_search_run
            .as_ref()
            .expect("root-parallel workers always finish their own search report");
        RootParallelWorker {
            action_totals: self.root_action_totals(),
            completed_iterations: self.stats.iter_count.load(Relaxed),
            tree_nodes: self.index.len(),
            accum_depth: self.stats.accum_depth.load(Relaxed),
            max_depth: self.stats.max_depth.load(Relaxed),
            tt_reads: run.tt_reads,
            tt_writes: run.tt_writes,
            tt_hits: run.tt_hits,
        }
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

        // PIMC: every worker (including the coordinator) gets its own
        // independent determinization of the root, sampled from its own
        // seeded rng, rather than all workers sharing the one literal
        // `state`.
        let determinize_root = self.config.determinize_root;

        let worker_states: Vec<G::S> = workers
            .iter_mut()
            .zip(&seeds)
            .map(|(worker, &seed)| {
                worker.config.num_threads = 1;
                worker.config.rng = SmallRng::seed_from_u64(seed);
                if determinize_root {
                    G::determinize(state.clone(), &mut worker.config.rng)
                } else {
                    state.clone()
                }
            })
            .collect();

        self.config.num_threads = 1;
        self.config.rng = SmallRng::seed_from_u64(seeds[num_threads - 1]);
        let own_state = if determinize_root {
            G::determinize(state.clone(), &mut self.config.rng)
        } else {
            state.clone()
        };

        std::thread::scope(|scope| {
            let handles: Vec<_> = workers
                .iter_mut()
                .zip(&worker_states)
                .map(|(worker, worker_state)| {
                    scope.spawn(move || {
                        worker.choose_action(worker_state);
                    })
                })
                .collect();

            self.choose_action(&own_state);

            for handle in handles {
                handle.join().unwrap();
            }
        });

        self.config.num_threads = num_threads;

        let mut worker_reports = vec![self.root_parallel_worker_report()];
        worker_reports.extend(workers.iter().map(Self::root_parallel_worker_report));
        let merged = merge_root_parallel_workers(worker_reports.clone());
        let selected = random_best(
            &merged.action_totals,
            &mut self.config.rng,
            |(_, visits, _)| *visits as f64,
        )
        .map(|(action, _, _)| action.clone())
        .unwrap();

        let (pv_worker, _) = worker_reports
            .iter()
            .enumerate()
            .map(|(i, worker)| {
                let visits = worker
                    .action_totals
                    .iter()
                    .find(|(action, _, _)| action == &selected)
                    .map_or(0, |(_, visits, _)| *visits);
                (i, visits)
            })
            .max_by_key(|&(i, visits)| (visits, std::cmp::Reverse(i)))
            .expect("at least the coordinator worker exists");
        let principal_variation = if pv_worker == 0 {
            self.compute_pv(state, Some(&selected));
            self.pv.clone()
        } else {
            let worker = &mut workers[pv_worker - 1];
            worker.compute_pv(state, Some(&selected));
            worker.pv.clone()
        };
        let principal_variation = if principal_variation.first() == Some(&selected) {
            principal_variation
        } else {
            vec![]
        };
        self.pv = principal_variation.clone();
        self.root_parallel_report = Some(super::RootParallelReport {
            actions: merged.action_totals,
            principal_variation,
            completed_iterations: merged.completed_iterations,
            tree_nodes: merged.tree_nodes,
            accum_depth: merged.accum_depth,
            max_depth: merged.max_depth,
            tt_reads: merged.tt_reads,
            tt_writes: merged.tt_writes,
            tt_hits: merged.tt_hits,
        });
        selected
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
                TranspositionKey::new(self.config.transposition_keying, hash, 0),
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
            keying: self.config.transposition_keying,
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

#[cfg(test)]
mod tests {
    use super::{merge_root_parallel_workers, RootParallelWorker};
    use crate::game::{Game, PlayerIndex};
    use crate::strategies::mcts::strategy::Ucb1;
    use crate::strategies::Search;
    use crate::{SearchConfig, TreeSearch};
    use rand::rngs::SmallRng;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    // A single-ply, one-action game: exists only to prove
    // `choose_action_root_parallel` calls `Game::determinize` once per
    // root-parallel worker (including the coordinator) when
    // `determinize_root` is on, and not at all when it's off -- `winner`
    // doesn't depend on anything `determinize` could change, so it's
    // deliberately not a test of search strength or of any particular
    // game's own hidden-information semantics.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
    struct Seat(usize);

    impl PlayerIndex for Seat {
        fn to_index(&self) -> usize {
            self.0
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
    struct CoinState {
        resolved: bool,
    }

    impl std::fmt::Display for CoinState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "resolved={}", self.resolved)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize)]
    struct Flip;

    #[derive(Clone)]
    struct Coin;

    static DETERMINIZE_CALLS: AtomicUsize = AtomicUsize::new(0);

    impl Game for Coin {
        type S = CoinState;
        type A = Flip;
        type P = Seat;

        fn apply(_state: Self::S, _action: &Self::A) -> Self::S {
            CoinState { resolved: true }
        }

        fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
            if !state.resolved {
                actions.push(Flip);
            }
        }

        fn winner(state: &Self::S) -> Option<Self::P> {
            state.resolved.then_some(Seat(0))
        }

        fn player_to_move(_state: &Self::S) -> Self::P {
            Seat(0)
        }

        fn determinize(state: Self::S, _rng: &mut SmallRng) -> Self::S {
            DETERMINIZE_CALLS.fetch_add(1, Relaxed);
            state
        }
    }

    // Runs the root-parallel search and returns how many times
    // `Coin::determinize` fired. `simulate_step` (`search/shared.rs`) also
    // calls `Game::determinize` once per rollout regardless of
    // `determinize_root`, so this count is never purely the root-level
    // calls this test cares about -- `num_threads` workers each running
    // exactly `max_iterations` single-threaded searches means that rollout
    // noise is the same fixed number on every call with the same
    // `max_iterations`, so comparing two runs' totals isolates the
    // root-level share.
    fn run_and_count_determinize_calls(num_threads: usize, determinize_root: bool) -> usize {
        DETERMINIZE_CALLS.store(0, Relaxed);
        let mut search = TreeSearch::<Coin, Ucb1>::new().config(
            SearchConfig::new()
                .max_iterations(4)
                .num_threads(num_threads)
                .determinize_root(determinize_root)
                .seed(1),
        );
        search.choose_action(&CoinState::default());
        DETERMINIZE_CALLS.load(Relaxed)
    }

    #[test]
    fn determinize_root_calls_game_determinize_once_per_worker() {
        let num_threads = 4;
        let without = run_and_count_determinize_calls(num_threads, false);
        let with = run_and_count_determinize_calls(num_threads, true);
        assert_eq!(
            with,
            without + num_threads,
            "determinize_root should add exactly one extra Game::determinize call per \
             root-parallel worker (including the coordinator) on top of whatever \
             rollout-triggered calls already happen regardless of the flag"
        );
    }

    #[test]
    fn merge_root_parallel_workers_sums_work_and_keeps_largest_depth() {
        let merged = merge_root_parallel_workers([
            RootParallelWorker {
                action_totals: vec![(1_u8, 3, vec![1.5, -1.5]), (2, 2, vec![0.0, 0.0])],
                completed_iterations: 5,
                tree_nodes: 7,
                accum_depth: 9,
                max_depth: 4,
                tt_reads: 11,
                tt_writes: 3,
                tt_hits: 2,
            },
            RootParallelWorker {
                action_totals: vec![(1, 4, vec![2.0, -2.0]), (3, 1, vec![-1.0, 1.0])],
                completed_iterations: 5,
                tree_nodes: 8,
                accum_depth: 12,
                max_depth: 6,
                tt_reads: 13,
                tt_writes: 5,
                tt_hits: 7,
            },
        ]);
        let action_one = merged
            .action_totals
            .iter()
            .find(|(action, _, _)| *action == 1)
            .unwrap();
        assert_eq!(action_one.1, 7);
        assert_eq!(action_one.2, vec![3.5, -3.5]);
        assert_eq!(merged.completed_iterations, 10);
        assert_eq!(merged.tree_nodes, 15);
        assert_eq!(merged.accum_depth, 21);
        assert_eq!(merged.max_depth, 6);
        assert_eq!(
            (merged.tt_reads, merged.tt_writes, merged.tt_hits),
            (24, 8, 9)
        );
    }
}
