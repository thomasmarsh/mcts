use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::Real;
use crate::strategies::mcts::config::GraphSearch;
use crate::strategies::mcts::config::GraphStats;
use crate::strategies::mcts::node::real_action;
use crate::strategies::mcts::node::Proven;
use crate::strategies::mcts::search::shared::SearchContext;
use crate::strategies::mcts::search::TreeSearch;
use crate::strategies::mcts::stack::NodeStack;
use crate::strategies::mcts::table::TranspositionKey;
use crate::strategies::{
    ActionReport, RootReport, Search, SearchActionReport, SearchGraphMode, SearchReport,
    SearchReportReason, SearchReportStatus, SearchTermination, SearchWarning,
};
use crate::symmetry::incoming_sym;

use std::sync::atomic::Ordering::Relaxed;

const ACTION_REPORT_LIMIT: usize = 1_024;
const PRINCIPAL_VARIATION_LIMIT: usize = 256;

/// Classifies a loop's boundary without inspecting wall-clock time. Keeping
/// this pure makes the precedence around a proof found on the final allowed
/// iteration explicit and independently testable.
pub(crate) fn classify_termination(
    iteration_limit: Option<usize>,
    completed_iterations: usize,
    time_expired: bool,
    solved: bool,
) -> SearchTermination {
    if solved {
        SearchTermination::Solved
    } else if time_expired {
        SearchTermination::Time
    } else if iteration_limit.is_some_and(|limit| completed_iterations >= limit) {
        SearchTermination::Iterations
    } else {
        SearchTermination::Unknown
    }
}

fn finite(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    fn begin_search_report(&mut self) -> super::SearchReportStart {
        self.last_search_run = None;
        self.root_parallel_report = None;
        super::SearchReportStart {
            started_at: std::time::Instant::now(),
            tt_reads: self.table.reads.load(Relaxed),
            tt_writes: self.table.writes.load(Relaxed),
            tt_hits: self.table.hits.load(Relaxed),
        }
    }

    pub(crate) fn finish_search_report(
        &mut self,
        start: super::SearchReportStart,
        time_expired: bool,
        solved: bool,
        partial_root_parallel: bool,
    ) {
        let elapsed_seconds = finite(start.started_at.elapsed().as_secs_f64());
        let completed_iterations = self.stats.iter_count.load(Relaxed);
        self.last_search_run = Some(super::SearchRun {
            elapsed_seconds,
            tt_reads: self.root_parallel_report.as_ref().map_or_else(
                || {
                    self.table
                        .reads
                        .load(Relaxed)
                        .saturating_sub(start.tt_reads)
                },
                |report| report.tt_reads,
            ),
            tt_writes: self.root_parallel_report.as_ref().map_or_else(
                || {
                    self.table
                        .writes
                        .load(Relaxed)
                        .saturating_sub(start.tt_writes)
                },
                |report| report.tt_writes,
            ),
            tt_hits: self.root_parallel_report.as_ref().map_or_else(
                || self.table.hits.load(Relaxed).saturating_sub(start.tt_hits),
                |report| report.tt_hits,
            ),
            termination: classify_termination(
                (self.config.max_iterations != usize::MAX).then_some(self.config.max_iterations),
                completed_iterations,
                time_expired,
                solved,
            ),
            root_parallel: partial_root_parallel.then(|| {
                self.root_parallel_report
                    .take()
                    .expect("root-parallel searches always produce an aggregate report")
            }),
        });
    }
}

impl<G, S> Search for TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Default,
    G::S: std::fmt::Display,
{
    type G = G;

    fn friendly_name(&self) -> String {
        self.config.name.clone()
    }

    fn choose_action(&mut self, state: &G::S) -> G::A {
        let report_start = self.begin_search_report();
        let explicit_dag = matches!(self.config.graph_search, GraphSearch::Dag(_));
        if explicit_dag {
            assert!(
                !self.config.use_transpositions,
                "graph_search replaces the legacy use_transpositions setting"
            );
        }
        // Order matters for hybrid root+tree parallelism: `num_threads`
        // (trees) is checked first so `choose_action_root_parallel` gets a
        // chance to spawn its independent trees; each of *those* then
        // recurses back into this same dispatch with `num_threads` forced to
        // `1` (see `choose_action_root_parallel`), so `num_tree_threads` is
        // what decides whether each individual tree is itself
        // tree-parallel. Checking `num_tree_threads` first would skip root
        // parallelism whenever both are set > 1, silently dropping the
        // "trees" half of a requested hybrid split.
        if self.config.num_threads > 1 {
            let selected = self.choose_action_root_parallel(state);
            self.finish_search_report(report_start, self.timer.done(), false, true);
            return selected;
        }
        if self.config.num_tree_threads > 1 {
            return self.choose_action_tree_parallel(state, report_start);
        }

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

        for _ in 0..self.config.max_iterations {
            if self.timer.done() {
                break;
            }
            // MCTS-Solver: the root itself is always the last node
            // `backprop`'s solver pass visits (see `derive_proven` in
            // backprop.rs), so by this point its `Proven` field already
            // reflects everything found so far -- fires the moment *a*
            // forced win is found (the `Win(p)` rule doesn't wait on
            // sibling root children), or once the position is fully solved
            // for the `Win(q)`/`Draw` cases. Single-threaded loop only --
            // the tree-/root-parallel loops need a shared/atomic stop
            // signal instead of this per-thread-local read, deliberately
            // deferred.
            if self.config.use_mcts_solver && self.index.get(root_id).proven() != Proven::Unproven {
                break;
            }
            self.reset_iter();
            let mut ctx = SearchContext::new(root_id, state.clone());

            if let Some(utilities) = self.select(&mut ctx) {
                self.backprop_correction(&utilities);
                continue;
            }

            let k = self.config.num_rollouts_per_leaf;
            let trials = if k > 1 {
                let stack = NodeStack::<G::A>::new(self.stack.clone());
                self.add_extra_virtual_loss(&stack, k - 1);
                self.simulate_many(&ctx.state, k)
            } else {
                vec![self.simulate(&ctx.state)]
            };

            for trial in trials {
                self.trial = Some(trial);
                self.backprop();
            }
        }

        let solved =
            self.config.use_mcts_solver && self.index.get(root_id).proven() != Proven::Unproven;
        let time_expired = self.timer.done();

        // NOTE: this can fail when root is a leaf. This happens if:
        //
        //     max_iterations < expand_threshold
        //
        // TODO: We might check for this and unconditionally expand root. I think
        // a lot of implementations fully expand root on the first iteration.
        let selected = self.select_final_action(state);
        self.compute_pv(state, Some(&selected));
        self.verbose_summary(state, 1);
        self.finish_search_report(report_start, time_expired, solved, false);
        selected
    }

    fn make_book_entry(
        &mut self,
        state: &<Self::G as Game>::S,
    ) -> (Vec<<Self::G as Game>::A>, Vec<f64>) {
        debug_assert_eq!(self.config.expand_threshold, 0);
        debug_assert_eq!(self.config.max_iterations, 1);

        // Run the search, with expand_threshold == 0, so we fully expand to the
        // terminal node.
        _ = self.choose_action(state);
        if self.stack.len() < 2 {
            return (vec![], vec![0.; G::num_players()]);
        }

        // The stack now contains the action path to the terminal state.
        // `stack.pairs()` walks root -> leaf, replaying real states (see
        // `crate::symmetry::incoming_sym`'s doc comment for why the
        // translation can't be cached across paths and must come from the
        // real state in hand).
        let mut actions = vec![];
        let stack = NodeStack::<G::A>::new(self.stack.clone());
        let canonicalizes = self.config.uses_transpositions();
        let mut replay_state = state.clone();
        for ((parent_id, _), (_, idx)) in stack.pairs() {
            let idx = *idx;
            let parent = self.index.get(*parent_id);
            let incoming_sym =
                incoming_sym::<G>(canonicalizes, parent.is_root(), Real(&replay_state));
            let action = real_action::<G>(parent.children(), idx, incoming_sym);
            replay_state = G::apply(replay_state, &action);
            actions.push(action);
        }

        let trial = self.trial.as_ref().unwrap();
        let utilities = trial
            .terminal
            .utilities(G::num_players())
            .or_else(|| trial.cutoff_utilities.clone())
            .unwrap_or_else(|| G::compute_utilities(&trial.state));

        (actions, utilities)
    }

    fn estimated_depth(&self) -> usize {
        (self.stats.accum_depth.load(Relaxed) as f64 / self.stats.iter_count.load(Relaxed) as f64)
            .round() as usize
    }

    fn arena_len(&self) -> usize {
        TreeSearch::arena_len(self)
    }

    fn principle_variation(&self) -> Vec<G::A> {
        self.pv.clone()
    }

    fn root_report(&self, state: &G::S) -> RootReport<G::A> {
        if let Some(root_parallel) = self
            .last_search_run
            .as_ref()
            .and_then(|run| run.root_parallel.as_ref())
        {
            return RootReport {
                actions: root_parallel
                    .actions
                    .iter()
                    .map(|(action, visits, scores)| ActionReport {
                        action: action.clone(),
                        visits: *visits,
                        mean_value: finite(
                            scores[G::player_to_move(state).to_index()] / *visits as f64,
                        )
                        .clamp(-1.0, 1.0),
                        is_proven: false,
                    })
                    .collect(),
                principal_variation: root_parallel.principal_variation.clone(),
                total_visits: root_parallel
                    .actions
                    .iter()
                    .fold(0_u32, |total, (_, visits, _)| total.saturating_add(*visits)),
            };
        }
        let player = G::player_to_move(state).to_index();
        let root = self.index.get(self.root_id);
        let children = root.children();
        let actions = (0..children.len())
            .filter(|&i| children.is_explored(i))
            .map(|i| {
                let child_id = children.node_id(i).unwrap();
                let is_proven = children
                    .node_id(i)
                    .is_some_and(|id| self.index.get(id).proven() != Proven::Unproven);
                let snap = if matches!(self.config.graph_stats(), Some(GraphStats::Nodes)) {
                    self.index.get(child_id).stats.snapshot(player)
                } else {
                    children.snapshot(i, player)
                };
                ActionReport {
                    action: children.action(i).clone(),
                    visits: snap.num_visits,
                    mean_value: snap.expected_score(),
                    is_proven,
                }
            })
            .collect();
        RootReport {
            actions,
            principal_variation: self.pv.clone(),
            total_visits: if self
                .config
                .graph_stats()
                .is_some_and(GraphStats::uses_nodes)
            {
                self.index.get(self.root_id).stats.num_visits()
            } else {
                self.root_stats.num_visits()
            },
        }
    }

    fn search_report(&self, state: &G::S, selected_action: &G::A) -> SearchReport<G::A> {
        let Some(run) = self.last_search_run.as_ref() else {
            return SearchReport::unavailable(SearchReportReason::SearchNotRun);
        };

        let player = G::player_to_move(state).to_index();
        let root_parallel = run.root_parallel.as_ref();
        let mut actions: Vec<_> = if let Some(root_parallel) = root_parallel {
            root_parallel
                .actions
                .iter()
                .enumerate()
                .map(|(i, (action, visits, scores))| {
                    (
                        i,
                        action.clone(),
                        *visits,
                        finite(scores[player] / *visits as f64),
                        false,
                    )
                })
                .collect()
        } else {
            let root = self.index.get(self.root_id);
            let children = root.children();
            (0..children.len())
                .filter(|&i| children.is_explored(i))
                .map(|i| {
                    let child_id = children.node_id(i).unwrap();
                    let is_proven = self.index.get(child_id).proven() != Proven::Unproven;
                    let snap = if matches!(self.config.graph_stats(), Some(GraphStats::Nodes)) {
                        self.index.get(child_id).stats.snapshot(player)
                    } else {
                        children.snapshot(i, player)
                    };
                    (
                        i,
                        children.action(i).clone(),
                        snap.num_visits,
                        finite(snap.expected_score()),
                        is_proven,
                    )
                })
                .collect()
        };
        let all_action_visits: u64 = actions
            .iter()
            .map(|(_, _, visits, _, _)| *visits as u64)
            .sum();
        let selected_row = actions
            .iter()
            .find(|(_, action, _, _, _)| action == selected_action)
            .cloned();
        actions.sort_by_key(|action| (std::cmp::Reverse(action.2), action.0));
        let actions_truncated = actions.len() > ACTION_REPORT_LIMIT;
        actions.truncate(ACTION_REPORT_LIMIT);
        if actions_truncated
            && !actions
                .iter()
                .any(|(_, action, _, _, _)| action == selected_action)
        {
            if let Some(selected) = selected_row {
                let _ = actions.pop();
                actions.push(selected);
                // Restore root-order tie breaking after retaining a selected
                // row that was outside the report cap.
                actions.sort_by_key(|action| (std::cmp::Reverse(action.2), action.0));
            }
        }
        let actions = actions
            .into_iter()
            .map(
                |(_, action, visits, mean_value, is_proven)| SearchActionReport {
                    action,
                    visits,
                    share: finite(if all_action_visits == 0 {
                        0.0
                    } else {
                        visits as f64 / all_action_visits as f64
                    }),
                    mean_value: mean_value.clamp(-1.0, 1.0),
                    is_proven,
                },
            )
            .collect();

        let pv = root_parallel.map_or(&self.pv, |report| &report.principal_variation);
        let pv_truncated = pv.len() > PRINCIPAL_VARIATION_LIMIT;
        let mut principal_variation = pv.clone();
        principal_variation.truncate(PRINCIPAL_VARIATION_LIMIT);
        let completed_iterations = root_parallel.map_or_else(
            || self.stats.iter_count.load(Relaxed),
            |report| report.completed_iterations,
        );
        let mean_depth = (completed_iterations > 0).then(|| {
            finite(
                root_parallel.map_or_else(
                    || self.stats.accum_depth.load(Relaxed),
                    |report| report.accum_depth,
                ) as f64
                    / completed_iterations as f64,
            )
        });
        let tt_hit_ratio =
            (run.tt_reads > 0).then(|| finite(run.tt_hits as f64 / run.tt_reads as f64));
        let iterations_per_second = (run.elapsed_seconds > 0.0)
            .then(|| completed_iterations as f64 / run.elapsed_seconds)
            .filter(|rate| rate.is_finite());
        let graph_mode = match self.config.graph_search {
            GraphSearch::Tree if self.config.use_transpositions => SearchGraphMode::Transpositions,
            GraphSearch::Tree => SearchGraphMode::Tree,
            GraphSearch::Dag(GraphStats::Edges) => SearchGraphMode::DagEdges,
            GraphSearch::Dag(GraphStats::Nodes) => SearchGraphMode::DagNodes,
            GraphSearch::Dag(GraphStats::Both) => SearchGraphMode::DagBoth,
        };
        let mut warnings = vec![SearchWarning::StructuralDiagnosticsOmitted];
        if root_parallel.is_some() {
            warnings.push(SearchWarning::RootParallelPvSingleTree);
        }
        if actions_truncated {
            warnings.push(SearchWarning::ActionsTruncated);
        }
        if pv_truncated {
            warnings.push(SearchWarning::PrincipalVariationTruncated);
        }
        SearchReport {
            schema_version: 1,
            status: if root_parallel.is_some() {
                SearchReportStatus::Partial
            } else {
                SearchReportStatus::Available
            },
            reason: run
                .root_parallel
                .is_some()
                .then_some(SearchReportReason::RootParallelPvSingleTree),
            elapsed_seconds: Some(run.elapsed_seconds),
            iteration_limit: (self.config.max_iterations != usize::MAX)
                .then_some(self.config.max_iterations),
            time_limit_seconds: (self.config.max_time != std::time::Duration::default())
                .then(|| finite(self.config.max_time.as_secs_f64())),
            completed_iterations,
            termination: Some(run.termination),
            selected_action: Some(selected_action.clone()),
            actions,
            principal_variation,
            root_visits: root_parallel.map_or_else(
                || {
                    if self
                        .config
                        .graph_stats()
                        .is_some_and(GraphStats::uses_nodes)
                    {
                        self.index.get(self.root_id).stats.num_visits()
                    } else {
                        self.root_stats.num_visits()
                    }
                },
                |report| {
                    report
                        .actions
                        .iter()
                        .fold(0_u32, |total, (_, visits, _)| total.saturating_add(*visits))
                },
            ),
            tree_nodes: root_parallel.map_or_else(|| self.index.len(), |report| report.tree_nodes),
            mean_depth,
            max_depth: (completed_iterations > 0).then(|| {
                root_parallel.map_or_else(
                    || self.stats.max_depth.load(Relaxed),
                    |report| report.max_depth,
                )
            }),
            graph_mode: Some(graph_mode),
            tt_reads: run.tt_reads,
            tt_writes: run.tt_writes,
            tt_hits: run.tt_hits,
            tt_hit_ratio,
            iterations_per_second,
            warnings,
        }
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.config.name = name.to_string();
    }
}
