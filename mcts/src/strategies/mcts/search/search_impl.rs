use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::Real;
use crate::strategies::mcts::config::GraphSearch;
use crate::strategies::mcts::config::GraphStats;
use crate::strategies::mcts::config::IsmctsMode;
use crate::strategies::mcts::node::real_action;
use crate::strategies::mcts::node::Proven;
use crate::strategies::mcts::search::shared::expand;
use crate::strategies::mcts::search::shared::SearchContext;
use crate::strategies::mcts::search::TreeSearch;
use crate::strategies::mcts::stack::NodeStack;
use crate::strategies::mcts::table::TranspositionKey;
use crate::strategies::{
    ActionReport, RootReport, Search, SearchReport, SearchReportReason, SearchTermination,
};
use crate::symmetry::incoming_sym;

use std::sync::atomic::Ordering::Relaxed;

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
        if self.config.ismcts_mode == IsmctsMode::MultiTree {
            let selected = self.choose_action_multi_tree(state);
            let time_expired = self.timer.done();
            self.compute_pv(state, Some(&selected));
            self.verbose_summary(state, 1);
            self.finish_search_report(report_start, time_expired, false, false);
            return selected;
        }

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
        if self.config.ismcts_mode == IsmctsMode::SingleTree {
            // The root's own position is never hidden from its mover --
            // unlike every other node, whose first expansion legitimately
            // reads whichever iteration happens to reach it first, the
            // root's legal-action list and terminal status must be resolved
            // against the literal caller-supplied `state`, not a
            // per-iteration `G::determinize`d guess. Without this, a game
            // whose terminal check can read hidden information (e.g.
            // Phantom's win check against a guessed board) could have its
            // very first iteration's guess permanently -- `expand`'s
            // `OnceLock` only ever resolves once -- and wrongly mark a real,
            // ongoing root position `Terminal`, which `select_final_action`
            // has no fallback for.
            let _ = expand::<G>(
                &self.index,
                root_id,
                state,
                false,
                self.config.requirements().amaf,
                false,
                true,
                None,
            );
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
            // SO-ISMCTS (`SearchConfig::ismcts_mode == IsmctsMode::SingleTree`):
            // every iteration descends its own fresh `G::determinize`d
            // sample of the root state, rather than the one literal `state`
            // every ordinary search (and PIMC's `determinize_root`, which
            // determinizes once per *worker tree* rather than once per
            // *iteration*) uses for every iteration of a given
            // `choose_action` call. When `ismcts_redeterminize` is also set,
            // `select_step` redraws this sample itself at every node it
            // visits during descent (including the root, on its first visit
            // this iteration), so the literal state is handed to it
            // unchanged here rather than determinizing the root twice.
            let iter_state = if self.config.ismcts_mode == IsmctsMode::SingleTree
                && !self.config.ismcts_redeterminize
            {
                G::determinize(state.clone(), &mut self.config.rng)
            } else {
                state.clone()
            };
            let mut ctx = SearchContext::new(root_id, iter_state);

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
        let canonicalizes = self.config.canonicalizes();
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
        self.last_search_run.as_ref().map_or(
            SearchReport::unavailable(SearchReportReason::SearchNotRun),
            |run| {
                super::report::build_search_report(
                    run,
                    &self.index,
                    self.root_id,
                    &self.pv,
                    &self.root_stats,
                    &self.stats,
                    self.config.graph_search,
                    self.config.use_transpositions,
                    self.config.max_iterations,
                    self.config.max_time,
                    state,
                    selected_action,
                )
            },
        )
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.config.name = name.to_string();
    }
}
