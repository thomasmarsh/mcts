use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::Real;
use crate::game::Transform;
use crate::strategies::mcts::config::{GraphSearch, GraphStats, TranspositionKeying};
use crate::strategies::mcts::index::Id;
use crate::strategies::mcts::node::{real_action, Node, NodeState, NodeStats};
use crate::strategies::mcts::search::shared::Shared;
use crate::strategies::mcts::search::shared::{
    add_path_virtual_loss, backprop_correction_step, backprop_step, expand, last_tree_action,
    proven_draw_child, proven_win_child, select_step, simulate_step,
};
use crate::strategies::mcts::search::SearchContext;
use crate::strategies::mcts::search::TreeSearch;
use crate::strategies::mcts::select::SelectContext;
use crate::strategies::mcts::select::SelectStrategy;
use crate::strategies::mcts::simulate::{SimulateStrategy, Trial};
use crate::strategies::mcts::stack::NodeStack;
use crate::symmetry::incoming_sym;
use crate::util::pv_string;

use rand::rngs::SmallRng;
use rand::Rng;
use rand_core::SeedableRng;
use std::sync::atomic::Ordering::Relaxed;

/// Byte totals are rough estimates (`std::mem::size_of` on fixed-size parts,
/// element-count * element-size for heap-allocated `Vec`s/maps, ignoring
/// allocator and hashmap bucket overhead) -- good enough to rank where
/// memory pressure actually comes from, not a precise accounting. See
/// `TreeSearch::memory_stats`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub terminal_nodes: usize,
    pub expanded_nodes: usize,
    /// Sum, over every expanded node, of its child array's length -- one
    /// slot per legal action, whether or not it was ever explored.
    pub total_child_slots: usize,
    /// Sum, over every expanded node, of its child array's explored slots
    /// (`ChildArray::explored_len`).
    pub explored_child_slots: usize,
    /// `total_nodes * size_of::<Node<A>>()` -- every arena entry's
    /// fixed-size footprint (this already includes its `ChildArray`'s own
    /// fixed-size fields, inlined via `OnceLock<NodeState<A>>`, regardless
    /// of whether that node ended up Leaf/Terminal/Expanded).
    pub node_bytes: usize,
    /// Estimated heap bytes owned by expanded nodes' `ChildArray`s (their
    /// parallel `Vec`s/`FxHashMap`), summed via `ChildArray::heap_bytes_estimate`.
    pub child_array_heap_bytes: usize,
    /// Heap bytes owned by every node's solver side block (`Box<SolverState>`
    /// when `SearchConfig::use_mcts_solver` is on, 0 otherwise), summed via
    /// `Node::solver_heap_bytes`. Unlike `node_bytes`, this reflects the
    /// runtime `use_mcts_solver` switch, not just `Node<A>`'s fixed type size.
    pub solver_bytes: usize,
    /// Total entry count in the transposition table.
    pub table_entries: usize,
    /// Entries keyed by root-relative `(position_hash, ply)` in explicit DAG
    /// search, as opposed to the legacy hash-only table.
    pub graph_table_entries: usize,
    /// Approximate key/value payload bytes for both transposition tables.
    pub table_bytes: usize,
}

/// Diagnostics-only snapshot of DAG/transposition-merge structure, meaningful
/// whenever `GraphSearch::Dag` or the legacy `use_transpositions(true)` is
/// active (all fields are legitimately zero for an ordinary tree). Walks
/// every arena entry, same caveat as `MemoryStats`. See `TreeSearch::
/// graph_diagnostics`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphDiagnostics {
    /// The keying mode this search is actually running under -- see
    /// `TranspositionKeying`'s doc comment. Every other field's meaning
    /// depends on it: under `StateOnly`, `transposition_nodes`/
    /// `max_incoming_edges` can reflect real cross-ply cycles, which are
    /// structurally impossible under `PerPly`/legacy keying.
    pub keying: TranspositionKeying,
    /// Cumulative transposition-table lookups (across both the legacy and
    /// ply-keyed graph tables) that found an already-resolved node rather
    /// than creating a new one -- `TranspositionTable::hits`.
    pub table_hits: usize,
    /// Distinct positions currently mapped in the transposition table(s) --
    /// `TranspositionTable::len`, i.e. every node reachable through a table
    /// lookup rather than only through a `ChildArray` slot.
    pub unique_graph_nodes: usize,
    /// Sum, over every expanded node, of its child array's resolved slots
    /// (`ChildArray::explored_len`) -- every edge that has actually been
    /// connected to a child node, whether newly created or shared.
    pub resolved_edges: usize,
    /// Nodes with more than one incoming edge (`Node::is_transposition`) --
    /// the nodes actually being shared by more than one parent, as opposed
    /// to every node merely reachable via the table.
    pub transposition_nodes: usize,
    /// The largest `Node::incoming_edges` seen on any node -- how heavily
    /// the single most-shared position is being reused.
    pub max_incoming_edges: u32,
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    #[inline]
    pub fn select(&mut self, ctx: &mut SearchContext<G>) -> Option<Vec<f64>> {
        debug_assert!(self.stack.is_empty());
        // `ctx` always starts a call at the root (`current_id`/`state` are
        // the root's own), so this is always correct and keeps
        // `self.root_state` populated for callers that drive `select`/
        // `simulate`/`backprop` directly rather than through `choose_action`
        // (which already sets it via `reuse_or_reset`, making this a no-op
        // refresh in that path).
        self.root_state = Some(ctx.state.clone());
        select_step(
            &Shared {
                index: &self.index,
                root_state: self.root_state.as_ref().unwrap(),
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
            },
            ctx,
            &mut self.stack,
            &mut self.config.select,
            &mut self.config.rng,
            self.config.prior.as_deref_mut(),
        )
    }

    #[inline]
    pub fn select_final_action(&mut self, state: &G::S) -> G::A {
        // A budget can end immediately after a root playout, before the
        // visit threshold permits normal selection to expand that root.  The
        // final-action policy still needs its legal child list even though no
        // child has search evidence yet.
        if self.index.get(self.root_id).is_leaf() {
            let _ = expand::<G>(
                &self.index,
                self.root_id,
                state,
                self.config.use_mcts_solver,
                self.config.requirements().amaf,
                self.config.uses_transpositions(),
                self.config.prior.as_deref_mut(),
            );
        }
        let player = G::player_to_move(state).to_index();
        if let Some(idx) = proven_win_child::<G>(
            self.config.use_mcts_solver,
            self.index.get(self.root_id),
            &self.index,
            player,
        ) {
            return self.index.get(self.root_id).children().action(idx).clone();
        }

        // Contempt factor (Kowalski et al. 2023, Section VII.C): no forced
        // win exists, and the root's own running average for `player` reads
        // worse than the configured threshold -- take a known draw over
        // gambling on whatever `final_action` would otherwise pick.
        if let Some(cf) = self.config.contempt_factor {
            let root_score = if self
                .config
                .graph_stats()
                .is_some_and(GraphStats::uses_nodes)
            {
                self.index.get(self.root_id).stats.expected_score(player)
            } else {
                self.root_stats.expected_score(player)
            };
            if root_score < cf {
                if let Some(idx) = proven_draw_child::<G>(
                    self.config.use_mcts_solver,
                    self.index.get(self.root_id),
                    &self.index,
                ) {
                    return self.index.get(self.root_id).children().action(idx).clone();
                }
            }
        }

        let stack = crate::strategies::mcts::stack::NodeStack::new(vec![(self.root_id, 0)]);
        let grave = self.stats.grave.read().unwrap();
        let idx = self.config.final_action.best_child(
            &SelectContext {
                q_init: self.config.q_init,
                stack: &stack,
                root_stats: &self.root_stats,
                root_state: state,
                canonicalizes: self.config.uses_transpositions(),
                player,
                state,
                index: &self.index,
                table: &self.table,
                grave: &grave,
                global: &self.stats,
                use_transpositions: self.config.uses_transpositions(),
                graph_stats: self.config.graph_stats(),
                solver_loss_threshold: self.config.solver_loss_threshold,
                incoming_sym: Transform::IDENTITY,
            },
            &mut self.config.rng,
        );

        self.index.get(self.root_id).children().action(idx).clone()
    }

    #[inline]
    pub fn simulate(&mut self, state: &G::S) -> Trial<G> {
        let prev_action = last_tree_action::<G>(
            &self.index,
            &self.stack,
            self.root_state.as_ref().unwrap(),
            self.config.uses_transpositions(),
        );
        simulate_step(
            self.config.max_playout_depth,
            &self.stats,
            &mut self.config.simulate,
            state,
            prev_action,
            &mut self.config.rng,
        )
    }

    pub fn simulate_many(&mut self, state: &G::S, k: usize) -> Vec<Trial<G>> {
        if k <= 1 {
            return vec![self.simulate(state)];
        }

        let seeds: Vec<u64> = (0..k).map(|_| self.config.rng.gen()).collect();
        let mut strategies: Vec<S::Simulate> =
            (0..k).map(|_| self.config.simulate.clone()).collect();
        let max_playout_depth = self.config.max_playout_depth;
        let stats = &self.stats;
        let prev_action = last_tree_action::<G>(
            &self.index,
            &self.stack,
            self.root_state.as_ref().unwrap(),
            self.config.uses_transpositions(),
        );

        std::thread::scope(|scope| {
            let handles: Vec<_> = strategies
                .iter_mut()
                .zip(seeds)
                .map(|(strategy, seed)| {
                    let state = state.clone();
                    let prev_action = prev_action.clone();
                    scope.spawn(move || {
                        let mut rng = SmallRng::seed_from_u64(seed);
                        simulate_step(
                            max_playout_depth,
                            stats,
                            strategy,
                            &state,
                            prev_action,
                            &mut rng,
                        )
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    pub(crate) fn add_extra_virtual_loss(&self, stack: &NodeStack<G::A>, extra: usize) {
        add_path_virtual_loss(&self.index, stack, extra, self.config.graph_stats());
    }

    #[inline]
    pub fn backprop(&mut self) {
        let trial = self.trial.as_ref().unwrap().clone();
        let flags = self.config.select.backprop_flags() | self.config.simulate.backprop_flags();
        backprop_step(
            &Shared {
                index: &self.index,
                root_state: self.root_state.as_ref().unwrap(),
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
            },
            &self.stack,
            &self.config.backprop,
            trial,
            flags,
        );
    }

    /// See `shared::backprop_correction_step`'s doc comment -- called instead
    /// of `simulate`/`backprop` whenever `select` returns `Some(utilities)`
    /// (an `McgsCorrection::Residual` check fired partway through descent).
    #[inline]
    pub fn backprop_correction(&mut self, utilities: &[f64]) {
        backprop_correction_step(
            &Shared {
                index: &self.index,
                root_state: self.root_state.as_ref().unwrap(),
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
            },
            &self.stack,
            utilities,
        );
    }

    pub fn arena_len(&self) -> usize {
        self.index.len()
    }

    /// Diagnostics-only snapshot of the arena's and transposition table's
    /// approximate memory footprint, broken down by category. Walks every
    /// arena entry, so this is a profiling tool, not something to call on
    /// any hot path.
    pub fn memory_stats(&self) -> MemoryStats {
        let mut stats = MemoryStats::default();
        self.index.for_each(|node: &Node<G::A>| {
            stats.total_nodes += 1;
            stats.node_bytes += std::mem::size_of::<Node<G::A>>();
            stats.solver_bytes += node.solver_heap_bytes();
            match node.status() {
                None => stats.leaf_nodes += 1,
                Some(NodeState::Terminal) => stats.terminal_nodes += 1,
                Some(NodeState::Expanded(children)) => {
                    stats.expanded_nodes += 1;
                    stats.total_child_slots += children.len();
                    stats.explored_child_slots += children.explored_len();
                    stats.child_array_heap_bytes += children.heap_bytes_estimate();
                }
            }
        });
        stats.table_entries = self.table.len();
        stats.graph_table_entries = self.table.graph_len();
        let legacy_entries = self.table.legacy_len();
        stats.table_bytes = legacy_entries
            * (std::mem::size_of::<u64>() + std::mem::size_of::<Id>())
            + stats.graph_table_entries
                * (std::mem::size_of::<crate::strategies::mcts::table::TranspositionKey>()
                    + std::mem::size_of::<Id>());
        stats
    }

    /// Diagnostics-only snapshot of DAG/transposition-merge structure -- see
    /// `GraphDiagnostics`. Cheap to call even for an ordinary tree (every
    /// field reads zero), but still walks the whole arena, so treat it the
    /// same as `memory_stats`: a profiling tool, not a hot-path call.
    pub fn graph_diagnostics(&self) -> GraphDiagnostics {
        let mut diag = GraphDiagnostics {
            keying: self.config.transposition_keying,
            table_hits: self.table.hits.load(Relaxed),
            unique_graph_nodes: self.table.len(),
            ..Default::default()
        };
        self.index.for_each(|node: &Node<G::A>| {
            let incoming = node.incoming_edges();
            if incoming > 1 {
                diag.transposition_nodes += 1;
            }
            diag.max_incoming_edges = diag.max_incoming_edges.max(incoming);
            if let Some(NodeState::Expanded(children)) = node.status() {
                diag.resolved_edges += children.explored_len();
            }
        });
        diag
    }

    pub fn verbose_summary(&self, state: &G::S, num_threads: usize) {
        if !self.config.verbose {
            return;
        }

        let root = self.index.get(self.root_id);
        let total_visits = if self
            .config
            .graph_stats()
            .is_some_and(GraphStats::uses_nodes)
        {
            root.stats.num_visits()
        } else {
            self.root_stats.num_visits()
        };
        let rate = total_visits as f64 / num_threads as f64 / self.timer.elapsed().as_secs_f64();
        eprintln!(
            "Using {} threads, did {} total simulations with {:.1} rollouts/sec/core",
            num_threads, total_visits, rate
        );

        let player = G::player_to_move(state);

        let children = root.children();
        let mut summaries = (0..children.len())
            .filter(|&i| children.is_explored(i))
            .map(|i| {
                let child_id = children.node_id(i).unwrap();
                if matches!(self.config.graph_stats(), Some(GraphStats::Nodes)) {
                    let child = self.index.get(child_id);
                    (
                        child.stats.num_visits(),
                        child.stats.score(player.to_index()),
                        children.action(i).clone(),
                    )
                } else {
                    (
                        children.num_visits(i),
                        children.score(i, player.to_index()),
                        children.action(i).clone(),
                    )
                }
            })
            .collect::<Vec<_>>();

        summaries.sort_by_key(|t| !t.0);

        for (visits, score, m) in summaries.into_iter().take(10) {
            let win_rate = (score + visits as f64) / (visits as f64 * 2.0);
            eprintln!(
                "{:>6} visits, {:.02}% wins: {}",
                visits,
                win_rate * 100.0,
                G::notation(state, &m),
            );
        }

        eprintln!("PV: {}", pv_string::<G>(self.pv.as_slice(), state));

        if self.config.uses_transpositions() {
            let diag = self.graph_diagnostics();
            eprintln!(
                "Graph ({:?}): {} table hits, {} unique nodes, {} resolved edges, \
                 {} transposition nodes, {} max incoming edges",
                diag.keying,
                diag.table_hits,
                diag.unique_graph_nodes,
                diag.resolved_edges,
                diag.transposition_nodes,
                diag.max_incoming_edges,
            );
        }
    }

    #[inline]
    pub fn reset_iter(&mut self) {
        self.stack.clear();
        self.trial = None;
    }

    #[inline]
    pub fn reset(&mut self, player_idx: usize, hash: u64) -> Id {
        self.index.clear();
        self.table.clear();
        self.stats.accum_depth.store(0, Relaxed);
        self.stats.max_depth.store(0, Relaxed);
        self.stats.iter_count.store(0, Relaxed);
        self.root_stats = NodeStats::new(G::num_players(), self.config.requirements().amaf);
        self.new_root(player_idx, hash)
    }

    pub(crate) fn compute_pv(&mut self, init_state: &G::S, selected_root_action: Option<&G::A>) {
        self.pv.clear();
        let mut node_id = self.root_id;
        let mut node = self.index.get(node_id);
        let mut state = init_state.clone();
        let mut stack = NodeStack::new(vec![(node_id, 0)]);
        let grave = self.stats.grave.read().unwrap();
        let canonicalizes = self.config.uses_transpositions();
        // Same cycle hazard `select_step`'s descent guard exists for (see
        // its comment in `search/shared.rs`): under
        // `TranspositionKeying::StateOnly` this replay can walk a real graph
        // cycle, and "still expanded" never stops being true on its own.
        let explicit_dag = matches!(self.config.graph_search, GraphSearch::Dag(_));
        let bound_descent =
            explicit_dag && self.config.transposition_keying == TranspositionKeying::StateOnly;
        while node.is_expanded() && !(bound_descent && stack.len() > self.config.max_playout_depth)
        {
            let player = node.player_idx;
            // Recomputed fresh from `state` every iteration, like
            // `select_step`'s local of the same name -- see `crate::
            // symmetry::incoming_sym`'s doc comment.
            let incoming_sym = incoming_sym::<G>(canonicalizes, node.is_root(), Real(&state));
            let select_ctx = SelectContext {
                q_init: self.config.q_init,
                player,
                stack: &stack,
                root_stats: &self.root_stats,
                root_state: init_state,
                canonicalizes,
                state: &state,
                index: &self.index,
                table: &self.table,
                grave: &grave,
                global: &self.stats,
                use_transpositions: self.config.uses_transpositions(),
                graph_stats: self.config.graph_stats(),
                solver_loss_threshold: self.config.solver_loss_threshold,
                incoming_sym,
            };

            let best_idx = if self.pv.is_empty() {
                selected_root_action
                    .and_then(|selected| {
                        (0..node.children().len())
                            .find(|&idx| node.children().action(idx) == selected)
                    })
                    .unwrap_or_else(|| {
                        proven_win_child::<G>(
                            self.config.use_mcts_solver,
                            node,
                            &self.index,
                            player,
                        )
                        .unwrap_or_else(|| {
                            self.config
                                .final_action
                                .best_child(&select_ctx, &mut self.config.rng)
                        })
                    })
            } else {
                proven_win_child::<G>(self.config.use_mcts_solver, node, &self.index, player)
                    .unwrap_or_else(|| {
                        self.config
                            .final_action
                            .best_child(&select_ctx, &mut self.config.rng)
                    })
            };

            let children = node.children();
            let Some(cached_child_id) = children.node_id(best_idx) else {
                break;
            };
            let action = real_action::<G>(children, best_idx, incoming_sym);
            state = G::apply(state, &action);
            // See `verified_child_id`'s doc comment: `cached_child_id` was
            // resolved by whichever real orientation reached this slot
            // first during search, which need not be the orientation this
            // PV replay is currently holding.
            node_id = crate::strategies::mcts::search::shared::verified_child_id::<G>(
                &crate::strategies::mcts::search::shared::TranspositionCtx {
                    index: &self.index,
                    table: &self.table,
                    explicit_dag: matches!(self.config.graph_search, GraphSearch::Dag(_)),
                    keying: self.config.transposition_keying,
                    use_transpositions: self.config.uses_transpositions(),
                    has_amaf: self.config.requirements().amaf,
                    use_mcts_solver: self.config.use_mcts_solver,
                },
                cached_child_id,
                &state,
                node.ply + 1,
            );
            node = self.index.get(node_id);
            self.pv.push(action);
            stack.push(node_id, best_idx);
        }
    }
}
