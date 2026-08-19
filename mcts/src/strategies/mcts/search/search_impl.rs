use crate::game::Game;
use crate::game::PlayerIndex;
use crate::strategies::mcts::config::GraphSearch;
use crate::strategies::mcts::config::GraphStats;
use crate::strategies::mcts::node;
use crate::strategies::mcts::node::real_action;
use crate::strategies::mcts::node::Proven;
use crate::strategies::mcts::search::shared::SearchContext;
use crate::strategies::mcts::search::TreeSearch;
use crate::strategies::mcts::stack::NodeStack;
use crate::strategies::mcts::table::TranspositionKey;
use crate::strategies::ActionReport;
use crate::strategies::RootReport;
use crate::strategies::Search;

use std::sync::atomic::Ordering::Relaxed;

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
        if matches!(self.config.graph_search, GraphSearch::Dag(_)) {
            assert!(
                !self.config.use_transpositions,
                "graph_search replaces the legacy use_transpositions setting"
            );
            assert!(
                !self.config.reuse_tree,
                "explicit graph search does not yet support tree reuse"
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
            return self.choose_action_root_parallel(state);
        }
        if self.config.num_tree_threads > 1 {
            return self.choose_action_tree_parallel(state);
        }

        let hash = G::zobrist_hash(state);
        let root_id = self.reuse_or_reset(G::player_to_move(state).to_index(), state);
        if matches!(self.config.graph_search, GraphSearch::Dag(_)) {
            assert!(
                !self.config.reuse_tree,
                "explicit graph search does not yet support tree reuse"
            );
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

            self.select(&mut ctx);

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

        self.compute_pv(state);
        self.verbose_summary(state, 1);

        // NOTE: this can fail when root is a leaf. This happens if:
        //
        //     max_iterations < expand_threshold
        //
        // TODO: We might check for this and unconditionally expand root. I think
        // a lot of implementations fully expand root on the first iteration.
        self.select_final_action(state)
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
        // `node::incoming_sym`'s doc comment for why the translation can't
        // be cached across paths and must come from the real state in hand).
        let mut actions = vec![];
        let stack = NodeStack::<G::A>::new(self.stack.clone());
        let explicit_dag = matches!(self.config.graph_search, GraphSearch::Dag(_));
        let mut replay_state = state.clone();
        for ((parent_id, _), (_, idx)) in stack.pairs() {
            let idx = *idx;
            let parent = self.index.get(*parent_id);
            let incoming_sym =
                node::incoming_sym::<G>(explicit_dag, parent.is_root(), &replay_state);
            let action = real_action::<G>(parent.children(), idx, incoming_sym);
            replay_state = G::apply(replay_state, &action);
            actions.push(action);
        }

        let trial = self.trial.as_ref().unwrap();
        let utilities = trial
            .terminal
            .utilities(G::num_players())
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

    // Reads `self.index`/`self.root_stats` directly, which is exactly right
    // for the single-threaded and tree-parallel paths (`choose_action_tree_parallel`
    // in parallel.rs): both leave the real, complete root stats in `self`
    // when `choose_action` returns, since there's one shared tree either way.
    // Root parallelism (`choose_action_root_parallel`) is the one path this
    // under-reports for: it merges each worker's totals into a local `merged`
    // map to pick the final action but never writes that merge back into
    // `self`, so `root_report` after a root-parallel call would only reflect
    // this thread's own final worker tree, not the true cross-worker totals.
    // Not fixed here because no current preset (`server/main.rs`'s
    // `build_ai`) sets `num_threads > 1` -- Strong/Master use tree
    // parallelism (`num_tree_threads`) instead: it strictly dominates root
    // parallelism at every tested board size. If a preset ever does turn
    // root parallelism back
    // on, `choose_action_root_parallel` would need to cache its merged
    // `ActionTotal`s somewhere `root_report` can read, mirroring this
    // method's shape.
    fn root_report(&self, state: &G::S) -> RootReport<G::A> {
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

    fn set_friendly_name(&mut self, name: &str) {
        self.config.name = name.to_string();
    }
}
