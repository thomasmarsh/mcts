use crate::strategies::mcts::index::Id;
use crate::strategies::mcts::node::NodeStats;
use crate::strategies::mcts::search::shared::Shared;
use crate::strategies::mcts::search::shared::{add_path_virtual_loss, backprop_step, last_tree_action, proven_win_child, select_step, simulate_step};
use crate::strategies::mcts::search::SearchContext;
use crate::strategies::mcts::search::TreeSearch;
use crate::strategies::mcts::select::SelectContext;
use crate::strategies::mcts::select::SelectStrategy;
use crate::strategies::mcts::simulate::{SimulateStrategy, Trial};
use crate::strategies::mcts::stack::NodeStack;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::util::pv_string;

use rand::rngs::SmallRng;
use rand::Rng;
use rand_core::SeedableRng;
use std::sync::atomic::Ordering::Relaxed;

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: crate::strategies::mcts::Strategy<G>,
    crate::strategies::mcts::SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    #[inline]
    pub fn select(&mut self, ctx: &mut SearchContext<G>) {
        debug_assert!(self.stack.is_empty());
        select_step(
            &Shared {
                index: &self.index,
                root_stats: &self.root_stats,
                table: &self.table,
                global: &self.stats,
                expand_threshold: self.config.expand_threshold,
                q_init: self.config.q_init,
                use_transpositions: self.config.use_transpositions,
                use_mcts_solver: self.config.use_mcts_solver,
                max_playout_depth: self.config.max_playout_depth,
            },
            ctx,
            &mut self.stack,
            &mut self.config.select,
            &mut self.config.rng,
        );
    }

    #[inline]
    pub(crate) fn select_final_action(&mut self, state: &G::S) -> G::A {
        let player = G::player_to_move(state).to_index();
        if let Some(idx) =
            proven_win_child::<G>(self.config.use_mcts_solver, self.index.get(self.root_id), &self.index, player)
        {
            return self.index.get(self.root_id).children().action(idx).clone();
        }

        let stack = crate::strategies::mcts::stack::NodeStack::new(vec![self.root_id]);
        let grave = self.stats.grave.read().unwrap();
        let idx = self.config.final_action.best_child(
            &SelectContext {
                q_init: self.config.q_init,
                stack: &stack,
                root_stats: &self.root_stats,
                player,
                state,
                index: &self.index,
                table: &self.table,
                grave: &grave,
                use_transpositions: self.config.use_transpositions,
            },
            &mut self.config.rng,
        );

        self.index.get(self.root_id).children().action(idx).clone()
    }

    #[inline]
    pub(crate) fn simulate(&mut self, state: &G::S) -> Trial<G> {
        let prev_action = last_tree_action::<G>(&self.index, &self.stack);
        simulate_step(
            self.config.max_playout_depth,
            &self.stats,
            &mut self.config.simulate,
            state,
            prev_action,
            &mut self.config.rng,
        )
    }

    pub(crate) fn simulate_many(&mut self, state: &G::S, k: usize) -> Vec<Trial<G>> {
        if k <= 1 {
            return vec![self.simulate(state)];
        }

        let seeds: Vec<u64> = (0..k).map(|_| self.config.rng.gen()).collect();
        let mut strategies: Vec<S::Simulate> =
            (0..k).map(|_| self.config.simulate.clone()).collect();
        let max_playout_depth = self.config.max_playout_depth;
        let stats = &self.stats;
        let prev_action = last_tree_action::<G>(&self.index, &self.stack);

        std::thread::scope(|scope| {
            let handles: Vec<_> = strategies
                .iter_mut()
                .zip(seeds)
                .map(|(strategy, seed)| {
                    let state = state.clone();
                    let prev_action = prev_action.clone();
                    scope.spawn(move || {
                        let mut rng = SmallRng::seed_from_u64(seed);
                        simulate_step(max_playout_depth, stats, strategy, &state, prev_action, &mut rng)
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    pub(crate) fn add_extra_virtual_loss(&self, stack: &NodeStack<G::A>, extra: usize) {
        add_path_virtual_loss(&self.index, stack, extra);
    }

    #[inline]
    pub(crate) fn backprop(&mut self) {
        let trial = self.trial.as_ref().unwrap().clone();
        let flags = self.config.select.backprop_flags() | self.config.simulate.backprop_flags();
        backprop_step(
            &Shared {
                index: &self.index,
                root_stats: &self.root_stats,
                table: &self.table,
                global: &self.stats,
                expand_threshold: self.config.expand_threshold,
                q_init: self.config.q_init,
                use_transpositions: self.config.use_transpositions,
                use_mcts_solver: self.config.use_mcts_solver,
                max_playout_depth: self.config.max_playout_depth,
            },
            &self.stack,
            &self.config.backprop,
            trial,
            flags,
        );
    }

    pub fn arena_len(&self) -> usize {
        self.index.len()
    }

    pub fn verbose_summary(&self, state: &G::S, num_threads: usize) {
        if !self.config.verbose {
            return;
        }

        let root = self.index.get(self.root_id);
        let total_visits = self.root_stats.num_visits();
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
                (
                    children.num_visits(i),
                    children.score(i, player.to_index()),
                    children.action(i).clone(),
                )
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

        eprintln!("PV: {}", pv_string::<G>(self.pv.as_slice(), state))
    }

    #[inline]
    pub(crate) fn reset_iter(&mut self) {
        self.stack.clear();
        self.trial = None;
    }

    #[inline]
    pub(crate) fn reset(&mut self, player_idx: usize, hash: u64) -> Id {
        self.index.clear();
        self.table.clear();
        self.stats.accum_depth.store(0, Relaxed);
        self.stats.iter_count.store(0, Relaxed);
        self.root_stats = NodeStats::new(G::num_players());
        self.new_root(player_idx, hash)
    }

    pub(crate) fn compute_pv(&mut self, init_state: &G::S) {
        self.pv.clear();
        let mut node_id = self.root_id;
        let mut node = self.index.get(node_id);
        let mut state = init_state.clone();
        let mut stack = NodeStack::new(vec![node_id]);
        let grave = self.stats.grave.read().unwrap();
        while node.is_expanded() {
            let player = node.player_idx;
            let select_ctx = SelectContext {
                q_init: self.config.q_init,
                player,
                stack: &stack,
                root_stats: &self.root_stats,
                state: &state,
                index: &self.index,
                table: &self.table,
                grave: &grave,
                use_transpositions: self.config.use_transpositions,
            };

            let best_idx = match proven_win_child::<G>(self.config.use_mcts_solver, node, &self.index, player) {
                Some(idx) => idx,
                None => self
                    .config
                    .final_action
                    .best_child(&select_ctx, &mut self.config.rng),
            };

            let children = node.children();
            let Some(child_id) = children.node_id(best_idx) else {
                break;
            };
            let action = children.action(best_idx).clone();
            node_id = child_id;
            node = self.index.get(node_id);
            state = G::apply(state, &action);
            self.pv.push(action);
            stack.push(node_id);
        }
    }
}