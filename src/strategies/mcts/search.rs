use super::backprop::BackpropStrategy;
use super::config::SearchConfig;
use super::config::Strategy;
use super::index;
use super::index::Id;
use super::node;
use super::node::Node;
use super::node::NodeState;
use super::node::NodeStats;
use super::select::SelectContext;
use super::select::SelectStrategy;
use super::simulate::SimulateStrategy;
use super::simulate::Trial;
use super::stack::NodeStack;
use super::table::TranspositionTable;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::strategies::mcts::node::Edge;
use crate::strategies::Search;
use crate::timer;
use crate::util::pv_string;

use rustc_hash::FxHashMap;

pub struct SearchContext<G: Game> {
    pub current_id: Id,
    pub state: G::S,
}

impl<G: Game> SearchContext<G> {
    pub fn new(current_id: Id, state: G::S) -> Self {
        Self { current_id, state }
    }

    #[inline]
    fn traverse_apply(&mut self, child_id: Id, action: &G::A) {
        self.traverse(child_id);
        self.state = G::apply(self.state.clone(), action);
    }

    #[inline]
    fn traverse(&mut self, child_id: Id) {
        self.current_id = child_id;
    }
}

#[derive(Clone, Debug)]
pub struct TreeStats<G: Game> {
    pub actions: FxHashMap<G::A, node::ActionStats>,
    pub grave: FxHashMap<u64, Vec<FxHashMap<G::A, node::ActionStats>>>,
    pub player_actions: Vec<FxHashMap<G::A, node::ActionStats>>,
    pub accum_depth: usize,
    pub iter_count: usize,
}

impl<G: Game> Default for TreeStats<G> {
    fn default() -> Self {
        Self {
            actions: FxHashMap::default(),
            grave: FxHashMap::default(),
            player_actions: vec![Default::default(); G::num_players()],
            accum_depth: 0,
            iter_count: 0,
        }
    }
}

pub type TreeIndex<A> = index::Arena<Node<A>>;

#[derive(Clone)]
pub struct TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    pub(crate) index: TreeIndex<G::A>,
    pub(crate) timer: timer::Timer,
    pub(crate) root_id: Id,
    pub(crate) root_stats: NodeStats,
    pub(crate) pv: Vec<G::A>,
    pub(crate) table: TranspositionTable<G::S>,

    pub config: SearchConfig<G, S>,
    pub stats: TreeStats<G>,
    pub stack: Vec<Id>,
    pub trial: Option<Trial<G>>,
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    G::S: std::fmt::Display,
{
    pub fn config(mut self, config: SearchConfig<G, S>) -> Self {
        self.config = config;
        self
    }
}

impl<G, S> Default for TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
    G::S: std::fmt::Display,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
    G::S: std::fmt::Display,
{
    pub fn new() -> Self {
        let mut index = index::Arena::new();
        let root_id = index.insert(Node::new_root(0, G::num_players(), 0));
        Self {
            root_id,
            root_stats: NodeStats::new(G::num_players()),
            pv: vec![],
            stack: vec![],
            table: TranspositionTable::default(),
            trial: None,
            index,
            config: S::config(),
            timer: timer::Timer::new(),
            stats: Default::default(),
        }
    }

    #[inline]
    pub(crate) fn new_root(&mut self, player_idx: usize, hash: u64) -> Id {
        let root = Node::new_root(player_idx, G::num_players(), hash);
        self.root_id = self.index.insert(root);
        self.root_id
    }

    #[inline]
    pub fn expand(&mut self, node_id: Id, state: &G::S) -> &NodeState<G::A> {
        let node = self.index.get_mut(node_id);
        if G::is_terminal(state) {
            node.state = NodeState::Terminal;
        } else {
            let mut actions = Vec::new();
            G::generate_actions(state, &mut actions);
            debug_assert!(!actions.is_empty());
            node.state = NodeState::Expanded(
                actions
                    .into_iter()
                    .map(|action| Edge::unexplored(action, G::num_players()))
                    .collect(),
            );
        }
        &node.state // .clone()
    }

    #[inline]
    pub fn select(&mut self, ctx: &mut SearchContext<G>) {
        let player = G::player_to_move(&ctx.state).to_index();
        debug_assert!(self.stack.is_empty());
        loop {
            self.stack.push(ctx.current_id);

            let stack = NodeStack::new(self.stack.clone());
            let num_visits = stack
                .current_stats(&self.index, &self.root_stats)
                .num_visits;
            let node = self.index.get(ctx.current_id);
            if node.is_terminal() || num_visits < self.config.expand_threshold {
                return;
            }

            // Get child actions
            if node.is_leaf() {
                let node_state = self.expand(ctx.current_id, &ctx.state);
                if matches!(node_state, NodeState::Terminal) {
                    return;
                }
            }

            let best_idx = {
                let select_ctx = SelectContext {
                    q_init: self.config.q_init,
                    stack: &stack,
                    root_stats: &self.root_stats,
                    player,
                    state: &ctx.state,
                    index: &self.index,
                    table: &self.table,
                    grave: &self.stats.grave,
                    use_transpositions: self.config.use_transpositions,
                };

                self.config
                    .select
                    .best_child(&select_ctx, &mut self.config.rng)
            };

            let NodeState::Expanded(ref edges) = &(self.index.get(ctx.current_id).state) else {
                unreachable!()
            };

            if let Some(child_id) = edges[best_idx].node_id {
                ctx.traverse_apply(child_id, &edges[best_idx].action);
            } else {
                {
                    let mut actions = vec![];
                    G::generate_actions(&ctx.state, &mut actions);
                    debug_assert_eq!(actions[best_idx], edges[best_idx].action);
                }

                let action = &edges[best_idx].action;
                let state = G::apply(ctx.state.clone(), action);

                let child_id = self.new_child(&state, best_idx, ctx.current_id);

                ctx.traverse(child_id);
                ctx.state = state;

                if self.config.expand_threshold > 0 {
                    self.stack.push(ctx.current_id);
                    return;
                }
            }
        }
    }

    fn new_child(&mut self, state: &G::S, best_idx: usize, current_id: Id) -> Id {
        let hash = G::zobrist_hash(state);
        let child_id = {
            if self.config.use_transpositions {
                // TODO: the following won't work with symmetries
                if let Some(entry) = self.table.get(hash, state.clone()) {
                    entry.node_id
                } else {
                    let child = Node::new(G::player_to_move(state).to_index(), hash);
                    let node_id = self.index.insert(child);
                    self.table.insert(hash, node_id, state.clone());
                    node_id
                }
            } else {
                let child_node = Node::new(G::player_to_move(state).to_index(), hash);
                self.index.insert(child_node)
            }
        };

        match &mut (self.index.get_mut(current_id).state) {
            NodeState::Expanded(edges) => {
                edges[best_idx].node_id = Some(child_id);
            }
            _ => unreachable!(),
        }

        child_id
    }

    #[inline]
    fn select_final_action(&mut self, state: &G::S) -> G::A {
        let stack = NodeStack::new(vec![self.root_id]);
        let idx = self.config.final_action.best_child(
            &SelectContext {
                q_init: self.config.q_init,
                stack: &stack,
                root_stats: &self.root_stats,
                player: G::player_to_move(state).to_index(),
                state,
                index: &self.index,
                table: &self.table,
                grave: &self.stats.grave,
                use_transpositions: self.config.use_transpositions,
            },
            &mut self.config.rng,
        );

        match &(self.index.get(self.root_id).state) {
            NodeState::Expanded(edges) => edges[idx].action.clone(),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn simulate(&mut self, state: &G::S, player: usize) -> Trial<G> {
        self.config.simulate.playout(
            G::determinize(state.clone(), &mut self.config.rng),
            self.config.max_playout_depth,
            &self.stats,
            player,
            &mut self.config.rng,
        )
    }

    #[inline]
    pub(crate) fn backprop(&mut self, player: usize) {
        self.stats.iter_count += 1;
        self.stats.accum_depth += self.trial.as_ref().unwrap().depth + self.stack.len() - 1;
        let flags = self.config.select.backprop_flags() | self.config.simulate.backprop_flags();
        let stack = NodeStack::new(self.stack.clone());
        self.config
            .backprop
            // TODO: may as well pass &mut self? Seems like the separation
            // of concerns is not ideal.
            .update(
                &stack,
                &mut self.stats,
                &mut self.index,
                &mut self.root_stats,
                self.trial.as_ref().unwrap().clone(),
                player,
                flags,
            );
    }

    #[allow(dead_code)]
    fn snapshot(&self, iteration: u32) {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;

        _ = std::fs::create_dir_all("snapshots");
        let path_str = format!("snapshots/{iteration}.json");
        let path = Path::new(path_str.as_str());
        let json = serde_json::to_string(&self.index).unwrap();
        let mut file = match File::options().create_new(true).write(true).open(path) {
            Err(why) => panic!("couldn't open {}: {}", path.to_str().unwrap(), why),
            Ok(file) => file,
        };

        file.write_all(json.as_bytes()).expect("can't write");
    }

    pub fn verbose_summary(&self, state: &G::S) {
        if !self.config.verbose {
            return;
        }

        let num_threads = 1;
        let root = self.index.get(self.root_id);
        let total_visits = self.root_stats.num_visits;
        let rate = total_visits as f64 / num_threads as f64 / self.timer.elapsed().as_secs_f64();
        eprintln!(
            "Using {} threads, did {} total simulations with {:.1} rollouts/sec/core",
            num_threads, total_visits, rate
        );

        let player = G::player_to_move(state);

        // Sort moves by visit count, largest first.
        let mut children = match &(root.state) {
            NodeState::Expanded(edges) => edges
                .iter()
                .filter(|edge| edge.is_explored())
                .map(|edge| {
                    (
                        edge.stats.num_visits,
                        edge.stats.player[player.to_index()].score,
                        edge.action.clone(),
                    )
                })
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };

        children.sort_by_key(|t| !t.0);

        // Dump stats about the top 10 nodes.
        for (visits, score, m) in children.into_iter().take(10) {
            // Normalized so all wins is 100%, all draws is 50%, and all losses is 0%.
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
        self.stats.accum_depth = 0;
        self.stats.iter_count = 0;
        self.new_root(player_idx, hash)
    }

    fn compute_pv(&mut self, init_state: &G::S) {
        self.pv.clear();
        let mut node_id = self.root_id;
        let mut node = self.index.get(node_id);
        let mut state = init_state.clone();
        let mut stack = NodeStack::new(vec![node_id]);
        let init_player = G::player_to_move(init_state).to_index();
        while node.is_expanded() {
            let select_ctx = SelectContext {
                q_init: self.config.q_init,
                player: init_player, // TODO: opponent perspective?
                stack: &stack,
                root_stats: &self.root_stats,
                state: &state,
                index: &self.index,
                table: &self.table,
                grave: &self.stats.grave,
                use_transpositions: self.config.use_transpositions,
            };

            let best_idx = self
                .config
                .final_action
                .best_child(&select_ctx, &mut self.config.rng);

            let edge = &node.edges()[best_idx];
            if let Some(child_id) = edge.node_id {
                node_id = child_id;
                node = self.index.get(node_id);
                state = G::apply(state, &edge.action);
                self.pv.push(edge.action.clone());
                stack.push(node_id);
            } else {
                break;
            }
        }
    }
}

impl<G, S> Search for TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
    G::S: std::fmt::Display,
{
    type G = G;

    fn friendly_name(&self) -> String {
        self.config.name.clone()
    }

    fn choose_action(&mut self, state: &G::S) -> G::A {
        let hash = G::zobrist_hash(state);
        let root_id = self.reset(G::player_to_move(state).to_index(), hash);
        if self.config.use_transpositions {
            self.table.insert(hash, root_id, state.clone());
        }

        self.timer.start(self.config.max_time);

        for _ in 0..self.config.max_iterations {
            if self.timer.done() {
                break;
            }
            self.reset_iter();
            let mut ctx = SearchContext::new(root_id, state.clone());

            self.select(&mut ctx);
            self.trial = Some(self.simulate(&ctx.state, G::player_to_move(state).to_index()));
            self.backprop(G::player_to_move(state).to_index());
        }

        self.compute_pv(state);
        self.verbose_summary(state);

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
        let mut actions = vec![];
        let stack = NodeStack::new(self.stack.clone());
        for (parent_id, child_id) in stack.pairs() {
            actions.push(
                stack
                    .edge(&self.index, *parent_id, *child_id)
                    .action
                    .clone(),
            );
        }

        let utilities = G::compute_utilities(&self.trial.as_ref().unwrap().state);

        (actions, utilities)
    }

    fn estimated_depth(&self) -> usize {
        (self.stats.accum_depth as f64 / self.stats.iter_count as f64).round() as usize
    }

    fn principle_variation(&self) -> Vec<G::A> {
        self.pv.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.config.name = name.to_string();
    }
}
