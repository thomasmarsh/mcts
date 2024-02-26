use super::backprop::BackpropStrategy;
use super::config::SearchConfig;
use super::config::Strategy;
use super::index;
use super::index::Id;
use super::node;
use super::node::Node;
use super::node::NodeState;
use super::select::SelectContext;
use super::select::SelectStrategy;
use super::simulate::SimulateStrategy;
use super::simulate::Trial;
use super::table::TranspositionTable;
use super::timer;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::strategies::Search;
use crate::util::pv_string;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rustc_hash::FxHashMap as HashMap;

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
    pub actions: HashMap<G::A, node::ActionStats>,
    pub player_actions: Vec<HashMap<G::A, node::ActionStats>>,
    pub accum_depth: usize,
    pub iter_count: usize,
}

impl<G: Game> Default for TreeStats<G> {
    fn default() -> Self {
        Self {
            actions: Default::default(),
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
{
    pub(crate) index: TreeIndex<G::A>,
    pub(crate) timer: timer::Timer,
    pub(crate) root_id: Id,
    pub(crate) init_state: Option<G::S>,
    pub(crate) pv: Vec<G::A>,
    pub(crate) table: TranspositionTable,

    pub config: SearchConfig<G, S>,
    pub stats: TreeStats<G>,
    pub stack: Vec<Id>,
    pub trial: Option<Trial<G>>,
    pub rng: SmallRng,
    pub verbose: bool,
    pub name: String,
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
{
    pub fn rng(mut self, rng: SmallRng) -> Self {
        self.rng = rng;
        self
    }

    pub fn config(mut self, config: SearchConfig<G, S>) -> Self {
        self.config = config;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}

impl<G, S> Default for TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
{
    fn default() -> Self {
        Self::new(SearchConfig::default(), SmallRng::from_entropy())
    }
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
{
    pub fn new(config: SearchConfig<G, S>, rng: SmallRng) -> Self {
        let mut index = index::Arena::new();
        let root_id = index.insert(Node::new_root(G::num_players()));
        Self {
            root_id,
            init_state: None,
            pv: vec![],
            stack: vec![],
            table: TranspositionTable::default(),
            trial: None,
            index,
            config,
            rng,
            timer: timer::Timer::new(),
            stats: Default::default(),
            verbose: false,
            name: format!("mcts[{}]", S::friendly_name()),
        }
    }

    #[inline]
    pub(crate) fn new_root(&mut self) -> Id {
        let root = Node::new_root(G::num_players());
        let root_id = self.index.insert(root);
        self.root_id = root_id;
        root_id
    }

    #[inline]
    pub fn expand(&mut self, node_id: Id, state: &G::S) -> NodeState<G::A> {
        let node = self.index.get_mut(node_id);
        if G::is_terminal(state) {
            node.state = NodeState::Terminal;
        } else {
            let mut actions = Vec::new();
            G::generate_actions(state, &mut actions);
            debug_assert!(!actions.is_empty());
            node.state = NodeState::Expanded {
                children: vec![None; actions.len()],
                actions,
            };
        }
        node.state.clone()
    }

    #[inline]
    pub fn select(&mut self, ctx: &mut SearchContext<G>) {
        let player = G::player_to_move(&ctx.state);
        debug_assert!(self.stack.is_empty());
        loop {
            self.stack.push(ctx.current_id);

            let node = self.index.get(ctx.current_id);
            if node.is_terminal() || node.stats.num_visits < self.config.expand_threshold {
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
                    current_id: ctx.current_id,
                    stack: self.stack.clone(),
                    player: player.to_index(),
                    player_to_move: G::player_to_move(&ctx.state).to_index(),
                    state: &ctx.state,
                    index: &self.index,
                };
                self.config.select.best_child(&select_ctx, &mut self.rng)
            };

            let NodeState::Expanded {
                ref children,
                actions,
            } = &(self.index.get(ctx.current_id).state)
            else {
                unreachable!()
            };

            if let Some(child_id) = children[best_idx] {
                let child = self.index.get(child_id);
                ctx.traverse_apply(child_id, &child.action(&self.index));
            } else {
                let action = &actions[best_idx];
                let state = G::apply(ctx.state.clone(), action);

                let child_id =
                    self.index
                        .insert(Node::new(ctx.current_id, best_idx, G::num_players()));

                match &mut (self.index.get_mut(ctx.current_id).state) {
                    NodeState::Expanded { children, .. } => {
                        children[best_idx] = Some(child_id);
                    }
                    _ => unreachable!(),
                }

                ctx.traverse(child_id);
                ctx.state = state;

                if self.config.expand_threshold > 0 {
                    self.stack.push(ctx.current_id);
                    return;
                }
            }
        }
    }

    #[inline]
    fn select_final_action(&mut self, state: &G::S) -> G::A {
        let idx = self.config.final_action.best_child(
            &SelectContext {
                q_init: self.config.q_init,
                current_id: self.root_id,
                stack: self.stack.clone(),
                player: G::player_to_move(state).to_index(),
                player_to_move: G::player_to_move(state).to_index(),
                state,
                index: &self.index,
            },
            &mut self.rng,
        );

        match &(self.index.get(self.root_id).state) {
            NodeState::Expanded { actions, .. } => actions[idx].clone(),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn simulate(&mut self, state: &G::S, player: usize) -> Trial<G> {
        self.config.simulate.playout(
            G::determinize(state.clone(), &mut self.rng),
            self.config.max_playout_depth,
            &self.stats,
            player,
            &mut self.rng,
        )
    }

    #[inline]
    pub(crate) fn backprop(&mut self, ctx: &mut SearchContext<G>, player: usize) {
        self.stats.iter_count += 1;
        self.stats.accum_depth += self.trial.as_ref().unwrap().depth + self.stack.len() - 1;
        let flags = self.config.select.backprop_flags() | self.config.simulate.backprop_flags();
        self.config
            .backprop
            // TODO: may as well pass &mut self? Seems like the separation
            // of concerns is not ideal.
            .update(
                ctx,
                self.stack.clone(),
                &mut self.stats,
                &mut self.index,
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
        if !self.verbose {
            return;
        }

        let num_threads = 1;
        let root = self.index.get(self.root_id);
        let total_visits = root.stats.num_visits;
        let rate = total_visits as f64 / num_threads as f64 / self.timer.elapsed().as_secs_f64();
        eprintln!(
            "Using {} threads, did {} total simulations with {:.1} rollouts/sec/core",
            num_threads, total_visits, rate
        );

        let player = G::player_to_move(state);

        // Sort moves by visit count, largest first.
        let mut children = match &(root.state) {
            NodeState::Expanded { children, .. } => children
                .iter()
                .flatten()
                .map(|node_id| {
                    let node = self.index.get(*node_id);
                    (
                        node.stats.num_visits,
                        node.stats.scores[player.to_index()],
                        node.action(&self.index).clone(),
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

        eprintln!(
            "PV: {}",
            pv_string::<G>(self.pv.as_slice(), self.init_state.as_ref().unwrap())
        )
    }

    #[inline]
    pub(crate) fn reset_iter(&mut self) {
        self.stack.clear();
        self.trial = None;
    }

    #[inline]
    pub(crate) fn reset(&mut self) -> Id {
        self.index.clear();
        self.table.clear();
        self.stats.accum_depth = 0;
        self.stats.iter_count = 0;
        self.new_root()
    }

    fn compute_pv(&mut self) {
        self.pv.clear();
        let mut node_id = self.root_id;
        let mut node = self.index.get(node_id);
        let mut state = self.init_state.clone().unwrap().clone();
        while node.is_expanded() {
            let player = G::player_to_move(&state);
            let select_ctx = SelectContext {
                q_init: self.config.q_init,
                current_id: node_id,
                player: player.to_index(),
                stack: self.stack.clone(),
                state: &state,
                player_to_move: player.to_index(),
                index: &self.index,
            };
            let best_idx = self
                .config
                .final_action
                .best_child(&select_ctx, &mut self.rng);
            if let Some(child_id) = node.children()[best_idx] {
                node_id = child_id;
                node = self.index.get(node_id);
                let action = node.action(&self.index);
                state = G::apply(state, &action);
                self.pv.push(action);
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
{
    type G = G;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn choose_action(&mut self, state: &G::S) -> G::A {
        let root_id = self.reset();
        self.timer.start(self.config.max_time);

        self.init_state = Some(state.clone());

        for _ in 0..self.config.max_iterations {
            if self.timer.done() {
                break;
            }
            self.reset_iter();
            let mut ctx = SearchContext::new(root_id, state.clone());

            self.select(&mut ctx);
            self.trial = Some(self.simulate(&ctx.state, G::player_to_move(state).to_index()));
            self.backprop(&mut ctx, G::player_to_move(state).to_index());
        }

        self.compute_pv();
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
        assert_eq!(self.config.expand_threshold, 0);
        assert_eq!(self.config.max_iterations, 1);

        // Run the search, with expand_threshold == 0, so we fully expand to the
        // terminal node.
        _ = self.choose_action(state);

        // The stack now contains the action path to the terminal state.
        let actions = self
            .stack
            .iter()
            .skip(1)
            .cloned()
            .map(|id| self.index.get(id).action(&self.index))
            .collect();

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
        self.name = name.to_string();
    }
}
