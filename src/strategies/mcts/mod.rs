pub mod backprop;
pub mod index;
pub mod node;
pub mod select;
pub mod simulate;
pub mod timer;
pub mod util;

use crate::game::{Game, PlayerIndex};
use backprop::BackpropStrategy;
use index::Id;
use node::Node;
use node::NodeState;
use node::UnvisitedValueEstimate;

use rand::{Rng, SeedableRng};
use select::SelectContext;
use select::SelectStrategy;
use simulate::SimulateStrategy;
use simulate::Trial;

// Uses Xoshiro256PlusPlus and seeds with a u64 using SplitMix64
use rand::rngs::SmallRng;

use rustc_hash::FxHashMap as HashMap;

////////////////////////////////////////////////////////////////////////////////

const GRAVE: usize = 0b001;
const GLOBAL: usize = 0b010;
const AMAF: usize = 0b100;

pub struct BackpropFlags(pub usize);

impl BackpropFlags {
    pub fn grave(&self) -> bool {
        self.0 & GRAVE == GRAVE
    }

    pub fn global(&self) -> bool {
        self.0 & GLOBAL == GLOBAL
    }

    pub fn amaf(&self) -> bool {
        self.0 & AMAF == AMAF
    }
}

impl std::ops::BitOr for BackpropFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

////////////////////////////////////////////////////////////////////////////////

pub trait Strategy<G: Game>: Clone {
    type Select: select::SelectStrategy<G::A>;
    type Simulate: simulate::SimulateStrategy<G>;
    type Backprop: backprop::BackpropStrategy;
    type FinalAction: select::SelectStrategy<G::A>;

    fn friendly_name() -> String;
}

#[derive(Clone)]
pub struct MctsStrategy<G, S>
where
    G: Game,
    S: Strategy<G>,
{
    pub select: S::Select,
    pub simulate: S::Simulate,
    pub backprop: S::Backprop,
    pub final_action: S::FinalAction,
    pub q_init: UnvisitedValueEstimate,
    pub playouts_before_expanding: u32,
    pub max_playout_depth: usize,
    pub max_iterations: usize,
    pub max_time: std::time::Duration,
}

impl<G, S> MctsStrategy<G, S>
where
    G: Game,
    S: Strategy<G>,
{
    pub fn select(mut self, select: S::Select) -> Self {
        self.select = select;
        self
    }

    pub fn simulate(mut self, simulate: S::Simulate) -> Self {
        self.simulate = simulate;
        self
    }

    pub fn backprop(mut self, backprop: S::Backprop) -> Self {
        self.backprop = backprop;
        self
    }

    pub fn final_action(mut self, final_action: S::FinalAction) -> Self {
        self.final_action = final_action;
        self
    }

    pub fn q_init(mut self, q_init: UnvisitedValueEstimate) -> Self {
        self.q_init = q_init;
        self
    }

    pub fn playouts_before_expanding(mut self, playouts_before_expanding: u32) -> Self {
        self.playouts_before_expanding = playouts_before_expanding;
        self
    }

    pub fn max_playout_depth(mut self, max_playout_depth: usize) -> Self {
        self.max_playout_depth = max_playout_depth;
        self
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    // NOTE: special logic here
    pub fn max_time(mut self, max_time: std::time::Duration) -> Self {
        self.max_time = max_time;
        if self.max_time != std::time::Duration::default() {
            self.max_iterations(usize::MAX)
        } else {
            self
        }
    }
}

pub struct SearchContext<G: Game> {
    pub current_id: Id,
    pub state: G::S,
    pub stack: Vec<Id>,
}

impl<G: Game> SearchContext<G> {
    pub fn new(current_id: Id, state: G::S) -> Self {
        Self {
            current_id,
            state,
            stack: vec![],
        }
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

#[derive(Debug)]
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

pub struct TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
{
    pub(crate) index: TreeIndex<G::A>,
    pub(crate) timer: timer::Timer,

    pub stats: TreeStats<G>,
    pub rng: SmallRng,
    pub strategy: MctsStrategy<G, S>,
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

    pub fn strategy(mut self, strategy: MctsStrategy<G, S>) -> Self {
        self.strategy = strategy;
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
    MctsStrategy<G, S>: Default,
{
    fn default() -> Self {
        Self::new(MctsStrategy::default(), SmallRng::from_entropy())
    }
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    MctsStrategy<G, S>: Default,
{
    pub fn new(strategy: MctsStrategy<G, S>, rng: SmallRng) -> Self {
        Self {
            index: index::Arena::new(),
            strategy,
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
        self.index.insert(root)
    }

    #[inline]
    pub fn select(&mut self, ctx: &mut SearchContext<G>) {
        let player = G::player_to_move(&ctx.state);
        loop {
            ctx.stack.push(ctx.current_id);

            let is_leaf = {
                let node = self.index.get(ctx.current_id);
                // TODO: when playouts_before_expanding == 0, then we should expand whole tree. This is done for realtime applications.
                if node.is_terminal()
                    || node.stats.num_visits < self.strategy.playouts_before_expanding
                {
                    return;
                }
                node.is_leaf()
            };

            // Get child actions
            if is_leaf {
                let node = self.index.get_mut(ctx.current_id);
                if G::is_terminal(&ctx.state) {
                    node.state = NodeState::Terminal;
                    return;
                }
                let mut actions = Vec::new();
                G::generate_actions(&ctx.state, &mut actions);
                assert!(!actions.is_empty());
                node.state = NodeState::Expanded {
                    children: vec![None; actions.len()],
                    actions,
                };
            }

            let best_idx = {
                let select_ctx = SelectContext {
                    q_init: self.strategy.q_init,
                    current_id: ctx.current_id,
                    player: player.to_index(),
                    index: &self.index,
                };
                self.strategy.select.best_child(&select_ctx, &mut self.rng)
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
                ctx.stack.push(ctx.current_id);
                return;
            }
        }
    }

    #[inline]
    fn select_final_action(&mut self, root_id: Id, state: &G::S) -> G::A {
        let idx = self.strategy.final_action.best_child(
            &SelectContext {
                q_init: self.strategy.q_init,
                current_id: root_id,
                player: G::player_to_move(state).to_index(),
                index: &self.index,
            },
            &mut self.rng,
        );

        match &(self.index.get(root_id).state) {
            NodeState::Expanded { actions, .. } => actions[idx].clone(),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn simulate(&mut self, state: &G::S, player: usize) -> Trial<G> {
        self.strategy.simulate.playout(
            G::determinize(state.clone(), &mut self.rng),
            self.strategy.max_playout_depth,
            &self.stats,
            player,
            &mut self.rng,
        )
    }

    #[inline]
    pub(crate) fn backprop(&mut self, ctx: &mut SearchContext<G>, trial: Trial<G>, player: usize) {
        self.stats.iter_count += 1;
        self.stats.accum_depth += trial.depth + ctx.stack.len() - 1;
        let flags = self.strategy.select.backprop_flags() | self.strategy.simulate.backprop_flags();
        self.strategy
            .backprop
            // TODO: may as well pass &mut self? Seems like the separation
            // of concerns is not ideal.
            .update(ctx, &mut self.stats, &mut self.index, trial, player, flags);
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

    pub fn verbose_summary(&self, root_id: Id, state: &G::S) {
        if !self.verbose {
            return;
        }

        let num_threads = 1;
        let root = self.index.get(root_id);
        let total_visits = root.stats.num_visits;
        let rate = total_visits as f64 / num_threads as f64 / self.timer.elapsed().as_secs_f64();
        eprintln!(
            "Using {} threads, did {} total simulations with {:.1} rollouts/sec/core",
            num_threads, total_visits, rate
        );

        let player = G::player_to_move(state);

        // Sort moves by visit count, largest first.
        let mut children = match &(self.index.get(root_id).state) {
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
    }

    fn reset(&mut self) {
        self.index.clear();
        self.stats.accum_depth = 0;
        self.stats.iter_count = 0;
    }
}

impl<G, S> super::Search for TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    MctsStrategy<G, S>: Default,
    <S as Strategy<G>>::Select: Sync + Send,
    <S as Strategy<G>>::FinalAction: Sync + Send,
    <S as Strategy<G>>::Backprop: Sync + Send,
    <S as Strategy<G>>::Simulate: Sync + Send,
{
    type G = G;

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn choose_action(&mut self, state: &G::S) -> G::A {
        self.reset();
        self.timer.start(self.strategy.max_time);

        let root_id = self.new_root();

        for _ in 0..self.strategy.max_iterations {
            if self.timer.done() {
                break;
            }
            let mut ctx = SearchContext::new(root_id, state.clone());
            self.select(&mut ctx);
            let trial = self.simulate(&ctx.state, G::player_to_move(state).to_index());
            self.backprop(&mut ctx, trial, G::player_to_move(state).to_index());
        }

        self.verbose_summary(root_id, state);
        self.select_final_action(root_id, state)
    }

    fn estimated_depth(&self) -> usize {
        (self.stats.accum_depth as f64 / self.stats.iter_count as f64).round() as usize
    }

    fn principle_variation(&self) -> Vec<&G::A>
    where
        G: Game,
    {
        unimplemented!()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }
}
