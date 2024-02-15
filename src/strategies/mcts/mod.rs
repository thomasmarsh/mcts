pub mod backprop;
pub mod index;
pub mod node;
pub mod select;
pub mod simulate;
pub mod timer;
pub mod util;

use crate::game::{Action, Game, PlayerIndex};
use backprop::BackpropStrategy;
use index::Id;
use node::Node;
use node::NodeState;
use node::UnvisitedValueEstimate;
use select::SelectContext;
use select::SelectStrategy;
use simulate::SimulateStrategy;
use simulate::Trial;

use rand::{Rng, SeedableRng};

// Uses Xoshiro256PlusPlus and seeds with a u64 using SplitMix64
type FastRng = rand::rngs::SmallRng;

use rustc_hash::FxHashMap as HashMap;

pub trait Strategy<A: Action> {
    type Select: select::SelectStrategy<A>;
    type Simulate: simulate::SimulateStrategy;
    type Backprop: backprop::BackpropStrategy;
    type FinalAction: select::SelectStrategy<A>;

    fn friendly_name() -> String;
}

pub struct MctsStrategy<S, A>
where
    S: Strategy<A>,
    A: Action,
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

pub struct TreeStats<G: Game> {
    pub actions: HashMap<G::A, node::ActionStats>,
}

impl<G: Game> Default for TreeStats<G> {
    fn default() -> Self {
        Self {
            actions: Default::default(),
        }
    }
}

pub type TreeIndex<A> = index::Arena<Node<A>>;

pub struct TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G::A>,
{
    pub index: TreeIndex<G::A>,
    pub rng: FastRng,
    pub strategy: MctsStrategy<S, G::A>,
    pub timer: timer::Timer,
    pub stats: TreeStats<G>,
    pub verbose: bool,
}

impl<G, S> Default for TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G::A>,
    MctsStrategy<S, G::A>: Default,
{
    fn default() -> Self {
        Self::new(Default::default(), FastRng::from_entropy())
    }
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G::A>,
{
    pub fn new(strategy: MctsStrategy<S, G::A>, rng: FastRng) -> Self {
        Self {
            index: index::Arena::new(),
            strategy,
            rng,
            timer: timer::Timer::new(),
            stats: Default::default(),
            verbose: false,
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
                let mut select_ctx = SelectContext {
                    q_init: self.strategy.q_init,
                    current_id: ctx.current_id,
                    player: player.to_index(),
                    index: &self.index,
                    rng: &mut self.rng,
                };
                self.strategy.select.best_child(&mut select_ctx)
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
        let idx = self.strategy.final_action.best_child(&mut SelectContext {
            q_init: self.strategy.q_init,
            current_id: root_id,
            player: G::player_to_move(state).to_index(),
            index: &self.index,
            rng: &mut self.rng,
        });

        match &(self.index.get(root_id).state) {
            NodeState::Expanded { actions, .. } => actions[idx].clone(),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn simulate(&mut self, state: &G::S) -> Trial<G> {
        self.strategy.simulate.playout::<G>(
            G::determinize(state.clone(), &mut self.rng),
            self.strategy.max_playout_depth,
            &mut self.rng,
        )
    }

    #[inline]
    pub(crate) fn backprop(&mut self, ctx: &mut SearchContext<G>, trial: Trial<G>) {
        self.strategy
            .backprop
            // TODO: may as well pass &mut self? Seems like the separation
            // of concerns is not ideal.
            .update(ctx, &mut self.stats, &mut self.index, trial);
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
}

impl<G, S> super::Search for TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G::A>,
{
    type G = G;

    fn friendly_name(&self) -> String {
        format!("mcts[{}]", S::friendly_name())
    }

    fn choose_action(&mut self, state: &G::S) -> G::A {
        self.index.clear();
        self.timer.start(self.strategy.max_time);

        let root_id = self.new_root();

        for _ in 0..self.strategy.max_iterations {
            if self.timer.done() {
                break;
            }
            let mut ctx = SearchContext::new(root_id, state.clone());
            self.select(&mut ctx);
            let trial = self.simulate(&ctx.state);
            self.backprop(&mut ctx, trial);
        }

        self.verbose_summary(root_id, state);
        self.select_final_action(root_id, state)
    }

    fn principle_variation(&self) -> Vec<&G::A>
    where
        G: Game,
    {
        unimplemented!()
    }
}
