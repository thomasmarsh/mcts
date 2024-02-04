use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::strategies::arena::{Arena, NodeRef};
use crate::util::random_best;
use rand::seq::SliceRandom;
use rand_core::SeedableRng;

use crate::game::Game;
use crate::strategies::Strategy;
use crate::util::{move_id, pv_string};

use super::sync_util::timeout_signal;

use log::{error, trace};

type Rng = rand_xorshift::XorShiftRng;

#[derive(Clone, Copy)]
pub enum SelectionStrategy {
    /// Select the root child with the highest reward.
    Max,

    /// Select the most visited root child.
    Robust,

    // theoretically sqrt(2)
    UCT(f64, f64),
    // Select the child which has both the highest visit count and the highest
    // value. If there is no max-robust child at the moment, it is better to
    // continue the search until a max-robust child is found rather than
    // returning a child with a low visit count
    // MaxRobust,

    // Select the child which maximizes a lower confidence bound.
    // SecureChild(f64)
}

#[derive(Copy, Clone, PartialEq)]
pub enum ExpansionStrategy {
    Single,
    Full,
}

impl ExpansionStrategy {
    fn is_single(self) -> bool {
        self == Self::Single
    }
}

#[inline]
fn uct<M>(c: f64, rave_param: f64, parent: &Node<M>, child: &Node<M>) -> f64 {
    let epsilon = 1e-6;
    let w = child.q as f64;
    let n = child.n as f64 + epsilon;
    let total = parent.n as f64;

    let uct_value = w / n + c * (2. * total.ln() / n).sqrt();

    let rave_value = if child.n_rave > 0 {
        child.q_rave as f64 / child.n_rave as f64
    } else {
        0.0
    };
    let beta = rave_param / (rave_param + n);

    (1. - beta) * uct_value + beta * rave_value
}

/*
use statrs::distribution::{Normal, Univariate};


#[inline]
fn secure_child(confidence_level: f64, child: &Node<M>) -> f64 {
    let mean = child.q as f64 / child.n as f64;
    let std_dev = (child.q_squared as f64 / child.n as f64 - mean.powi(2)).sqrt();
    let normal = Normal::new(mean, std_dev).unwrap();
    normal.inverse_cdf(confidence_level)
}
*/

impl SelectionStrategy {
    fn score<M>(&self, parent: &Node<M>, child: &Node<M>) -> f64 {
        match self {
            SelectionStrategy::Max => child.q as f64,
            SelectionStrategy::Robust => child.n as f64,
            SelectionStrategy::UCT(c, rave_param) => uct(*c, *rave_param, parent, child),
        }
    }
}

// TODO: I'd like to make this more type safe with an enum
pub(crate) struct Node<M> {
    q: i32,
    n: u32,
    q_rave: i32,
    n_rave: u32,
    action: M,
    is_terminal: bool, // ignored when ExpansionStrategy is Full
    unexplored: Vec<M>,
    children: Vec<NodeRef>,
}

impl<M> std::fmt::Debug for Node<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node {{ q: {}, n: {}, q_rave: {}, n_rave: {}, is_terminal: {}, children: {:?}, unexplored: [...] (len={}) action: {{...}} }}", self.q, self.n, self.q_rave, self.n_rave, self.is_terminal, self.children, self.unexplored.len())
    }
}

impl<M> Node<M> {
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    fn new_root(unexplored: Vec<M>) -> Self {
        Node {
            q: 0,
            n: 0,
            q_rave: 0,
            n_rave: 0,
            action: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            unexplored,
            is_terminal: false,
            children: Vec::new(),
        }
    }

    #[inline]
    fn new(m: M, unexplored: Vec<M>) -> Self {
        Node {
            q: 0,
            n: 0,
            q_rave: 0,
            n_rave: 0,
            action: m,
            unexplored,
            is_terminal: false,
            children: Vec::new(),
        }
    }

    #[inline]
    fn update(&mut self, reward: i32) {
        self.q += reward;
        self.n += 1;
    }
}

struct Timer {
    start_time: Instant,
}

impl Timer {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    fn elapsed(&self) -> Duration {
        Instant::now().duration_since(self.start_time)
    }
}

pub struct Config {
    pub rng: Rng,
    pub max_time: Duration,
    pub use_rave: bool,
    pub action_selection_strategy: SelectionStrategy,
    pub tree_selection_strategy: SelectionStrategy,
    pub expansion_strategy: ExpansionStrategy,
    pub rollouts_before_expanding: u32,
    pub max_rollouts: u32,
    pub verbose: bool,
    pub max_simulate_depth: u32,
}

impl Config {
    fn new() -> Self {
        Self {
            rng: Rng::from_entropy(),
            max_time: Duration::from_secs(5),
            use_rave: true,
            action_selection_strategy: SelectionStrategy::UCT(2.0_f64.sqrt(), 3000.0),
            tree_selection_strategy: SelectionStrategy::UCT(2.0_f64.sqrt(), 3000.0),
            expansion_strategy: ExpansionStrategy::Single,
            rollouts_before_expanding: 5,
            max_rollouts: u32::MAX,
            verbose: false,
            max_simulate_depth: 20000,
        }
    }
}

pub struct TreeSearch<G: Game> {
    arena: Arena<G::M>,
    pv: Vec<NodeRef>,
    timeout: Arc<AtomicBool>,
    pub config: Config,
}

impl<G: Game> std::fmt::Debug for TreeSearch<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TreeSearch<G>")
    }
}

impl<G: Game> Default for TreeSearch<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Game> TreeSearch<G> {
    pub fn new() -> Self {
        Self {
            arena: Arena::new(),
            pv: Vec::new(),
            timeout: Arc::new(AtomicBool::new(false)),
            config: Config::new(),
        }
    }

    #[inline]
    fn best_child(&mut self, strategy: SelectionStrategy, node_id: NodeRef) -> Option<NodeRef> {
        let parent = self.arena.get(node_id);
        if parent.children.is_empty() {
            return None;
        }
        random_best(
            parent.children.as_slice(),
            &mut self.config.rng,
            |child_id| {
                let child = self.arena.get(*child_id);
                strategy.score(parent, child)
            },
        )
        .copied()
    }

    fn set_pv(&mut self, mut node_id: NodeRef) {
        self.pv.clear();
        loop {
            match self.best_child(self.config.action_selection_strategy, node_id) {
                None => break,
                Some(child_id) => {
                    self.pv.push(child_id);
                    node_id = child_id;
                }
            }
        }
    }

    #[inline]
    fn is_terminal(&self, node: &Node<G::M>, state: &G::S) -> bool {
        if self.config.expansion_strategy.is_single() {
            node.is_terminal
        } else {
            G::is_terminal(state)
        }
    }

    fn select(&mut self, mut node_id: NodeRef, init_state: &G::S) -> (NodeRef, G::S, Vec<NodeRef>) {
        let mut stack = vec![node_id];
        let mut state = init_state.clone();
        loop {
            let node = self.arena.get(node_id);
            if self.is_terminal(node, &state) {
                return (node_id, state.clone(), stack);
            }

            let is_leaf = node.children.is_empty();
            let needs_rollouts = node.n <= self.config.rollouts_before_expanding;
            let unexplored =
                self.config.expansion_strategy.is_single() && !node.unexplored.is_empty();

            if is_leaf || needs_rollouts || unexplored {
                if !needs_rollouts {
                    // Perform expansion
                    return self.expand(node_id, &state, stack);
                } else {
                    return (node_id, state.clone(), stack);
                }
            } else {
                let child_id = self
                    .best_child(self.config.tree_selection_strategy, node_id)
                    .unwrap();
                let child = self.arena.get(child_id);
                state = G::apply(&state, child.action.clone());

                node_id = child_id;
                stack.push(node_id);
            }
        }
    }

    #[inline]
    fn expand(
        &mut self,
        node_id: NodeRef,
        init_state: &G::S,
        stack: Vec<NodeRef>,
    ) -> (NodeRef, G::S, Vec<NodeRef>) {
        debug_assert!(!G::is_terminal(init_state));
        match self.config.expansion_strategy {
            ExpansionStrategy::Single => self.expand_single(node_id, init_state, stack),
            ExpansionStrategy::Full => self.expand_full(node_id, init_state, stack),
        }
    }

    fn expand_full(
        &mut self,
        node_id: NodeRef,
        init_state: &G::S,
        mut stack: Vec<NodeRef>,
    ) -> (NodeRef, G::S, Vec<NodeRef>) {
        let moves = G::gen_moves(init_state)
            .iter()
            .map(|m| self.arena.add(Node::new(m.clone(), Vec::new())))
            .collect::<Vec<_>>();

        let node = self.arena.get_mut(node_id);
        assert!(node.children.is_empty());
        node.children.extend(moves);

        let child_id = node.children[0];
        let child = self.arena.get(child_id);
        let state = G::apply(init_state, child.action.clone());
        stack.push(child_id);
        (child_id, state, stack)
    }

    fn expand_single(
        &mut self,
        node_id: NodeRef,
        init_state: &G::S,
        mut stack: Vec<NodeRef>,
    ) -> (NodeRef, G::S, Vec<NodeRef>) {
        let node = self.arena.get(node_id);
        debug_assert!(!node.unexplored.is_empty());
        if let Some(action) = node.unexplored.last() {
            let state = G::apply(init_state, action.clone());
            let is_terminal = G::is_terminal(&state);
            let moves = if is_terminal {
                vec![]
            } else {
                G::gen_moves(&state)
            };
            let child_id = self.arena.add(Node {
                is_terminal,
                ..Node::new(action.clone(), moves)
            });
            stack.push(child_id);
            let node = self.arena.get_mut(node_id);
            node.children.push(child_id);
            node.unexplored.pop();
            (child_id, state, stack)
        } else {
            error!("No unexplored actions left for this node: {:?}", node_id);
            error!("state: {:?}", init_state);
            (node_id, init_state.clone(), stack)
        }
    }

    fn step(&mut self, root_id: NodeRef, init_state: &G::S) {
        let (node_id, state, stack) = self.select(root_id, init_state);
        let (reward, history) = self.simulate(node_id, init_state, &state);
        self.backpropagate(stack, reward, history);
    }

    // TODO: move to a separate Rollout<S: Strategy>, noting that RAVE needs the PV
    #[inline]
    fn simulate(&mut self, node_id: NodeRef, init_state: &G::S, state: &G::S) -> (i32, Vec<G::M>) {
        debug_assert!(self.arena.get(node_id).children.is_empty());
        if self.arena.get(node_id).is_terminal {
            return (G::get_reward(init_state, state), Vec::new());
        }

        let mut state = state.clone();
        let mut depth = 0;
        let mut history = Vec::new();
        loop {
            if G::is_terminal(&state) {
                return (G::get_reward(init_state, &state), history);
            }
            let m = G::gen_moves(&state)
                .choose(&mut self.config.rng)
                .unwrap()
                .clone();
            self.config.use_rave.then(|| history.push(m.clone()));
            state = G::apply(&state, m);
            depth += 1;
            if depth >= self.config.max_simulate_depth {
                return (0, history);
            }
        }
    }

    fn update_rave(&mut self, node_id: NodeRef, reward: i32, history: &Vec<G::M>) {
        if !self.config.use_rave {
            return;
        }

        let child_ms = self
            .arena
            .get(node_id)
            .children
            .iter()
            .map(|child_id| (*child_id, self.arena.get(*child_id).action.clone()))
            .collect::<Vec<_>>();

        for m in history {
            for (child_id, child_m) in &child_ms {
                if *m == *child_m {
                    let child = self.arena.get_mut(*child_id);
                    child.q_rave += reward;
                    child.n_rave += 1;
                }
            }
        }
    }

    fn backpropagate(
        &mut self,
        // node_id: NodeRef,
        mut stack: Vec<NodeRef>,
        reward: i32,
        history: Vec<G::M>,
    ) {
        while let Some(node_id) = stack.pop() {
            let node = self.arena.get_mut(node_id);
            node.update(reward);
            self.update_rave(node_id, reward, &history);
        }
    }

    pub fn choose_move(&mut self, state: &G::S) -> Option<G::M> {
        if G::is_terminal(state) {
            return None;
        }
        let timer = self.start_timer();
        self.arena.clear();
        let root = Node::new_root(match self.config.expansion_strategy {
            ExpansionStrategy::Full => Vec::new(),
            ExpansionStrategy::Single => G::gen_moves(state),
        });

        let root_id = self.arena.add(root);

        for _ in 0..self.config.max_rollouts {
            if self.timeout.load(Ordering::Relaxed) {
                break;
            }
            self.step(root_id, state);
        }

        self.set_pv(root_id);
        self.verbose_summary(root_id, &timer, state);
        self.best_child(self.config.action_selection_strategy, root_id)
            .map(|child_id| self.arena.get(child_id).action.clone())
    }

    fn start_timer(&mut self) -> Timer {
        self.timeout = if self.config.max_time == Duration::default() {
            Arc::new(AtomicBool::new(false))
        } else {
            timeout_signal(self.config.max_time)
        };

        Timer::new()
    }

    pub fn set_timeout(&mut self, timeout: std::time::Duration) {
        self.config.max_rollouts = u32::MAX;
        self.config.max_time = timeout;
    }

    pub fn set_max_rollouts(&mut self, max_rollouts: u32) {
        self.config.max_time = Duration::default();
        self.config.max_rollouts = max_rollouts;
    }

    pub fn set_max_depth(&mut self, depth: u32) {
        self.config.max_simulate_depth = depth;
    }

    fn verbose_summary(&self, root_id: NodeRef, timer: &Timer, state: &G::S) {
        if !self.config.verbose {
            return;
        }
        let num_threads = 1;
        let root = self.arena.get(root_id);
        let total_visits = root.n;
        let rate = total_visits as f64 / num_threads as f64 / timer.elapsed().as_secs_f64();
        eprintln!(
            "Using {} threads, did {} total simulations with {:.1} rollouts/sec/core",
            num_threads, total_visits, rate
        );
        // Sort moves by visit count, largest first.
        let mut children = self
            .arena
            .get(root_id)
            .children
            .iter()
            .map(|node_id| {
                let node = self.arena.get(*node_id);
                (node.n, node.q, node.action.clone())
            })
            .collect::<Vec<_>>();

        children.sort_by_key(|t| !t.0);

        // Dump stats about the top 10 nodes.
        for (visits, score, m) in children.into_iter().take(10) {
            // Normalized so all wins is 100%, all draws is 50%, and all losses is 0%.
            let win_rate = (score as f64 + visits as f64) / (visits as f64 * 2.0);
            eprintln!(
                "{:>6} visits, {:.02}% wins: {}",
                visits,
                win_rate * 100.0,
                move_id::<G>(state, Some(m))
            );
        }

        // Dump PV.
        let pv_m = self
            .pv
            .iter()
            .map(|node_id| self.arena.get(*node_id).action.clone())
            .collect::<Vec<_>>();
        eprintln!("Principal variation: {}", pv_string::<G>(&pv_m[..], state));
    }
}

pub struct TreeSearchStrategy<G: Game>(TreeSearch<G>);

impl<G: Game> Default for TreeSearchStrategy<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Game> TreeSearchStrategy<G> {
    pub fn new() -> Self {
        Self(TreeSearch::new())
    }
}

impl<G: Game> Strategy<G> for TreeSearchStrategy<G> {
    fn choose_move(&mut self, state: &G::S) -> Option<G::M> {
        self.0.choose_move(state)
    }

    fn set_timeout(&mut self, timeout: std::time::Duration) {
        self.0.set_timeout(timeout)
    }

    fn set_max_depth(&mut self, depth: u32) {
        self.0.set_max_depth(depth);
    }

    fn set_max_rollouts(&mut self, max_rollouts: u32) {
        self.0.set_max_rollouts(max_rollouts);
    }

    fn set_verbose(&mut self) {
        self.0.config.verbose = true;
    }

    fn principal_variation(&self) -> Vec<G::M> {
        self.0
            .pv
            .iter()
            .map(|node_id| self.0.arena.get(*node_id).action.clone())
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::games::ttt::{HashedPosition, Move, TicTacToe};

    #[test]
    fn test_select_trivial() {
        let mut t: TreeSearch<TicTacToe> = Default::default();

        let init_node = Node::new(Move(0), Vec::new());
        let init_node_id = t.arena.add(init_node);

        let init_state = HashedPosition::new();

        //  Check if the function returns the correct node when the node has not
        // been expanded yet
        let (node_id, state, _) = t.select(init_node_id, &init_state);
        assert_eq!(node_id, init_node_id);
        assert_eq!(state, init_state);
    }

    #[test]
    fn test_select() {
        let mut t: TreeSearch<TicTacToe> = Default::default();

        let node = Node::new(Move(0), Vec::new());
        let node_id = t.arena.add(node);

        let state = HashedPosition::new();

        //  Check if the function returns the correct node when the node has not
        // been expanded yet
        let (selected_id, selected_state, _) = t.select(node_id, &state);
        assert_eq!(node_id, selected_id);
        assert_eq!(state, selected_state);
    }
}
