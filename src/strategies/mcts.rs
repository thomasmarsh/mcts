use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::strategies::index::{Index, NodeRef};
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
    UCT(f32),
    // Select the child which has both the highest visit count and the highest
    // value. If there is no max-robust child at the moment, it is better to
    // continue the search until a max-robust child is found rather than
    // returning a child with a low visit count
    // MaxRobust,

    // Select the child which maximizes a lower confidence bound.
    // SecureChild(f32)
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
fn uct<M>(c: f32, parent: &Node<M>, child: &Node<M>) -> f32 {
    let epsilon = 1e-6;
    let w = child.q as f32;
    let n = child.n as f32 + epsilon;
    let total = parent.n as f32;

    w / n + c * (2. * total.ln() / n).sqrt()
}

/*
use statrs::distribution::{Normal, Univariate};


#[inline]
fn secure_child(confidence_level: f32, child: &Node<M>) -> f32 {
    let mean = child.q as f32 / child.n as f32;
    let std_dev = (child.q_squared as f32 / child.n as f32 - mean.powi(2)).sqrt();
    let normal = Normal::new(mean, std_dev).unwrap();
    normal.inverse_cdf(confidence_level)
}
*/

impl SelectionStrategy {
    fn score<M>(&self, parent: &Node<M>, child: &Node<M>) -> f32 {
        match self {
            SelectionStrategy::Max => child.q as f32,
            SelectionStrategy::Robust => child.n as f32,
            SelectionStrategy::UCT(c) => uct(*c, parent, child),
        }
    }
}

// TODO: I'd like to make this more type safe with an enum
pub(crate) struct Node<M> {
    q: i32,
    n: u32,
    action: M,
    unexplored: Vec<M>,
    is_terminal: bool, // ignored when ExpansionStrategy is Full
}
// rave_q: HashMap<M, i32>,
// rave_n: HashMap<M, u32>,

impl<M> Node<M> {
    #[inline]
    fn new(m: M, unexplored: Vec<M>) -> Self {
        Node {
            q: 0,
            n: 0,
            action: m,
            unexplored,
            is_terminal: false,
            // rave_q: HashMap::new(),
            // rave_n: HashMap::new(),
        }
    }

    #[inline]
    fn update(&mut self, reward: i32 /* actions: &[M]*/) {
        self.q += reward;
        self.n += 1;

        // for action in actions {
        //     *self.rave_q.entry(action.clone()).or_insert(0) += reward;
        //     *self.rave_n.entry(action.clone()).or_insert(0) += 1;
        // }
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
    pub action_selection_strategy: SelectionStrategy,
    pub tree_selection_strategy: SelectionStrategy,
    pub expansion_strategy: ExpansionStrategy,
    pub rollouts_before_expanding: u32,
    pub max_rollouts: u32,
    pub verbose: bool,
}

impl Config {
    fn new() -> Self {
        Self {
            rng: Rng::from_entropy(),
            max_time: Duration::from_secs(5),
            action_selection_strategy: SelectionStrategy::UCT(2.0_f32.sqrt()),
            tree_selection_strategy: SelectionStrategy::UCT(2.0_f32.sqrt()),
            expansion_strategy: ExpansionStrategy::Single,
            rollouts_before_expanding: 5,
            max_rollouts: u32::MAX,
            verbose: false,
        }
    }
}

pub struct TreeSearch<G: Game> {
    index: Index<G::M>,
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
            index: Index::new(),
            pv: Vec::new(),
            timeout: Arc::new(AtomicBool::new(false)),
            config: Config::new(),
        }
    }

    #[inline]
    fn best_child(&mut self, strategy: SelectionStrategy, node_id: NodeRef) -> Option<NodeRef> {
        let parent = self.index.get(node_id);
        let children = self.index.children(node_id).collect::<Vec<_>>();
        if children.is_empty() {
            return None;
        }
        random_best(children.as_slice(), &mut self.config.rng, |child_id| {
            let child = self.index.get(*child_id);
            strategy.score(parent, child)
        })
        .cloned()
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

    fn select(&mut self, mut node_id: NodeRef, init_state: &G::S) -> (NodeRef, G::S) {
        let mut state = init_state.clone();
        loop {
            let node = self.index.get(node_id);
            if self.is_terminal(node, &state) {
                return (node_id, state.clone());
            }

            let is_leaf = self.index.children(node_id).count() == 0;
            let needs_rollouts = node.n <= self.config.rollouts_before_expanding;
            let unexplored =
                self.config.expansion_strategy.is_single() && !node.unexplored.is_empty();

            if is_leaf || needs_rollouts || unexplored {
                if !needs_rollouts {
                    // Perform expansion
                    return self.expand(node_id, &state);
                } else {
                    return (node_id, state.clone());
                }
            } else {
                let child_id = self
                    .best_child(self.config.tree_selection_strategy, node_id)
                    .unwrap();
                let child = self.index.get(child_id);
                state = G::apply(&state, child.action.clone());
                node_id = child_id;
            }
        }
    }

    #[inline]
    fn expand(&mut self, node_id: NodeRef, init_state: &G::S) -> (NodeRef, G::S) {
        debug_assert!(!G::is_terminal(init_state));
        match self.config.expansion_strategy {
            ExpansionStrategy::Single => self.expand_single(node_id, init_state),
            ExpansionStrategy::Full => self.expand_full(node_id, init_state),
        }
    }

    fn expand_full(&mut self, node_id: NodeRef, init_state: &G::S) -> (NodeRef, G::S) {
        let child_id = G::gen_moves(init_state)
            .iter()
            .map(|m| {
                self.index
                    .add_child(node_id, Node::new(m.clone(), Vec::new()))
            })
            .collect::<Vec<_>>()[0];

        let child = self.index.get(child_id);
        let state = G::apply(init_state, child.action.clone());
        (child_id, state)
    }

    fn expand_single(&mut self, node_id: NodeRef, init_state: &G::S) -> (NodeRef, G::S) {
        let node = self.index.get_mut(node_id);
        if let Some(action) = node.unexplored.pop() {
            let state = G::apply(init_state, action.clone());
            let is_terminal = G::is_terminal(&state);
            let moves = if is_terminal {
                vec![]
            } else {
                G::gen_moves(&state)
            };
            let child_id = self.index.add_child(
                node_id,
                Node {
                    is_terminal,
                    ..Node::new(action.clone(), moves)
                },
            );
            (child_id, state)
        } else {
            error!("No unexplored actions left for this node: {:?}", node_id);
            error!("state: {:?}", init_state);
            (node_id, init_state.clone())
        }
    }

    fn step(&mut self, root_id: NodeRef, init_state: &G::S) {
        let (node_id, state) = self.select(root_id, init_state);
        let reward = self.simulate(node_id, init_state, &state);
        self.backpropagate(node_id, reward);
    }

    // TODO: move to a separate Rollout<S: Strategy>, noting that RAVE needs the PV
    #[inline]
    fn simulate(&mut self, node_id: NodeRef, init_state: &G::S, state: &G::S) -> i32 {
        debug_assert!(self.index.children(node_id).count() == 0);
        if self.index.get(node_id).is_terminal {
            return G::get_reward(init_state, state);
        }

        let mut state = state.clone();
        loop {
            if G::is_terminal(&state) {
                return G::get_reward(init_state, &state);
            }
            state = G::apply(
                &state,
                G::gen_moves(&state)
                    .choose(&mut self.config.rng)
                    .unwrap()
                    .clone(),
            );
        }
    }

    fn backpropagate(&mut self, node_id: NodeRef, reward: i32 /* actions */) {
        let mut node_id_opt = Some(node_id);
        loop {
            match node_id_opt {
                None => break,
                Some(node_id) => {
                    let node = self.index.get_mut(node_id);
                    node.update(reward /*, actions */);
                    node_id_opt = self.index.get_parent(node_id);
                }
            }
        }
    }
    pub fn choose_move(&mut self, state: &G::S) -> Option<G::M> {
        let timer = self.start_timer();
        self.index.clear();
        let root = match self.config.expansion_strategy {
            ExpansionStrategy::Full => Node::new(G::empty_move(state), Vec::new()),
            ExpansionStrategy::Single => Node::new(G::empty_move(state), G::gen_moves(state)),
        };

        let root_id = self.index.add(root);

        for _ in 0..self.config.max_rollouts {
            if self.timeout.load(Ordering::Relaxed) {
                break;
            }
            self.step(root_id, state);
        }

        self.set_pv(root_id);
        self.verbose_summary(root_id, &timer, state);
        self.best_child(self.config.action_selection_strategy, root_id)
            .map(|child_id| self.index.get(child_id).action.clone())
    }

    fn start_timer(&mut self) -> Timer {
        self.timeout = if self.config.max_time == Duration::default() {
            Arc::new(AtomicBool::new(false))
        } else {
            timeout_signal(self.config.max_time)
        };

        Timer::new()
    }

    fn set_timeout(&mut self, timeout: std::time::Duration) {
        self.config.max_rollouts = u32::MAX;
        self.config.max_time = timeout;
    }

    fn set_max_rollouts(&mut self, max_rollouts: u32) {
        self.config.max_time = Duration::default();
        self.config.max_rollouts = max_rollouts;
    }

    fn set_max_depth(&mut self, depth: u8) {
        // Set some arbitrary function of rollouts.
        self.config.max_time = Duration::default();
        self.config.max_rollouts = 5u32
            .saturating_pow(depth as u32)
            .saturating_mul(self.config.rollouts_before_expanding + 1);
    }

    fn verbose_summary(&self, root_id: NodeRef, timer: &Timer, state: &G::S) {
        if !self.config.verbose {
            return;
        }
        let num_threads = 1;
        let root = self.index.get(root_id);
        let total_visits = root.n;
        let rate = total_visits as f64 / num_threads as f64 / timer.elapsed().as_secs_f64();
        eprintln!(
            "Using {} threads, did {} total simulations with {:.1} rollouts/sec/core",
            num_threads, total_visits, rate
        );
        // Sort moves by visit count, largest first.
        let mut children = self
            .index
            .children(root_id)
            .map(|node_id| {
                let node = self.index.get(node_id);
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
            .map(|node_id| self.index.get(*node_id).action.clone())
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

    fn set_max_depth(&mut self, depth: u8) {
        self.0.set_max_depth(depth);
    }

    fn set_max_rollouts(&mut self, max_rollouts: u32) {
        self.0.set_max_rollouts(max_rollouts);
    }

    fn set_verbose(&mut self) {
        self.0.config.verbose = true;
    }

    fn principal_variation(&self) -> Vec<G::M> {
        unimplemented!();
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
        let init_node_id = t.index.add(init_node);

        let init_state = HashedPosition::new();

        //  Check if the function returns the correct node when the node has not
        // been expanded yet
        let (node_id, state) = t.select(init_node_id, &init_state);
        assert_eq!(node_id, init_node_id);
        assert_eq!(state, init_state);
    }

    #[test]
    fn test_select() {
        let mut t: TreeSearch<TicTacToe> = Default::default();

        let node = Node::new(Move(0), Vec::new());
        let node_id = t.index.add(node);

        let state = HashedPosition::new();

        //  Check if the function returns the correct node when the node has not
        // been expanded yet
        let (selected_id, selected_state) = t.select(node_id, &state);
        assert_eq!(node_id, selected_id);
        assert_eq!(state, selected_state);
    }
}
