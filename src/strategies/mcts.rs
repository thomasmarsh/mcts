use log::error;
use rand::seq::SliceRandom;
use rustc_hash::FxHashMap as HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use super::arena::{Arena, Ref};
use super::config::*;
use super::sync_util::{timeout_signal, Timer};
use super::Strategy;
use crate::game::{Game, ZobristHash};
use crate::util::random_best;
use crate::util::{move_id, pv_string};

#[inline]
fn uct<G: Game>(c: f64, rave_param: f64, parent: &Node<G>, child: &Node<G>) -> f64 {
    let epsilon = 1e-6;
    let w = child.q as f64;
    let n = child.n as f64 + epsilon;
    let total = parent.n as f64;

    let uct_value = w / n + c * (2. * total.ln() / n).sqrt();

    // TODO: also use player q/n_player_amaf values
    let rave_value = if child.n_all_amaf > 0 {
        child.q_all_amaf as f64 / child.n_all_amaf as f64
    } else {
        0.0
    };
    let beta = rave_param / (rave_param + n);

    (1. - beta) * uct_value + beta * rave_value
}

#[allow(dead_code)]
#[derive(Debug)]
struct TranspositionEntry {
    is_terminal: bool,
    node_id: Ref, // The first node which encountered this state
    access_count: u32,
}

// TODO: placeholder code
#[allow(dead_code)]
#[derive(Debug, Default)]
struct TranspositionTable(HashMap<ZobristHash, TranspositionEntry>);

#[allow(dead_code)]
impl TranspositionTable {
    fn get(&self, hash: ZobristHash) -> Option<&TranspositionEntry> {
        self.0.get(&hash)
    }
}

// Move Average Sampling Technique (incorrect placeholder implementation)
#[derive(Debug)]
struct Mast<M: std::hash::Hash + Eq + Clone> {
    action_value: HashMap<M, i32>,
    action_count: HashMap<M, u32>,
}

impl<M: std::hash::Hash + Eq + Clone> Mast<M> {
    fn new() -> Self {
        Self {
            action_value: HashMap::default(),
            action_count: HashMap::default(),
        }
    }

    fn update(&mut self, actions: &[M]) {
        actions.iter().for_each(|action| {
            let value = self.get_value(&action.clone()).round() as i32;
            self.action_count
                .entry(action.clone())
                .and_modify(|x| *x += 1)
                .or_insert(1);
            self.action_value
                .entry(action.clone())
                .and_modify(|x| *x += value)
                .or_insert(0);
        });
    }

    fn get_value(&self, action: &M) -> f64 {
        if !self.action_count.contains_key(action) || self.action_count[action] == 0 {
            0.
        } else {
            self.action_value[action] as f64 / self.action_count[action] as f64
        }
    }
}

impl SelectionStrategy {
    fn score<G: Game>(&self, parent: &Node<G>, child: &Node<G>) -> f64 {
        match self {
            SelectionStrategy::Max => child.q as f64,
            SelectionStrategy::Robust => child.n as f64,
            SelectionStrategy::UCT(c, rave_param) => uct(*c, *rave_param, parent, child),
        }
    }
}

// TODO: I'd like to make this more type safe with an enum
pub(crate) struct Node<G: Game> {
    q: i32,
    n: u32,
    #[allow(dead_code)]
    q_player_amaf: HashMap<G::P, i32>,
    #[allow(dead_code)]
    n_player_amaf: HashMap<G::P, i32>,
    q_all_amaf: i32,
    n_all_amaf: u32,
    action: G::M,
    is_terminal: bool, // ignored when ExpansionStrategy is Full
    unexplored: Vec<G::M>,
    children: Vec<Ref>,
}

impl<G: Game> std::fmt::Debug for Node<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node {{ q: {}, n: {}, q_amaf: {}, n_amaf: {}, is_terminal: {}, children: {:?}, unexplored: [...] (len={}) action: {{...}} }}", self.q, self.n, self.q_all_amaf, self.n_all_amaf, self.is_terminal, self.children, self.unexplored.len())
    }
}

impl<G: Game> Node<G> {
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    fn new_root(unexplored: Vec<G::M>) -> Self {
        Node {
            q: 0,
            n: 0,
            q_player_amaf: Default::default(),
            n_player_amaf: Default::default(),
            q_all_amaf: 0,
            n_all_amaf: 0,
            action: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            unexplored,
            is_terminal: false,
            children: Vec::new(),
        }
    }

    #[inline]
    fn new(m: G::M, unexplored: Vec<G::M>) -> Self {
        Node {
            q: 0,
            n: 0,
            q_player_amaf: Default::default(),
            n_player_amaf: Default::default(),
            q_all_amaf: 0,
            n_all_amaf: 0,
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

pub struct TreeSearch<G: Game> {
    arena: Arena<Node<G>>,
    pv: Vec<Ref>,
    timeout: Arc<AtomicBool>,
    mast: Mast<G::M>,
    #[allow(dead_code)]
    transpositions: TranspositionTable,
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
            mast: Mast::new(),
            transpositions: Default::default(),
            config: Config::new(),
        }
    }

    #[inline]
    fn best_child(&mut self, strategy: SelectionStrategy, node_id: Ref) -> Option<Ref> {
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

    fn set_pv(&mut self, mut node_id: Ref) {
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
    fn is_terminal(&self, node: &Node<G>, state: &G::S) -> bool {
        if self.config.expansion_strategy.is_single() {
            node.is_terminal
        } else {
            G::is_terminal(state)
        }
    }

    fn select(&mut self, mut node_id: Ref, init_state: &G::S) -> (Ref, G::S, Vec<Ref>) {
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
        node_id: Ref,
        init_state: &G::S,
        stack: Vec<Ref>,
    ) -> (Ref, G::S, Vec<Ref>) {
        debug_assert!(!G::is_terminal(init_state));
        match self.config.expansion_strategy {
            ExpansionStrategy::Single => self.expand_single(node_id, init_state, stack),
            ExpansionStrategy::Full => self.expand_full(node_id, init_state, stack),
        }
    }

    fn expand_full(
        &mut self,
        node_id: Ref,
        init_state: &G::S,
        mut stack: Vec<Ref>,
    ) -> (Ref, G::S, Vec<Ref>) {
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
        node_id: Ref,
        init_state: &G::S,
        mut stack: Vec<Ref>,
    ) -> (Ref, G::S, Vec<Ref>) {
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

    fn step(&mut self, root_id: Ref, init_state: &G::S) {
        let (node_id, state, stack) = self.select(root_id, init_state);
        let (reward, history) = self.simulate(node_id, init_state, &state);
        self.backpropagate(stack, reward, history);
    }

    // TODO: move to a separate Rollout<S: Strategy>, noting that RAVE needs the PV
    #[inline]
    fn simulate(&mut self, node_id: Ref, init_state: &G::S, state: &G::S) -> (i32, Vec<G::M>) {
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

            let m = if self.config.use_mast {
                random_best(
                    G::gen_moves(&state).as_slice(),
                    &mut self.config.rng,
                    |action| self.mast.get_value(action),
                )
                .cloned()
                .unwrap()
            } else {
                G::gen_moves(&state)
                    .choose(&mut self.config.rng)
                    .unwrap()
                    .clone()
            };

            self.config.use_rave.then(|| history.push(m.clone()));
            state = G::apply(&state, m);
            depth += 1;
            if depth >= self.config.max_simulate_depth {
                return (0, history);
            }
        }
    }

    fn update_rave(&mut self, node_id: Ref, reward: i32, history: &Vec<G::M>) {
        if !self.config.use_rave {
            return;
        }

        let child_ms: HashMap<_, _> = self
            .arena
            .get(node_id)
            .children
            .iter()
            .map(|child_id| (self.arena.get(*child_id).action.clone(), *child_id))
            .collect();

        for m in history {
            if let Some(child_id) = child_ms.get(m) {
                let child = self.arena.get_mut(*child_id);
                // TODO: add player to history
                child.q_all_amaf += reward;
                child.n_all_amaf += 1;
            }
        }
    }

    fn backpropagate(
        &mut self,
        // node_id: Ref,
        mut stack: Vec<Ref>,
        reward: i32,
        history: Vec<G::M>,
    ) {
        if self.config.use_mast {
            self.mast.update(&history);
        }

        while let Some(node_id) = stack.pop() {
            let node = self.arena.get_mut(node_id);
            node.update(reward);
            if self.config.use_mast {
                self.mast.update(&[node.action.clone()]);
            }
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

    fn verbose_summary(&self, root_id: Ref, timer: &Timer, state: &G::S) {
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

        eprintln!("{:?}", self.mast);
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
