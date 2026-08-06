use super::*;
use crate::game::Action;

use std::str::FromStr;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::*;
use std::sync::OnceLock;
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, Default)]
pub struct ActionStats {
    pub num_visits: u32,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct PlayerStats {
    pub score: f64,
    pub sum_squared_score: f64,
    pub amaf: ActionStats,
}

impl Default for PlayerStats {
    fn default() -> Self {
        Self {
            score: 0.,
            sum_squared_score: 0.,
            amaf: ActionStats::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseQInitError;

impl FromStr for QInit {
    type Err = ParseQInitError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Draw" => Ok(QInit::Draw),
            "Infinity" => Ok(QInit::Infinity),
            "Loss" => Ok(QInit::Loss),
            "Parent" => Ok(QInit::Parent),
            "Win" => Ok(QInit::Win),
            _ => Err(ParseQInitError),
        }
    }
}

/// QInit is an unvisited value estimate, the Q value assigned to a node
/// that has not been expanded or explored. The choice of a default unvisited
/// child value will bias the search. Choosing win, loss, or draw can prompt
/// an optimistic (greedy) or pessimistic move selection. Using the parent's
/// value is a common approach and the default used here. Using infinity will
/// encourage exploration of unvisited child nodes.
///
/// TODO: there are other strategies we could employ:
///
///   - Average: the average value from historical outcomes in simulation in this
///     subtree. This increases the memory requirement but is a middle ground
///     compared to setting the expansion threshold to 0.
///
///   - Custom: the client could provide an implementation rather than coupling
///     this to the implementation of `SelectStratey`.
#[allow(unused)]
#[derive(Clone, Copy, Default)]
pub enum QInit {
    #[default]
    Parent,
    Win,
    Loss,
    Draw,
    Infinity,
}

#[derive(Clone, Debug)]
pub struct Edge<A: Action> {
    pub action: A,
    node_id: OnceLock<index::Id>,
    pub stats: NodeStats,
}

/// The mutable core of `NodeStats`: everything backprop accumulates into.
/// Kept behind one lock so a single write covers a whole `update` call.
#[derive(Debug, Clone)]
struct NodeStatsData {
    num_visits: u32,
    player: Vec<PlayerStats>,
}

#[derive(Debug)]
pub struct NodeStats {
    // For virtual loss -- lock-free since it's touched on every descent/
    // backprop, the hottest path in tree-parallel search.
    pub num_visits_virtual: AtomicU32,

    data: RwLock<NodeStatsData>,
}

impl Clone for NodeStats {
    fn clone(&self) -> Self {
        let data = self.data.read().unwrap();
        Self {
            num_visits_virtual: AtomicU32::new(self.num_visits_virtual.load(Relaxed)),
            data: RwLock::new(data.clone()),
        }
    }
}

impl<A: Action> Edge<A> {
    pub fn is_explored(&self) -> bool {
        self.node_id.get().is_some()
    }

    pub fn node_id(&self) -> Option<index::Id> {
        self.node_id.get().copied()
    }

    /// Resolves the edge-creation race: if two threads land on this
    /// unexplored edge concurrently, only the first `create` closure to
    /// arrive actually runs (allocating a new arena node); the other blocks
    /// and then reads the same `Id` back, rather than both creating separate
    /// children for the same edge.
    pub fn get_or_create_child(&self, create: impl FnOnce() -> index::Id) -> index::Id {
        *self.node_id.get_or_init(create)
    }

    pub fn unexplored(action: A, num_players: usize) -> Edge<A> {
        Self {
            action,
            node_id: OnceLock::new(),
            stats: NodeStats::new(num_players),
        }
    }
}

impl NodeStats {
    pub fn new(num_players: usize) -> Self {
        Self {
            num_visits_virtual: AtomicU32::new(0),
            data: RwLock::new(NodeStatsData {
                num_visits: 0,
                player: vec![PlayerStats::default(); num_players],
            }),
        }
    }

    pub fn num_visits(&self) -> u32 {
        self.data.read().unwrap().num_visits
    }

    pub fn score(&self, player_index: usize) -> f64 {
        self.data.read().unwrap().player[player_index].score
    }

    pub fn sum_squared_score(&self, player_index: usize) -> f64 {
        self.data.read().unwrap().player[player_index].sum_squared_score
    }

    pub fn amaf(&self, player_index: usize) -> ActionStats {
        self.data.read().unwrap().player[player_index].amaf
    }

    pub fn total_visits(&self) -> u32 {
        self.num_visits() + self.num_visits_virtual.load(Relaxed)
    }

    /// Marks this edge as "in flight" for a concurrent tree-parallel search:
    /// a thread has committed to this path but hasn't backpropagated a result
    /// yet, so other threads scoring the same edge see it as worse/busier
    /// than its real stats suggest. Must be paired with `remove_virtual_loss`
    /// once that thread's simulation result is backpropagated.
    pub fn add_virtual_loss(&self) {
        self.num_visits_virtual.fetch_add(1, Relaxed);
    }

    pub fn remove_virtual_loss(&self) {
        let prev = self.num_visits_virtual.fetch_sub(1, Relaxed);
        debug_assert!(prev >= 1, "virtual loss removed without a matching add");
    }

    pub fn update(&self, utilities: &[f64]) {
        let mut data = self.data.write().unwrap();
        data.num_visits += 1;
        utilities.iter().enumerate().for_each(|(p, reward)| {
            data.player[p].score += reward;
            data.player[p].sum_squared_score += utilities[p] * utilities[p];
        });
    }

    pub fn add_amaf(&self, player_index: usize, utility: f64) {
        let mut data = self.data.write().unwrap();
        let amaf = &mut data.player[player_index].amaf;
        amaf.num_visits += 1;
        amaf.score += utility;
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn expected_score(&self, player_index: usize) -> f64 {
        let data = self.data.read().unwrap();
        if data.num_visits == 0 {
            0.
        } else {
            let loss_visits = self.num_visits_virtual.load(Relaxed) as f64;

            (data.player[player_index].score - loss_visits) / (data.num_visits as f64 + loss_visits)
        }
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn exploitation_score(&self, player_index: usize) -> f64 {
        self.expected_score(player_index)
    }

    // These numbers come from Ludii
    pub fn value_estimate_unvisited(&self, player_index: usize, q_init: QInit) -> f64 {
        use QInit::*;
        match q_init {
            Draw => 0.,
            Infinity => 10000.0,
            Loss => -1.,
            Parent => {
                if self.num_visits() == 0 {
                    10000.
                } else {
                    self.expected_score(player_index)
                }
            }
            Win => 1.,
        }
    }
}

#[derive(Clone, Debug)]
pub enum NodeState<A: Action> {
    Terminal,
    // NOTE: this Vec necessitates O(n) lookups. Consider FxHashMap
    Expanded(Vec<Edge<A>>),
}

#[derive(Clone, Debug)]
pub struct Node<A: Action> {
    pub player_idx: usize,
    pub hash: u64,
    pub is_root: bool,
    // Unset == not yet expanded ("leaf"). `expand` resolves this exactly
    // once via `get_or_init`, so concurrent threads landing on the same
    // unexpanded node race for free: only the winner runs `G::is_terminal`/
    // `generate_actions`, the rest block and read its result.
    state: OnceLock<NodeState<A>>,
}

impl<A: Action> Node<A>
where
    A: Clone + std::hash::Hash,
{
    pub fn new(player_idx: usize, hash: u64) -> Self {
        Self {
            player_idx,
            hash,
            is_root: false,
            state: OnceLock::new(),
        }
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(self.state.get(), Some(NodeState::Terminal))
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.state.get().is_none()
    }

    #[inline]
    pub fn is_expanded(&self) -> bool {
        matches!(self.state.get(), Some(NodeState::Expanded { .. }))
    }

    /// A single, self-consistent snapshot of this node's Leaf/Terminal/
    /// Expanded status. Callers that need to branch on more than one
    /// aspect of this status (e.g. "is it terminal" and, separately, "is it
    /// still a leaf") MUST derive both from one call to this method rather
    /// than calling `is_terminal()`/`is_leaf()` back to back: those are
    /// each their own independent `OnceLock::get()` read, and under tree
    /// parallelism a concurrent `expand()` elsewhere -- e.g. a transposed
    /// node shared with another thread's path -- can resolve Leaf ->
    /// Terminal in the gap between two such reads. Each individual read is
    /// locally correct at the instant it happens, but the combination can
    /// fall through both branches: `is_terminal()` (checked first) sees the
    /// still-unresolved leaf and returns `false`, then `is_leaf()` (checked
    /// moments later) sees the now-resolved node and *also* returns
    /// `false`, leaving neither branch's handling applied to a node that's
    /// actually `Terminal` -- which then panics the first time something
    /// calls `edges()` on it.
    #[inline]
    pub fn status(&self) -> Option<&NodeState<A>> {
        self.state.get()
    }

    /// Resolves this node's Leaf -> {Terminal, Expanded} transition exactly
    /// once (see the `state` field doc comment).
    pub fn expand(&self, init: impl FnOnce() -> NodeState<A>) -> &NodeState<A> {
        self.state.get_or_init(init)
    }

    #[inline]
    pub fn edges(&self) -> &Vec<Edge<A>> {
        let Some(NodeState::Expanded(edges)) = self.state.get() else {
            unreachable!()
        };
        edges
    }

    pub fn child_edge(&self, child_id: index::Id) -> &Edge<A> {
        // NOTE: O(n) lookup
        self.edges()
            .iter()
            .find(|e| e.node_id() == Some(child_id))
            .unwrap()
    }

    pub fn actions(&self) -> Vec<A> {
        self.edges()
            .iter()
            .map(|edge| edge.action.clone())
            .collect()
    }

    pub fn node_ids(&self) -> Vec<Option<index::Id>> {
        self.edges().iter().map(|edge| edge.node_id()).collect()
    }

    pub fn new_root(player: usize, num_players: usize, hash: u64) -> Self {
        debug_assert!((num_players == 0 && player == 0) || player < num_players);
        Self {
            is_root: true,
            ..Self::new(player, hash)
        }
    }

    pub fn update(&self, action_idx: usize, utilities: &[f64]) {
        self.edges()[action_idx].stats.update(utilities);
    }

    pub fn is_root(&self) -> bool {
        self.is_root
    }
}
