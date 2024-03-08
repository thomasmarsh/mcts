use super::*;
use crate::game::Action;
use crate::game::HashKey;
use crate::game::PlayerIndex;

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use serde::Serialize;
use std::str::FromStr;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::*;

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ActionStats {
    pub num_visits: u32,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Clone, Serialize, Debug)]
pub struct Edge<A: Action> {
    pub node_id: Option<index::Id>,
    pub action: A,
    pub stats: NodeStats,
}

#[derive(Serialize, Debug)]
pub struct NodeStats {
    pub num_visits: u32,

    // For virtual loss
    pub num_visits_virtual: AtomicU32,

    pub player: Vec<PlayerStats>,
}

impl Clone for NodeStats {
    fn clone(&self) -> Self {
        Self {
            num_visits: self.num_visits,
            num_visits_virtual: AtomicU32::new(self.num_visits_virtual.load(Relaxed)),
            player: self.player.clone(),
        }
    }
}

impl<A: Action> Edge<A> {
    pub fn is_explored(&self) -> bool {
        self.node_id.is_some()
    }

    pub fn unexplored(action: A, num_players: usize) -> Edge<A> {
        Self {
            action,
            node_id: None,
            stats: NodeStats::new(num_players),
        }
    }
}

impl NodeStats {
    pub fn new(num_players: usize) -> Self {
        Self {
            num_visits: 0,
            num_visits_virtual: AtomicU32::new(0),
            player: vec![PlayerStats::default(); num_players],
        }
    }

    pub fn total_visits(&self) -> u32 {
        self.num_visits + self.num_visits_virtual.load(Relaxed)
    }

    pub fn update(&mut self, utilities: &[f64]) {
        self.num_visits += 1;
        utilities.iter().enumerate().for_each(|(p, reward)| {
            self.player[p].score += reward;
            self.player[p].sum_squared_score += utilities[p] * utilities[p];
        });
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn expected_score(&self, player_index: PlayerIndex) -> f64 {
        if self.num_visits == 0 {
            0.
        } else {
            let loss_visits = self.num_visits_virtual.load(Relaxed) as f64;

            (self.player[player_index].score - loss_visits) / (self.num_visits as f64 + loss_visits)
        }
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn exploitation_score(&self, player_index: PlayerIndex) -> f64 {
        self.expected_score(player_index)
    }

    // These numbers come from Ludii
    pub fn value_estimate_unvisited(&self, player_index: PlayerIndex, q_init: QInit) -> f64 {
        use QInit::*;
        match q_init {
            Draw => 0.,
            Infinity => 10000.0,
            Loss => -1.,
            Parent => {
                if self.num_visits == 0 {
                    10000.
                } else {
                    self.expected_score(player_index)
                }
            }
            Win => 1.,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Edges<A: Action> {
    edges: Vec<Edge<A>>,
    explored: FxHashMap<index::Id, usize>,
}

impl<A: Action> Edges<A> {
    pub fn new_unexplored(edges: Vec<Edge<A>>) -> Self {
        Self {
            edges,
            explored: FxHashMap::default(),
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Edge<A>> {
        self.edges.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Edge<A>> {
        self.edges.iter_mut()
    }

    pub fn set_child(&mut self, index: usize, node_id: index::Id) {
        debug_assert!(self.edges[index].node_id.is_none());
        self.edges[index].node_id = Some(node_id);
        // debug_assert!(!self.explored.contains_key(&node_id));
        self.explored.insert(node_id, index);
    }

    pub fn get(&self, node_id: index::Id) -> Option<&Edge<A>> {
        self.explored.get(&node_id).map(|index| &self.edges[*index])
    }

    pub fn get_mut(&mut self, node_id: index::Id) -> Option<&mut Edge<A>> {
        self.explored
            .get_mut(&node_id)
            .map(|index| &mut self.edges[*index])
    }

    pub fn as_slice(&self) -> &[Edge<A>] {
        self.edges.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [Edge<A>] {
        self.edges.as_mut_slice()
    }
}

#[derive(Clone, Debug, Serialize)]
pub enum NodeState<A: Action> {
    Terminal,
    Leaf,
    Expanded(Edges<A>),
}

#[derive(Clone, Debug, Serialize)]
pub struct Node<A: Action, K: HashKey> {
    pub ancestors: FxHashSet<index::Id>,
    pub player_idx: PlayerIndex,
    pub state: NodeState<A>,
    pub hash: K,
    pub aggregate_stats: NodeStats,
    pub is_root: bool,
}

impl<A: Action, K: HashKey> Node<A, K> {
    pub fn new(
        parent: Option<index::Id>,
        num_players: usize,
        player_idx: PlayerIndex,
        hash: K,
    ) -> Self {
        Self {
            ancestors: parent.iter().cloned().collect(),
            player_idx,
            state: NodeState::Leaf,
            hash,
            aggregate_stats: NodeStats::new(num_players),
            is_root: false,
        }
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(&self.state, NodeState::Terminal)
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        matches!(&self.state, NodeState::Leaf)
    }

    #[inline]
    pub fn is_expanded(&self) -> bool {
        matches!(&self.state, NodeState::Expanded { .. })
    }

    #[inline]
    pub fn edges(&self) -> &Edges<A> {
        let NodeState::Expanded(edges) = &self.state else {
            unreachable!()
        };
        edges
    }

    #[inline]
    pub fn edges_mut(&mut self) -> &mut Edges<A> {
        let NodeState::Expanded(edges) = &mut self.state else {
            unreachable!()
        };
        edges
    }

    pub fn actions(&self) -> Vec<A> {
        self.edges()
            .iter()
            .map(|edge| edge.action.clone())
            .collect()
    }

    pub fn node_ids(&self) -> Vec<Option<index::Id>> {
        self.edges().iter().map(|edge| edge.node_id).collect()
    }

    pub fn new_root(player: PlayerIndex, num_players: usize, hash: K) -> Self {
        debug_assert!((num_players == 0 && player.0 == 0) || player.0 < num_players);
        Self {
            is_root: true,
            ..Self::new(None, num_players, player, hash)
        }
    }

    pub fn update(&mut self, action_idx: usize, utilities: &[f64]) {
        self.edges_mut().as_mut_slice()[action_idx]
            .stats
            .update(utilities);
    }

    pub fn is_root(&self) -> bool {
        self.is_root
    }
}
