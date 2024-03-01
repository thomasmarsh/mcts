use super::*;
use crate::game::Action;

use rustc_hash::FxHashMap as HashMap;
use serde::Serialize;
use std::ops::Add;
use std::str::FromStr;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::*;

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ActionStats {
    pub num_visits: u32,
    pub score: f64,
}

impl Add for ActionStats {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ActionStats {
            num_visits: self.num_visits + rhs.num_visits,
            score: self.score + rhs.score,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PlayerStats {
    pub score: f64,
    pub sum_squared_score: f64,
    pub amaf: ActionStats,
}

impl Add for PlayerStats {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            score: self.score + rhs.score,
            sum_squared_score: self.sum_squared_score + rhs.sum_squared_score,
            amaf: self.amaf + rhs.amaf,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct NodeStats<A: Action> {
    pub num_visits: u32,

    // For virtual loss
    pub num_visits_virtual: AtomicU32,

    pub player: Vec<PlayerStats>,

    // TODO: Only used for GRAVE; how to parameterize
    #[serde(skip_serializing)]
    pub grave_stats: HashMap<A, ActionStats>,
}

impl<A: Action> Clone for NodeStats<A> {
    fn clone(&self) -> Self {
        Self {
            num_visits: self.num_visits,
            num_visits_virtual: AtomicU32::new(self.num_visits_virtual.load(Relaxed)),
            player: self.player.clone(),
            grave_stats: self.grave_stats.clone(),
        }
    }
}

impl<A: Action> NodeStats<A> {
    pub fn new(num_players: usize) -> Self {
        Self {
            num_visits: 0,
            num_visits_virtual: AtomicU32::new(0),
            player: vec![PlayerStats::default(); num_players],
            grave_stats: Default::default(),
        }
    }

    pub fn update(&mut self, utilities: &[f64]) {
        self.num_visits += 1;
        utilities.iter().enumerate().for_each(|(p, reward)| {
            self.player[p].score += reward;
            self.player[p].sum_squared_score += utilities[p] * utilities[p];
        });
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn expected_score(&self, player_index: usize) -> f64 {
        if self.num_visits == 0 {
            0.
        } else {
            let loss_visits = self.num_visits_virtual.load(Relaxed) as f64;

            (self.player[player_index].score - loss_visits) / (self.num_visits as f64 + loss_visits)
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

impl<A: Action> Add for NodeStats<A> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        NodeStats {
            num_visits: self.num_visits + rhs.num_visits,
            num_visits_virtual: AtomicU32::new(
                self.num_visits_virtual.load(Relaxed) + rhs.num_visits_virtual.load(Relaxed),
            ),
            // TODO: group per-player stats to avoid N*M loops
            player: self
                .player
                .into_iter()
                .zip(rhs.player)
                .map(|(x, y)| x + y)
                .collect(),
            // NOTE: GRAVE is not currently supported with transpositions
            grave_stats: HashMap::default(),
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

#[derive(Clone, Debug, Serialize)]
pub enum NodeState<A: Action> {
    Terminal,
    Leaf,
    Expanded {
        children: Vec<Option<index::Id>>, // TODO: consider storing this in arena
        actions: Vec<A>,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Node<A: Action> {
    pub parent_id: index::Id, // TODO: consider storing this in arena
    pub action_idx: usize,
    pub player_idx: usize,
    pub stats: NodeStats<A>,
    pub state: NodeState<A>,
}

impl<A: Action> Node<A>
where
    A: Clone + std::hash::Hash,
{
    pub fn new(
        parent_id: index::Id,
        action_idx: usize,
        player_idx: usize,
        num_players: usize,
    ) -> Self {
        Self {
            parent_id,
            action_idx,
            player_idx,
            stats: NodeStats::new(num_players),
            state: NodeState::Leaf,
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
    pub fn children(&self) -> &Vec<Option<index::Id>> {
        // NOTE: unchecked
        let NodeState::Expanded { children, .. } = &self.state else {
            unreachable!()
        };
        children
    }

    #[inline]
    pub fn actions(&self) -> &Vec<A> {
        // NOTE: unchecked
        let NodeState::Expanded { actions, .. } = &self.state else {
            unreachable!()
        };
        actions
    }

    pub fn new_root(player: usize, num_players: usize) -> Self {
        debug_assert!((num_players == 0 && player == 0) || player < num_players);
        Self::new(index::Id::invalid_id(), usize::MAX, player, num_players)
    }

    pub fn update(&mut self, utilities: &[f64]) {
        self.stats.update(utilities);
    }

    pub fn action(&self, index: &TreeIndex<A>) -> A {
        match &(index.get(self.parent_id).state) {
            NodeState::Expanded { actions, .. } => actions[self.action_idx].clone(),
            _ => unreachable!(),
        }
    }

    pub fn is_root(&self) -> bool {
        self.parent_id == index::Id::invalid_id()
    }
}
