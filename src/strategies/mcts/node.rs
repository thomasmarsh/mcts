use serde::Serialize;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::*;

use crate::game::Action;

use super::*;

#[derive(Debug, Clone, Default, Serialize)]
pub struct ActionStats {
    pub num_visits: u32,
    pub score: f64,
}

#[derive(Debug, Serialize)]
pub struct NodeStats<A: Action> {
    pub num_visits: u32,

    // For virtual loss
    pub num_visits_virtual: AtomicU32,

    pub scores: Vec<f64>,

    // Only used for UCB1Tuned; how to parameterize
    pub sum_squared_scores: Vec<f64>,

    // TODO: what is this actually? I can't find this in the literature, but it's
    // like a coarser version of RAVE/AMAF.
    pub scalar_amaf: ActionStats,

    // TODO: Only used for GRAVE; how to parameterize
    #[serde(skip_serializing)]
    pub grave_stats: HashMap<A, ActionStats>,
}

impl<A: Action> Clone for NodeStats<A> {
    fn clone(&self) -> Self {
        Self {
            num_visits: self.num_visits,
            num_visits_virtual: AtomicU32::new(self.num_visits_virtual.load(Relaxed)),
            scores: self.scores.clone(),
            sum_squared_scores: self.sum_squared_scores.clone(),
            scalar_amaf: self.scalar_amaf.clone(),
            grave_stats: self.grave_stats.clone(),
        }
    }
}

impl<A: Action> NodeStats<A> {
    pub fn new(num_players: usize) -> Self {
        Self {
            num_visits: 0,
            num_visits_virtual: AtomicU32::new(0),
            scores: vec![0.; num_players],
            sum_squared_scores: vec![0.; num_players],
            scalar_amaf: Default::default(),
            grave_stats: Default::default(),
        }
    }

    pub fn update(&mut self, utilities: &[f64]) {
        self.num_visits += 1;
        utilities.iter().enumerate().for_each(|(p, reward)| {
            self.scores[p] += reward;
            self.sum_squared_scores[p] += utilities[p] * utilities[p];
        });
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn expected_score(&self, player_index: usize) -> f64 {
        if self.num_visits == 0 {
            0.
        } else {
            let loss_visits = self.num_visits_virtual.load(Relaxed) as f64;

            (self.scores[player_index] - loss_visits) / (self.num_visits as f64 + loss_visits)
        }
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn exploitation_score(&self, player_index: usize) -> f64 {
        self.expected_score(player_index)
    }

    // These numbers come from Ludii
    pub fn value_estimate_unvisited(
        &self,
        player_index: usize,
        q_init: UnvisitedValueEstimate,
    ) -> f64 {
        use UnvisitedValueEstimate::*;
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

// QInit:
// - MC-GRAVE: Infinity
// - MC-BRAVE: Infinity
// - UCB1Tuned: Parent
// - ScoreBounded: Parent
// - ProgressiveHistory: Parent
// - ProgressiveBias: Infinity
// - MAST: Parent
// - NST: Parent
// - UCB1GRAVE: default
// ===
// - default: Parent
// - createBiased: Win
#[allow(unused)]
#[derive(Clone, Copy)]
pub enum UnvisitedValueEstimate {
    Draw,
    Infinity,
    Loss,
    Parent,
    Win,
}

impl Default for UnvisitedValueEstimate {
    fn default() -> Self {
        Self::Parent
    }
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
    pub stats: NodeStats<A>,
    pub state: NodeState<A>,
}

impl<A: Action> Node<A>
where
    A: Clone + std::hash::Hash,
{
    pub fn new(parent_id: index::Id, action_idx: usize, num_players: usize) -> Self {
        Self {
            parent_id,
            action_idx,
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

    pub fn new_root(num_players: usize) -> Self {
        Self::new(index::Id::invalid_id(), usize::MAX, num_players)
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
