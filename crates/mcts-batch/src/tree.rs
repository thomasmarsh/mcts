//! A batch of MCTS trees stored as one structure of fixed-size arrays,
//! ported from AlphaZero.jl's `BatchedMcts.Tree` (Guillaume Thomas' 2022
//! GSoC "batched MCTS" work, `jonathan-laurent/AlphaZero.jl#147`): `B`
//! independent trees, advanced one simulation round at a time in lockstep,
//! so their leaf evaluations can be scored in one batched oracle call
//! instead of one call per leaf -- a different shape from the arena
//! `crates/mcts::algorithms::mcts::TreeSearch` already uses per game
//! (deliberately so; see this crate's own top-level docs).
//!
//! # Layout
//!
//! Node index `0` is reserved as a sentinel (`NO_PARENT`/`UNVISITED`/never a
//! real node); the root of every tree is node `1`. Real nodes occupy
//! `1..=num_simulations`. This mirrors the Julia source's own 1-indexed
//! `NO_PARENT = 0`/`ROOT = 1` convention exactly, rather than reusing node
//! `0` as both "the root" and "no such node" the way a 0-indexed port might
//! be tempted to -- keeping the sentinel numerically distinct from any real
//! node is what lets `parent[cid] != NO_PARENT` double as both "not the
//! root" and "a real link", matching every comparison in the source
//! algorithm without an off-by-one translation at each site.
//!
//! Arrays are flat `Vec`s in `(batch, node)` or `(batch, node, action)`
//! order -- row-major with the batch dimension slowest-varying, so each
//! batch item's entire per-node and per-action data is one contiguous
//! slice. `select`/`backpropagate` parallelize over the batch dimension
//! (`crate::search`), so this is the layout that gives each parallel task
//! its own contiguous memory, the Rust (row-major) analogue of the
//! source's own reasoning for preferring `(N, B)` in Julia's column-major
//! convention (see the source `Tree` docstring's "Remarks" section).
//!
//! Action ids are `0..num_actions` (unlike the Julia source's 1-indexed
//! actions); `NO_ACTION` is `u16::MAX`, a value no real action id can reach
//! for any game this crate targets (`num_actions` is always small -- 65 for
//! Othello).

pub const NO_PARENT: i32 = 0;
pub const UNVISITED: i32 = 0;
pub const ROOT: i32 = 1;
pub const NO_ACTION: u16 = u16::MAX;

pub struct Tree<Env> {
    pub(crate) num_actions: usize,
    /// Node capacity per batch item, including the reserved sentinel slot
    /// at index 0 -- `num_simulations + 1`.
    pub(crate) cap: usize,
    pub(crate) batch_size: usize,

    // Dynamic stats, (batch, node).
    pub(crate) parent: Vec<i32>,
    pub(crate) num_visits: Vec<i32>,
    pub(crate) total_values: Vec<f32>,
    // (batch, node, action).
    pub(crate) children: Vec<i32>,

    // Cached oracle output, (batch, node) unless noted.
    pub(crate) state: Vec<Env>,
    pub(crate) terminal: Vec<bool>,
    pub(crate) valid_actions: Vec<bool>, // (batch, node, action)
    pub(crate) prev_action: Vec<u16>,
    pub(crate) prev_reward: Vec<f32>,
    pub(crate) prev_switched: Vec<bool>,
    pub(crate) policy_prior: Vec<f32>, // (batch, node, action)
    pub(crate) value_prior: Vec<f32>,
}

impl<Env: Clone> Tree<Env> {
    #[inline]
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    #[inline]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Maximum number of simulations this tree was built for (its per-batch
    /// node capacity, minus the reserved sentinel slot).
    #[inline]
    pub fn num_simulations(&self) -> usize {
        self.cap - 1
    }

    #[inline]
    pub(crate) fn node_idx(&self, bid: usize, nid: i32) -> usize {
        bid * self.cap + nid as usize
    }

    #[inline]
    pub(crate) fn action_idx(&self, bid: usize, nid: i32, aid: u16) -> usize {
        self.node_idx(bid, nid) * self.num_actions + aid as usize
    }

    #[inline]
    pub fn state(&self, bid: usize, nid: i32) -> &Env {
        &self.state[self.node_idx(bid, nid)]
    }

    #[inline]
    pub fn is_terminal(&self, bid: usize, nid: i32) -> bool {
        self.terminal[self.node_idx(bid, nid)]
    }

    #[inline]
    pub fn is_valid_action(&self, bid: usize, nid: i32, aid: u16) -> bool {
        self.valid_actions[self.action_idx(bid, nid, aid)]
    }

    #[inline]
    pub fn child(&self, bid: usize, nid: i32, aid: u16) -> i32 {
        self.children[self.action_idx(bid, nid, aid)]
    }

    #[inline]
    pub fn prev_action(&self, bid: usize, nid: i32) -> u16 {
        self.prev_action[self.node_idx(bid, nid)]
    }

    #[inline]
    pub fn num_visits(&self, bid: usize, nid: i32) -> i32 {
        self.num_visits[self.node_idx(bid, nid)]
    }

    /// The root of the tree for a given batch item -- always node `ROOT`.
    #[inline]
    pub fn root(&self) -> i32 {
        ROOT
    }

    /// The absolute value of a node's game position: the mean of every
    /// value backpropagated through it (its own value prior included, via
    /// the `total_values`/`num_visits` initialization `create_tree` does).
    ///
    /// See AlphaZero.jl `BatchedMcts.value`.
    #[inline]
    pub fn value(&self, bid: usize, nid: i32) -> f32 {
        let i = self.node_idx(bid, nid);
        self.total_values[i] / self.num_visits[i] as f32
    }

    /// A node's value from its *parent's* perspective -- negated exactly
    /// when the mover switched on the way in, matching the nega convention
    /// `mcts::evaluator::Evaluator::evaluate` already uses in this
    /// workspace.
    ///
    /// See AlphaZero.jl `BatchedMcts.qvalue`.
    #[inline]
    pub fn qvalue(&self, bid: usize, nid: i32) -> f32 {
        let i = self.node_idx(bid, nid);
        if self.prev_switched[i] {
            -self.value(bid, nid)
        } else {
            self.value(bid, nid)
        }
    }

    /// A value estimate for `nid` blending its own value prior with its
    /// visited children's qvalues, weighted by prior and visit count --
    /// used as the "as if visited" value for `completed_qvalues`' currently
    /// -unvisited actions.
    ///
    /// See AlphaZero.jl `BatchedMcts.root_value_estimate` (not root-only
    /// despite the name -- called on any node, always with `nid` itself,
    /// matching the source).
    pub fn root_value_estimate(&self, bid: usize, nid: i32) -> f32 {
        let mut total_qvalues = 0.0f32;
        let mut total_prior = 0.0f32;
        let mut total_visits: i32 = 0;
        for aid in 0..self.num_actions as u16 {
            let cnid = self.child(bid, nid, aid);
            if cnid == UNVISITED {
                continue;
            }
            let prior = self.policy_prior[self.action_idx(bid, nid, aid)];
            total_qvalues += prior * self.qvalue(bid, cnid);
            total_prior += prior;
            total_visits += self.num_visits(bid, cnid);
        }
        let mut children_value = total_qvalues;
        if total_prior > 0.0 {
            children_value /= total_prior;
        }
        let i = self.node_idx(bid, nid);
        (self.value_prior[i] + total_visits as f32 * children_value) / (1 + total_visits) as f32
    }

    /// Every action's estimated qvalue from `nid`: a visited child's real
    /// `qvalue`, or `root_value_estimate(nid)` for an unvisited one --
    /// what makes the policy target meaningful at low simulation counts
    /// (Grill et al., "Monte-Carlo tree search as regularized policy
    /// optimization").
    ///
    /// See AlphaZero.jl `BatchedMcts.completed_qvalues` (per-node variant).
    pub fn completed_qvalues(&self, bid: usize, nid: i32) -> Vec<f32> {
        let root_value = self.root_value_estimate(bid, nid);
        (0..self.num_actions as u16)
            .map(|aid| {
                if !self.is_valid_action(bid, nid, aid) {
                    f32::NEG_INFINITY
                } else {
                    let cnid = self.child(bid, nid, aid);
                    if cnid != UNVISITED {
                        self.qvalue(bid, cnid)
                    } else {
                        root_value
                    }
                }
            })
            .collect()
    }

    /// Visit count of each of `nid`'s children, `0` for an unvisited action.
    ///
    /// See AlphaZero.jl `BatchedMcts.get_num_child_visits`.
    pub fn child_visits(&self, bid: usize, nid: i32) -> Vec<i32> {
        (0..self.num_actions as u16)
            .map(|aid| {
                let cnid = self.child(bid, nid, aid);
                if cnid != UNVISITED {
                    self.num_visits(bid, cnid)
                } else {
                    0
                }
            })
            .collect()
    }
}
