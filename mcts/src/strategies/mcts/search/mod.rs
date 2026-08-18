pub mod compact;
pub mod core;
pub mod parallel;
pub mod reuse;
pub mod search_impl;
pub mod shared;

pub use core::MemoryStats;
pub use shared::SearchContext;
pub use shared::Shared;
pub use shared::TreeIndex;
pub use shared::TreeStats;

use super::config::SearchConfig;
use super::config::Strategy;
use super::index;
use super::index::Id;
use super::node::Node;
use super::node::NodeStats;
use super::simulate::Trial;
use super::table::TranspositionTable;
use crate::game::Game;
use crate::timer;

#[derive(Clone)]
pub struct TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    pub index: shared::TreeIndex<G::A>,
    pub timer: timer::Timer,
    pub root_id: Id,
    pub root_stats: NodeStats,
    pub pv: Vec<G::A>,
    pub table: TranspositionTable,
    /// The real state `root_id` represents, tracked purely so
    /// `reuse_or_reset` (`SearchConfig::reuse_tree`) can replay a candidate
    /// path and verify full state equality
    /// before promoting onto it -- a `Node` only stores its Zobrist hash,
    /// not its state, and a bare `u64` match isn't proof (a real, if
    /// astronomically rare, 64-bit collision would otherwise silently
    /// promote onto the wrong position and corrupt the whole tree). The
    /// transposition table accepts that same risk on every lookup (trusting
    /// the hash outright, see `table.rs`) since a false merge there just
    /// reuses one node's stats for two positions -- re-rooting onto the
    /// wrong position outright is a worse failure mode, so it gets this
    /// dedicated check instead of trusting the hash.
    /// `None` only before the very first `choose_action` call.
    pub root_state: Option<G::S>,

    pub config: SearchConfig<G, S>,
    pub stats: TreeStats<G>,
    pub stack: Vec<Id>,
    pub trial: Option<Trial<G>>,
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    G::S: std::fmt::Display,
{
    pub fn config(mut self, config: SearchConfig<G, S>) -> Self {
        self.config = config;
        self
    }
}

impl<G, S> Default for TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
    G::S: std::fmt::Display,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
    G::S: std::fmt::Display,
{
    pub fn new() -> Self {
        let config = S::config();
        let has_amaf = config.requirements().amaf;
        let index = index::Arena::new();
        let root_id = index.insert(Node::new_root(0, G::num_players(), 0, has_amaf));
        Self {
            root_id,
            root_stats: NodeStats::new(G::num_players(), has_amaf),
            root_state: None,
            pv: vec![],
            stack: vec![],
            table: TranspositionTable::default(),
            trial: None,
            index,
            config,
            timer: timer::Timer::new(),
            stats: Default::default(),
        }
    }

    #[inline]
    pub fn new_root(&mut self, player_idx: usize, hash: u64) -> Id {
        let root = Node::new_root(
            player_idx,
            G::num_players(),
            hash,
            self.config.requirements().amaf,
        );
        self.root_id = self.index.insert(root);
        self.root_id
    }
}
