pub mod compact;
pub mod core;
pub mod multi_tree;
pub mod parallel;
pub(crate) mod report;
pub mod reroot;
pub mod reuse;
pub mod search_impl;
pub mod shared;

pub use core::MemoryStats;
pub use shared::SearchContext;
pub use shared::Shared;
pub use shared::TreeIndex;
pub use shared::TreeStats;

use super::config::SearchConfig;
use super::config::PolicyProfile;
use super::index;
use super::index::Id;
use super::node::Node;
use super::node::NodeStats;
use super::simulate::Trial;
use super::table::TranspositionTable;
use crate::game::Game;
use crate::algorithms::mcts::search::shared::ActionTotal;
use crate::timer;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchReportStart {
    pub started_at: Instant,
    pub tt_reads: usize,
    pub tt_writes: usize,
    pub tt_hits: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SearchRun<A> {
    pub elapsed_seconds: f64,
    pub tt_reads: usize,
    pub tt_writes: usize,
    pub tt_hits: usize,
    pub termination: crate::algorithms::SearchTermination,
    pub root_parallel: Option<RootParallelReport<A>>,
}

/// Final evidence aggregated from independent root-parallel worker trees.
/// There is deliberately no merged arena: the PV remains the best available
/// path from one contributing worker, while every numeric field is a sum (or
/// the appropriate aggregate) across all workers.
#[derive(Debug, Clone)]
pub(crate) struct RootParallelReport<A> {
    pub actions: Vec<ActionTotal<A>>,
    pub principal_variation: Vec<A>,
    pub completed_iterations: usize,
    pub tree_nodes: usize,
    pub accum_depth: usize,
    pub max_depth: usize,
    pub tt_reads: usize,
    pub tt_writes: usize,
    pub tt_hits: usize,
}

#[derive(Clone)]
pub struct TreeSearch<G, S>
where
    G: Game,
    S: PolicyProfile<G>,
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
    /// Evidence for the last completed `choose_action`; absent until an
    /// action has actually been selected.
    pub(crate) last_search_run: Option<SearchRun<G::A>>,
    /// Root-parallel aggregation produced during the current call. It is
    /// moved into `last_search_run` when the outer call finishes, and cleared
    /// before every new call so it cannot leak into a later serial search.
    pub(crate) root_parallel_report: Option<RootParallelReport<G::A>>,
    /// The root->leaf descent path from the most recent `select`/`select_step`
    /// call, as `(Id, idx)` pairs -- `idx` is the slot in the *previous*
    /// entry's `ChildArray` that was actually selected to reach this entry
    /// (unused placeholder for the root's own entry). Carried explicitly
    /// rather than a bare `Vec<Id>` because a `ChildArray`'s `id_index`
    /// reverse map can't disambiguate which slot was used once
    /// symmetry-aware graph merging lets several actions from one parent
    /// canonicalize to the same shared child (see `stack::StackEntry`'s doc
    /// comment).
    pub stack: Vec<(Id, usize)>,
    pub trial: Option<Trial<G>>,
}

impl<G, S> TreeSearch<G, S>
where
    G: Game,
    S: PolicyProfile<G>,
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
    S: PolicyProfile<G>,
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
    S: PolicyProfile<G>,
    SearchConfig<G, S>: Default,
    G::S: std::fmt::Display,
{
    pub fn new() -> Self {
        let config = S::config();
        let has_amaf = config.requirements().amaf;
        let index = index::Arena::new();
        let root_id = index.insert(Node::new_root(
            0,
            G::num_players(),
            0,
            has_amaf,
            config.use_mcts_solver,
        ));
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
            last_search_run: None,
            root_parallel_report: None,
        }
    }

    #[inline]
    pub fn new_root(&mut self, player_idx: usize, hash: u64) -> Id {
        let root = Node::new_root(
            player_idx,
            G::num_players(),
            hash,
            self.config.requirements().amaf,
            self.config.use_mcts_solver,
        );
        self.root_id = self.index.insert(root);
        self.root_id
    }
}
