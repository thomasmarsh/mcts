use crate::strategies::mcts::backprop::BackpropStrategy;
use crate::strategies::mcts::config::BackpropFlags;
use crate::strategies::mcts::index;
use crate::strategies::mcts::index::Id;
use crate::strategies::mcts::node;
use crate::strategies::mcts::node::Node;
use crate::strategies::mcts::node::NodeState;
use crate::strategies::mcts::node::NodeStats;
use crate::strategies::mcts::node::Proven;
use crate::strategies::mcts::node::Edge;
use crate::strategies::mcts::select::SelectContext;
use crate::strategies::mcts::select::SelectStrategy;
use crate::strategies::mcts::simulate::SimulateStrategy;
use crate::strategies::mcts::simulate::Trial;
use crate::strategies::mcts::stack::NodeStack;
use crate::strategies::mcts::table::TranspositionTable;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::TerminalStatus;

use rand::rngs::SmallRng;
use rustc_hash::FxHashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::RwLock;

pub struct SearchContext<G: Game> {
    pub current_id: Id,
    pub state: G::S,
}

impl<G: Game> SearchContext<G> {
    pub fn new(current_id: Id, state: G::S) -> Self {
        Self { current_id, state }
    }

    #[inline]
    fn traverse_apply(&mut self, child_id: Id, action: &G::A) {
        self.traverse(child_id);
        self.state = G::apply(self.state.clone(), action);
    }

    #[inline]
    fn traverse(&mut self, child_id: Id) {
        self.current_id = child_id;
    }
}

/// Per-hash, per-player action stats -- GRAVE's accumulator, keyed by the
/// hash of the subtree root the stats were collected under.
type GraveTable<A> = FxHashMap<u64, Vec<FxHashMap<A, node::ActionStats>>>;

/// GRAVE/global-MAST accumulators, read during select/simulate and written
/// during backprop. Each map gets its own lock (rather than one lock over
/// the whole struct) so, e.g., a GRAVE reader in `select` doesn't contend
/// with an unrelated MAST reader in `simulate`. `accum_depth`/`iter_count`
/// are plain atomics since they're incremented once per iteration and never
/// read alongside the maps.
/// Per-player bigram table: `(prev_action, action) -> ActionStats`, keyed by
/// the mover of `action` (not `prev_action`, which may belong to either
/// player) -- mirrors `player_actions`'s per-mover indexing. NST's context is
/// scoped to the current search only (the tree root's own predecessor is
/// unknown -- see `Nst`'s doc comment), so this never carries a real
/// game-history action in from outside the tree being searched.
type BigramTable<A> = FxHashMap<(A, A), node::ActionStats>;

#[derive(Debug)]
pub struct TreeStats<G: Game> {
    pub actions: RwLock<FxHashMap<G::A, node::ActionStats>>,
    pub grave: RwLock<GraveTable<G::A>>,
    pub player_actions: Vec<RwLock<FxHashMap<G::A, node::ActionStats>>>,
    pub player_bigram_actions: Vec<RwLock<BigramTable<G::A>>>,
    pub accum_depth: AtomicUsize,
    pub iter_count: AtomicUsize,
}

impl<G: Game> Default for TreeStats<G> {
    fn default() -> Self {
        Self {
            actions: RwLock::new(FxHashMap::default()),
            grave: RwLock::new(FxHashMap::default()),
            player_actions: (0..G::num_players())
                .map(|_| RwLock::new(FxHashMap::default()))
                .collect(),
            player_bigram_actions: (0..G::num_players())
                .map(|_| RwLock::new(FxHashMap::default()))
                .collect(),
            accum_depth: AtomicUsize::new(0),
            iter_count: AtomicUsize::new(0),
        }
    }
}

impl<G: Game> Clone for TreeStats<G> {
    fn clone(&self) -> Self {
        Self {
            actions: RwLock::new(self.actions.read().unwrap().clone()),
            grave: RwLock::new(self.grave.read().unwrap().clone()),
            player_actions: self
                .player_actions
                .iter()
                .map(|m| RwLock::new(m.read().unwrap().clone()))
                .collect(),
            player_bigram_actions: self
                .player_bigram_actions
                .iter()
                .map(|m| RwLock::new(m.read().unwrap().clone()))
                .collect(),
            accum_depth: AtomicUsize::new(self.accum_depth.load(Relaxed)),
            iter_count: AtomicUsize::new(self.iter_count.load(Relaxed)),
        }
    }
}

pub type TreeIndex<A> = index::Arena<Node<A>>;

/// How many plies of catch-up `TreeSearch::find_reachable` will search past
/// the current root before giving up and falling back to a full reset. Deep
/// enough to cover the two real call patterns in this codebase --
/// `util::self_play`'s single search instance advancing by exactly its own
/// move each call (1 ply) and `util::round_robin`'s alternating instances
/// advancing by its own move plus the opponent's reply between calls (2
/// plies) -- with slack for a couple more without unbounded search if
/// something calls `choose_action` less often than every ply.
pub(crate) const MAX_REROOT_DEPTH: usize = 4;

/// A root child's (action, visits, per-player score) triple -- the unit
/// root-parallel search merges across independently-searched trees.
pub(crate) type ActionTotal<A> = (A, u32, Vec<f64>);

/// Bundles the parts of a `TreeSearch` that every tree-parallel worker
/// thread reads (and, via interior mutability, writes) concurrently --
/// everything except per-thread scratch (rng, strategy instances, and the
/// in-progress select stack/trial).
///
/// `select_step`/`new_child`/`simulate_step`/`backprop_step` below take this
/// (plus explicit scratch parameters) instead of being `TreeSearch` methods,
/// specifically so the single-threaded path can call them as
/// `select_step(&Shared { index: &self.index, .. }, &mut self.stack, ...)`:
/// the borrow checker accepts disjoint borrows of `self`'s fields expressed
/// as direct field projections like that, but not the same borrows routed
/// through a `&self`/`&mut self` method receiver (which conceptually claims
/// all of `self`). That's what lets both the untouched single-threaded loop
/// and the new tree-parallel worker loop share one implementation instead of
/// two copies that could drift.
pub struct Shared<'a, G: Game> {
    pub index: &'a TreeIndex<G::A>,
    pub root_stats: &'a NodeStats,
    pub table: &'a TranspositionTable<G::S>,
    pub global: &'a TreeStats<G>,
    pub expand_threshold: u32,
    pub q_init: node::QInit,
    pub use_transpositions: bool,
    pub use_mcts_solver: bool,
    pub max_playout_depth: usize,
}

/// Resolves a node's Leaf -> {Terminal, Expanded} transition exactly once,
/// even under concurrent callers (see `Node::expand`). When `use_mcts_solver`
/// is set, also decodes and writes this node's `Proven` status the moment a
/// terminal position is found here -- this is the one legitimate proof
/// source for a tree node's own position (see PLAN-DRUID.md session 3 point
/// 1; a rollout's endpoint past this leaf is not).
#[inline]
pub fn expand<'a, G: Game>(
    index: &'a TreeIndex<G::A>,
    node_id: Id,
    state: &G::S,
    use_mcts_solver: bool,
) -> &'a NodeState<G::A> {
    let node = index.get(node_id);
    node.expand(|| {
        let status = G::terminal_status(state);
        if matches!(status, TerminalStatus::NotTerminal) {
            let mut actions = Vec::new();
            G::generate_actions(state, &mut actions);
            debug_assert!(!actions.is_empty());
            NodeState::Expanded(
                actions
                    .into_iter()
                    .map(|action| Edge::unexplored(action, G::num_players()))
                    .collect(),
            )
        } else {
            if use_mcts_solver {
                debug_assert!(G::num_players() <= 2);
                let proven = match status {
                    TerminalStatus::NotTerminal => unreachable!(),
                    TerminalStatus::Draw => Proven::Draw,
                    TerminalStatus::Winner(w) => Proven::Win(w.to_index()),
                };
                node.try_prove(proven);
            }
            NodeState::Terminal
        }
    })
}

/// Resolves an unexplored edge's child, creating it if this is the first
/// caller to arrive (see `Edge::get_or_create_child` and
/// `TranspositionTable::get_or_insert` for how each half of the
/// edge-creation/transposition race is handled).
pub fn new_child<G: Game>(shared: &Shared<'_, G>, state: &G::S, best_idx: usize, current_id: Id) -> Id {
    let hash = G::zobrist_hash(state);
    let parent = shared.index.get(current_id);
    let edge = &parent.edges()[best_idx];
    edge.get_or_create_child(|| {
        if shared.use_transpositions {
            // TODO: the following won't work with symmetries
            shared.table.get_or_insert(hash, state.clone(), || {
                let child = Node::new(G::player_to_move(state).to_index(), hash);
                shared.index.insert(child)
            })
        } else {
            let child_node = Node::new(G::player_to_move(state).to_index(), hash);
            shared.index.insert(child_node)
        }
    })
}

/// A child of `node` already proven to win for `player`, if one exists --
/// `None` when the solver is off, or none of `node`'s *explored* children
/// happen to be a proven win (yet). Shared by every place that would
/// otherwise call `SelectStrategy::best_child` (`select_step` below,
/// `select_final_action`, and `compute_pv`) and needs to bypass it outright
/// once the answer is already known: `Score` is a strategy-specific
/// associated type with no generic "infinity" to bias by that would work
/// the same way across all 10 `SelectStrategy` impls, so this lives here
/// instead of in `select.rs`. Fires on the first such child found, matching
/// the "any winning child" rule `derive_proven` (backprop.rs) uses to
/// decide `node` is itself a proven win one level up.
#[inline]
pub fn proven_win_child<G: Game>(
    use_mcts_solver: bool,
    node: &Node<G::A>,
    index: &TreeIndex<G::A>,
    player: usize,
) -> Option<usize> {
    if !use_mcts_solver {
        return None;
    }
    node.edges().iter().position(|edge| {
        edge.node_id()
            .is_some_and(|child_id| index.get(child_id).proven() == Proven::Win(player))
    })
}

/// Descend from `ctx.current_id` to a leaf, expanding/creating nodes as
/// needed, leaving the root->leaf path in `stack`. Shared by the
/// single-threaded path and every tree-parallel worker.
pub fn select_step<G: Game>(
    shared: &Shared<'_, G>,
    ctx: &mut SearchContext<G>,
    stack: &mut Vec<Id>,
    select_strategy: &mut impl SelectStrategy<G>,
    rng: &mut SmallRng,
) {
    debug_assert!(stack.is_empty());
    let grave = shared.global.grave.read().unwrap();
    loop {
        stack.push(ctx.current_id);

        let node_stack = NodeStack::new(stack.clone());
        let num_visits = node_stack
            .current_stats(shared.index, shared.root_stats)
            .num_visits();
        let node = shared.index.get(ctx.current_id);
        let player = node.player_idx;

        // A single snapshot of this node's status -- see `Node::status`'s
        // doc comment for why this can't be two separate `is_terminal()`/
        // `is_leaf()` calls (a concurrent `expand()` elsewhere, e.g. on a
        // transposed node shared with another thread's path, can resolve
        // Leaf -> Terminal in the gap between them and slip past both
        // branches).
        match node.status() {
            Some(NodeState::Terminal) => return,
            Some(NodeState::Expanded(_)) => {
                if num_visits < shared.expand_threshold {
                    return;
                }
            }
            None => {
                if num_visits < shared.expand_threshold {
                    return;
                }
                let node_state = expand::<G>(
                    shared.index,
                    ctx.current_id,
                    &ctx.state,
                    shared.use_mcts_solver,
                );
                if matches!(node_state, NodeState::Terminal) {
                    return;
                }
            }
        }

        let best_idx = match proven_win_child::<G>(shared.use_mcts_solver, node, shared.index, player) {
            Some(idx) => idx,
            None => {
                let select_ctx = SelectContext {
                    q_init: shared.q_init,
                    stack: &node_stack,
                    root_stats: shared.root_stats,
                    player,
                    state: &ctx.state,
                    index: shared.index,
                    table: shared.table,
                    grave: &grave,
                    use_transpositions: shared.use_transpositions,
                };

                select_strategy.best_child(&select_ctx, rng)
            }
        };

        let edges = shared.index.get(ctx.current_id).edges();

        // Claim this edge for the duration of the iteration so other
        // tree-parallel threads see it as less attractive until `backprop`
        // removes the virtual loss again -- this is what keeps concurrent
        // descents from all piling onto the same path.
        edges[best_idx].stats.add_virtual_loss();

        if let Some(child_id) = edges[best_idx].node_id() {
            ctx.traverse_apply(child_id, &edges[best_idx].action);
        } else {
            {
                let mut actions = vec![];
                G::generate_actions(&ctx.state, &mut actions);
                debug_assert_eq!(actions[best_idx], edges[best_idx].action);
            }

            let action = &edges[best_idx].action;
            let state = G::apply(ctx.state.clone(), action);

            let child_id = new_child::<G>(shared, &state, best_idx, ctx.current_id);

            ctx.traverse(child_id);
            ctx.state = state;

            if shared.expand_threshold > 0 {
                stack.push(ctx.current_id);
                return;
            }
        }
    }
}

/// The action on the edge leading to `stack`'s last node, i.e. the most
/// recent move played during tree descent -- `None` when `stack` is just the
/// root (no descent happened this iteration, e.g. `expand_threshold` not yet
/// met there). This is the bigram context `Nst::select_move` needs for the
/// playout's first ply; `playout` tracks its own running context for every
/// ply after that.
#[inline]
pub fn last_tree_action<G: Game>(index: &TreeIndex<G::A>, stack: &[Id]) -> Option<G::A> {
    if stack.len() < 2 {
        return None;
    }
    let parent_id = stack[stack.len() - 2];
    let child_id = stack[stack.len() - 1];
    Some(index.get(parent_id).child_edge(child_id).action.clone())
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_step<G: Game>(
    max_playout_depth: usize,
    global: &TreeStats<G>,
    simulate_strategy: &mut impl SimulateStrategy<G>,
    state: &G::S,
    prev_action: Option<G::A>,
    rng: &mut SmallRng,
) -> Trial<G> {
    simulate_strategy.playout(
        G::determinize(state.clone(), rng),
        max_playout_depth,
        global,
        prev_action,
        rng,
    )
}

/// Adds `extra` additional units of virtual loss to every edge on `stack`'s
/// root->leaf path. `select_step` already added one unit per edge on the
/// path as it descended; when a batch of `k` rollouts is about to fire from
/// that same leaf (leaf parallelism, or tree parallelism's per-worker
/// `num_rollouts_per_leaf` loop), the path needs `k` units in flight total
/// (one released per rollout's `backprop_step` call), so this tops up the
/// `k - 1` the batch adds beyond the one `select_step` already placed.
pub fn add_path_virtual_loss<A: crate::game::Action>(
    index: &TreeIndex<A>,
    stack: &NodeStack<A>,
    extra: usize,
) {
    for (parent_id, child_id) in stack.pairs() {
        let edge = stack.edge(index, *parent_id, *child_id);
        for _ in 0..extra {
            edge.stats.add_virtual_loss();
        }
    }
}

pub fn backprop_step<G: Game>(
    shared: &Shared<'_, G>,
    stack: &[Id],
    backprop_strategy: &impl BackpropStrategy,
    trial: Trial<G>,
    flags: BackpropFlags,
) {
    shared.global.iter_count.fetch_add(1, Relaxed);
    shared
        .global
        .accum_depth
        .fetch_add(trial.depth + stack.len() - 1, Relaxed);
    let node_stack = NodeStack::new(stack.to_vec());
    backprop_strategy.update(
        &node_stack,
        shared.global,
        shared.index,
        shared.root_stats,
        trial,
        flags,
        shared.use_mcts_solver,
    );
}