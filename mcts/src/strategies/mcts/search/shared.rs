use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::Real;
use crate::game::TerminalStatus;
use crate::strategies::mcts::backprop::BackpropStrategy;
use crate::strategies::mcts::config::BackpropFlags;
use crate::strategies::mcts::config::GraphStats;
use crate::strategies::mcts::index;
use crate::strategies::mcts::index::Id;
use crate::strategies::mcts::node;
use crate::strategies::mcts::node::real_action;
use crate::strategies::mcts::node::ChildArray;
use crate::strategies::mcts::node::Node;
use crate::strategies::mcts::node::NodeState;
use crate::strategies::mcts::node::NodeStats;
use crate::strategies::mcts::node::Proven;
use crate::strategies::mcts::select::SelectContext;
use crate::strategies::mcts::select::SelectStrategy;
use crate::strategies::mcts::simulate::SimulateStrategy;
use crate::strategies::mcts::simulate::Trial;
use crate::strategies::mcts::stack::NodeStack;
use crate::strategies::mcts::table::TranspositionKey;
use crate::strategies::mcts::table::TranspositionTable;

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

/// LGR's per-player reply table: the opponent's move that preceded this
/// player's move -> the last move this player played in reply that went on
/// to win the playout. Unlike `player_actions`/`player_bigram_actions`
/// (running averages), this is a plain last-write-wins map -- no
/// `ActionStats` accumulation, since LGR-1 has no notion of a score, only
/// "the most recent winning reply". Scoped to the current search only, same
/// reasoning as `BigramTable`'s doc comment above.
type ReplyTable<A> = FxHashMap<A, A>;

/// LGRF-2's per-player 2-ply reply table: (this player's own previous move,
/// the opponent's reply to it) -> the last move this player played in that
/// context that went on to win. Same last-write-wins shape as `ReplyTable`,
/// plus forgetting (see `backprop.rs`'s `flags.lgr2()` block): a losing
/// reply is removed from here rather than just left unwritten.
type Reply2Table<A> = FxHashMap<(A, A), A>;

#[derive(Debug)]
pub struct TreeStats<G: Game> {
    pub actions: RwLock<FxHashMap<G::A, node::ActionStats>>,
    pub grave: RwLock<GraveTable<G::A>>,
    pub player_actions: Vec<RwLock<FxHashMap<G::A, node::ActionStats>>>,
    pub player_bigram_actions: Vec<RwLock<BigramTable<G::A>>>,
    pub player_replies: Vec<RwLock<ReplyTable<G::A>>>,
    pub player_replies2: Vec<RwLock<Reply2Table<G::A>>>,
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
            player_replies: (0..G::num_players())
                .map(|_| RwLock::new(FxHashMap::default()))
                .collect(),
            player_replies2: (0..G::num_players())
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
            player_replies: self
                .player_replies
                .iter()
                .map(|m| RwLock::new(m.read().unwrap().clone()))
                .collect(),
            player_replies2: self
                .player_replies2
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
    /// The root's own literal-board state -- the anchor every `node::
    /// incoming_sym` translation replays real states forward from (see its
    /// doc comment for why this can't be a cached per-edge value).
    pub root_state: &'a G::S,
    pub root_stats: &'a NodeStats,
    pub table: &'a TranspositionTable,
    pub global: &'a TreeStats<G>,
    pub expand_threshold: u32,
    pub q_init: node::QInit,
    pub use_transpositions: bool,
    /// `Some` for every statistics-owning graph mode, including the legacy
    /// edge-only compatibility path.
    pub graph_stats: Option<GraphStats>,
    /// Only explicit `GraphSearch::Dag`: legacy transpositions deliberately
    /// retain their historic hash-only key and reuse behavior.
    pub explicit_dag: bool,
    pub use_mcts_solver: bool,
    pub max_playout_depth: usize,
    pub solver_loss_threshold: u32,
    /// From `SearchConfig::requirements().amaf` -- whether the active
    /// `Strategy` reads per-child AMAF stats, gating whether newly created
    /// `Node`/`ChildArray`s allocate their AMAF side table at all.
    pub has_amaf: bool,
}

/// Resolves a node's Leaf -> {Terminal, Expanded} transition exactly once,
/// even under concurrent callers (see `Node::expand`). When `use_mcts_solver`
/// is set, also decodes and writes this node's `Proven` status the moment a
/// terminal position is found here -- this is the one legitimate proof
/// source for a tree node's own position (a rollout's endpoint past this
/// leaf is not).
#[inline]
pub fn expand<'a, G: Game>(
    index: &'a TreeIndex<G::A>,
    node_id: Id,
    state: &G::S,
    use_mcts_solver: bool,
    has_amaf: bool,
    canonicalize: bool,
) -> &'a NodeState<G::A> {
    let node = index.get(node_id);
    node.expand(|| {
        let status = G::terminal_status(state);
        if matches!(status, TerminalStatus::NotTerminal) {
            // The root's own action list stays in the literal, uncanonicalized
            // orientation regardless of `canonicalize`: it has no incoming
            // edge, and it's the one node whose actions must already be
            // directly playable against the real game state
            // (`select_final_action`, `verbose_summary`). Every other node's
            // action list is generated from whichever real state first
            // reached it, canonicalized -- `canonical_representation` is
            // deterministic on the equivalence class, so it doesn't matter
            // which of possibly several transposed parents that state came
            // from; every caller translates back via `node::incoming_sym`,
            // recomputed fresh from its own real state rather than cached
            // (see that function's doc comment).
            let gen_state = if canonicalize && !node.is_root() {
                G::canonical_representation(Real(state.clone()))
                    .0
                    .into_inner()
            } else {
                state.clone()
            };
            let mut actions = Vec::new();
            G::generate_actions(&gen_state, &mut actions);
            debug_assert!(!actions.is_empty());
            NodeState::Expanded(ChildArray::new(actions, G::num_players(), has_amaf))
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
pub fn new_child<G: Game>(
    shared: &Shared<'_, G>,
    state: &G::S,
    best_idx: usize,
    current_id: Id,
) -> Id {
    let parent = shared.index.get(current_id);
    let children = parent.children();
    children.get_or_create_child(best_idx, || {
        let child_id = if shared.explicit_dag {
            let ply = parent.ply + 1;
            let canon_state = G::canonical_representation(Real(state.clone()))
                .0
                .into_inner();
            let canon_hash = G::zobrist_hash(&canon_state);
            shared.table.get_or_insert_graph(
                TranspositionKey {
                    position_hash: canon_hash,
                    ply,
                },
                || {
                    shared.index.insert(Node::new_at_ply(
                        G::player_to_move(state).to_index(),
                        canon_hash,
                        ply,
                        G::num_players(),
                        shared.has_amaf,
                        shared.use_mcts_solver,
                    ))
                },
            )
        } else if shared.use_transpositions {
            let hash = G::zobrist_hash(state);
            shared.table.get_or_insert(hash, || {
                shared.index.insert(Node::new_at_ply(
                    G::player_to_move(state).to_index(),
                    hash,
                    parent.ply + 1,
                    G::num_players(),
                    shared.has_amaf,
                    shared.use_mcts_solver,
                ))
            })
        } else {
            let hash = G::zobrist_hash(state);
            shared.index.insert(Node::new_at_ply(
                G::player_to_move(state).to_index(),
                hash,
                parent.ply + 1,
                G::num_players(),
                shared.has_amaf,
                shared.use_mcts_solver,
            ))
        };
        shared.index.get(child_id).add_incoming_edge();
        child_id
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
    let children = node.children();
    (0..children.len()).find(|&i| {
        children
            .node_id(i)
            .is_some_and(|child_id| index.get(child_id).proven() == Proven::Win(player))
    })
}

/// A child of `node` already proven a draw, if one exists -- the
/// contempt-factor counterpart to `proven_win_child` above (Kowalski et al.
/// 2023, Section VII.C's "Final Move Selection Contempt Factor"). `None`
/// when the solver is off, or none of `node`'s *explored* children happen to
/// be a proven draw (yet). Unlike `proven_win_child`, doesn't need the
/// second-layer `pn2`/`dpn2` machinery: `Proven::Draw` is already an exact,
/// unambiguous terminal category on this codebase's `Proven` (see its doc
/// comment) -- the paper's own binary PNS bookkeeping is what conflates draw
/// and loss, and only the live-ranking `pn`/`dpn` (and hence `UctPn`) ever
/// inherits that ambiguity.
#[inline]
pub fn proven_draw_child<G: Game>(
    use_mcts_solver: bool,
    node: &Node<G::A>,
    index: &TreeIndex<G::A>,
) -> Option<usize> {
    if !use_mcts_solver {
        return None;
    }
    let children = node.children();
    (0..children.len()).find(|&i| {
        children
            .node_id(i)
            .is_some_and(|child_id| index.get(child_id).proven() == Proven::Draw)
    })
}

/// Descend from `ctx.current_id` to a leaf, expanding/creating nodes as
/// needed, leaving the root->leaf path in `stack`. Shared by the
/// single-threaded path and every tree-parallel worker.
pub fn select_step<G: Game>(
    shared: &Shared<'_, G>,
    ctx: &mut SearchContext<G>,
    stack: &mut Vec<(Id, usize)>,
    select_strategy: &mut impl SelectStrategy<G>,
    rng: &mut SmallRng,
) {
    debug_assert!(stack.is_empty());
    let grave = shared.global.grave.read().unwrap();
    // The idx (in the previous node's `ChildArray`) that reached
    // `ctx.current_id` -- unused for the very first push (the root has no
    // predecessor, see `stack::StackEntry`'s doc comment), then updated to
    // `best_idx` every iteration right after it's chosen, since that's the
    // same slot whichever branch below (existing vs. newly created child)
    // ends up using.
    let mut incoming_idx = 0usize;
    loop {
        stack.push((ctx.current_id, incoming_idx));

        let node_stack = NodeStack::new(stack.clone());
        let num_visits = node_stack
            .current_stats(shared.index, shared.root_stats, shared.graph_stats)
            .num_visits();
        let node = shared.index.get(ctx.current_id);
        let player = node.player_idx;
        // Recomputed fresh from `ctx.state` (this node's own real board
        // state) every iteration rather than carried from the previous one
        // -- see `node::incoming_sym`'s doc comment for why a value cached
        // on the incoming edge would be wrong whenever this node's own
        // parent is itself a transposition (reached via more than one real
        // orientation across different iterations).
        let incoming_sym =
            node::incoming_sym::<G>(shared.explicit_dag, node.is_root(), Real(&ctx.state));

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
                    shared.has_amaf,
                    shared.explicit_dag,
                );
                if matches!(node_state, NodeState::Terminal) {
                    return;
                }
            }
        }

        let best_idx =
            match proven_win_child::<G>(shared.use_mcts_solver, node, shared.index, player) {
                Some(idx) => idx,
                None => {
                    let select_ctx = SelectContext {
                        q_init: shared.q_init,
                        stack: &node_stack,
                        root_stats: shared.root_stats,
                        root_state: shared.root_state,
                        explicit_dag: shared.explicit_dag,
                        player,
                        state: &ctx.state,
                        index: shared.index,
                        table: shared.table,
                        grave: &grave,
                        global: shared.global,
                        use_transpositions: shared.use_transpositions,
                        graph_stats: shared.graph_stats,
                        solver_loss_threshold: shared.solver_loss_threshold,
                        incoming_sym,
                    };

                    select_strategy.best_child(&select_ctx, rng)
                }
            };
        incoming_idx = best_idx;

        let children = shared.index.get(ctx.current_id).children();

        // Claim this edge for the duration of the iteration so other
        // tree-parallel threads see it as less attractive until `backprop`
        // removes the virtual loss again -- this is what keeps concurrent
        // descents from all piling onto the same path.
        if shared.graph_stats.is_none_or(GraphStats::uses_edges) {
            children.add_virtual_loss(best_idx);
        }

        if let Some(child_id) = children.node_id(best_idx) {
            if shared.graph_stats.is_some_and(GraphStats::uses_nodes) {
                shared.index.get(child_id).stats.add_virtual_loss();
            }
            // `children.action(best_idx)` is in *this* node's own
            // orientation (canonical if this node isn't the root -- see
            // `expand`), translated via `incoming_sym`.
            let action = real_action::<G>(children, best_idx, incoming_sym);
            ctx.traverse_apply(child_id, &action);
        } else {
            let action = real_action::<G>(children, best_idx, incoming_sym);
            {
                let mut actions = vec![];
                G::generate_actions(&ctx.state, &mut actions);
                debug_assert!(actions.contains(&action));
            }

            let state = G::apply(ctx.state.clone(), &action);

            let child_id = new_child::<G>(shared, &state, best_idx, ctx.current_id);

            if shared.graph_stats.is_some_and(GraphStats::uses_nodes) {
                shared.index.get(child_id).stats.add_virtual_loss();
            }

            ctx.traverse(child_id);
            ctx.state = state;

            if shared.expand_threshold > 0 {
                stack.push((ctx.current_id, incoming_idx));
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
/// ply after that. Replays real states from `root_state` (see `NodeStack::
/// incoming_syms`'s doc comment for why a cached per-edge value can't be
/// trusted here) -- O(depth), same as `backprop_step`'s own per-iteration
/// walk.
#[inline]
pub fn last_tree_action<G: Game>(
    index: &TreeIndex<G::A>,
    stack: &[(Id, usize)],
    root_state: &G::S,
    explicit_dag: bool,
) -> Option<G::A> {
    if stack.len() < 2 {
        return None;
    }
    let node_stack = NodeStack::new(stack.to_vec());
    let (_, actions) = node_stack.incoming_syms::<G>(index, root_state, explicit_dag);
    actions.get(&stack[stack.len() - 1].0).cloned()
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
    graph_stats: Option<GraphStats>,
) {
    for ((parent_id, _), (child_id, idx)) in stack.pairs() {
        let parent = index.get(*parent_id);
        let children = parent.children();
        for _ in 0..extra {
            if graph_stats.is_none_or(GraphStats::uses_edges) {
                children.add_virtual_loss(*idx);
            }
            if graph_stats.is_some_and(GraphStats::uses_nodes) {
                index.get(*child_id).stats.add_virtual_loss();
            }
        }
    }
}

pub fn backprop_step<G: Game>(
    shared: &Shared<'_, G>,
    stack: &[(Id, usize)],
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
        shared.root_state,
        shared.explicit_dag,
        trial,
        flags,
        shared.use_mcts_solver,
        shared.graph_stats,
    );
}
