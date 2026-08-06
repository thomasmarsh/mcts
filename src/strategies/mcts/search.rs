use super::backprop::BackpropStrategy;
use super::config::BackpropFlags;
use super::config::SearchConfig;
use super::config::Strategy;
use super::index;
use super::index::Id;
use super::node;
use super::node::Node;
use super::node::NodeState;
use super::node::NodeStats;
use super::node::Proven;
use super::select::SelectContext;
use super::select::SelectStrategy;
use super::simulate::SimulateStrategy;
use super::simulate::Trial;
use super::stack::NodeStack;
use super::table::TranspositionTable;
use crate::game::Game;
use crate::game::PlayerIndex;
use crate::game::TerminalStatus;
use crate::strategies::mcts::node::Edge;
use crate::strategies::Search;
use crate::timer;
use crate::util::pv_string;
use crate::util::random_best;

use rand::rngs::SmallRng;
use rand::Rng;
use rand_core::SeedableRng;
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

/// A root child's (action, visits, per-player score) triple -- the unit
/// root-parallel search merges across independently-searched trees.
type ActionTotal<A> = (A, u32, Vec<f64>);

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
struct Shared<'a, G: Game> {
    index: &'a TreeIndex<G::A>,
    root_stats: &'a NodeStats,
    table: &'a TranspositionTable<G::S>,
    global: &'a TreeStats<G>,
    expand_threshold: u32,
    q_init: node::QInit,
    use_transpositions: bool,
    use_mcts_solver: bool,
    max_playout_depth: usize,
}

/// Resolves a node's Leaf -> {Terminal, Expanded} transition exactly once,
/// even under concurrent callers (see `Node::expand`). When `use_mcts_solver`
/// is set, also decodes and writes this node's `Proven` status the moment a
/// terminal position is found here -- this is the one legitimate proof
/// source for a tree node's own position (see PLAN-DRUID.md session 3 point
/// 1; a rollout's endpoint past this leaf is not).
#[inline]
fn expand<'a, G: Game>(
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
fn new_child<G: Game>(shared: &Shared<'_, G>, state: &G::S, best_idx: usize, current_id: Id) -> Id {
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
fn proven_win_child<G: Game>(
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
fn select_step<G: Game>(
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
fn last_tree_action<G: Game>(index: &TreeIndex<G::A>, stack: &[Id]) -> Option<G::A> {
    if stack.len() < 2 {
        return None;
    }
    let parent_id = stack[stack.len() - 2];
    let child_id = stack[stack.len() - 1];
    Some(index.get(parent_id).child_edge(child_id).action.clone())
}

#[allow(clippy::too_many_arguments)]
fn simulate_step<G: Game>(
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
fn add_path_virtual_loss<A: crate::game::Action>(
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

fn backprop_step<G: Game>(
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

#[derive(Clone)]
pub struct TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Sync + Send,
    G::S: std::fmt::Display,
{
    pub(crate) index: TreeIndex<G::A>,
    pub(crate) timer: timer::Timer,
    pub(crate) root_id: Id,
    pub(crate) root_stats: NodeStats,
    pub(crate) pv: Vec<G::A>,
    pub(crate) table: TranspositionTable<G::S>,

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
        let index = index::Arena::new();
        let root_id = index.insert(Node::new_root(0, G::num_players(), 0));
        Self {
            root_id,
            root_stats: NodeStats::new(G::num_players()),
            pv: vec![],
            stack: vec![],
            table: TranspositionTable::default(),
            trial: None,
            index,
            config: S::config(),
            timer: timer::Timer::new(),
            stats: Default::default(),
        }
    }

    #[inline]
    pub(crate) fn new_root(&mut self, player_idx: usize, hash: u64) -> Id {
        let root = Node::new_root(player_idx, G::num_players(), hash);
        self.root_id = self.index.insert(root);
        self.root_id
    }

    #[inline]
    pub fn select(&mut self, ctx: &mut SearchContext<G>) {
        debug_assert!(self.stack.is_empty());
        select_step(
            &Shared {
                index: &self.index,
                root_stats: &self.root_stats,
                table: &self.table,
                global: &self.stats,
                expand_threshold: self.config.expand_threshold,
                q_init: self.config.q_init,
                use_transpositions: self.config.use_transpositions,
                use_mcts_solver: self.config.use_mcts_solver,
                max_playout_depth: self.config.max_playout_depth,
            },
            ctx,
            &mut self.stack,
            &mut self.config.select,
            &mut self.config.rng,
        );
    }

    #[inline]
    fn select_final_action(&mut self, state: &G::S) -> G::A {
        let player = G::player_to_move(state).to_index();
        // MCTS-Solver: `choose_action`'s iteration loop may have already
        // stopped the moment the root was proven a win (search.rs's
        // `use_mcts_solver` break condition), well before the winning
        // child necessarily accumulated the most visits -- so a plain
        // most-visited/highest-score `final_action` pick here can't be
        // trusted to land on it. Reading the proof directly is what
        // actually guarantees the move `choose_action` returns matches the
        // one it just finished proving.
        if let Some(idx) =
            proven_win_child::<G>(self.config.use_mcts_solver, self.index.get(self.root_id), &self.index, player)
        {
            return self.index.get(self.root_id).edges()[idx].action.clone();
        }

        let stack = NodeStack::new(vec![self.root_id]);
        let grave = self.stats.grave.read().unwrap();
        let idx = self.config.final_action.best_child(
            &SelectContext {
                q_init: self.config.q_init,
                stack: &stack,
                root_stats: &self.root_stats,
                player,
                state,
                index: &self.index,
                table: &self.table,
                grave: &grave,
                use_transpositions: self.config.use_transpositions,
            },
            &mut self.config.rng,
        );

        self.index.get(self.root_id).edges()[idx].action.clone()
    }

    #[inline]
    pub(crate) fn simulate(&mut self, state: &G::S) -> Trial<G> {
        let prev_action = last_tree_action::<G>(&self.index, &self.stack);
        simulate_step(
            self.config.max_playout_depth,
            &self.stats,
            &mut self.config.simulate,
            state,
            prev_action,
            &mut self.config.rng,
        )
    }

    /// Leaf parallelism: run `k` playouts from the same selected leaf's
    /// `state` on separate threads (each with its own reseeded RNG and
    /// cloned `SimulateStrategy`, since `SmallRng` isn't `Sync` and
    /// `playout` takes `&mut self`), rather than just the one `simulate`
    /// does. Only the rollouts run concurrently -- selection stays
    /// single-threaded and none of this touches the shared arena, since
    /// `playout` already only reads `&TreeStats<G>` and operates on a
    /// state cloned for the rollout.
    fn simulate_many(&mut self, state: &G::S, k: usize) -> Vec<Trial<G>> {
        if k <= 1 {
            return vec![self.simulate(state)];
        }

        let seeds: Vec<u64> = (0..k).map(|_| self.config.rng.gen()).collect();
        let mut strategies: Vec<S::Simulate> =
            (0..k).map(|_| self.config.simulate.clone()).collect();
        let max_playout_depth = self.config.max_playout_depth;
        let stats = &self.stats;
        let prev_action = last_tree_action::<G>(&self.index, &self.stack);

        std::thread::scope(|scope| {
            let handles: Vec<_> = strategies
                .iter_mut()
                .zip(seeds)
                .map(|(strategy, seed)| {
                    let state = state.clone();
                    let prev_action = prev_action.clone();
                    scope.spawn(move || {
                        let mut rng = SmallRng::seed_from_u64(seed);
                        simulate_step(max_playout_depth, stats, strategy, &state, prev_action, &mut rng)
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    fn add_extra_virtual_loss(&self, stack: &NodeStack<G::A>, extra: usize) {
        add_path_virtual_loss(&self.index, stack, extra);
    }

    #[inline]
    pub(crate) fn backprop(&mut self) {
        let trial = self.trial.as_ref().unwrap().clone();
        let flags = self.config.select.backprop_flags() | self.config.simulate.backprop_flags();
        backprop_step(
            &Shared {
                index: &self.index,
                root_stats: &self.root_stats,
                table: &self.table,
                global: &self.stats,
                expand_threshold: self.config.expand_threshold,
                q_init: self.config.q_init,
                use_transpositions: self.config.use_transpositions,
                use_mcts_solver: self.config.use_mcts_solver,
                max_playout_depth: self.config.max_playout_depth,
            },
            &self.stack,
            &self.config.backprop,
            trial,
            flags,
        );
    }

    pub fn verbose_summary(&self, state: &G::S, num_threads: usize) {
        if !self.config.verbose {
            return;
        }

        let root = self.index.get(self.root_id);
        let total_visits = self.root_stats.num_visits();
        let rate = total_visits as f64 / num_threads as f64 / self.timer.elapsed().as_secs_f64();
        eprintln!(
            "Using {} threads, did {} total simulations with {:.1} rollouts/sec/core",
            num_threads, total_visits, rate
        );

        let player = G::player_to_move(state);

        // Sort moves by visit count, largest first.
        let mut children = root
            .edges()
            .iter()
            .filter(|edge| edge.is_explored())
            .map(|edge| {
                (
                    edge.stats.num_visits(),
                    edge.stats.score(player.to_index()),
                    edge.action.clone(),
                )
            })
            .collect::<Vec<_>>();

        children.sort_by_key(|t| !t.0);

        // Dump stats about the top 10 nodes.
        for (visits, score, m) in children.into_iter().take(10) {
            // Normalized so all wins is 100%, all draws is 50%, and all losses is 0%.
            let win_rate = (score + visits as f64) / (visits as f64 * 2.0);
            eprintln!(
                "{:>6} visits, {:.02}% wins: {}",
                visits,
                win_rate * 100.0,
                G::notation(state, &m),
            );
        }

        eprintln!("PV: {}", pv_string::<G>(self.pv.as_slice(), state))
    }

    #[inline]
    pub(crate) fn reset_iter(&mut self) {
        self.stack.clear();
        self.trial = None;
    }

    #[inline]
    pub(crate) fn reset(&mut self, player_idx: usize, hash: u64) -> Id {
        self.index.clear();
        self.table.clear();
        self.stats.accum_depth.store(0, Relaxed);
        self.stats.iter_count.store(0, Relaxed);
        self.new_root(player_idx, hash)
    }

    fn compute_pv(&mut self, init_state: &G::S) {
        self.pv.clear();
        let mut node_id = self.root_id;
        let mut node = self.index.get(node_id);
        let mut state = init_state.clone();
        let mut stack = NodeStack::new(vec![node_id]);
        let grave = self.stats.grave.read().unwrap();
        while node.is_expanded() {
            let player = node.player_idx;
            let select_ctx = SelectContext {
                q_init: self.config.q_init,
                player,
                stack: &stack,
                root_stats: &self.root_stats,
                state: &state,
                index: &self.index,
                table: &self.table,
                grave: &grave,
                use_transpositions: self.config.use_transpositions,
            };

            let best_idx = match proven_win_child::<G>(self.config.use_mcts_solver, node, &self.index, player) {
                Some(idx) => idx,
                None => self
                    .config
                    .final_action
                    .best_child(&select_ctx, &mut self.config.rng),
            };

            let edge = &node.edges()[best_idx];
            if let Some(child_id) = edge.node_id() {
                node_id = child_id;
                node = self.index.get(node_id);
                state = G::apply(state, &edge.action);
                self.pv.push(edge.action.clone());
                stack.push(node_id);
            } else {
                break;
            }
        }
    }

    /// Root child visit counts/scores for this tree, keyed by action --
    /// the summary a root-parallel search merges across threads. Only
    /// explored edges are included (unexplored ones contribute nothing).
    fn root_action_totals(&self) -> Vec<ActionTotal<G::A>> {
        self.index
            .get(self.root_id)
            .edges()
            .iter()
            .filter(|edge| edge.is_explored())
            .map(|edge| {
                let scores = (0..G::num_players())
                    .map(|p| edge.stats.score(p))
                    .collect();
                (edge.action.clone(), edge.stats.num_visits(), scores)
            })
            .collect()
    }

    /// Root parallelism: run `config.num_threads` independent trees to
    /// completion (each its own `TreeSearch`, reseeded so they don't all
    /// explore identically), then merge by summing visit counts/scores per
    /// action across trees and picking the action with the most total
    /// visits. Doesn't touch the shared arena/stats -- each thread owns its
    /// own tree -- so unlike tree parallelism this needs no interior
    /// mutability anywhere in the search.
    ///
    /// Each worker (and `self`'s own in-place tree) has `num_threads` forced
    /// to `1` before its recursive `choose_action` call below, but
    /// `num_tree_threads` is left untouched -- so if it's also `> 1`, that
    /// recursive call dispatches into `choose_action_tree_parallel` instead
    /// of the plain single-tree loop, making every one of these `num_threads`
    /// trees itself tree-parallel: a hybrid split, e.g. `num_threads(4)` +
    /// `num_tree_threads(2)` for 4 trees x 2 threads each on an 8-core
    /// machine.
    ///
    /// Not `use_mcts_solver`-safe: each worker's recursive `choose_action`
    /// call can stop early the moment *its own* tree proves a line, leaving
    /// it with far fewer visits than trees that ran the full budget. The
    /// merge below picks the action with the most *summed* visits across
    /// trees, so a proven-winning action found quickly by one tree can be
    /// silently outvoted by an unproven action other trees merely visited
    /// more, simply because they weren't the one that found the proof.
    /// Fixing this needs proven-aware merging, not just summing visits, so
    /// it's guarded off below rather than left to mis-serve silently.
    fn choose_action_root_parallel(&mut self, state: &G::S) -> G::A {
        let num_threads = self.config.num_threads.max(1);
        debug_assert!(num_threads > 1);
        debug_assert!(
            !self.config.use_mcts_solver,
            "root parallelism's visit-sum merge doesn't account for trees that stop \
             early on a solver proof -- combining num_threads > 1 with use_mcts_solver \
             is not supported yet"
        );

        // One deterministic seed per worker, derived from this search's own
        // RNG, so a fixed `.seed(...)` still gives reproducible results.
        let seeds: Vec<u64> = (0..num_threads).map(|_| self.config.rng.gen()).collect();

        // `num_threads - 1` extra trees run on their own threads; `self`
        // runs the last one in place (on the calling thread) rather than
        // sitting idle, so afterward its own `index`/`pv` reflect one real
        // completed tree instead of being discarded -- picked up for free
        // by the normal single-tree `choose_action` path's `compute_pv`/
        // `verbose_summary` calls below.
        let mut workers: Vec<Self> = (0..num_threads - 1).map(|_| self.clone()).collect();

        let totals = std::thread::scope(|scope| {
            let handles: Vec<_> = workers
                .iter_mut()
                .zip(&seeds)
                .map(|(worker, &seed)| {
                    worker.config.num_threads = 1;
                    worker.config.rng = SmallRng::seed_from_u64(seed);
                    scope.spawn(move || {
                        worker.choose_action(state);
                        worker.root_action_totals()
                    })
                })
                .collect();

            self.config.num_threads = 1;
            self.config.rng = SmallRng::seed_from_u64(seeds[num_threads - 1]);
            self.choose_action(state);

            let mut totals = vec![self.root_action_totals()];
            totals.extend(handles.into_iter().map(|h| h.join().unwrap()));
            totals
        });

        self.config.num_threads = num_threads;

        let mut merged: FxHashMap<G::A, (u32, Vec<f64>)> = FxHashMap::default();
        for worker_totals in totals {
            for (action, visits, scores) in worker_totals {
                let entry = merged
                    .entry(action)
                    .or_insert_with(|| (0, vec![0.; scores.len()]));
                entry.0 += visits;
                for (i, s) in scores.into_iter().enumerate() {
                    entry.1[i] += s;
                }
            }
        }

        let merged: Vec<ActionTotal<G::A>> = merged
            .into_iter()
            .map(|(action, (visits, scores))| (action, visits, scores))
            .collect();
        random_best(&merged, &mut self.config.rng, |(_, visits, _)| {
            *visits as f64
        })
        .map(|(action, _, _)| action.clone())
        .unwrap()
    }

    /// Tree parallelism: `config.num_tree_threads` worker threads descend
    /// *one* shared `index`/`root_stats`/`table`/`stats` concurrently,
    /// racing the same `timer` and a shared iteration budget, relying on
    /// virtual loss (added in `select_step`, released in `backprop_step`) so
    /// concurrent descents spread out across the tree instead of piling onto
    /// the same path. Unlike root parallelism, this shares search effort
    /// across threads rather than duplicating it -- the whole point of the
    /// "make the arena/stats concurrent-safe" work above.
    ///
    /// Composes with root parallelism for a hybrid split (a handful of
    /// independent trees, each internally tree-parallel): `choose_action`'s
    /// dispatch checks `num_threads` first, so by the time this is reached
    /// -- either directly (pure tree parallelism) or via a root-parallel
    /// worker's recursive `choose_action` call -- `num_threads` is always
    /// `1` for *this* tree; the assert below documents that invariant
    /// rather than an exclusion between the two modes.
    fn choose_action_tree_parallel(&mut self, state: &G::S) -> G::A {
        let num_threads = self.config.num_tree_threads.max(1);
        debug_assert!(num_threads > 1);
        debug_assert_eq!(self.config.num_threads, 1);

        let hash = G::zobrist_hash(state);
        let root_id = self.reset(G::player_to_move(state).to_index(), hash);
        if self.config.use_transpositions {
            self.table.insert(hash, root_id, state.clone());
        }

        self.timer.start(self.config.max_time);

        let shared = Shared {
            index: &self.index,
            root_stats: &self.root_stats,
            table: &self.table,
            global: &self.stats,
            expand_threshold: self.config.expand_threshold,
            q_init: self.config.q_init,
            use_transpositions: self.config.use_transpositions,
            use_mcts_solver: self.config.use_mcts_solver,
            max_playout_depth: self.config.max_playout_depth,
        };
        let iterations_remaining = AtomicUsize::new(self.config.max_iterations);
        let k = self.config.num_rollouts_per_leaf.max(1);
        let timer = &self.timer;
        let backprop_strategy = &self.config.backprop;

        let seeds: Vec<u64> = (0..num_threads).map(|_| self.config.rng.gen()).collect();
        let mut select_strategies: Vec<S::Select> =
            (0..num_threads).map(|_| self.config.select.clone()).collect();
        let mut simulate_strategies: Vec<S::Simulate> = (0..num_threads)
            .map(|_| self.config.simulate.clone())
            .collect();

        std::thread::scope(|scope| {
            for ((seed, select_strategy), simulate_strategy) in seeds
                .into_iter()
                .zip(select_strategies.iter_mut())
                .zip(simulate_strategies.iter_mut())
            {
                let shared = &shared;
                let iterations_remaining = &iterations_remaining;
                scope.spawn(move || {
                    let mut rng = SmallRng::seed_from_u64(seed);
                    loop {
                        if timer.done() {
                            break;
                        }
                        // MCTS-Solver: no separate atomic flag needed --
                        // `root_id`'s `Proven` field (an `AtomicU8` on the
                        // shared arena `Node`, see `derive_proven` in
                        // backprop.rs) is already written by every worker's
                        // own `backprop_step` call and visible to every
                        // other worker through the same `shared.index`
                        // they're already reading/writing every iteration.
                        // Mirrors the single-threaded loop's check in
                        // `choose_action` above, just read through the
                        // shared arena instead of `self.index` directly.
                        if shared.use_mcts_solver
                            && shared.index.get(root_id).proven() != Proven::Unproven
                        {
                            break;
                        }
                        if iterations_remaining
                            .fetch_update(Relaxed, Relaxed, |n| n.checked_sub(1))
                            .is_err()
                        {
                            break;
                        }

                        let mut stack = Vec::new();
                        let mut ctx = SearchContext::new(root_id, state.clone());
                        select_step(shared, &mut ctx, &mut stack, select_strategy, &mut rng);

                        let node_stack = NodeStack::<G::A>::new(stack.clone());
                        if k > 1 {
                            add_path_virtual_loss(shared.index, &node_stack, k - 1);
                        }
                        let prev_action = last_tree_action::<G>(shared.index, &stack);
                        for _ in 0..k {
                            let trial = simulate_step(
                                shared.max_playout_depth,
                                shared.global,
                                simulate_strategy,
                                &ctx.state,
                                prev_action.clone(),
                                &mut rng,
                            );
                            let flags = select_strategy.backprop_flags()
                                | simulate_strategy.backprop_flags();
                            backprop_step(shared, &stack, backprop_strategy, trial, flags);
                        }
                    }
                });
            }
        });

        self.compute_pv(state);
        self.verbose_summary(state, num_threads);
        self.select_final_action(state)
    }
}

impl<G, S> Search for TreeSearch<G, S>
where
    G: Game,
    S: Strategy<G>,
    SearchConfig<G, S>: Default,
    G::S: std::fmt::Display,
{
    type G = G;

    fn friendly_name(&self) -> String {
        self.config.name.clone()
    }

    fn choose_action(&mut self, state: &G::S) -> G::A {
        // Order matters for hybrid root+tree parallelism: `num_threads`
        // (trees) is checked first so `choose_action_root_parallel` gets a
        // chance to spawn its independent trees; each of *those* then
        // recurses back into this same dispatch with `num_threads` forced to
        // `1` (see `choose_action_root_parallel`), so `num_tree_threads` is
        // what decides whether each individual tree is itself
        // tree-parallel. Checking `num_tree_threads` first would skip root
        // parallelism whenever both are set > 1, silently dropping the
        // "trees" half of a requested hybrid split.
        if self.config.num_threads > 1 {
            return self.choose_action_root_parallel(state);
        }
        if self.config.num_tree_threads > 1 {
            return self.choose_action_tree_parallel(state);
        }

        let hash = G::zobrist_hash(state);
        let root_id = self.reset(G::player_to_move(state).to_index(), hash);
        if self.config.use_transpositions {
            self.table.insert(hash, root_id, state.clone());
        }

        self.timer.start(self.config.max_time);

        for _ in 0..self.config.max_iterations {
            if self.timer.done() {
                break;
            }
            // MCTS-Solver: the root itself is always the last node
            // `backprop`'s solver pass visits (see `derive_proven` in
            // backprop.rs), so by this point its `Proven` field already
            // reflects everything found so far -- fires the moment *a*
            // forced win is found (the `Win(p)` rule doesn't wait on
            // sibling root children), or once the position is fully solved
            // for the `Win(q)`/`Draw` cases. Single-threaded loop only --
            // the tree-/root-parallel loops need a shared/atomic stop
            // signal instead of this per-thread-local read (see
            // PLAN-DRUID.md session 3 point 5), deliberately deferred.
            if self.config.use_mcts_solver && self.index.get(root_id).proven() != Proven::Unproven
            {
                break;
            }
            self.reset_iter();
            let mut ctx = SearchContext::new(root_id, state.clone());

            self.select(&mut ctx);

            let k = self.config.num_rollouts_per_leaf;
            let trials = if k > 1 {
                let stack = NodeStack::new(self.stack.clone());
                self.add_extra_virtual_loss(&stack, k - 1);
                self.simulate_many(&ctx.state, k)
            } else {
                vec![self.simulate(&ctx.state)]
            };

            for trial in trials {
                self.trial = Some(trial);
                self.backprop();
            }
        }

        self.compute_pv(state);
        self.verbose_summary(state, 1);

        // NOTE: this can fail when root is a leaf. This happens if:
        //
        //     max_iterations < expand_threshold
        //
        // TODO: We might check for this and unconditionally expand root. I think
        // a lot of implementations fully expand root on the first iteration.
        self.select_final_action(state)
    }

    fn make_book_entry(
        &mut self,
        state: &<Self::G as Game>::S,
    ) -> (Vec<<Self::G as Game>::A>, Vec<f64>) {
        debug_assert_eq!(self.config.expand_threshold, 0);
        debug_assert_eq!(self.config.max_iterations, 1);

        // Run the search, with expand_threshold == 0, so we fully expand to the
        // terminal node.
        _ = self.choose_action(state);
        if self.stack.len() < 2 {
            return (vec![], vec![0.; G::num_players()]);
        }

        // The stack now contains the action path to the terminal state.
        let mut actions = vec![];
        let stack = NodeStack::new(self.stack.clone());
        for (parent_id, child_id) in stack.pairs() {
            actions.push(
                stack
                    .edge(&self.index, *parent_id, *child_id)
                    .action
                    .clone(),
            );
        }

        let trial = self.trial.as_ref().unwrap();
        let utilities = trial
            .terminal
            .utilities(G::num_players())
            .unwrap_or_else(|| G::compute_utilities(&trial.state));

        (actions, utilities)
    }

    fn estimated_depth(&self) -> usize {
        (self.stats.accum_depth.load(Relaxed) as f64 / self.stats.iter_count.load(Relaxed) as f64)
            .round() as usize
    }

    fn principle_variation(&self) -> Vec<G::A> {
        self.pv.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.config.name = name.to_string();
    }
}
