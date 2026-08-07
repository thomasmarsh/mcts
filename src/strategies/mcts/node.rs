use super::*;
use crate::game::Action;

use rustc_hash::FxHashMap;
use std::str::FromStr;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::*;
use std::sync::OnceLock;
use std::sync::RwLock;

/// MCTS-Solver's proof status for a node's position. `Win(p)` names the
/// winning player by index rather than by `Game::P`, matching `player_idx`'s
/// own convention (see `Node::player_idx`) so `Node<A>` doesn't need a second
/// generic bound just to name a player. "Loss for player p" is deliberately
/// not represented -- with `num_players() <= 2` (this is scoped to that, see
/// `debug_assert!`s where the solver is wired in), "not a win for me" among
/// resolved options collapses unambiguously to "the other player wins", so
/// it's derived at read time instead of stored twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proven {
    Unproven,
    Draw,
    Win(usize),
}

impl Proven {
    const UNPROVEN_U8: u8 = 0;
    const DRAW_U8: u8 = 1;

    fn to_u8(self) -> u8 {
        match self {
            Proven::Unproven => Self::UNPROVEN_U8,
            Proven::Draw => Self::DRAW_U8,
            Proven::Win(p) => 2 + p as u8,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            Self::UNPROVEN_U8 => Proven::Unproven,
            Self::DRAW_U8 => Proven::Draw,
            p => Proven::Win((p - 2) as usize),
        }
    }
}

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

/// Shared by `NodeStats::expected_score` and `ChildSnapshot::expected_score`
/// so the two representations of "one node/child's accumulated stats"
/// (owned lock vs. a row in a `ChildArray`) can't silently drift apart.
#[inline]
fn expected_score_from(num_visits: u32, num_visits_virtual: u32, score: f64) -> f64 {
    if num_visits == 0 {
        0.
    } else {
        let loss_visits = num_visits_virtual as f64;
        (score - loss_visits) / (num_visits as f64 + loss_visits)
    }
}

/// Shared by `NodeStats::value_estimate_unvisited` and
/// `ChildArray::value_estimate_unvisited`. `expected_score` is a closure
/// (rather than a plain `f64`) so the `Parent`+visited case is the only one
/// that actually pays for a stats read.
#[inline]
fn value_estimate_unvisited_from(
    q_init: QInit,
    num_visits: u32,
    expected_score: impl FnOnce() -> f64,
) -> f64 {
    use QInit::*;
    match q_init {
        Draw => 0.,
        Infinity => 10000.0,
        Loss => -1.,
        Parent => {
            if num_visits == 0 {
                10000.
            } else {
                expected_score()
            }
        }
        Win => 1.,
    }
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
        let virtual_loss = self.num_visits_virtual.load(Relaxed);
        expected_score_from(data.num_visits, virtual_loss, data.player[player_index].score)
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn exploitation_score(&self, player_index: usize) -> f64 {
        self.expected_score(player_index)
    }

    // These numbers come from Ludii
    pub fn value_estimate_unvisited(&self, player_index: usize, q_init: QInit) -> f64 {
        value_estimate_unvisited_from(q_init, self.num_visits(), || {
            self.expected_score(player_index)
        })
    }
}

/// One lock acquisition's worth of a single child's stats for a single
/// player -- what `SelectStrategy::score_child` typically needs (visits,
/// score, sum-squared-score, AMAF). Reading these individually through
/// `ChildArray`'s per-field accessors (as `NodeStats`'s edge-owned
/// predecessor required) means a separate lock acquisition per field;
/// `ChildArray::snapshot` takes the lock once and returns all of them
/// together. See MEMORY.md's SoA charter ("composite read win").
#[derive(Debug, Clone, Copy, Default)]
pub struct ChildSnapshot {
    pub num_visits: u32,
    pub num_visits_virtual: u32,
    pub score: f64,
    pub sum_squared_score: f64,
    pub amaf: ActionStats,
}

impl ChildSnapshot {
    pub fn total_visits(&self) -> u32 {
        self.num_visits + self.num_visits_virtual
    }

    pub fn expected_score(&self) -> f64 {
        expected_score_from(self.num_visits, self.num_visits_virtual, self.score)
    }

    pub fn exploitation_score(&self) -> f64 {
        self.expected_score()
    }
}

/// The mutable core of `ChildArray`: one node's worth of children's stats,
/// as flat parallel arrays instead of N independently-locked `NodeStats`
/// (one per child, as a `Vec<Edge<A>>` AoS layout would have). `select`'s
/// hot loop over a node's children takes one read lock total here instead
/// of N -- under tree parallelism, where multiple workers call
/// `select_step` on the same node concurrently, that's an N-fold reduction
/// in lock contention right at the point virtual loss exists to reduce it.
#[derive(Debug, Clone)]
struct ChildArrayData {
    // len == num_children
    num_visits: Vec<u32>,
    // len == num_children * num_players, row-major by child
    player: Vec<PlayerStats>,
}

/// A node's children, stored struct-of-arrays instead of as a
/// `Vec<Edge<A>>` of independently-owned, independently-locked structs.
/// `Node` itself deliberately stays array-of-structs (see `Node::children`'s
/// doc comment) -- this is the one part of the tree with a real dense
/// per-node hot loop (`select` scoring every child of the node it's
/// currently at), which is what makes the SoA layout's cache-locality and
/// lock-consolidation wins actually pay for their extra indexing.
#[derive(Debug)]
pub struct ChildArray<A: Action> {
    actions: Vec<A>,
    child_ids: Vec<OnceLock<index::Id>>,
    // Reverse of `child_ids`, populated as each child is first resolved so
    // `child_index` (id -> idx, needed by every path that only has an `Id`
    // in hand -- backprop walking the stack, tree reuse matching a promoted
    // child, ...) is an O(1) lookup instead of an O(n) scan over
    // `child_ids`. Deliberately not derivable from `child_ids` alone without
    // a scan, hence kept as its own index.
    id_index: RwLock<FxHashMap<index::Id, usize>>,
    // Lock-free, one per child -- same reasoning as `NodeStats`'s field of
    // the same name.
    num_visits_virtual: Vec<AtomicU32>,
    data: RwLock<ChildArrayData>,
    num_players: usize,
}

impl<A: Action> Clone for ChildArray<A> {
    fn clone(&self) -> Self {
        let data = self.data.read().unwrap();
        let id_index = self.id_index.read().unwrap();
        Self {
            actions: self.actions.clone(),
            child_ids: self.child_ids.clone(),
            id_index: RwLock::new(id_index.clone()),
            num_visits_virtual: self
                .num_visits_virtual
                .iter()
                .map(|v| AtomicU32::new(v.load(Relaxed)))
                .collect(),
            data: RwLock::new(data.clone()),
            num_players: self.num_players,
        }
    }
}

impl<A: Action> ChildArray<A> {
    pub fn new(actions: Vec<A>, num_players: usize) -> Self {
        let n = actions.len();
        Self {
            child_ids: (0..n).map(|_| OnceLock::new()).collect(),
            id_index: RwLock::new(FxHashMap::default()),
            num_visits_virtual: (0..n).map(|_| AtomicU32::new(0)).collect(),
            data: RwLock::new(ChildArrayData {
                num_visits: vec![0; n],
                player: vec![PlayerStats::default(); n * num_players],
            }),
            actions,
            num_players,
        }
    }

    #[inline]
    fn player_index(&self, idx: usize, player_index: usize) -> usize {
        idx * self.num_players + player_index
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn action(&self, idx: usize) -> &A {
        &self.actions[idx]
    }

    pub fn is_explored(&self, idx: usize) -> bool {
        self.child_ids[idx].get().is_some()
    }

    pub fn node_id(&self, idx: usize) -> Option<index::Id> {
        self.child_ids[idx].get().copied()
    }

    /// Resolves the edge-creation race: if two threads land on this
    /// unexplored child concurrently, only the first `create` closure to
    /// arrive actually runs (allocating a new arena node); the other blocks
    /// and then reads the same `Id` back, rather than both creating separate
    /// children for the same slot.
    pub fn get_or_create_child(&self, idx: usize, create: impl FnOnce() -> index::Id) -> index::Id {
        // `id_index`'s insert has to happen *inside* the `OnceLock`'s own
        // init closure, not after `get_or_init` returns: `OnceLock`
        // guarantees the closure fully completes before *any* caller --
        // the winner or a blocked racer -- observes the resolved value, on
        // this call or a later one (`OnceLock::get`). Checking
        // `child_ids[idx].get()` first and inserting into `id_index`
        // afterward looked like a harmless fast-path, but it opened a
        // window where another thread's *own* `get_or_create_child`/`get`
        // call on the same idx could observe the resolved id before this
        // call's `id_index` insert had run -- `child_index` on that id
        // would then find nothing. `OnceLock::get_or_init` already has its
        // own lock-free fast path for the already-resolved case, so this
        // isn't giving up meaningful performance, just the (incorrect)
        // extra one.
        *self.child_ids[idx].get_or_init(|| {
            let id = create();
            self.id_index.write().unwrap().insert(id, idx);
            id
        })
    }

    /// O(1) reverse lookup of `child_ids` -- see `id_index`'s doc comment.
    /// Only ever called with an `Id` that was itself returned by
    /// `get_or_create_child` on this same `ChildArray`, so the entry is
    /// always present.
    pub fn child_index(&self, child_id: index::Id) -> usize {
        *self.id_index.read().unwrap().get(&child_id).unwrap()
    }

    pub fn virtual_loss(&self, idx: usize) -> u32 {
        self.num_visits_virtual[idx].load(Relaxed)
    }

    /// See `NodeStats::add_virtual_loss`'s doc comment -- same mechanism,
    /// keyed by array index instead of by an owning struct.
    pub fn add_virtual_loss(&self, idx: usize) {
        self.num_visits_virtual[idx].fetch_add(1, Relaxed);
    }

    pub fn remove_virtual_loss(&self, idx: usize) {
        let prev = self.num_visits_virtual[idx].fetch_sub(1, Relaxed);
        debug_assert!(prev >= 1, "virtual loss removed without a matching add");
    }

    pub fn num_visits(&self, idx: usize) -> u32 {
        self.data.read().unwrap().num_visits[idx]
    }

    pub fn total_visits(&self, idx: usize) -> u32 {
        self.num_visits(idx) + self.virtual_loss(idx)
    }

    pub fn score(&self, idx: usize, player_index: usize) -> f64 {
        self.data.read().unwrap().player[self.player_index(idx, player_index)].score
    }

    pub fn sum_squared_score(&self, idx: usize, player_index: usize) -> f64 {
        self.data.read().unwrap().player[self.player_index(idx, player_index)].sum_squared_score
    }

    pub fn amaf(&self, idx: usize, player_index: usize) -> ActionStats {
        self.data.read().unwrap().player[self.player_index(idx, player_index)].amaf
    }

    pub fn expected_score(&self, idx: usize, player_index: usize) -> f64 {
        let data = self.data.read().unwrap();
        let virtual_loss = self.virtual_loss(idx);
        expected_score_from(
            data.num_visits[idx],
            virtual_loss,
            data.player[self.player_index(idx, player_index)].score,
        )
    }

    pub fn exploitation_score(&self, idx: usize, player_index: usize) -> f64 {
        self.expected_score(idx, player_index)
    }

    pub fn value_estimate_unvisited(&self, idx: usize, player_index: usize, q_init: QInit) -> f64 {
        value_estimate_unvisited_from(q_init, self.num_visits(idx), || {
            self.expected_score(idx, player_index)
        })
    }

    /// One lock acquisition covering every field a `SelectStrategy` typically
    /// needs for one (child, player) pair, instead of the separate lock per
    /// field that reading through `Edge<A>`'s owned `NodeStats` required.
    pub fn snapshot(&self, idx: usize, player_index: usize) -> ChildSnapshot {
        let data = self.data.read().unwrap();
        let p = &data.player[self.player_index(idx, player_index)];
        ChildSnapshot {
            num_visits: data.num_visits[idx],
            num_visits_virtual: self.virtual_loss(idx),
            score: p.score,
            sum_squared_score: p.sum_squared_score,
            amaf: p.amaf,
        }
    }

    pub fn update(&self, idx: usize, utilities: &[f64]) {
        let mut data = self.data.write().unwrap();
        data.num_visits[idx] += 1;
        let base = idx * self.num_players;
        utilities.iter().enumerate().for_each(|(p, reward)| {
            data.player[base + p].score += reward;
            data.player[base + p].sum_squared_score += reward * reward;
        });
    }

    pub fn add_amaf(&self, idx: usize, player_index: usize, utility: f64) {
        let mut data = self.data.write().unwrap();
        let amaf = &mut data.player[self.player_index(idx, player_index)].amaf;
        amaf.num_visits += 1;
        amaf.score += utility;
    }

    /// Lifts one child's accumulated stats out into a standalone
    /// `NodeStats` -- used when tree reuse (`reuse.rs`'s `try_promote`)
    /// promotes a child into the new root. `root_stats` is never itself a
    /// row in some parent's `ChildArray` (the root has no incoming edge),
    /// so promoting a child means copying its row out rather than just
    /// re-pointing a reference.
    pub fn extract_stats(&self, idx: usize) -> NodeStats {
        let data = self.data.read().unwrap();
        let base = idx * self.num_players;
        let player = data.player[base..base + self.num_players].to_vec();
        NodeStats {
            num_visits_virtual: AtomicU32::new(self.virtual_loss(idx)),
            data: RwLock::new(NodeStatsData {
                num_visits: data.num_visits[idx],
                player,
            }),
        }
    }
}

/// A node's stats, viewed either as the standalone root (`root_stats` is
/// never a row in any parent's `ChildArray`, since the root has no incoming
/// edge) or as one child row of a parent's `ChildArray`. Lets
/// `NodeStack::current_stats`/`get_stats` return one type regardless of
/// which case applies, instead of allocating a fresh `NodeStats` for the
/// child case just to unify them.
pub enum StatsRef<'a, A: Action> {
    Root(&'a NodeStats),
    Child(&'a ChildArray<A>, usize),
}

impl<A: Action> StatsRef<'_, A> {
    pub fn num_visits(&self) -> u32 {
        match self {
            StatsRef::Root(s) => s.num_visits(),
            StatsRef::Child(c, i) => c.num_visits(*i),
        }
    }

    pub fn total_visits(&self) -> u32 {
        match self {
            StatsRef::Root(s) => s.total_visits(),
            StatsRef::Child(c, i) => c.total_visits(*i),
        }
    }

    pub fn value_estimate_unvisited(&self, player_index: usize, q_init: QInit) -> f64 {
        match self {
            StatsRef::Root(s) => s.value_estimate_unvisited(player_index, q_init),
            StatsRef::Child(c, i) => c.value_estimate_unvisited(*i, player_index, q_init),
        }
    }
}

#[derive(Clone, Debug)]
pub enum NodeState<A: Action> {
    Terminal,
    Expanded(ChildArray<A>),
}

#[derive(Debug)]
pub struct Node<A: Action> {
    pub player_idx: usize,
    pub hash: u64,
    pub is_root: bool,
    // Unset == not yet expanded ("leaf"). `expand` resolves this exactly
    // once via `get_or_init`, so concurrent threads landing on the same
    // unexpanded node race for free: only the winner runs `G::is_terminal`/
    // `generate_actions`, the rest block and read its result.
    state: OnceLock<NodeState<A>>,
    // MCTS-Solver proof status, `0 = Unproven` by default. Not a `OnceLock`:
    // unlike `state`, there's no real init work to gate behind "first caller
    // wins" -- concurrent derivations of the same node can't disagree, so a
    // plain compare-exchange-from-Unproven is simpler and sufficient.
    // `Relaxed` throughout, matching
    // `num_visits_virtual` on `NodeStats` above.
    proven: AtomicU8,
}

// Manual impl: `AtomicU8` isn't `Clone`, so this can no longer be derived.
impl<A: Action> Clone for Node<A> {
    fn clone(&self) -> Self {
        Self {
            player_idx: self.player_idx,
            hash: self.hash,
            is_root: self.is_root,
            state: self.state.clone(),
            proven: AtomicU8::new(self.proven.load(Relaxed)),
        }
    }
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
            proven: AtomicU8::new(Proven::UNPROVEN_U8),
        }
    }

    #[inline]
    pub fn proven(&self) -> Proven {
        Proven::from_u8(self.proven.load(Relaxed))
    }

    /// Writes `status`, but only if this node is still `Unproven` -- once
    /// proven, a node's status is final. Safe to call redundantly from
    /// multiple threads deriving the same (correct) status concurrently: a
    /// CAS that loses the race is a harmless no-op, not a conflict --
    /// concurrent derivations of a fixed, real set of children can't
    /// disagree.
    pub fn try_prove(&self, status: Proven) {
        debug_assert_ne!(status, Proven::Unproven);
        let _ = self
            .proven
            .compare_exchange(Proven::UNPROVEN_U8, status.to_u8(), Relaxed, Relaxed);
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
    /// calls `children()` on it.
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
    pub fn children(&self) -> &ChildArray<A> {
        let Some(NodeState::Expanded(children)) = self.state.get() else {
            unreachable!()
        };
        children
    }

    pub fn child_index(&self, child_id: index::Id) -> usize {
        self.children().child_index(child_id)
    }

    pub fn new_root(player: usize, num_players: usize, hash: u64) -> Self {
        debug_assert!((num_players == 0 && player == 0) || player < num_players);
        Self {
            is_root: true,
            ..Self::new(player, hash)
        }
    }

    pub fn is_root(&self) -> bool {
        self.is_root
    }
}
