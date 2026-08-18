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

#[derive(Debug, Clone, Default)]
pub struct PlayerStats {
    pub score: f64,
    pub sum_squared_score: f64,
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
    // Side table, parallel to `player`. Only populated (length
    // `player.len()`) when `has_amaf` is set on the owning `NodeStats`;
    // left empty otherwise so a `Strategy` that never requests AMAF doesn't
    // pay for it.
    amaf: Vec<ActionStats>,
}

#[derive(Debug)]
pub struct NodeStats {
    // For virtual loss -- lock-free since it's touched on every descent/
    // backprop, the hottest path in tree-parallel search.
    pub num_visits_virtual: AtomicU32,

    // Set once at construction from `Requirements.amaf`; gates whether
    // `data.amaf` is populated and whether the accessors below read it.
    has_amaf: bool,
    data: RwLock<NodeStatsData>,
}

impl Clone for NodeStats {
    fn clone(&self) -> Self {
        let data = self.data.read().unwrap();
        Self {
            num_visits_virtual: AtomicU32::new(self.num_visits_virtual.load(Relaxed)),
            has_amaf: self.has_amaf,
            data: RwLock::new(data.clone()),
        }
    }
}

impl NodeStats {
    pub fn new(num_players: usize, has_amaf: bool) -> Self {
        Self {
            num_visits_virtual: AtomicU32::new(0),
            has_amaf,
            data: RwLock::new(NodeStatsData {
                num_visits: 0,
                player: vec![PlayerStats::default(); num_players],
                amaf: if has_amaf {
                    vec![ActionStats::default(); num_players]
                } else {
                    Vec::new()
                },
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

    pub fn snapshot(&self, player_index: usize) -> ChildSnapshot {
        let data = self.data.read().unwrap();
        let p = &data.player[player_index];
        ChildSnapshot {
            num_visits: data.num_visits,
            num_visits_virtual: self.num_visits_virtual.load(Relaxed),
            score: p.score,
            sum_squared_score: p.sum_squared_score,
            amaf: if self.has_amaf {
                data.amaf[player_index]
            } else {
                ActionStats::default()
            },
        }
    }

    // NOTE: needs to be overridden for score bounded search
    pub fn expected_score(&self, player_index: usize) -> f64 {
        let data = self.data.read().unwrap();
        let virtual_loss = self.num_visits_virtual.load(Relaxed);
        expected_score_from(
            data.num_visits,
            virtual_loss,
            data.player[player_index].score,
        )
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
    // Side table, parallel to `player`. Only populated (same length as
    // `player`) when `has_amaf` is set on the owning `ChildArray`; left
    // empty otherwise so a `Strategy` that never requests AMAF doesn't pay
    // for it -- this is the actually-multiplied structure (num_children *
    // num_players), so this is the real payoff of gating on `has_amaf`.
    amaf: Vec<ActionStats>,
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
    // Set once at construction from `Requirements.amaf`; gates whether
    // `data.amaf` is populated and whether the accessors below read it.
    has_amaf: bool,
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
            has_amaf: self.has_amaf,
        }
    }
}

impl<A: Action> ChildArray<A> {
    pub fn new(actions: Vec<A>, num_players: usize, has_amaf: bool) -> Self {
        let n = actions.len();
        Self {
            child_ids: (0..n).map(|_| OnceLock::new()).collect(),
            id_index: RwLock::new(FxHashMap::default()),
            num_visits_virtual: (0..n).map(|_| AtomicU32::new(0)).collect(),
            data: RwLock::new(ChildArrayData {
                num_visits: vec![0; n],
                player: vec![PlayerStats::default(); n * num_players],
                amaf: if has_amaf {
                    vec![ActionStats::default(); n * num_players]
                } else {
                    Vec::new()
                },
            }),
            actions,
            num_players,
            has_amaf,
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
        if !self.has_amaf {
            return ActionStats::default();
        }
        self.data.read().unwrap().amaf[self.player_index(idx, player_index)]
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
        let i = self.player_index(idx, player_index);
        let p = &data.player[i];
        ChildSnapshot {
            num_visits: data.num_visits[idx],
            num_visits_virtual: self.virtual_loss(idx),
            score: p.score,
            sum_squared_score: p.sum_squared_score,
            amaf: if self.has_amaf {
                data.amaf[i]
            } else {
                ActionStats::default()
            },
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
        if !self.has_amaf {
            return;
        }
        let mut data = self.data.write().unwrap();
        let i = self.player_index(idx, player_index);
        let amaf = &mut data.amaf[i];
        amaf.num_visits += 1;
        amaf.score += utility;
    }

    /// Number of child slots with a resolved arena node (`is_explored`) --
    /// i.e. actually visited at least once, vs. merely allocated as a legal
    /// action when the parent expanded. `len() - explored_len()` is exactly
    /// what a scheme that only allocates a child slot once an action has
    /// actually been sampled (e.g. progressive widening) would avoid
    /// allocating. Diagnostics-only (memory profiling).
    pub fn explored_len(&self) -> usize {
        (0..self.len()).filter(|&idx| self.is_explored(idx)).count()
    }

    /// Rough estimate, in bytes, of this child array's heap footprint: every
    /// parallel array/map this struct owns, at its current length, ignoring
    /// allocator and hashmap bucket overhead. Diagnostics-only (memory
    /// profiling) -- not precise enough to drive anything but a relative
    /// comparison between categories.
    pub fn heap_bytes_estimate(&self) -> usize {
        let n = self.len();
        let explored = self.explored_len();
        n * std::mem::size_of::<A>()
            + n * std::mem::size_of::<OnceLock<index::Id>>()
            + explored * (std::mem::size_of::<index::Id>() + std::mem::size_of::<usize>())
            + n * std::mem::size_of::<AtomicU32>()
            + n * std::mem::size_of::<u32>()
            + n * self.num_players * std::mem::size_of::<PlayerStats>()
            + if self.has_amaf {
                n * self.num_players * std::mem::size_of::<ActionStats>()
            } else {
                0
            }
    }

    /// Rewrites every resolved child id through `old_to_new`, and rebuilds
    /// `id_index` to match -- used by arena compaction (`search/compact.rs`'s
    /// `TreeSearch::compact`) after cloning this array's owning node into a
    /// freshly built, garbage-free arena, when the array's `child_ids` still
    /// point at the *old* arena's ids. Every resolved id here is guaranteed
    /// present in `old_to_new`: compaction's reachability walk enqueues every
    /// explored child of every node it visits, so no id resolved here can be
    /// missing from the map.
    pub fn remap_child_ids(&mut self, old_to_new: &FxHashMap<index::Id, index::Id>) {
        let mut new_id_index = FxHashMap::default();
        for (idx, slot) in self.child_ids.iter_mut().enumerate() {
            if let Some(old_id) = slot.get().copied() {
                let new_id = *old_to_new
                    .get(&old_id)
                    .expect("compaction: reachable child missing from id map");
                *slot = OnceLock::from(new_id);
                new_id_index.insert(new_id, idx);
            }
        }
        self.id_index = RwLock::new(new_id_index);
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
        let amaf = if self.has_amaf {
            data.amaf[base..base + self.num_players].to_vec()
        } else {
            Vec::new()
        };
        NodeStats {
            num_visits_virtual: AtomicU32::new(self.virtual_loss(idx)),
            has_amaf: self.has_amaf,
            data: RwLock::new(NodeStatsData {
                num_visits: data.num_visits[idx],
                player,
                amaf,
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
    Node(&'a NodeStats),
    Child(&'a ChildArray<A>, usize),
}

impl<A: Action> StatsRef<'_, A> {
    pub fn num_visits(&self) -> u32 {
        match self {
            StatsRef::Root(s) | StatsRef::Node(s) => s.num_visits(),
            StatsRef::Child(c, i) => c.num_visits(*i),
        }
    }

    pub fn total_visits(&self) -> u32 {
        match self {
            StatsRef::Root(s) | StatsRef::Node(s) => s.total_visits(),
            StatsRef::Child(c, i) => c.total_visits(*i),
        }
    }

    pub fn value_estimate_unvisited(&self, player_index: usize, q_init: QInit) -> f64 {
        match self {
            StatsRef::Root(s) | StatsRef::Node(s) => {
                s.value_estimate_unvisited(player_index, q_init)
            }
            StatsRef::Child(c, i) => c.value_estimate_unvisited(*i, player_index, q_init),
        }
    }
}

// `ChildArray<A>` carries its stats inline (behind an `RwLock`, not boxed),
// so `Terminal` is far smaller than `Expanded` by construction, not by
// oversight -- boxing `ChildArray<A>` here would ripple into every
// `NodeState::Expanded` construction site across the crate for a variant
// that's already the common case once a node has been visited once.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum NodeState<A: Action> {
    Terminal,
    Expanded(ChildArray<A>),
}

// MCTS-Solver proof status plus PN-MCTS (Kowalski et al. 2023) proof/
// disproof numbers, split out of `Node` into their own boxed, optional block
// (`Node::solver`) so a search with `use_mcts_solver` off doesn't pay for
// them on every node -- only the pointer-width `None` tag. `pn`/`dpn` are
// seeded at `1` (PNS's "unknown leaf" case) rather than `0 = Unproven`; see
// `Node::pn`/`Node::dpn`'s doc comments for why these fields are only read
// directly while `proven` is still `Unproven`. `proven` itself is write-once
// (`Node::try_prove` only ever transitions away from `Unproven`), but
// `pn`/`dpn`/`pn2`/`dpn2` are not: `set_pn_dpn`/`set_pn_dpn2` (called from
// `derive_pn_dpn`/`derive_pn_dpn2` in backprop.rs) overwrite them every time
// this node's children's counts change, only settling once `proven` itself
// resolves. `pn2`/`dpn2` are the second-layer numbers (Kowalski et al. 2023,
// Section VII "Double-Layer PN-MCTS"), tracking "not lost" (won or drawn)
// instead of `pn`/`dpn`'s "won" -- see `Node::pn2`/`Node::dpn2`'s doc
// comments. `Relaxed` throughout, matching `num_visits_virtual` on
// `NodeStats`.
#[derive(Debug)]
struct SolverState {
    proven: AtomicU8,
    pn: AtomicU32,
    dpn: AtomicU32,
    pn2: AtomicU32,
    dpn2: AtomicU32,
}

impl SolverState {
    fn unproven() -> Self {
        Self {
            proven: AtomicU8::new(Proven::UNPROVEN_U8),
            pn: AtomicU32::new(1),
            dpn: AtomicU32::new(1),
            pn2: AtomicU32::new(1),
            dpn2: AtomicU32::new(1),
        }
    }
}

// Manual impl: `AtomicU8`/`AtomicU32` aren't `Clone`.
impl Clone for SolverState {
    fn clone(&self) -> Self {
        Self {
            proven: AtomicU8::new(self.proven.load(Relaxed)),
            pn: AtomicU32::new(self.pn.load(Relaxed)),
            dpn: AtomicU32::new(self.dpn.load(Relaxed)),
            pn2: AtomicU32::new(self.pn2.load(Relaxed)),
            dpn2: AtomicU32::new(self.dpn2.load(Relaxed)),
        }
    }
}

#[derive(Debug)]
pub struct Node<A: Action> {
    pub player_idx: usize,
    pub hash: u64,
    pub ply: u32,
    pub is_root: bool,
    pub stats: NodeStats,
    incoming_edges: AtomicU32,
    // Unset == not yet expanded ("leaf"). `expand` resolves this exactly
    // once via `get_or_init`, so concurrent threads landing on the same
    // unexpanded node race for free: only the winner runs `G::is_terminal`/
    // `generate_actions`, the rest block and read its result.
    state: OnceLock<NodeState<A>>,
    // `None` when `SearchConfig::use_mcts_solver` is off for the active
    // search -- see `SolverState`'s doc comment. Decided once at
    // construction from that same flag, never populated afterward.
    solver: Option<Box<SolverState>>,
}

// Manual impl: `AtomicU8`/`AtomicU32` aren't `Clone`, so this can no longer
// be derived.
impl<A: Action> Clone for Node<A> {
    fn clone(&self) -> Self {
        Self {
            player_idx: self.player_idx,
            hash: self.hash,
            ply: self.ply,
            is_root: self.is_root,
            stats: self.stats.clone(),
            incoming_edges: AtomicU32::new(self.incoming_edges.load(Relaxed)),
            state: self.state.clone(),
            solver: self.solver.clone(),
        }
    }
}

impl<A: Action> Node<A>
where
    A: Clone + std::hash::Hash,
{
    pub fn new(player_idx: usize, hash: u64) -> Self {
        Self::new_at_ply(player_idx, hash, 0, 2, true, true)
    }

    pub fn new_at_ply(
        player_idx: usize,
        hash: u64,
        ply: u32,
        num_players: usize,
        has_amaf: bool,
        has_solver: bool,
    ) -> Self {
        Self {
            player_idx,
            hash,
            ply,
            is_root: false,
            stats: NodeStats::new(num_players, has_amaf),
            incoming_edges: AtomicU32::new(0),
            state: OnceLock::new(),
            solver: has_solver.then(|| Box::new(SolverState::unproven())),
        }
    }

    #[inline]
    pub fn add_incoming_edge(&self) {
        self.incoming_edges.fetch_add(1, Relaxed);
    }

    #[inline]
    pub fn incoming_edges(&self) -> u32 {
        self.incoming_edges.load(Relaxed)
    }

    #[inline]
    pub fn is_transposition(&self) -> bool {
        self.incoming_edges() > 1
    }

    #[inline]
    pub fn proven(&self) -> Proven {
        self.solver.as_ref().map_or(Proven::Unproven, |s| {
            Proven::from_u8(s.proven.load(Relaxed))
        })
    }

    /// Writes `status`, but only if this node is still `Unproven` -- once
    /// proven, a node's status is final. Safe to call redundantly from
    /// multiple threads deriving the same (correct) status concurrently: a
    /// CAS that loses the race is a harmless no-op, not a conflict --
    /// concurrent derivations of a fixed, real set of children can't
    /// disagree. No-ops when the solver is off for this search (`solver` is
    /// `None`) -- callers only reach here from code already gated on
    /// `use_mcts_solver`, so this is a defensive fallback, not a live path.
    pub fn try_prove(&self, status: Proven) {
        debug_assert_ne!(status, Proven::Unproven);
        let Some(solver) = self.solver.as_ref() else {
            return;
        };
        let _ =
            solver
                .proven
                .compare_exchange(Proven::UNPROVEN_U8, status.to_u8(), Relaxed, Relaxed);
    }

    /// Proof number: PN-MCTS's (Kowalski et al. 2023) generalization of
    /// `proven()` from a ternary status to a magnitude -- the minimum number
    /// of leaf nodes that still need to resolve to prove *this node's own
    /// mover* forces a win. `0` once `proven()` is `Win(player_idx)`;
    /// saturated (`u32::MAX`) once it's proven anything else (`Draw` or the
    /// opponent's win -- see `Proven`'s doc comment on why "loss" isn't
    /// stored separately); otherwise the live count `derive_pn_dpn`
    /// (backprop.rs) maintains, seeded at `1` for an unvisited leaf (PNS's
    /// "unknown leaf" case). Only meaningful when `use_mcts_solver` is on --
    /// with it off, `proven()` never leaves `Unproven` and `pn`/`dpn` never
    /// move off their seed value (see `select::UctPn`'s doc comment).
    #[inline]
    pub fn pn(&self) -> u32 {
        match self.proven() {
            Proven::Win(w) if w == self.player_idx => 0,
            Proven::Win(_) | Proven::Draw => u32::MAX,
            Proven::Unproven => self.solver.as_ref().map_or(1, |s| s.pn.load(Relaxed)),
        }
    }

    /// Disproof number -- the mirror image of `pn()` (see its doc comment):
    /// the minimum number of leaf nodes that still need to resolve to
    /// disprove this node's own mover forces a win.
    #[inline]
    pub fn dpn(&self) -> u32 {
        match self.proven() {
            Proven::Win(w) if w == self.player_idx => u32::MAX,
            Proven::Win(_) | Proven::Draw => 0,
            Proven::Unproven => self.solver.as_ref().map_or(1, |s| s.dpn.load(Relaxed)),
        }
    }

    /// Overwrites the live proof/disproof counts. Called only from
    /// `derive_pn_dpn` (backprop.rs); see the `pn`/`dpn` fields' doc comment
    /// for why this isn't write-once like `try_prove`. No-ops when the
    /// solver is off, same as `try_prove`.
    #[inline]
    pub fn set_pn_dpn(&self, pn: u32, dpn: u32) {
        let Some(solver) = self.solver.as_ref() else {
            return;
        };
        solver.pn.store(pn, Relaxed);
        solver.dpn.store(dpn, Relaxed);
    }

    /// Second-layer proof number (Kowalski et al. 2023, Section VII): the
    /// same PNS magnitude as `pn()`, but for the goal "not lost" (won or
    /// drawn) instead of "won". This is what lets PN-MCTS distinguish a
    /// drawn subtree from a lost one in games with draws -- `pn()`/`dpn()`
    /// alone can't, since `Proven::Win(_)` (the opponent's win) and
    /// `Proven::Draw` both collapse to the same "disproven" magnitude there
    /// (see `pn()`'s doc comment and the paper's Table II). `0` once
    /// `proven()` is `Win(player_idx)` *or* `Draw` (both satisfy "not
    /// lost"); saturated once it's the opponent's win (the only way to
    /// actually lose); otherwise the live count `derive_pn_dpn2`
    /// (backprop.rs) maintains, seeded at `1` like `pn()`. Only meaningful
    /// when `use_mcts_solver` is on, same caveat as `pn()`.
    #[inline]
    pub fn pn2(&self) -> u32 {
        match self.proven() {
            Proven::Win(w) if w == self.player_idx => 0,
            Proven::Draw => 0,
            Proven::Win(_) => u32::MAX,
            Proven::Unproven => self.solver.as_ref().map_or(1, |s| s.pn2.load(Relaxed)),
        }
    }

    /// Second-layer disproof number -- the mirror image of `pn2()`, i.e. the
    /// PNS magnitude for "lost" (the negation of "not lost").
    #[inline]
    pub fn dpn2(&self) -> u32 {
        match self.proven() {
            Proven::Win(w) if w == self.player_idx => u32::MAX,
            Proven::Draw => u32::MAX,
            Proven::Win(_) => 0,
            Proven::Unproven => self.solver.as_ref().map_or(1, |s| s.dpn2.load(Relaxed)),
        }
    }

    /// Overwrites the live second-layer proof/disproof counts. Called only
    /// from `derive_pn_dpn2` (backprop.rs); mirrors `set_pn_dpn`, including
    /// the solver-off no-op case.
    #[inline]
    pub fn set_pn_dpn2(&self, pn2: u32, dpn2: u32) {
        let Some(solver) = self.solver.as_ref() else {
            return;
        };
        solver.pn2.store(pn2, Relaxed);
        solver.dpn2.store(dpn2, Relaxed);
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

    /// Mutable access to this node's `ChildArray`, if expanded -- `None` for
    /// a Leaf or Terminal node. Only used by arena compaction
    /// (`search/compact.rs`), which owns its freshly built arena exclusively
    /// (no concurrent search in flight), so `OnceLock::get_mut` -- which
    /// needs no locking, unlike every other read path here -- is sound.
    #[inline]
    pub(crate) fn children_mut(&mut self) -> Option<&mut ChildArray<A>> {
        match self.state.get_mut() {
            Some(NodeState::Expanded(children)) => Some(children),
            _ => None,
        }
    }

    pub fn new_root(
        player: usize,
        num_players: usize,
        hash: u64,
        has_amaf: bool,
        has_solver: bool,
    ) -> Self {
        debug_assert!((num_players == 0 && player == 0) || player < num_players);
        Self {
            is_root: true,
            ..Self::new_at_ply(player, hash, 0, num_players, has_amaf, has_solver)
        }
    }

    pub fn is_root(&self) -> bool {
        self.is_root
    }

    /// Test-only introspection hook: whether the solver side block was
    /// actually allocated, mirroring `ChildArray`/`NodeStats`'s
    /// `heap_bytes_estimate`-based check for the AMAF side table -- there's
    /// no per-node heap-bytes accounting to piggyback on here, so this is a
    /// direct (but crate-private) look at `solver` instead.
    #[cfg(test)]
    pub(crate) fn has_solver(&self) -> bool {
        self.solver.is_some()
    }
}
