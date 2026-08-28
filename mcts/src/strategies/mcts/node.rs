use super::*;
use crate::game::Action;
use crate::game::Canonical;
use crate::game::Game;
use crate::game::Transform;

use rustc_hash::FxHashMap;
use std::str::FromStr;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::*;
use std::sync::OnceLock;
use std::sync::RwLock;

/// MCTS-Solver's proof status for a node's position. `Win(p)` names the
/// winning player by index rather than by `Game::P`, matching `player_idx`'s
/// own convention (see `Node::player_idx`) so `Node<A>` doesn't need a second
/// generic bound just to name a player. "Loss for player p" is deliberately
/// not represented at all, at any player count: `Win(w)` for `w != p`, or
/// `Draw`, already unambiguously mean "not a win for p" wherever that's what
/// a reader (e.g. `Node::pn`/`Node::dpn`) needs, without naming *which*
/// other player wins -- see `backprop::derive_proven`'s doc comment for how
/// a node with more than one possible opponent decides whether it can name
/// one at all (its "Standard" update rule only does so when every resolved
/// child agrees).
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
    // Bayesian posterior (mean, variance) of this player's true value at
    // this node/edge -- only written by `BayesGaussian`/`BayesNumeric`
    // backprop (`backprop.rs`), read by `BayesUct1`/`BayesUct2`
    // (`select/bayes.rs`). Left at `0.0` (meaningless, never read) for
    // every other strategy pairing -- `Requirements::needs_posterior` +
    // `BackpropStrategy::provides_posterior` (`config.rs`) reject any
    // config where a select strategy that reads these is paired with a
    // backprop strategy that doesn't write them.
    pub posterior_mean: f64,
    pub posterior_variance: f64,
    // `BayesNumeric` backprop's discretized posterior PDF over
    // `bayes::BAYES_GRID_SIZE` points, lazily allocated on first write so
    // every other strategy pairing (including `BayesGaussian`, which only
    // ever needs `posterior_mean`/`posterior_variance`) pays nothing but
    // this one pointer-sized `None`.
    pub posterior_grid: Option<Box<[f64; super::backprop::BAYES_GRID_SIZE]>>,
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

    /// Bayesian posterior `(mean, variance)` -- see `PlayerStats::posterior_mean`'s
    /// doc comment. `(0.0, 0.0)` unless a `BayesGaussian`/`BayesNumeric`
    /// backprop strategy has written it.
    pub fn posterior(&self, player_index: usize) -> (f64, f64) {
        let p = &self.data.read().unwrap().player[player_index];
        (p.posterior_mean, p.posterior_variance)
    }

    pub fn set_posterior(&self, player_index: usize, mean: f64, variance: f64) {
        let mut data = self.data.write().unwrap();
        let p = &mut data.player[player_index];
        p.posterior_mean = mean;
        p.posterior_variance = variance;
    }

    pub fn posterior_grid(
        &self,
        player_index: usize,
    ) -> Option<[f64; super::backprop::BAYES_GRID_SIZE]> {
        self.data.read().unwrap().player[player_index]
            .posterior_grid
            .as_deref()
            .copied()
    }

    pub fn set_posterior_grid(
        &self,
        player_index: usize,
        grid: [f64; super::backprop::BAYES_GRID_SIZE],
    ) {
        self.data.write().unwrap().player[player_index].posterior_grid = Some(Box::new(grid));
    }

    pub fn total_visits(&self) -> u32 {
        self.num_visits() + self.num_visits_virtual.load(Relaxed)
    }

    /// Overwrites this node's own accumulated score for `player_index` so
    /// its next `expected_score` read returns exactly `mean` -- used by
    /// `backprop::MinimaxBackprop` (MCTS-MB-n) to replace a node's Monte-
    /// Carlo average with a minimax-derived value in place, rather than
    /// blending it in the way `update` does. Leaves `num_visits`/
    /// `sum_squared_score` untouched: a later real playout through this
    /// node still accumulates onto the same visit count, and a variance-
    /// based select strategy reading `sum_squared_score` (UCB1-Tuned, RAVE)
    /// keeps seeing the real sample spread rather than one implied by the
    /// overwritten mean -- the same known approximation `expected_score`'s
    /// own "needs to be overridden for score bounded search" note already
    /// flags.
    pub fn overwrite_score(&self, player_index: usize, mean: f64) {
        let mut data = self.data.write().unwrap();
        let n = data.num_visits.max(1) as f64;
        data.player[player_index].score = mean * n;
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
            posterior_mean: p.posterior_mean,
            posterior_variance: p.posterior_variance,
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
    pub posterior_mean: f64,
    pub posterior_variance: f64,
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
    // ISMCTS's per-child availability count (Cowling, Powley & Whitehouse
    // 2012): how many iterations' determinized sample made this action
    // legal at this node, as opposed to `num_visits`'s "how many iterations
    // actually chose it". Only populated (same length as `num_visits`) when
    // `growable` is set on the owning `ChildArray`, same gating as `amaf`.
    availability: Vec<u32>,
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
    // Set once at construction from `SearchConfig::ismcts_mode != IsmctsMode::
    // Off`. `false` for every ordinary search: `extra` then stays
    // permanently empty and every accessor below resolves entirely within
    // the fixed-size fields above, unchanged from before this type
    // supported growth at all. Only an ISMCTS search -- one shared tree
    // (`IsmctsMode::SingleTree`, `search/shared.rs::select_step`) or one
    // tree per player (`IsmctsMode::MultiTree`, `search/multi_tree.rs`),
    // either way walked under a fresh `Game::determinize` sample every
    // iteration -- ever calls `grow`, since only there can a later
    // iteration's legal-move set contain an action no earlier iteration
    // reaching this node ever saw.
    growable: bool,
    // Children discovered by `grow` after construction, appended here
    // instead of into `actions`/`child_ids`/`num_visits_virtual`/`data` so
    // every pre-existing slot keeps its lock-free access path -- growth pays
    // for one `RwLock` acquisition per access to a *new* slot, not to the
    // whole array. Indices `>= actions.len()` resolve here, offset by
    // `actions.len()`.
    extra: RwLock<Vec<ExtraChild<A>>>,
}

impl<A: Action> Clone for ChildArray<A> {
    fn clone(&self) -> Self {
        let data = self.data.read().unwrap();
        let id_index = self.id_index.read().unwrap();
        let extra = self.extra.read().unwrap();
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
            growable: self.growable,
            extra: RwLock::new(extra.iter().map(ExtraChild::clone_from).collect()),
        }
    }
}

/// One child slot appended after construction by `ChildArray::grow` -- see
/// `ChildArray::extra`'s doc comment. Its own `RwLock` per mutable field
/// (rather than one lock for the whole slot) so that scoring one appended
/// child doesn't block a concurrent visit/score update to a sibling appended
/// child, matching the granularity `ChildArrayData` gives the fixed range.
#[derive(Debug)]
struct ExtraChild<A: Action> {
    action: A,
    child_id: OnceLock<index::Id>,
    num_visits_virtual: AtomicU32,
    data: RwLock<ExtraChildData>,
}

#[derive(Debug, Clone)]
struct ExtraChildData {
    num_visits: u32,
    availability: u32,
    player: Vec<PlayerStats>,
}

impl<A: Action> ExtraChild<A> {
    fn new(action: A, num_players: usize) -> Self {
        Self {
            action,
            child_id: OnceLock::new(),
            num_visits_virtual: AtomicU32::new(0),
            data: RwLock::new(ExtraChildData {
                num_visits: 0,
                availability: 0,
                player: vec![PlayerStats::default(); num_players],
            }),
        }
    }

    fn clone_from(this: &Self) -> Self {
        Self {
            action: this.action.clone(),
            child_id: this.child_id.clone(),
            num_visits_virtual: AtomicU32::new(this.num_visits_virtual.load(Relaxed)),
            data: RwLock::new(this.data.read().unwrap().clone()),
        }
    }
}

impl<A: Action> ChildArray<A> {
    pub fn new(actions: Vec<A>, num_players: usize, has_amaf: bool, growable: bool) -> Self {
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
                availability: if growable { vec![0; n] } else { Vec::new() },
            }),
            actions,
            num_players,
            has_amaf,
            growable,
            extra: RwLock::new(Vec::new()),
        }
    }

    #[inline]
    fn player_index(&self, idx: usize, player_index: usize) -> usize {
        idx * self.num_players + player_index
    }

    /// Number of slots present in the fixed-size fields -- the range every
    /// accessor below resolves lock-free, before falling back to `extra`.
    #[inline]
    fn base_len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_growable(&self) -> bool {
        self.growable
    }

    pub fn len(&self) -> usize {
        self.base_len() + self.extra.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn action(&self, idx: usize) -> A {
        let base_len = self.base_len();
        if idx < base_len {
            self.actions[idx].clone()
        } else {
            self.extra.read().unwrap()[idx - base_len].action.clone()
        }
    }

    /// Widens this array to include every action in `new_actions`, appending
    /// whichever of them (by `Eq`) aren't already present -- base or
    /// previously-appended -- as a fresh `extra` slot, and returns each
    /// input action's resulting index (existing or newly created) in the
    /// same order. Only ever called on a `growable` array (ISMCTS's
    /// per-iteration "restrict to compatible children" step, `search/
    /// shared.rs::select_step`): a non-growable array's action set is fixed
    /// at construction, exactly as every search other than ISMCTS assumes.
    pub fn grow(&self, new_actions: &[A]) -> Vec<usize> {
        debug_assert!(
            self.growable,
            "grow() called on a non-growable ChildArray -- only an ISMCTS search should ever \
             widen a node's action set after construction"
        );
        let base_len = self.base_len();
        let mut extra = self.extra.write().unwrap();
        new_actions
            .iter()
            .map(|action| {
                if let Some(i) = self.actions.iter().position(|a| a == action) {
                    return i;
                }
                if let Some(i) = extra.iter().position(|c| &c.action == action) {
                    return base_len + i;
                }
                extra.push(ExtraChild::new(action.clone(), self.num_players));
                base_len + extra.len() - 1
            })
            .collect()
    }

    pub fn availability(&self, idx: usize) -> u32 {
        let base_len = self.base_len();
        if idx < base_len {
            self.data.read().unwrap().availability[idx]
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .data
                .read()
                .unwrap()
                .availability
        }
    }

    pub fn add_availability(&self, idx: usize) {
        let base_len = self.base_len();
        if idx < base_len {
            self.data.write().unwrap().availability[idx] += 1;
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .data
                .write()
                .unwrap()
                .availability += 1;
        }
    }

    /// Rewrites every action in place via `f` -- used only when a graph
    /// node promotes to root (`search/reroot.rs`'s DAG re-rooting). Its
    /// actions were generated in *its own* canonical orientation (see
    /// `expand`'s doc comment: every non-root node's action list is), but
    /// `symmetry::incoming_sym` hard-codes `Transform::IDENTITY` for any node
    /// with `is_root() == true`, on the assumption that a from-birth root's
    /// actions are already directly playable against the real state. A
    /// promoted node wasn't root when its actions were generated, so that
    /// assumption doesn't hold for it until this translates them back to
    /// the literal board exactly once, before `is_root` flips and every
    /// future `incoming_sym`/`real_action` call for this node stops
    /// translating at all.
    pub(crate) fn retranslate_actions(&mut self, mut f: impl FnMut(&A) -> A) {
        for a in self.actions.iter_mut() {
            *a = f(a);
        }
    }

    pub fn is_explored(&self, idx: usize) -> bool {
        let base_len = self.base_len();
        if idx < base_len {
            self.child_ids[idx].get().is_some()
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .child_id
                .get()
                .is_some()
        }
    }

    pub fn node_id(&self, idx: usize) -> Option<index::Id> {
        let base_len = self.base_len();
        if idx < base_len {
            self.child_ids[idx].get().copied()
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .child_id
                .get()
                .copied()
        }
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
        let base_len = self.base_len();
        if idx < base_len {
            *self.child_ids[idx].get_or_init(|| {
                let id = create();
                self.id_index.write().unwrap().insert(id, idx);
                id
            })
        } else {
            *self.extra.read().unwrap()[idx - base_len]
                .child_id
                .get_or_init(|| {
                    let id = create();
                    self.id_index.write().unwrap().insert(id, idx);
                    id
                })
        }
    }

    /// O(1) reverse lookup of `child_ids` -- see `id_index`'s doc comment.
    /// Only ever called with an `Id` that was itself returned by
    /// `get_or_create_child` on this same `ChildArray`, so the entry is
    /// always present.
    pub fn child_index(&self, child_id: index::Id) -> usize {
        *self.id_index.read().unwrap().get(&child_id).unwrap()
    }

    pub fn virtual_loss(&self, idx: usize) -> u32 {
        let base_len = self.base_len();
        if idx < base_len {
            self.num_visits_virtual[idx].load(Relaxed)
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .num_visits_virtual
                .load(Relaxed)
        }
    }

    /// See `NodeStats::add_virtual_loss`'s doc comment -- same mechanism,
    /// keyed by array index instead of by an owning struct.
    pub fn add_virtual_loss(&self, idx: usize) {
        let base_len = self.base_len();
        if idx < base_len {
            self.num_visits_virtual[idx].fetch_add(1, Relaxed);
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .num_visits_virtual
                .fetch_add(1, Relaxed);
        }
    }

    pub fn remove_virtual_loss(&self, idx: usize) {
        let base_len = self.base_len();
        let prev = if idx < base_len {
            self.num_visits_virtual[idx].fetch_sub(1, Relaxed)
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .num_visits_virtual
                .fetch_sub(1, Relaxed)
        };
        debug_assert!(prev >= 1, "virtual loss removed without a matching add");
    }

    pub fn num_visits(&self, idx: usize) -> u32 {
        let base_len = self.base_len();
        if idx < base_len {
            self.data.read().unwrap().num_visits[idx]
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .data
                .read()
                .unwrap()
                .num_visits
        }
    }

    pub fn total_visits(&self, idx: usize) -> u32 {
        self.num_visits(idx) + self.virtual_loss(idx)
    }

    pub fn score(&self, idx: usize, player_index: usize) -> f64 {
        self.player_stats(idx, player_index).score
    }

    pub fn sum_squared_score(&self, idx: usize, player_index: usize) -> f64 {
        self.player_stats(idx, player_index).sum_squared_score
    }

    /// Reads child `idx`'s `player_index`'th `PlayerStats` row, from the
    /// fixed row-major `data.player` array or, past `base_len`, from an
    /// appended `ExtraChild`'s own small per-child `Vec` -- see `extra`'s
    /// doc comment for why the two ranges are indexed differently.
    fn player_stats(&self, idx: usize, player_index: usize) -> PlayerStats {
        let base_len = self.base_len();
        if idx < base_len {
            self.data.read().unwrap().player[self.player_index(idx, player_index)].clone()
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .data
                .read()
                .unwrap()
                .player[player_index]
                .clone()
        }
    }

    /// See `NodeStats::posterior`'s doc comment -- same fields, edge-indexed.
    pub fn posterior(&self, idx: usize, player_index: usize) -> (f64, f64) {
        let i = self.player_index(idx, player_index);
        let p = &self.data.read().unwrap().player[i];
        (p.posterior_mean, p.posterior_variance)
    }

    pub fn set_posterior(&self, idx: usize, player_index: usize, mean: f64, variance: f64) {
        let i = self.player_index(idx, player_index);
        let mut data = self.data.write().unwrap();
        let p = &mut data.player[i];
        p.posterior_mean = mean;
        p.posterior_variance = variance;
    }

    pub fn posterior_grid(
        &self,
        idx: usize,
        player_index: usize,
    ) -> Option<[f64; super::backprop::BAYES_GRID_SIZE]> {
        let i = self.player_index(idx, player_index);
        self.data.read().unwrap().player[i]
            .posterior_grid
            .as_deref()
            .copied()
    }

    pub fn set_posterior_grid(
        &self,
        idx: usize,
        player_index: usize,
        grid: [f64; super::backprop::BAYES_GRID_SIZE],
    ) {
        let i = self.player_index(idx, player_index);
        self.data.write().unwrap().player[i].posterior_grid = Some(Box::new(grid));
    }

    pub fn amaf(&self, idx: usize, player_index: usize) -> ActionStats {
        if !self.has_amaf {
            return ActionStats::default();
        }
        self.data.read().unwrap().amaf[self.player_index(idx, player_index)]
    }

    pub fn expected_score(&self, idx: usize, player_index: usize) -> f64 {
        expected_score_from(
            self.num_visits(idx),
            self.virtual_loss(idx),
            self.player_stats(idx, player_index).score,
        )
    }

    /// Edge-indexed counterpart to `NodeStats::overwrite_score` -- see its
    /// doc comment.
    pub fn overwrite_score(&self, idx: usize, player_index: usize, mean: f64) {
        let n = self.num_visits(idx).max(1) as f64;
        let base_len = self.base_len();
        if idx < base_len {
            let i = self.player_index(idx, player_index);
            self.data.write().unwrap().player[i].score = mean * n;
        } else {
            self.extra.read().unwrap()[idx - base_len]
                .data
                .write()
                .unwrap()
                .player[player_index]
                .score = mean * n;
        }
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
        let base_len = self.base_len();
        let p = self.player_stats(idx, player_index);
        let amaf = if idx < base_len && self.has_amaf {
            self.data.read().unwrap().amaf[self.player_index(idx, player_index)]
        } else {
            ActionStats::default()
        };
        ChildSnapshot {
            num_visits: self.num_visits(idx),
            num_visits_virtual: self.virtual_loss(idx),
            score: p.score,
            sum_squared_score: p.sum_squared_score,
            posterior_mean: p.posterior_mean,
            posterior_variance: p.posterior_variance,
            amaf,
        }
    }

    pub fn update(&self, idx: usize, utilities: &[f64]) {
        let base_len = self.base_len();
        if idx < base_len {
            let mut data = self.data.write().unwrap();
            data.num_visits[idx] += 1;
            let base = idx * self.num_players;
            utilities.iter().enumerate().for_each(|(p, reward)| {
                data.player[base + p].score += reward;
                data.player[base + p].sum_squared_score += reward * reward;
            });
        } else {
            let extra = self.extra.read().unwrap();
            let mut data = extra[idx - base_len].data.write().unwrap();
            data.num_visits += 1;
            utilities.iter().enumerate().for_each(|(p, reward)| {
                data.player[p].score += reward;
                data.player[p].sum_squared_score += reward * reward;
            });
        }
    }

    /// Seeds child `idx` with `pseudo_visits` fictitious visits at `value`
    /// (this node's own player-to-move's perspective), before any real
    /// `Node`/`Id` exists for that slot -- `prior::PriorStrategy`'s
    /// expansion-time hook (MCTS-IP/MS). Reuses `update` rather than writing
    /// `ChildArrayData`'s fields directly, so the seeded visits are ordinary
    /// visits as far as every other accessor (`expected_score`, `snapshot`,
    /// ...) is concerned -- indistinguishable from `pseudo_visits` real
    /// playouts that all happened to return `value`. Two-player zero-sum
    /// only: `player_idx`'s row gets `value`, every other player's row gets
    /// `-value` (see `prior::PriorStrategy`'s doc comment on this same
    /// restriction).
    pub(crate) fn seed_prior(&self, idx: usize, player_idx: usize, value: f64, pseudo_visits: u32) {
        if pseudo_visits == 0 {
            return;
        }
        debug_assert!(self.num_players <= 2);
        let mut utilities = vec![0.0; self.num_players];
        utilities[player_idx] = value;
        for (p, u) in utilities.iter_mut().enumerate() {
            if p != player_idx {
                *u = -value;
            }
        }
        for _ in 0..pseudo_visits {
            self.update(idx, &utilities);
        }
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
        debug_assert!(
            self.extra.get_mut().unwrap().is_empty(),
            "compaction is incompatible with ISMCTS's growable arrays -- SearchConfig::validate \
             already rejects ismcts_mode paired with reuse_tree, so this should be unreachable"
        );
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
        debug_assert!(
            idx < self.base_len(),
            "tree reuse is incompatible with ISMCTS's growable arrays -- SearchConfig::validate \
             already rejects ismcts_mode paired with reuse_tree, so this should be unreachable"
        );
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

/// The literal-board action `children.action(idx)` corresponds to.
/// `children` is stored in *its owning node's own* canonical orientation
/// (see `Game::canonical_representation`), which generally differs from
/// whatever real board orientation a given path actually reached that node
/// through. The caller supplies that translation as `incoming_sym` -- see
/// `crate::symmetry::incoming_sym`'s doc comment for why it must always be
/// recomputed fresh from a real game state, never cached on the edge.
///
/// Every consumer that feeds a `ChildArray` action into `Game::apply`, or
/// into a table keyed by literal board actions (GRAVE/MAST/NST/history/
/// AMAF), must go through this instead of reading `action(idx)` directly, or
/// it silently applies/keys a canonical-orientation action against a
/// literal-orientation state. A byte-for-byte no-op (`invert_action`
/// defaults to the identity) for every game that hasn't overridden
/// `canonical_representation`, regardless of `incoming_sym`.
pub fn real_action<G: Game>(
    children: &ChildArray<G::A>,
    idx: usize,
    incoming_sym: Transform,
) -> G::A {
    G::invert_action(Canonical(children.action(idx).clone()), incoming_sym).into_inner()
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
    // Generalized Proof-Number MCTS (Kowalski, Soemers, Kosakowski &
    // Winands, arXiv:2506.13249, 2025): one proof number *per player*,
    // instead of `pn`/`dpn`'s single per-mover negamax pair. `player_pn[p]`
    // is the minimum number of leaf nodes still to resolve to prove a win
    // for player `p`, under the paranoid assumption that every other player
    // cooperates against `p` (§3.1). Because a win for `p` is proven the
    // moment any one of `p`'s own moves forces it but disproven only when
    // every opponent reply fails, the node is an OR node in the layer where
    // `p` moves and an AND node everywhere else -- so no disproof numbers
    // are needed, and the scheme generalizes to any player count. Length
    // `num_players`, maintained by `backprop::derive_player_pn`, seeded at
    // `1` (PNS's "unknown leaf"). Read by `select::GpnUct`; inert (never
    // read) when `use_mcts_solver` is off or no GPN select strategy is
    // active, same as `pn`/`dpn`.
    player_pn: Box<[AtomicU32]>,
    // Score-Bounded MCTS (Cazenave & Saffidine, CG 2010): the pessimistic
    // and optimistic bounds on this node's graded score, always from
    // player 0's ("Max's") perspective, maintained by
    // `backprop::derive_score_bounds`. Seeded at `i32::MIN`/`i32::MAX` --
    // wider than any real game score, so an ancestor's first
    // min/max-over-children combination that includes a still-unseeded
    // child slot degrades gracefully (the seed acts as the "no information
    // yet" bound and is clamped back into the game's real
    // `Game::score_bounds()` range at the end of the recurrence). Only
    // moved off the seed when `use_mcts_solver` is on *and* the game
    // overrides `Game::score_bounds()`; inert (and never read) otherwise,
    // same as `pn`/`dpn` with the solver off.
    pess: AtomicI32,
    opti: AtomicI32,
}

impl SolverState {
    fn unproven(num_players: usize) -> Self {
        Self {
            proven: AtomicU8::new(Proven::UNPROVEN_U8),
            pn: AtomicU32::new(1),
            dpn: AtomicU32::new(1),
            pn2: AtomicU32::new(1),
            dpn2: AtomicU32::new(1),
            player_pn: (0..num_players.max(1)).map(|_| AtomicU32::new(1)).collect(),
            pess: AtomicI32::new(i32::MIN),
            opti: AtomicI32::new(i32::MAX),
        }
    }
}

// Manual impl: `AtomicU8`/`AtomicU32`/`AtomicI32` aren't `Clone`.
impl Clone for SolverState {
    fn clone(&self) -> Self {
        Self {
            proven: AtomicU8::new(self.proven.load(Relaxed)),
            pn: AtomicU32::new(self.pn.load(Relaxed)),
            dpn: AtomicU32::new(self.dpn.load(Relaxed)),
            pn2: AtomicU32::new(self.pn2.load(Relaxed)),
            dpn2: AtomicU32::new(self.dpn2.load(Relaxed)),
            player_pn: self
                .player_pn
                .iter()
                .map(|a| AtomicU32::new(a.load(Relaxed)))
                .collect(),
            pess: AtomicI32::new(self.pess.load(Relaxed)),
            opti: AtomicI32::new(self.opti.load(Relaxed)),
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
            solver: has_solver.then(|| Box::new(SolverState::unproven(num_players))),
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

    /// Generalized Proof-Number MCTS's per-player proof number (Kowalski et
    /// al., arXiv:2506.13249, §3.1): the minimum number of leaf nodes still
    /// to resolve to prove a paranoid forced win for player `p`. `0` once
    /// `proven()` is `Win(p)`; saturated (`u32::MAX`) once `proven()` is
    /// anything else -- another player's win *or* a draw, since only `p`'s
    /// own win counts as a proof for `p` under the paranoid assumption
    /// (§3.1). Otherwise the live count `backprop::derive_player_pn`
    /// maintains, seeded at `1` for an unvisited leaf. Only meaningful with
    /// `use_mcts_solver` on; `u32::MAX` (via the `None` arm) with the solver
    /// off or `p` out of range.
    #[inline]
    pub fn player_pn(&self, p: usize) -> u32 {
        match self.proven() {
            Proven::Win(w) if w == p => 0,
            Proven::Win(_) | Proven::Draw => u32::MAX,
            Proven::Unproven => self
                .solver
                .as_ref()
                .and_then(|s| s.player_pn.get(p))
                .map_or(u32::MAX, |a| a.load(Relaxed)),
        }
    }

    /// Overwrites player `p`'s live proof number. Called only from
    /// `backprop::derive_player_pn`; not write-once (it tightens over
    /// successive backprops until `proven()` resolves). No-ops with the
    /// solver off or `p` out of range.
    #[inline]
    pub fn set_player_pn(&self, p: usize, pn: u32) {
        if let Some(a) = self.solver.as_ref().and_then(|s| s.player_pn.get(p)) {
            a.store(pn, Relaxed);
        }
    }

    /// Score-Bounded MCTS's pessimistic bound: a lower bound on this
    /// node's graded score under optimal play, from player 0's ("Max's")
    /// perspective. `i32::MIN` when nothing has been derived yet (the seed
    /// -- see `SolverState::pess`) or the solver is off. Maintained by
    /// `backprop::derive_score_bounds`; `pess() == opti()` means the
    /// node's score is solved exactly.
    #[inline]
    pub fn pess(&self) -> i32 {
        self.solver
            .as_ref()
            .map_or(i32::MIN, |s| s.pess.load(Relaxed))
    }

    /// Score-Bounded MCTS's optimistic bound -- the mirror of `pess()`: an
    /// upper bound on this node's graded score, from player 0's
    /// perspective. `i32::MAX` when unset / solver off.
    #[inline]
    pub fn opti(&self) -> i32 {
        self.solver
            .as_ref()
            .map_or(i32::MAX, |s| s.opti.load(Relaxed))
    }

    /// Overwrites the live score bounds. Called only from
    /// `backprop::derive_score_bounds`; not write-once (the interval
    /// tightens over successive backprops until `pess == opti`). No-ops
    /// when the solver is off, same as `set_pn_dpn`.
    #[inline]
    pub fn set_score_bounds(&self, pess: i32, opti: i32) {
        let Some(solver) = self.solver.as_ref() else {
            return;
        };
        solver.pess.store(pess, Relaxed);
        solver.opti.store(opti, Relaxed);
    }

    /// Pins this node's score interval to an exact `score` -- for a
    /// `NodeState::Terminal` node, whose real value `Game::terminal_score`
    /// reports directly. Idempotent: a terminal node's score never
    /// changes, so a redundant call from a racing thread writes the same
    /// value. No-op with the solver off.
    #[inline]
    pub fn set_terminal_score(&self, score: i32) {
        self.set_score_bounds(score, score);
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

    /// Heap bytes owned by this node's solver side block (`Box<SolverState>`
    /// when allocated, 0 otherwise) -- the per-node analogue of
    /// `ChildArray::heap_bytes_estimate`'s AMAF term, used by
    /// `TreeSearch::memory_stats` to size the storage win from gating
    /// `Node`'s solver fields on `SearchConfig::use_mcts_solver`.
    pub(crate) fn solver_heap_bytes(&self) -> usize {
        match self.solver.as_ref() {
            Some(s) => {
                std::mem::size_of::<SolverState>()
                    + s.player_pn.len() * std::mem::size_of::<AtomicU32>()
            }
            None => 0,
        }
    }
}

#[cfg(test)]
mod prior_tests {
    use super::ChildArray;

    // Hand-verifiable: seeding child 0 with value 0.5 at 4 pseudo-visits (for
    // player 0 of a 2-player game) should read back exactly as if 4 real
    // playouts had all scored 0.5 for player 0 and -0.5 for player 1 --
    // `expected_score` averages, so the seeded value itself, not `4 * 0.5`.
    #[test]
    fn test_seed_prior_writes_two_player_zero_sum_stats() {
        let children: ChildArray<u32> = ChildArray::new(vec![10, 20], 2, false, false);

        children.seed_prior(0, 0, 0.5, 4);

        assert_eq!(children.num_visits(0), 4);
        assert_eq!(children.expected_score(0, 0), 0.5);
        assert_eq!(children.expected_score(0, 1), -0.5);
        // Untouched sibling.
        assert_eq!(children.num_visits(1), 0);
        assert!(
            children.node_id(0).is_none(),
            "seeding never creates a Node"
        );
    }

    #[test]
    fn test_seed_prior_zero_pseudo_visits_is_a_no_op() {
        let children: ChildArray<u32> = ChildArray::new(vec![10], 2, false, false);
        children.seed_prior(0, 0, 0.9, 0);
        assert_eq!(children.num_visits(0), 0);
    }
}
