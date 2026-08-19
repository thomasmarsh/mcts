use super::config::GraphStats;
use super::index::Id;
use super::node;
use super::node::NodeStats;
use super::node::StatsRef;
use super::search::TreeIndex;
use crate::game::Action;
use crate::game::Game;
use crate::game::Real;
use crate::game::Transform;
use crate::util::Pairs;
use crate::util::ReversePairs;
use crate::util::ReversePairs2;

use rustc_hash::FxHashMap;

/// One entry on a root->leaf descent stack: the `Id` reached, and the index
/// (in its immediate predecessor's `ChildArray`) that was actually selected
/// to reach it. The idx is carried explicitly rather than reconstructed
/// later from `(predecessor, Id)` alone via `ChildArray`'s `id_index`
/// reverse map -- that reverse map is only sound when a parent's children
/// map 1:1 to arena ids, which symmetry-aware graph merging breaks: several
/// of a node's actions can canonicalize to the *same* shared child (e.g.
/// ttt's four D4-symmetric corner moves from an empty board), so a reverse
/// lookup by `Id` alone can't tell which of those slots a given traversal
/// actually used. The entry at index 0 (the stack's root) has no
/// predecessor, so its `usize` is an unused placeholder (`0`) -- every
/// consumer here only ever reads an entry's idx when it's acting as the
/// *child* half of a pair, never the root.
type StackEntry = (Id, usize);

#[derive(Debug, Clone)]
pub struct NodeStack<A> {
    stack: Vec<StackEntry>,
    marker: std::marker::PhantomData<A>,
}

impl<A: Action> NodeStack<A> {
    pub fn new(stack: Vec<StackEntry>) -> Self {
        Self {
            stack,
            marker: std::marker::PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn pairs(&self) -> Pairs<'_, StackEntry> {
        Pairs::new(&self.stack)
    }

    pub fn reverse_pairs(&self) -> ReversePairs<'_, StackEntry> {
        ReversePairs::new(&self.stack)
    }

    pub fn reverse_pairs2(&self) -> ReversePairs2<'_, StackEntry> {
        ReversePairs2::new(&self.stack)
    }

    /// The symmetry index of the edge leading into each `Id` on this stack,
    /// and the literal-board action taken on that edge -- what
    /// `node::real_action`'s callers need translate a stack node's own
    /// `ChildArray` actions back to the literal board (see `node::
    /// incoming_sym`'s doc comment for why this must be derived by replaying
    /// real states from the root, never read off a `ChildArray` alone: a
    /// node's own incoming symmetry depends on which *real* orientation of
    /// its parent a given path reached it through, and a parent that's
    /// itself a transposition can be reached through different real
    /// orientations on different iterations). `G::A` here must match `A`;
    /// only ever called with a `G: Game<A = A>` that matches this stack's
    /// own arena.
    ///
    /// The idx used to translate each edge's action comes directly from
    /// that child's own stack entry (`child_idx` below) -- never from a
    /// `ChildArray` reverse lookup, which can't disambiguate when several of
    /// a parent's actions canonicalize to the same shared child (see
    /// `StackEntry`'s doc comment).
    pub fn incoming_syms<G: Game<A = A>>(
        &self,
        index: &TreeIndex<A>,
        root_state: &G::S,
        explicit_dag: bool,
    ) -> (FxHashMap<Id, Transform>, FxHashMap<Id, G::A>) {
        let mut syms = FxHashMap::default();
        let mut actions = FxHashMap::default();
        syms.insert(self.root(), Transform::IDENTITY);
        let mut state = root_state.clone();
        for ((parent_id, _), (child_id, child_idx)) in self.pairs() {
            let parent = index.get(*parent_id);
            let parent_sym = *syms.get(parent_id).unwrap();
            let action = node::real_action::<G>(parent.children(), *child_idx, parent_sym);
            state = G::apply(state, &action);
            let child_sym = node::incoming_sym::<G>(explicit_dag, false, Real(&state));
            syms.insert(*child_id, child_sym);
            actions.insert(*child_id, action);
        }
        (syms, actions)
    }

    pub fn root(&self) -> Id {
        debug_assert!(!self.stack.is_empty());
        self.stack[0].0
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Appends `node_id`, reached via slot `idx` in the current last entry's
    /// `ChildArray` (or an unused placeholder if `node_id` is itself the
    /// root -- see `StackEntry`'s doc comment).
    pub fn push(&mut self, node_id: Id, idx: usize) {
        self.stack.push((node_id, idx))
    }

    pub fn parent_id(&self) -> Id {
        debug_assert!(self.stack.len() > 1);
        self.stack[self.stack.len() - 2].0
    }

    pub fn current_id(&self) -> Id {
        debug_assert!(!self.stack.is_empty());
        self.stack.last().unwrap().0
    }

    #[inline]
    pub fn current_stats<'a>(
        &self,
        index: &'a TreeIndex<A>,
        root_stats: &'a NodeStats,
        graph_stats: Option<GraphStats>,
    ) -> StatsRef<'a, A> {
        if graph_stats.is_some_and(GraphStats::uses_nodes) {
            return StatsRef::Node(&index.get(self.current_id()).stats);
        }
        if index.get(self.current_id()).is_root() {
            StatsRef::Root(root_stats)
        } else {
            let (child_id, idx) = *self.stack.last().unwrap();
            self.get_stats(
                index,
                root_stats,
                graph_stats,
                self.parent_id(),
                child_id,
                idx,
            )
        }
    }

    /// `idx` is the slot in `parent_id`'s `ChildArray` that `child_id`
    /// occupies on *this* stack's path -- supplied by the caller (from a
    /// `StackEntry`), never reconstructed here via a `ChildArray` reverse
    /// lookup (see `StackEntry`'s doc comment for why that's unsound under
    /// symmetry-aware merging).
    pub fn get_stats<'a>(
        &self,
        index: &'a TreeIndex<A>,
        root_stats: &'a NodeStats,
        graph_stats: Option<GraphStats>,
        parent_id: Id,
        child_id: Id,
        idx: usize,
    ) -> StatsRef<'a, A> {
        if index.get(child_id).is_root() {
            StatsRef::Root(root_stats)
        } else if graph_stats.is_some_and(GraphStats::uses_nodes) {
            StatsRef::Node(&index.get(child_id).stats)
        } else {
            debug_assert_ne!(parent_id, child_id);
            StatsRef::Child(index.get(parent_id).children(), idx)
        }
    }
}
