use super::config::GraphStats;
use super::index::Id;
use super::node::NodeStats;
use super::node::StatsRef;
use super::search::TreeIndex;
use crate::game::Action;
use crate::util::Pairs;
use crate::util::ReversePairs;
use crate::util::ReversePairs2;

#[derive(Debug, Clone)]
pub struct NodeStack<A> {
    stack: Vec<Id>,
    marker: std::marker::PhantomData<A>,
}

impl<A: Action> NodeStack<A> {
    pub fn new(stack: Vec<Id>) -> Self {
        Self {
            stack,
            marker: std::marker::PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Id> {
        self.stack.iter()
    }

    pub fn pairs(&self) -> Pairs<'_, Id> {
        Pairs::new(&self.stack)
    }

    pub fn reverse_pairs(&self) -> ReversePairs<'_, Id> {
        ReversePairs::new(&self.stack)
    }

    pub fn reverse_pairs2(&self) -> ReversePairs2<'_, Id> {
        ReversePairs2::new(&self.stack)
    }

    pub fn root(&self) -> Id {
        debug_assert!(!self.stack.is_empty());
        self.stack[0]
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub fn push(&mut self, node_id: Id) {
        self.stack.push(node_id)
    }

    pub fn parent_id(&self) -> Id {
        debug_assert!(self.stack.len() > 1);
        self.stack.get(self.stack.len() - 2).cloned().unwrap()
    }

    pub fn current_id(&self) -> Id {
        debug_assert!(!self.stack.is_empty());
        *self.stack.last().unwrap()
    }

    /// The index of `child_id` among `parent_id`'s children -- replaces the
    /// old `&Edge<A>`-returning lookup now that a node's children live in a
    /// `ChildArray` instead of individually-addressable `Edge` structs.
    pub fn child_index(&self, index: &TreeIndex<A>, parent_id: Id, child_id: Id) -> usize {
        index.get(parent_id).child_index(child_id)
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
            self.get_stats(
                index,
                root_stats,
                graph_stats,
                self.parent_id(),
                self.current_id(),
            )
        }
    }

    pub fn get_stats<'a>(
        &self,
        index: &'a TreeIndex<A>,
        root_stats: &'a NodeStats,
        graph_stats: Option<GraphStats>,
        parent_id: Id,
        child_id: Id,
    ) -> StatsRef<'a, A> {
        if index.get(child_id).is_root() {
            StatsRef::Root(root_stats)
        } else if graph_stats.is_some_and(GraphStats::uses_nodes) {
            StatsRef::Node(&index.get(child_id).stats)
        } else {
            debug_assert_ne!(parent_id, child_id);
            let parent = index.get(parent_id);
            let idx = parent.child_index(child_id);
            StatsRef::Child(parent.children(), idx)
        }
    }
}
