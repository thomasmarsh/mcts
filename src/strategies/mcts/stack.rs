use super::index::Id;
use super::node::Edge;
use super::node::NodeState;
use super::node::NodeStats;
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

    // NOTE: O(n) lookup. TODO: benchmark against FxHashMap
    pub fn child_index(&self, index: &TreeIndex<A>, parent_id: Id, child_id: Id) -> usize {
        index
            .get(parent_id)
            .edges()
            .iter()
            .position(|e| e.node_id.is_some_and(|id| id == child_id))
            .unwrap()
    }

    pub fn edge<'a>(&self, index: &'a TreeIndex<A>, parent_id: Id, child_id: Id) -> &'a Edge<A> {
        let action_index = self.child_index(index, parent_id, child_id);
        let NodeState::Expanded(ref edges) = &(index.get(parent_id).state) else {
            unreachable!()
        };
        &edges[action_index]
    }

    #[inline]
    pub fn current_stats<'a>(
        &self,
        index: &'a TreeIndex<A>,
        root_stats: &'a NodeStats,
    ) -> &'a NodeStats {
        if index.get(self.current_id()).is_root() {
            root_stats
        } else {
            self.get_stats(index, root_stats, self.parent_id(), self.current_id())
        }
    }

    pub fn get_stats<'a>(
        &self,
        index: &'a TreeIndex<A>,
        root_stats: &'a NodeStats,
        parent_id: Id,
        child_id: Id,
    ) -> &'a NodeStats {
        if index.get(child_id).is_root() {
            root_stats
        } else {
            debug_assert_ne!(parent_id, child_id);
            &self.edge(index, parent_id, child_id).stats
        }
    }
}
