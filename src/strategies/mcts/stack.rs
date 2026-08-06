use super::index::Id;
use super::node::Edge;
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

    pub fn edge<'a>(&self, index: &'a TreeIndex<A>, parent_id: Id, child_id: Id) -> &'a Edge<A> {
        index.get(parent_id).child_edge(child_id)
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
