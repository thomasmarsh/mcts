use super::mcts::Node;

#[derive(Debug)]
pub struct Arena<M>(pub Vec<Node<M>>);

#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug, Hash)]
pub struct NodeRef(usize);

// indextree probably gets better cache locality by maintaining a next_sibling:
// Option<indextree:Node<T>>. Although this introduces branching, it may
// be preferable than storing Node<M>.children on the heap. That would need
// benchmarking, and is probably a negligible difference. For now, we push the
// requirement to track the children to the user of the arena.

impl<M: std::fmt::Debug> Arena<M> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[allow(unused)]
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    #[inline]
    pub fn add(&mut self, node: Node<M>) -> NodeRef {
        let node_id = NodeRef(self.0.len());
        self.0.push(node);
        node_id
    }

    #[inline]
    pub fn get(&self, node_id: NodeRef) -> &Node<M> {
        self.0.get(node_id.0).unwrap()
    }

    #[inline]
    pub fn get_mut(&mut self, node_id: NodeRef) -> &mut Node<M> {
        self.0.get_mut(node_id.0).unwrap()
    }
}
