use super::mcts::Node;

// TODO: indextree stores too much data per node and needs trimming.
// Consider forking or reimplementing our own arena. The use of indextree
// is isolated to this file.
pub struct Index<M> {
    arena: indextree::Arena<Node<M>>,
}

// We need a wrapper so we can attach our own ergonomics.
#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug, Hash)]
pub struct NodeRef(indextree::NodeId);

// There might be an easier way to handle this map
pub struct Children<'a, M>(indextree::Children<'a, Node<M>>);

impl<'a, M> Iterator for Children<'a, M> {
    type Item = NodeRef;

    fn next(&mut self) -> Option<NodeRef> {
        self.0.next().map(NodeRef)
    }
}

impl<'a, M> core::iter::FusedIterator for Children<'a, M> {}

impl<M> Index<M> {
    pub fn new() -> Self {
        Self {
            arena: indextree::Arena::new(),
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.arena.clear();
    }

    #[allow(unused)]
    pub fn reserve(&mut self, additional: usize) {
        self.arena.reserve(additional);
    }

    #[inline]
    pub fn add(&mut self, node: Node<M>) -> NodeRef {
        NodeRef(self.arena.new_node(node))
    }

    #[inline]
    pub fn get(&self, node_id: NodeRef) -> &Node<M> {
        self.arena.get(node_id.0).unwrap().get()
    }

    #[inline]
    pub fn get_parent(&self, node_id: NodeRef) -> Option<NodeRef> {
        node_id.0.ancestors(&self.arena).nth(1).map(NodeRef)
    }

    #[inline]
    pub fn get_mut(&mut self, node_id: NodeRef) -> &mut Node<M> {
        self.arena.get_mut(node_id.0).unwrap().get_mut()
    }

    #[inline]
    pub fn children(&self, node_id: NodeRef) -> Children<'_, M> {
        Children(node_id.0.children(&self.arena))
    }

    #[inline]
    pub fn add_child(&mut self, parent: NodeRef, child: Node<M>) -> NodeRef {
        let child_id = self.arena.new_node(child);
        parent.0.append(child_id, &mut self.arena);
        NodeRef(child_id)
    }
}
