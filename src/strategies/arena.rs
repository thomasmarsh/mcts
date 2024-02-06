#[derive(Debug)]
pub struct Arena<T>(pub Vec<T>);

#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug, Hash)]
pub struct Ref(usize);

// indextree probably gets better cache locality by maintaining a next_sibling:
// Option<indextree:Node<T>>. Although this introduces branching, it may
// be preferable than storing Node<M>.children on the heap. That would need
// benchmarking, and is probably a negligible difference. For now, we push the
// requirement to track the children to the user of the arena.

impl<T: std::fmt::Debug> Arena<T> {
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
    pub fn add(&mut self, value: T) -> Ref {
        let node_id = Ref(self.0.len());
        self.0.push(value);
        node_id
    }

    #[inline]
    pub fn get(&self, node_id: Ref) -> &T {
        self.0.get(node_id.0).unwrap()
    }

    #[inline]
    pub fn get_mut(&mut self, node_id: Ref) -> &mut T {
        self.0.get_mut(node_id.0).unwrap()
    }
}
