use std::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Id(usize);

impl Id {
    pub fn invalid_id() -> Id {
        Id(usize::MAX)
    }

    pub fn get_raw(&self) -> usize {
        self.0
    }
}

// TODO: benchmark keeping child/sibling relationships here vs. on Node (space vs. time)
//
/// Entries are heap-allocated (`Box`) so that growing `entries` (a write-locked
/// `Vec<Box<T>>`) never moves an already-inserted `T` -- only the `Box`
/// pointers get relocated, not their pointees. That's what makes `get`'s
/// unsafe lifetime extension below sound.
#[derive(Default, Debug)]
pub struct Arena<T>(RwLock<Vec<Box<T>>>);

impl<T: Clone> Clone for Arena<T> {
    fn clone(&self) -> Self {
        let entries = self.0.read().unwrap();
        Self(RwLock::new(entries.clone()))
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self(RwLock::new(Vec::new()))
    }

    pub fn clear(&mut self) {
        self.0.get_mut().unwrap().clear();
    }

    /// Concurrent-safe: multiple threads may call `insert` while other
    /// threads hold references returned by `get`, since insertion never
    /// moves an existing entry (see the `Arena` doc comment).
    pub fn insert(&self, value: T) -> Id {
        let mut entries = self.0.write().unwrap();
        let id = entries.len();
        entries.push(Box::new(value));
        Id(id)
    }

    /// Returns a reference bound to `&self` rather than to the transient
    /// read-lock guard.
    ///
    /// SAFETY: `entries[id.0]` is a `Box<T>`, whose heap allocation address
    /// is stable regardless of how the outer `Vec` grows/reallocates (growth
    /// only relocates the `Box` pointers, never the pointee). Entries are
    /// never removed except by `clear`, which takes `&mut self` and
    /// therefore (by the borrow checker, at every call site) cannot run
    /// while any `&T` returned from an earlier `get` is still alive. So
    /// extending this reference's lifetime past the read-lock guard, which
    /// we drop immediately below, is sound.
    pub fn get(&self, id: Id) -> &T {
        let entries = self.0.read().unwrap();
        let ptr: *const T = entries.get(id.0).unwrap().as_ref();
        unsafe { &*ptr }
    }

    /// Exclusive access, only usable when the caller already holds `&mut
    /// Arena` (e.g. single-threaded callers like `OpeningBook`) -- no
    /// locking involved.
    pub fn get_mut(&mut self, id: Id) -> &mut T {
        &mut self.0.get_mut().unwrap()[id.0]
    }

    pub fn len(&self) -> usize {
        self.0.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.read().unwrap().is_empty()
    }

    /// Diagnostics-only: applies `f` to every entry currently in the arena,
    /// holding the read lock for the whole walk. `get`/`len` alone can't
    /// enumerate the arena's contents -- this is for memory-profiling code,
    /// not any hot search path.
    pub fn for_each(&self, mut f: impl FnMut(&T)) {
        let entries = self.0.read().unwrap();
        for entry in entries.iter() {
            f(entry);
        }
    }
}
