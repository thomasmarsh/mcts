use super::index;
use crate::zobrist::ZobristHashMap;

use rustc_hash::FxHashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::RwLock;

#[derive(Clone, Debug)]
pub struct TableEntry<S: Eq> {
    pub node_id: index::Id,
    pub state: S,
}

#[derive(Debug)]
pub struct TranspositionTable<S: Eq> {
    table: RwLock<ZobristHashMap<Vec<TableEntry<S>>>>,
    pub reads: AtomicUsize,
    pub writes: AtomicUsize,
    pub hits: AtomicUsize,
}

impl<S: Eq> Default for TranspositionTable<S> {
    fn default() -> Self {
        Self {
            table: RwLock::new(ZobristHashMap::default()),
            reads: AtomicUsize::new(0),
            writes: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
        }
    }
}

impl<S: Clone + Eq> Clone for TranspositionTable<S> {
    fn clone(&self) -> Self {
        Self {
            table: RwLock::new(self.table.read().unwrap().clone()),
            reads: AtomicUsize::new(self.reads.load(Relaxed)),
            writes: AtomicUsize::new(self.writes.load(Relaxed)),
            hits: AtomicUsize::new(self.hits.load(Relaxed)),
        }
    }
}

impl<S: Clone + Eq> TranspositionTable<S> {
    #[inline]
    pub fn clear(&mut self) {
        self.table.get_mut().unwrap().clear();
        self.reads = AtomicUsize::new(0);
        self.writes = AtomicUsize::new(0);
        self.hits = AtomicUsize::new(0);
    }

    #[inline]
    pub fn get_const(&self, k: u64, state: S) -> Option<TableEntry<S>> {
        let table = self.table.read().unwrap();
        table
            .get(k)
            .and_then(|entries| entries.iter().find(|entry| entry.state == state).cloned())
    }

    /// Concurrent-safe get-or-insert: the whole check-then-insert happens
    /// under one write lock, so two threads racing on the same `(k, state)`
    /// can't both decide "not found" and each insert a duplicate node --
    /// `create` only actually runs (and its resulting node only actually
    /// gets inserted) for whichever thread's check comes first.
    #[inline]
    pub fn get_or_insert(&self, k: u64, state: S, create: impl FnOnce() -> index::Id) -> index::Id {
        self.reads.fetch_add(1, Relaxed);
        let mut table = self.table.write().unwrap();
        if let Some(entries) = table.get(k) {
            if let Some(entry) = entries.iter().find(|entry| entry.state == state) {
                self.hits.fetch_add(1, Relaxed);
                return entry.node_id;
            }
        }
        let node_id = create();
        let entries = table.entry(k).or_default();
        if !entries.is_empty() {
            log::debug!("collision: key={k:0x} len={}!", entries.len() + 1);
        }
        entries.push(TableEntry { node_id, state });
        self.writes.fetch_add(1, Relaxed);
        node_id
    }

    /// Unconditional insert-if-absent for callers that already have a node
    /// id in hand (e.g. seeding the root) rather than needing one created
    /// on demand.
    #[inline(always)]
    pub fn insert(&self, k: u64, node_id: index::Id, state: S) {
        self.get_or_insert(k, state, || node_id);
    }

    /// Arena compaction's table half (`search/compact.rs`'s
    /// `TreeSearch::compact`): every entry's `node_id` refers to the arena
    /// compaction just discarded, not the freshly built one, so each entry
    /// either gets remapped (its node survived, i.e. was reachable from the
    /// new root) or dropped (it wasn't -- the position is still a real
    /// position, but the table no longer has a node for it, exactly as if
    /// it had simply never been inserted). Cheaper than rebuilding by
    /// re-walking the compacted tree: this is one pass over the existing
    /// table instead of re-deriving every entry's key from scratch, and
    /// entries for positions outside the reachable subtree (a real
    /// possibility -- the table isn't scoped to any one subtree) are
    /// naturally dropped rather than needing a separate reachability check.
    /// `&mut self`, matching `clear` -- compaction only ever runs between
    /// moves, with no concurrent search in flight.
    pub fn compact(&mut self, old_to_new: &FxHashMap<index::Id, index::Id>) {
        let table = self.table.get_mut().unwrap();
        table.0.retain(|_, entries| {
            entries.retain_mut(|entry| match old_to_new.get(&entry.node_id) {
                Some(&new_id) => {
                    entry.node_id = new_id;
                    true
                }
                None => false,
            });
            !entries.is_empty()
        });
    }

    /// Test/debug helper: the number of distinct states currently sharing
    /// each Zobrist hash bucket. A bucket length above 1 is a real 64-bit
    /// hash collision (as opposed to the pre-session-2 bit-width bug, which
    /// made same-hash-but-different-state collisions routine).
    pub fn bucket_lens(&self) -> Vec<usize> {
        self.table.read().unwrap().0.values().map(Vec::len).collect()
    }
}
