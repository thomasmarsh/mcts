use super::index;
use crate::zobrist::ZobristHashMap;

use rustc_hash::FxHashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::RwLock;

/// A root-relative key for explicit graph search. Including ply means every
/// graph edge advances strictly forward and a repeated board position cannot
/// introduce a cycle into the search structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TranspositionKey {
    pub position_hash: u64,
    pub ply: u32,
}

/// Maps a Zobrist hash to the arena node for that position. Stores no state
/// at all -- a same-hash lookup is trusted outright rather than verified
/// against a stored clone. This is sound because the table already buckets
/// on the *full* 64-bit hash (unlike a fixed-size, index-truncated table,
/// which needs a separate verification tag): a same-bucket "collision" here
/// would have to be a genuine full 64-bit Zobrist collision, vanishingly
/// rare for a well-distributed hash. The accepted tradeoff: if a genuine
/// 64-bit collision ever does occur, the second
/// position silently reuses the first position's node rather than getting
/// its own -- first write wins, no error, no detection. At real table sizes
/// (single-digit millions of entries per game) the odds of that are
/// astronomically below other failure sources (e.g. cosmic-ray bit flips).
#[derive(Debug)]
pub struct TranspositionTable {
    table: RwLock<ZobristHashMap<index::Id>>,
    graph_table: RwLock<FxHashMap<TranspositionKey, index::Id>>,
    pub reads: AtomicUsize,
    pub writes: AtomicUsize,
    pub hits: AtomicUsize,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self {
            table: RwLock::new(ZobristHashMap::default()),
            graph_table: RwLock::new(FxHashMap::default()),
            reads: AtomicUsize::new(0),
            writes: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
        }
    }
}

impl Clone for TranspositionTable {
    fn clone(&self) -> Self {
        Self {
            table: RwLock::new(self.table.read().unwrap().clone()),
            graph_table: RwLock::new(self.graph_table.read().unwrap().clone()),
            reads: AtomicUsize::new(self.reads.load(Relaxed)),
            writes: AtomicUsize::new(self.writes.load(Relaxed)),
            hits: AtomicUsize::new(self.hits.load(Relaxed)),
        }
    }
}

impl TranspositionTable {
    #[inline]
    pub fn clear(&mut self) {
        self.table.get_mut().unwrap().clear();
        self.graph_table.get_mut().unwrap().clear();
        self.reads = AtomicUsize::new(0);
        self.writes = AtomicUsize::new(0);
        self.hits = AtomicUsize::new(0);
    }

    #[inline]
    pub fn get_const(&self, k: u64) -> Option<index::Id> {
        self.table.read().unwrap().get(k).copied()
    }

    /// Number of entries currently in the table (diagnostics only).
    pub fn len(&self) -> usize {
        self.table.read().unwrap().0.len() + self.graph_table.read().unwrap().len()
    }

    pub fn legacy_len(&self) -> usize {
        self.table.read().unwrap().0.len()
    }

    pub fn graph_len(&self) -> usize {
        self.graph_table.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Concurrent-safe get-or-insert: the whole check-then-insert happens
    /// under one write lock, so two threads racing on the same `k` can't
    /// both decide "not found" and each insert a duplicate node -- `create`
    /// only actually runs (and its resulting node only actually gets
    /// inserted) for whichever thread's check comes first.
    #[inline]
    pub fn get_or_insert(&self, k: u64, create: impl FnOnce() -> index::Id) -> index::Id {
        self.reads.fetch_add(1, Relaxed);
        let mut table = self.table.write().unwrap();
        if let Some(&id) = table.get(k) {
            self.hits.fetch_add(1, Relaxed);
            return id;
        }
        let node_id = create();
        table.insert(k, node_id);
        self.writes.fetch_add(1, Relaxed);
        node_id
    }

    /// The explicit MCGS table has a compound key, so it cannot use the
    /// legacy Zobrist-only map. The entire check and insertion stays under
    /// one lock just like `get_or_insert`.
    #[inline]
    pub fn get_or_insert_graph(
        &self,
        key: TranspositionKey,
        create: impl FnOnce() -> index::Id,
    ) -> index::Id {
        self.reads.fetch_add(1, Relaxed);
        let mut table = self.graph_table.write().unwrap();
        if let Some(&id) = table.get(&key) {
            self.hits.fetch_add(1, Relaxed);
            return id;
        }
        let id = create();
        table.insert(key, id);
        self.writes.fetch_add(1, Relaxed);
        id
    }

    #[inline]
    pub fn insert_graph(&self, key: TranspositionKey, node_id: index::Id) {
        self.get_or_insert_graph(key, || node_id);
    }

    /// Unconditional insert-if-absent for callers that already have a node
    /// id in hand (e.g. seeding the root) rather than needing one created
    /// on demand.
    #[inline(always)]
    pub fn insert(&self, k: u64, node_id: index::Id) {
        self.get_or_insert(k, || node_id);
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
        table.0.retain(|_, id| match old_to_new.get(id) {
            Some(&new_id) => {
                *id = new_id;
                true
            }
            None => false,
        });
        self.graph_table
            .get_mut()
            .unwrap()
            .retain(|_, id| match old_to_new.get(id) {
                Some(&new_id) => {
                    *id = new_id;
                    true
                }
                None => false,
            });
    }
}
