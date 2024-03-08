use rustc_hash::FxHashMap;

use crate::game::{Game, Symmetry, ZobristHash};

use super::index;

#[derive(Clone, Debug)]
pub struct TableEntry {
    pub node_id: index::Id,
    pub symmetry: Symmetry,
}

#[derive(Clone)]
pub struct TranspositionTable<G: Game> {
    pub table: FxHashMap<G::K, TableEntry>,
    pub reads: usize,
    pub writes: usize,
    pub hits: usize,

    // TODO: state collision checks
    pub state: FxHashMap<G::K, G::S>,
    pub check_collisions: bool,
}

impl<G: Game> std::fmt::Debug for TranspositionTable<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranspositionTable")
            .field("table", &self.table)
            .field("state", &self.state)
            .field("check_collisions", &self.check_collisions)
            .field("reads", &self.reads)
            .field("writes", &self.writes)
            .field("hits", &self.hits)
            .finish()
    }
}

impl<G: Game> Default for TranspositionTable<G> {
    fn default() -> Self {
        Self {
            table: FxHashMap::default(),
            reads: 0,
            writes: 0,
            hits: 0,

            state: FxHashMap::default(),
            check_collisions: false,
        }
    }
}

impl<G: Game> TranspositionTable<G> {
    #[inline]
    pub fn clear(&mut self) {
        self.table.clear();
        self.state.clear();
        self.reads = 0;
        self.writes = 0;
        self.hits = 0;
    }

    #[inline]
    pub fn get(&mut self, k: &G::K, _state: G::S) -> Option<&TableEntry> {
        self.reads += 1;
        let result = self.table.get(k);
        if let Some(entry) = result {
            self.hits += 1;
            return Some(entry);
        }
        None
    }

    #[inline]
    pub fn get_const(&self, k: &G::K, _state: G::S) -> Option<&TableEntry> {
        let result = self.table.get(k);
        if let Some(entry) = result {
            return Some(entry);
        }
        None
    }

    #[inline(always)]
    pub fn insert(&mut self, k: &ZobristHash<G::K>, node_id: index::Id, _state: G::S) {
        self.writes += 1;
        self.table.insert(
            k.hash,
            TableEntry {
                node_id,
                symmetry: k.symmetry,
            },
        );
    }
}
