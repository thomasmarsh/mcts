use super::index;
use crate::zobrist::ZobristHashMap;
use rustc_hash::FxHashSet;

type IdSet = FxHashSet<index::Id>;

#[derive(Clone, Default)]
pub struct TranspositionTable {
    pub table: ZobristHashMap<IdSet>,
    pub reads: usize,
    pub writes: usize,
    pub hits: usize,
}

impl TranspositionTable {
    #[inline]
    pub fn clear(&mut self) {
        self.table.clear();
        self.reads = 0;
        self.writes = 0;
        self.hits = 0;
    }

    #[inline]
    pub fn get(&mut self, k: u64) -> Option<&IdSet> {
        let result = self.table.get(k);
        if result.is_some() {
            self.hits += 1;
        }
        self.reads += 1;
        result
    }

    #[inline]
    pub fn get_const(&self, k: u64) -> Option<&IdSet> {
        self.table.0.get(&crate::zobrist::ZobristHash(k))
    }

    #[inline(always)]
    pub fn insert(&mut self, k: u64, node_id: index::Id) {
        self.writes += 1;
        let mut found = true;
        self.table
            .entry(k)
            .or_insert_with(|| {
                found = false;
                IdSet::default()
            })
            .insert(node_id);
        if found {
            self.hits += 1;
        }
    }
}
