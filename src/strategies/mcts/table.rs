use super::index;
use crate::zobrist::ZobristHashMap;

#[derive(Clone, Debug)]
pub struct TableEntry<S: Eq> {
    pub node_id: index::Id,
    pub state: S,
}

#[derive(Clone, Debug)]
pub struct TranspositionTable<S: Eq> {
    pub table: ZobristHashMap<Vec<TableEntry<S>>>,
    pub reads: usize,
    pub writes: usize,
    pub hits: usize,
}

impl<S: Eq> Default for TranspositionTable<S> {
    fn default() -> Self {
        Self {
            table: ZobristHashMap::default(),
            reads: 0,
            writes: 0,
            hits: 0,
        }
    }
}

impl<S: Clone + Eq> TranspositionTable<S> {
    #[inline]
    pub fn clear(&mut self) {
        self.table.clear();
        self.reads = 0;
        self.writes = 0;
        self.hits = 0;
    }

    #[inline]
    pub fn get(&mut self, k: u64, state: S) -> Option<&TableEntry<S>> {
        self.reads += 1;
        let result = self.table.get(k);
        if let Some(entries) = result {
            self.hits += 1;
            for entry in entries {
                if entry.state == state {
                    return Some(entry);
                }
            }
            return None;
        }
        None
    }

    #[inline]
    pub fn get_const(&self, k: u64, state: S) -> Option<&TableEntry<S>> {
        let result = self.table.get(k);
        if let Some(entries) = result {
            for entry in entries {
                if entry.state == state {
                    return Some(entry);
                }
            }
            return None;
        }
        None
    }

    #[inline(always)]
    pub fn insert(&mut self, k: u64, node_id: index::Id, state: S) {
        if self.get(k, state.clone()).is_some() {
            return;
        }
        let entries = self.table.entry(k).or_default();
        if !entries.is_empty() {
            eprintln!("collision: key={k:0x} len={}!", entries.len() + 1);
        }
        entries.push(TableEntry { node_id, state });
    }
}
