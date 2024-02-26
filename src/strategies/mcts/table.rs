use super::index;
use crate::zobrist::ZobristHash;
use rustc_hash::FxHashMap as HashMap;

#[derive(Clone, Default)]
pub struct TranspositionTable {
    table: HashMap<ZobristHash, index::Id>,
    hits: usize,
}

impl TranspositionTable {
    pub fn clear(&mut self) {
        self.table.clear();
        self.hits = 0;
    }
}
