use rand::rngs::StdRng;
use rand::Rng;
use rand_core::SeedableRng;
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::sync::OnceLock;

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug, Serialize)]
pub struct ZobristKey {
    pub state: u64,
    pub path: u64,
}

impl ZobristKey {
    pub const fn new() -> Self {
        ZobristKey { state: 0, path: 0 }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Serialize, Debug)]
pub struct ZobristHashMap<T> {
    table: FxHashMap<u64, FxHashMap<u64, T>>,
}

impl<T> Default for ZobristHashMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ZobristHashMap<T> {
    pub fn new() -> Self {
        Self {
            table: FxHashMap::default(),
        }
    }

    pub fn clear(&mut self) {
        self.table.clear()
    }

    pub fn get(&self, k: &ZobristKey) -> Option<&T> {
        self.table.get(&k.state).and_then(|t| t.get(&k.path))
    }

    pub fn entry(&self, k: ZobristKey) -> std::collections::hash_map::Entry<'_, u64, T> {
        // self.table.entry(k.state).or_default().entry(k.path)
        todo!()
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub struct ZobristTable<const N: usize, const D: usize> {
    state: Box<[u64; N]>,
    path: Vec<u64>,
}

impl<const N: usize, const D: usize> ZobristTable<N, D> {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut state = Box::new([0; N]);
        for h in &mut *state {
            *h = rng.gen::<u64>();
        }

        let mut path = Vec::with_capacity(N * D);
        (0..N * D).for_each(|_| {
            path.push(rng.gen::<u64>());
        });
        ZobristTable { state, path }
    }

    fn state_hash(&self, index: usize) -> u64 {
        debug_assert!(index < N);
        self.state[index]
    }

    fn path_hash(&self, index: usize, depth: usize) -> u64 {
        debug_assert!(index < N);
        debug_assert!(depth < D);
        self.path[index * D + depth]
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub struct LazyZobristTable<const N: usize, const D: usize> {
    once: OnceLock<ZobristTable<N, D>>,
    seed: u64,
}

impl<const N: usize, const D: usize> LazyZobristTable<N, D> {
    pub const fn new(seed: u64) -> Self {
        LazyZobristTable {
            once: OnceLock::new(),
            seed,
        }
    }

    #[inline(always)]
    fn get_or_init(&self) -> &ZobristTable<N, D> {
        self.once.get_or_init(|| ZobristTable::new(self.seed))
    }

    #[inline(always)]
    pub fn state_hash(&self, index: usize) -> u64 {
        self.get_or_init().state_hash(index)
    }

    #[inline(always)]
    pub fn path_hash(&self, index: usize, depth: usize) -> u64 {
        if depth >= D {
            panic!("maximum depth {depth} exceeded for ZobristTable");
        }
        self.get_or_init().path_hash(index, depth)
    }

    #[inline]
    pub fn apply(&self, index: usize, depth: usize, key: &mut ZobristKey) {
        if depth >= D {
            panic!("maximum depth {depth} exceeded for ZobristTable");
        }
        let table = self.get_or_init();
        key.state ^= table.state_hash(index);
        key.path ^= table.path_hash(index, depth);
    }
}
