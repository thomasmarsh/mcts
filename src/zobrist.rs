use rand::rngs::StdRng;
use rand::Rng;
use rand_core::SeedableRng;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::sync::OnceLock;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ZobristHash(pub u64);

////////////////////////////////////////////////////////////////////////////////////////

// TODO: this is meant to avoid any further work on an already precomputed
// Zobrist hash, but it hasn't been benchmarked to see if it's worth it.
impl Hasher for ZobristHash {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {}

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
pub struct ZobristHashBuilder {
    hash: u64,
}

impl BuildHasher for ZobristHashBuilder {
    type Hasher = ZobristHash;

    fn build_hasher(&self) -> ZobristHash {
        ZobristHash(self.hash)
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Default)]
pub struct ZobristHashMap<T>(pub HashMap<ZobristHash, T, ZobristHashBuilder>);

impl<T> ZobristHashMap<T> {
    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[inline]
    pub fn get(&mut self, k: u64) -> Option<&T> {
        self.0.get(&ZobristHash(k))
    }

    #[inline]
    pub fn entry(&mut self, k: u64) -> Entry<'_, ZobristHash, T> {
        self.0.entry(ZobristHash(k))
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub struct ZobristTable<const N: usize> {
    hashes: [u64; N],
    initial: u64,
    // We have a unique path via node_id in mcts, but other approaches might
    // benefit from having a path hash.  See: Kishimoto, A., Müller, M., A
    // General Solution to the Graph History Interaction Problem.
    //
    // path: [[u64; D]; N],
}

impl<const N: usize> ZobristTable<N> {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut hashes = [0; N];
        for h in &mut hashes {
            *h = rng.gen::<u64>();
        }

        ZobristTable {
            hashes,
            initial: rng.gen::<u64>(),
        }
    }

    fn hash(&self, index: usize) -> u64 {
        self.hashes[index]
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub struct LazyZobristTable<const N: usize> {
    once: OnceLock<ZobristTable<N>>,
    seed: u64,
}

impl<const N: usize> LazyZobristTable<N> {
    pub const fn new(seed: u64) -> Self {
        LazyZobristTable {
            once: OnceLock::new(),
            seed,
        }
    }

    #[inline(always)]
    fn get_or_init(&self) -> &ZobristTable<N> {
        self.once.get_or_init(|| ZobristTable::new(self.seed))
    }

    #[inline(always)]
    pub fn hash(&self, index: usize) -> u64 {
        self.get_or_init().hash(index)
    }

    /// The initial value should be used as the "empty" or initial state
    /// of the game. I've seen implementations that initialize the initial
    /// board to zero. TODO: I'm not sure that it matters or not. Maybe zero
    /// initialization is fine.
    #[inline(always)]
    pub fn initial(&self) -> u64 {
        self.get_or_init().initial
    }
}
