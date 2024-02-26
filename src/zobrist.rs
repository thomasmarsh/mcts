use rand::rngs::StdRng;
use rand::Rng;
use rand_core::SeedableRng;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::OnceLock;

#[derive(Copy, Clone)]
pub struct ZobristHash(pub u64);

impl Hash for ZobristHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

pub struct ZobristTable<const N: usize, const D: usize> {
    action: [u64; N],
    path: [[u64; D]; N],
}

// TODO: Consider splitting moves for the path only:
//
// > We note that the size of the random table is small enough for current
// > hardware. For example, in our experiments on 19 × 19 Go, setting MaxMove =
// > 362 and MaxDepth = 50 the size is about 140KB. In games with a large number
// > of different possible moves, such as Shogi or Amazons, a move can be split
// > into two or three partial moves, for example by separating the from-square
// > information from the to-square information. This way MaxMove can be greatly
// > reduced, while MaxDepth increases by a factor of 2 or 3.
//
// See: Kishimoto, A., Müller, M., A General Solution to the Graph History
// Interaction Problem.
//
// Splitting, in general, is usually beneficial, but has pathological cases
// where it is not desirable for MCTS. See: Kowalski, et al, Split Moves for
// Monte-Carlo Tree Search
impl<const N: usize, const D: usize> ZobristTable<N, D> {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut action = [0; N];
        let mut path = [[0; D]; N];

        for i in 0..N {
            action[i] = rng.gen::<u64>();
            for j in 0..D {
                path[i][j] = rng.gen::<u64>();
            }
        }

        ZobristTable { action, path }
    }

    fn action_hash(&self, index: usize) -> u64 {
        self.action[index]
    }
}

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

    fn get_or_init(&self) -> &ZobristTable<N, D> {
        self.once.get_or_init(|| ZobristTable::new(self.seed))
    }

    pub fn action_hash(&self, index: usize) -> u64 {
        self.get_or_init().action_hash(index)
    }
}
