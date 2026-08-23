use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::{Board, Direction};
use crate::dim::Dim;
use crate::storage::Storage;

//////////////////////////////////////////////////////////////////////////////////////////////////

// Shifts (carry a bit shift across word boundaries, bignum-style -- same
// approach as `BigBitBoard`'s `Shl`/`Shr`). Only fixed offsets less than 64
// are supported: shift_north/south use `cols()` and shift_east/west use `1`,
// both always well under 64 for any board size these games use.

impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    #[inline]
    fn raw_shl(self, rhs: usize) -> Self {
        debug_assert!(rhs < 64);
        if rhs == 0 {
            return self;
        }
        let mut out = self;
        for w in (0..S::CAPACITY_WORDS).rev() {
            let mut value = self.bits.word(w) << rhs;
            if w > 0 {
                value |= self.bits.word(w - 1) >> (64 - rhs);
            }
            *out.bits.word_mut(w) = value;
        }
        out
    }

    #[inline]
    fn raw_shr(self, rhs: usize) -> Self {
        debug_assert!(rhs < 64);
        if rhs == 0 {
            return self;
        }
        let mut out = self;
        for w in 0..S::CAPACITY_WORDS {
            let mut value = self.bits.word(w) >> rhs;
            if w + 1 < S::CAPACITY_WORDS {
                value |= self.bits.word(w + 1) << (64 - rhs);
            }
            *out.bits.word_mut(w) = value;
        }
        out
    }
}

/// Raw, arbitrary-distance word shifts (as opposed to `shift_north`/etc.,
/// which shift by exactly one cell and mask off the wrapped-around wall).
/// Callers implementing their own multi-step flood (e.g. Othello's
/// Kogge-Stone dumb7fill, which shifts by 1/7/8/9 directly) need the raw
/// operation; `shift_north`/`shift_east`/etc. are built on top of these.
impl<S: Storage, R: Dim, C: Dim> std::ops::Shl<usize> for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        self.raw_shl(rhs)
    }
}

impl<S: Storage, R: Dim, C: Dim> std::ops::Shr<usize> for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        self.raw_shr(rhs)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Wall masks. `wall_words_uncached` is the one operation in this module that's
// a real loop over the board's dimension rather than O(1) word arithmetic.
// Whether that loop's result is worth memoizing depends on the dim kind --
// see `Dim::CACHE_WALLS`'s doc comment -- so `wall_words` dispatches on it:
// `Dyn` boards (many sizes, one monomorphization) hit the process-wide cache
// keyed by `TypeId` (a local `static` can't name a generic parameter from
// its enclosing `impl`, i.e. `S` here, so this stands in for "one table per
// `S`"); `Const<N>, Const<M>` boards (one monomorphization *per* size)
// recompute directly every call instead, skipping the lock/hashmap/downcast
// entirely -- for a loop bounded by `rows`/`cols` (at most 19 in every game
// this crate serves today), that's cheaper than the synchronization it would
// otherwise pay on every `wall()`/shift call.
impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    fn wall_words_uncached(rows: usize, cols: usize, direction: Direction) -> S {
        let mut out = S::zero();
        let limit = match direction {
            Direction::North | Direction::South => cols,
            Direction::East | Direction::West => rows,
        };
        for i in 0..limit {
            let k = match direction {
                Direction::North => (rows - 1) * cols + i,
                Direction::East => (i + 1) * cols - 1,
                Direction::South => i,
                Direction::West => i * cols,
            };
            *out.word_mut(k / 64) |= 1u64 << (k % 64);
        }
        out
    }

    fn compute_walls(rows: usize, cols: usize) -> [S; 4] {
        [
            Self::wall_words_uncached(rows, cols, Direction::North),
            Self::wall_words_uncached(rows, cols, Direction::East),
            Self::wall_words_uncached(rows, cols, Direction::South),
            Self::wall_words_uncached(rows, cols, Direction::West),
        ]
    }

    fn wall_words(rows: usize, cols: usize, direction: Direction) -> S {
        if !(R::CACHE_WALLS || C::CACHE_WALLS) {
            return Self::wall_words_uncached(rows, cols, direction);
        }

        use std::any::{Any, TypeId};

        type WallCache = Mutex<HashMap<(TypeId, usize, usize), Box<dyn Any + Send>>>;
        static CACHE: OnceLock<WallCache> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().unwrap();
        let key = (TypeId::of::<S>(), rows, cols);
        let walls = guard
            .entry(key)
            .or_insert_with(|| Box::new(Self::compute_walls(rows, cols)) as Box<dyn Any + Send>)
            .downcast_ref::<[S; 4]>()
            .expect("wall cache entry type mismatch for this key");
        walls[direction as usize]
    }

    /// The board-shaped mask of every cell along `direction`'s edge.
    pub fn wall(&self, direction: Direction) -> Self {
        Self {
            bits: Self::wall_words(self.rows(), self.cols(), direction),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Board displacement

impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    #[inline]
    pub fn shift_north(self) -> Self {
        (self & !self.wall(Direction::North)).raw_shl(self.cols())
    }

    #[inline]
    pub fn shift_east(self) -> Self {
        (self & !self.wall(Direction::East)).raw_shl(1)
    }

    #[inline]
    pub fn shift_south(self) -> Self {
        self.raw_shr(self.cols())
    }

    #[inline]
    pub fn shift_west(self) -> Self {
        (self & !self.wall(Direction::West)).raw_shr(1)
    }

    /// The hex-adjacent diagonal `flood6` needs -- see `flood6`'s own doc
    /// comment for why northeast/southwest (not northwest/southeast) is the
    /// pair that turns 4-way adjacency into 6-way.
    #[inline]
    pub fn shift_northeast(self) -> Self {
        (self & !self.wall(Direction::North) & !self.wall(Direction::East)).raw_shl(self.cols() + 1)
    }

    #[inline]
    pub fn shift_southwest(self) -> Self {
        (self & !self.wall(Direction::South) & !self.wall(Direction::West)).raw_shr(self.cols() + 1)
    }

    /// The *other* diagonal -- deliberately excluded from `flood6`'s 6-way
    /// adjacency (see `shift_northeast`'s doc comment), but still needed by a
    /// caller that wants full 8-way (queen-move) adjacency, e.g. a chess-like
    /// game or `gdl`'s `Connectivity::Eight`.
    #[inline]
    pub fn shift_northwest(self) -> Self {
        (self & !self.wall(Direction::North) & !self.wall(Direction::West)).raw_shl(self.cols() - 1)
    }

    #[inline]
    pub fn shift_southeast(self) -> Self {
        (self & !self.wall(Direction::South) & !self.wall(Direction::East)).raw_shr(self.cols() - 1)
    }

    #[inline]
    pub fn shift(self, direction: Direction) -> Self {
        match direction {
            Direction::North => self.shift_north(),
            Direction::East => self.shift_east(),
            Direction::South => self.shift_south(),
            Direction::West => self.shift_west(),
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Adjacency

impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    #[inline]
    pub fn adjacency_mask(self) -> Self {
        (self.shift_north() | self.shift_east() | self.shift_south() | self.shift_west()) & !self
    }
}
