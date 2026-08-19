use std::collections::HashMap;
use std::ops::{BitAnd, BitOr, BitXor, Not};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::dim::Dim;
use crate::storage::Storage;

/// A cardinal direction on a `Board`, used to pick a wall mask or a shift.
/// The discriminant order (`North` = 0, ..., `West` = 3) is load-bearing:
/// `Board::compute_walls` relies on it to index the cached `[S; 4]` wall
/// array without a match.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

/// A generic `rows x cols` bitboard: `S` picks the storage backend (`u64`
/// for a single-word board, `[u64; WORDS]` for a multi-word one), `R`/`C`
/// pick whether the row/column counts are compile-time (`Const<N>`) or
/// runtime (`Dyn`) values. Indexing is row-major (`row * cols + col`),
/// matching `BitBoard`/`BigBitBoard`'s existing wire format.
///
/// Supports `get`/`set`/`clear`/`count_ones`/iteration over set bits, the
/// `&`/`|`/`^`/`!` binary ops, serde, cardinal/hex-diagonal shifts, wall
/// masks, `flood4`/`flood6`/`flood8`, and opposite-wall connectivity tests.
/// Go-specific capture logic (`check_go_move`) is not yet implemented.
#[derive(Clone, Copy, Debug)]
pub struct Board<S: Storage, R: Dim, C: Dim> {
    bits: S,
    rows: R,
    cols: C,
}

impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    #[inline(always)]
    pub fn new(rows: R, cols: C) -> Self {
        Self {
            bits: S::zero(),
            rows,
            cols,
        }
    }

    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows.get()
    }

    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.cols.get()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.rows() * self.cols()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    fn index_of(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.rows());
        debug_assert!(col < self.cols());
        row * self.cols() + col
    }

    /// Gets a single bit by its row-major index (`row * cols + col`), rather
    /// than by `(row, col)` -- e.g. for a value already produced by
    /// `iter_set`/an action index, where recovering `(row, col)` first would
    /// be pure overhead.
    #[inline(always)]
    pub fn get_index(&self, index: usize) -> bool {
        (self.bits.word(index / 64) >> (index % 64)) & 1 != 0
    }

    /// Sets a single bit by its row-major index -- see `get_index`.
    #[inline(always)]
    pub fn set_index(&mut self, index: usize) {
        *self.bits.word_mut(index / 64) |= 1u64 << (index % 64);
    }

    /// Clears a single bit by its row-major index -- see `get_index`.
    #[inline(always)]
    pub fn clear_index(&mut self, index: usize) {
        *self.bits.word_mut(index / 64) &= !(1u64 << (index % 64));
    }

    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> bool {
        self.get_index(self.index_of(row, col))
    }

    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize) {
        let index = self.index_of(row, col);
        self.set_index(index);
    }

    #[inline(always)]
    pub fn clear(&mut self, row: usize, col: usize) {
        let index = self.index_of(row, col);
        self.clear_index(index);
    }

    pub fn count_ones(&self) -> u32 {
        (0..S::CAPACITY_WORDS)
            .map(|w| self.bits.word(w).count_ones())
            .sum()
    }

    /// The raw backing words, low word first -- for a caller that needs to fold over every word
    /// generically (e.g. hashing), independent of `S`'s concrete layout. Mirrors
    /// `BigBitBoard::words`, generalized over any storage rather than only `[u64; WORDS]`.
    pub fn words(&self) -> impl Iterator<Item = u64> + '_ {
        (0..S::CAPACITY_WORDS).map(move |w| self.bits.word(w))
    }

    /// Iterates the row-major indices (`row * cols + col`) of set bits, in
    /// ascending order. Pops the lowest set bit via `trailing_zeros` (a
    /// single BSF/TZCNT) each step, so cost is O(popcount) per word rather
    /// than a fixed O(64) scan -- the same idiom `nego`'s `BitBoard` uses for
    /// its `Iterator` impl, which matters on the mostly-empty boards this
    /// crate's flood/connectivity ops iterate most (e.g. a near-empty 19x19
    /// Go board still costs a full 6-word x 64-bit scan under the naive
    /// version).
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        (0..S::CAPACITY_WORDS).flat_map(move |w| {
            let mut word = self.bits.word(w);
            std::iter::from_fn(move || {
                if word == 0 {
                    None
                } else {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some(w * 64 + bit)
                }
            })
        })
    }

    /// True if no bits are set (independent of `is_empty`, which reports
    /// whether the board has zero cells at all).
    #[inline]
    fn bits_empty(&self) -> bool {
        (0..S::CAPACITY_WORDS).all(|w| self.bits.word(w) == 0)
    }

    /// True if no bits are set. Public counterpart to `bits_empty`, for
    /// callers that need to ask a board (as opposed to `is_empty`, which asks
    /// the board's declared dimensions) whether it currently holds any bits.
    #[inline]
    pub fn none_set(&self) -> bool {
        self.bits_empty()
    }

    #[inline]
    pub fn intersects(self, rhs: Self) -> bool {
        !(self & rhs).bits_empty()
    }

    #[inline]
    pub fn is_subset(self, rhs: Self) -> bool {
        (self & rhs) == self
    }

    #[inline]
    pub fn is_disjoint(self, rhs: Self) -> bool {
        (self & rhs).bits_empty()
    }

    /// Combines `self` and `rhs` word-by-word under `f`, keeping `self`'s
    /// dims -- the caller (a same-type binary op) guarantees both boards
    /// share the same `rows`/`cols`.
    #[inline]
    fn combine(mut self, rhs: Self, f: impl Fn(u64, u64) -> u64) -> Self {
        for w in 0..S::CAPACITY_WORDS {
            let value = f(self.bits.word(w), rhs.bits.word(w));
            *self.bits.word_mut(w) = value;
        }
        self
    }

    /// The bitmask for word `w` covering only bits within `0..len()`, used
    /// to keep `Not` from setting padding bits past the board's real cell
    /// count in the last word.
    #[inline]
    fn word_mask(&self, w: usize) -> u64 {
        let total = self.len();
        let word_start = w * 64;
        if word_start >= total {
            0
        } else if total - word_start >= 64 {
            u64::MAX
        } else {
            (1u64 << (total - word_start)) - 1
        }
    }
}

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

// Wall masks. `wall_words_uncached` is the one operation in this file that's
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

//////////////////////////////////////////////////////////////////////////////////////////////////

// Flood fill

impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    /// Performs a four-way floodfill traversing set bits, starting from
    /// `start` (a row-major index, matching `iter_set`'s output). It might
    /// seem more natural to fill unset bits, but that requires one
    /// additional operation in this function, so that decision is up to the
    /// client.
    pub fn flood4(self, start: usize) -> Self {
        debug_assert!(start < self.len());
        let mut seed = Self::new(self.rows, self.cols);
        seed.set_index(start);
        let mut flood = seed & self;

        if flood.bits_empty() {
            return flood;
        }

        loop {
            let temp = flood;
            flood = flood
                | flood.shift_north()
                | flood.shift_east()
                | flood.shift_south()
                | flood.shift_west();
            flood &= self;
            if flood == temp {
                break;
            }
        }
        flood
    }

    /// Performs an eight-way floodfill traversing set bits, starting from
    /// `start`. Mirrors `BitBoard::flood8` (`BigBitBoard` doesn't implement
    /// this): unlike `flood4`, the north/south shifts are OR'd into `flood`
    /// in one statement *before* the east/west shifts are computed off that
    /// updated value, in a second statement -- so `flood.shift_east()` in
    /// the second statement also reaches cells diagonal to the original
    /// `flood` (`east(north(x))` = northeast of `x`, etc.), giving full
    /// 8-way connectivity through compounding, without a real diagonal
    /// shift primitive.
    pub fn flood8(self, start: usize) -> Self {
        debug_assert!(start < self.len());
        let mut seed = Self::new(self.rows, self.cols);
        seed.set_index(start);
        let mut flood = seed & self;

        if flood.bits_empty() {
            return flood;
        }

        loop {
            let temp = flood;
            flood = flood | flood.shift_north() | flood.shift_south();
            flood = flood | flood.shift_east() | flood.shift_west();
            flood &= self;
            if flood == temp {
                break;
            }
        }
        flood
    }

    /// Performs a six-way (hex) floodfill traversing set bits, seeded from
    /// every set bit of `seed` rather than a single start index -- unlike
    /// `flood4`/`flood8`, a hex connection check (e.g. "does the mover's
    /// group touch both of their board edges") needs to seed from an entire
    /// edge region, which may contain several disconnected stones.
    ///
    /// All six shifts must be OR'd into `flood` in a single expression, as
    /// `flood4` does (unlike `flood8`, which splits its four shifts across
    /// two statements) -- splitting them here would let one direction's
    /// shift compound off a *previous statement's* unmasked result within
    /// the same iteration, bridging through a cell not actually in `self`.
    /// Six-way adjacency deliberately excludes the northwest/southeast
    /// diagonal, so that compounding would wrongly treat it as connected.
    pub fn flood6(self, seed: Self) -> Self {
        let mut flood = seed & self;

        if flood.bits_empty() {
            return flood;
        }

        loop {
            let temp = flood;
            flood = flood
                | flood.shift_north()
                | flood.shift_south()
                | flood.shift_east()
                | flood.shift_west()
                | flood.shift_northeast()
                | flood.shift_southwest();
            flood &= self;
            if flood == temp {
                break;
            }
        }
        flood
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Connectivity tests

impl<S: Storage, R: Dim, C: Dim> Board<S, R, C> {
    #[inline]
    pub fn has_opposite_connection4(self, start: usize) -> bool {
        let n = self.wall(Direction::North);
        let e = self.wall(Direction::East);
        let s = self.wall(Direction::South);
        let w = self.wall(Direction::West);

        let mut seed = Self::new(self.rows, self.cols);
        seed.set_index(start);
        let mut flood = seed & self;

        if flood.bits_empty() {
            return false;
        }

        loop {
            let temp = flood;
            flood = flood
                | flood.shift_north()
                | flood.shift_east()
                | flood.shift_south()
                | flood.shift_west();
            flood &= self;
            if (flood.intersects(n) && flood.intersects(s))
                || (flood.intersects(e) && flood.intersects(w))
            {
                return true;
            } else if flood == temp {
                return false;
            }
        }
    }

    /// The eight-way counterpart to `has_opposite_connection4`. Reaches
    /// diagonal neighbors the same way `flood8` does -- see its doc comment.
    #[inline]
    pub fn has_opposite_connection8(self, start: usize) -> bool {
        let n = self.wall(Direction::North);
        let e = self.wall(Direction::East);
        let s = self.wall(Direction::South);
        let w = self.wall(Direction::West);

        let mut seed = Self::new(self.rows, self.cols);
        seed.set_index(start);
        let mut flood = seed & self;

        if flood.bits_empty() {
            return false;
        }

        loop {
            let temp = flood;
            flood = flood | flood.shift_north() | flood.shift_south();
            flood = flood | flood.shift_east() | flood.shift_west();
            flood &= self;
            if (flood.intersects(n) && flood.intersects(s))
                || (flood.intersects(e) && flood.intersects(w))
            {
                return true;
            } else if flood == temp {
                return false;
            }
        }
    }
}

impl<S: Storage, const N: usize, const M: usize>
    Board<S, crate::dim::Const<N>, crate::dim::Const<M>>
{
    #[inline(always)]
    pub fn new_const() -> Self {
        Self::new(crate::dim::Const, crate::dim::Const)
    }
}

impl<R: Dim, C: Dim> Board<u64, R, C> {
    /// The raw backing word, for wire formats that serialize a single-word
    /// board as plain hex (mirroring `BitBoard::bits`) rather than through
    /// `Board`'s own `Serialize` impl.
    #[inline(always)]
    pub fn bits(&self) -> u64 {
        self.bits
    }
}

impl<const N: usize, const M: usize> Board<u64, crate::dim::Const<N>, crate::dim::Const<M>> {
    /// Builds a board directly from a raw row-major bitmask -- e.g. a
    /// literal winning-line pattern a game (or codegen) already knows at
    /// compile time, mirroring `BitBoard::new(value: u64)`. Only defined for
    /// single-word (`u64`) storage at `Const` dims, the shape every such
    /// literal mask this crate serves fits in.
    #[inline(always)]
    pub const fn from_bits(bits: u64) -> Self {
        Self {
            bits,
            rows: crate::dim::Const,
            cols: crate::dim::Const,
        }
    }

    /// The board-shaped constant with no bits set.
    pub const EMPTY: Self = Self::from_bits(0);

    /// The board-shaped constant with every in-bounds cell (`0..N*M`) set --
    /// mirrors `BitBoard::ONES`, used as the "no wall guard needed" mask for
    /// shifts that can't wrap off either edge.
    pub const ONES: Self = Self::from_bits(if N * M == 64 {
        u64::MAX
    } else {
        (1u64 << (N * M)) - 1
    });

    /// A board with only row-major index `index` set, matching
    /// `BitBoard::from_index`'s static call form -- only defined at `Const`
    /// dims, where `N`/`M` are known without an existing instance to
    /// template off of.
    #[inline(always)]
    pub const fn from_index(index: usize) -> Self {
        debug_assert!(index < N * M);
        Self::from_bits(1u64 << index)
    }

    /// A board with only `(row, col)` set.
    #[inline(always)]
    pub fn from_coord(row: usize, col: usize) -> Self {
        debug_assert!(row < N);
        debug_assert!(col < M);
        Self::from_index(Self::to_index(row, col))
    }

    /// The row-major index of `(row, col)`.
    #[inline(always)]
    pub const fn to_index(row: usize, col: usize) -> usize {
        row * M + col
    }

    /// The inverse of `to_index`.
    #[inline(always)]
    pub const fn to_coord(index: usize) -> (usize, usize) {
        (index / M, index % M)
    }
}

impl<S: Storage, R: Dim, C: Dim> PartialEq for Board<S, R, C> {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows()
            && self.cols() == other.cols()
            && (0..S::CAPACITY_WORDS).all(|w| self.bits.word(w) == other.bits.word(w))
    }
}

impl<S: Storage, R: Dim, C: Dim> Eq for Board<S, R, C> {}

impl<S: Storage, R: Dim, C: Dim> BitAnd for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.combine(rhs, |a, b| a & b)
    }
}

impl<S: Storage, R: Dim, C: Dim> BitOr for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.combine(rhs, |a, b| a | b)
    }
}

impl<S: Storage, R: Dim, C: Dim> BitXor for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.combine(rhs, |a, b| a ^ b)
    }
}

impl<S: Storage, R: Dim, C: Dim> Not for Board<S, R, C> {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        let mut out = self;
        for w in 0..S::CAPACITY_WORDS {
            let mask = self.word_mask(w);
            *out.bits.word_mut(w) = !self.bits.word(w) & mask;
        }
        out
    }
}

impl<S: Storage, R: Dim, C: Dim> std::ops::BitAndAssign for Board<S, R, C> {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<S: Storage, R: Dim, C: Dim> std::ops::BitOrAssign for Board<S, R, C> {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<S: Storage, R: Dim, C: Dim> std::ops::BitXorAssign for Board<S, R, C> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

/// Consumes set bits lowest-index-first, same idiom as
/// `game_core::bitboard::BitBoard`'s `Iterator` impl -- lets a `for src in
/// board` loop (over a `Copy` board, so the loop body's `board` binding is
/// unaffected) walk row-major indices without going through `iter_set`'s
/// borrow.
impl<S: Storage, R: Dim, C: Dim> Iterator for Board<S, R, C> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        for w in 0..S::CAPACITY_WORDS {
            let word = self.bits.word(w);
            if word != 0 {
                let bit = word.trailing_zeros() as usize;
                *self.bits.word_mut(w) = word & (word - 1);
                return Some(w * 64 + bit);
            }
        }
        None
    }
}

// Serde. `S` (`u64` or `[u64; WORDS]`) doesn't itself implement
// `Serialize`/`Deserialize` generically over a const `WORDS`, so words are
// collected into a plain `Vec<u64>` via `Storage::word`/`word_mut` instead.
// `rows`/`cols` ride along as plain `usize`s so a `Dyn`-dimensioned board's
// runtime size survives the round trip; `Const<N>` verifies the
// deserialized length still matches `N` via `Dim::from_len`.

#[derive(Serialize)]
struct BoardDataRef {
    rows: usize,
    cols: usize,
    words: Vec<u64>,
}

#[derive(Deserialize)]
struct BoardDataOwned {
    rows: usize,
    cols: usize,
    words: Vec<u64>,
}

impl<S: Storage, R: Dim, C: Dim> Serialize for Board<S, R, C> {
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        BoardDataRef {
            rows: self.rows(),
            cols: self.cols(),
            words: (0..S::CAPACITY_WORDS).map(|w| self.bits.word(w)).collect(),
        }
        .serialize(serializer)
    }
}

impl<'de, S: Storage, R: Dim, C: Dim> Deserialize<'de> for Board<S, R, C> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let data = BoardDataOwned::deserialize(deserializer)?;
        if data.words.len() != S::CAPACITY_WORDS {
            return Err(D::Error::invalid_length(
                data.words.len(),
                &format!("{} words", S::CAPACITY_WORDS).as_str(),
            ));
        }

        let mut bits = S::zero();
        for (w, word) in data.words.into_iter().enumerate() {
            *bits.word_mut(w) = word;
        }

        let rows = R::from_len(data.rows);
        let cols = C::from_len(data.cols);
        // `Const<N>::from_len` ignores its argument (it has no runtime state
        // to restore), so a mismatched `data.rows`/`data.cols` -- e.g.
        // deserializing a 11x11 board's JSON as `Const<9>` -- must be caught
        // here instead, by comparing what was actually reconstructed against
        // what was on the wire.
        if rows.get() != data.rows || cols.get() != data.cols {
            return Err(D::Error::custom(format!(
                "Board: dims on the wire ({}x{}) don't match the target type's dims ({}x{})",
                data.rows,
                data.cols,
                rows.get(),
                cols.get()
            )));
        }

        Ok(Board { bits, rows, cols })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::{Const, Dyn};
    use proptest::prelude::*;

    // Array-backed oracle: an independent, obviously-correct `Vec<bool>`
    // model checked against `Board` across the same representative sizes
    // `bigbitboard.rs`'s oracle tests cover -- a sub-word board, an
    // exact-word-boundary board, and every WORDS from 1..6 -- at *both*
    // `Const` and `Dyn` dims, since both must agree bit-for-bit.

    fn check_against_oracle<S: Storage, R: Dim, C: Dim>(
        rows: R,
        cols: C,
        sets: &[usize],
        clears: &[usize],
    ) {
        let (n, m) = (rows.get(), cols.get());
        let bits = n * m;
        let mut oracle = vec![false; bits];
        let mut board: Board<S, R, C> = Board::new(rows, cols);
        let to_coord = |i: usize| (i / m, i % m);

        for &i in sets {
            let i = i % bits;
            oracle[i] = true;
            let (r, c) = to_coord(i);
            board.set(r, c);
        }
        for &i in clears {
            let i = i % bits;
            oracle[i] = false;
            let (r, c) = to_coord(i);
            board.clear(r, c);
        }

        for (i, &expected) in oracle.iter().enumerate() {
            let (r, c) = to_coord(i);
            assert_eq!(board.get(r, c), expected, "get({i}) mismatch");
        }

        assert_eq!(
            board.count_ones() as usize,
            oracle.iter().filter(|&&b| b).count(),
            "count_ones mismatch"
        );

        let mut got: Vec<usize> = board.iter_set().collect();
        got.sort_unstable();
        let expected: Vec<usize> = (0..bits).filter(|&i| oracle[i]).collect();
        assert_eq!(got, expected, "iterated set bits mismatch");
    }

    fn check_binary_ops_against_oracle<S: Storage, R: Dim, C: Dim>(
        rows: R,
        cols: C,
        a_bits: &[usize],
        b_bits: &[usize],
    ) {
        let (n, m) = (rows.get(), cols.get());
        let bits = n * m;
        let mut oa = vec![false; bits];
        let mut ob = vec![false; bits];
        let mut a: Board<S, R, C> = Board::new(rows, cols);
        let mut b: Board<S, R, C> = Board::new(rows, cols);
        let to_coord = |i: usize| (i / m, i % m);

        for &i in a_bits {
            let i = i % bits;
            oa[i] = true;
            let (r, c) = to_coord(i);
            a.set(r, c);
        }
        for &i in b_bits {
            let i = i % bits;
            ob[i] = true;
            let (r, c) = to_coord(i);
            b.set(r, c);
        }

        let union = a | b;
        let inter = a & b;
        let xor = a ^ b;
        let not_a = !a;

        for i in 0..bits {
            let (r, c) = to_coord(i);
            assert_eq!(union.get(r, c), oa[i] || ob[i], "union mismatch at {i}");
            assert_eq!(inter.get(r, c), oa[i] && ob[i], "intersect mismatch at {i}");
            assert_eq!(xor.get(r, c), oa[i] ^ ob[i], "xor mismatch at {i}");
            assert_eq!(not_a.get(r, c), !oa[i], "not mismatch at {i}");
        }

        assert_eq!(
            a.intersects(b),
            (0..bits).any(|i| oa[i] && ob[i]),
            "intersects mismatch"
        );
        assert_eq!(
            a.is_subset(b),
            (0..bits).all(|i| !oa[i] || ob[i]),
            "is_subset mismatch"
        );
        assert_eq!(
            a.is_disjoint(b),
            (0..bits).all(|i| !(oa[i] && ob[i])),
            "is_disjoint mismatch"
        );
    }

    macro_rules! oracle_tests {
        ($mod_name:ident, $n:expr, $m:expr, $storage:ty, $max_index:expr) => {
            mod $mod_name {
                use super::*;

                proptest! {
                    #[test]
                    fn const_get_set_clear_count_iter(
                        sets in proptest::collection::vec(0usize..$max_index, 0..200),
                        clears in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &sets, &clears);
                    }

                    #[test]
                    fn dyn_get_set_clear_count_iter(
                        sets in proptest::collection::vec(0usize..$max_index, 0..200),
                        clears in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &sets, &clears);
                    }

                    #[test]
                    fn const_binary_ops(
                        a_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        b_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_binary_ops_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &a_bits, &b_bits);
                    }

                    #[test]
                    fn dyn_binary_ops(
                        a_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        b_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_binary_ops_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &a_bits, &b_bits);
                    }
                }
            }
        };
    }

    // Sub-word board.
    oracle_tests!(oracle_3x3, 3, 3, u64, 9);
    // Exact single-word boundary (64 bits, remainder == 0).
    oracle_tests!(oracle_8x8, 8, 8, u64, 64);
    // Multi-word sizes, matching `bigbitboard.rs`'s coverage.
    oracle_tests!(oracle_9x9, 9, 9, [u64; 2], 81);
    oracle_tests!(oracle_11x11, 11, 11, [u64; 2], 121);
    oracle_tests!(oracle_13x13, 13, 13, [u64; 3], 169);
    oracle_tests!(oracle_19x19, 19, 19, [u64; 6], 361);

    #[test]
    fn not_masks_padding_bits_in_last_word() {
        // A 9x9 board (81 bits) in 2 words leaves 47 padding bits past bit
        // 80 in word 1; complementing an empty board must not set them, or
        // count_ones would report 128 instead of 81.
        let empty: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
        let full = !empty;
        assert_eq!(full.count_ones(), 81);
    }

    #[test]
    fn serde_round_trips_across_word_boundary() {
        let mut board: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
        board.set(0, 0);
        board.set(7, 0); // index 63, last bit of word 0
        board.set(7, 1); // index 64, first bit of word 1
        board.set(8, 8); // index 80, last valid bit

        let json = serde_json::to_string(&board).unwrap();
        let round_tripped: Board<[u64; 2], Const<9>, Const<9>> =
            serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, board);
    }

    #[test]
    fn serde_round_trips_dyn_dims() {
        let mut board: Board<[u64; 6], Dyn, Dyn> = Board::new(Dyn(13), Dyn(13));
        board.set(0, 0);
        board.set(12, 12);

        let json = serde_json::to_string(&board).unwrap();
        let round_tripped: Board<[u64; 6], Dyn, Dyn> = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, board);
        assert_eq!(round_tripped.rows(), 13);
        assert_eq!(round_tripped.cols(), 13);
    }

    #[test]
    fn deserialize_rejects_const_length_mismatch() {
        let mut board: Board<[u64; 6], Dyn, Dyn> = Board::new(Dyn(9), Dyn(9));
        board.set(0, 0);
        let json = serde_json::to_string(&board).unwrap();

        let result: Result<Board<[u64; 6], Const<9>, Const<9>>, _> = serde_json::from_str(&json);
        // rows/cols (9, 9) match Const<9> here, so this should succeed --
        // the interesting negative case is a genuine size mismatch.
        assert!(result.is_ok());

        let mismatched_json = json.replace("\"rows\":9", "\"rows\":11");
        let result: Result<Board<[u64; 6], Const<9>, Const<9>>, _> =
            serde_json::from_str(&mismatched_json);
        assert!(result.is_err());
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Phase 2: shift/wall/adjacency/flood4/flood8/flood6/connectivity oracle
    // tests, ported from `bigbitboard.rs`'s `check_shifts_against_oracle`/
    // `check_flood4_against_oracle`/`check_flood6_against_oracle` to run
    // against `Board` at both `Const` and `Dyn` dims.

    /// Independent row/col-arithmetic oracle for the four cardinal shifts.
    fn check_shifts_against_oracle<S: Storage, R: Dim, C: Dim>(rows: R, cols: C, bits: &[usize]) {
        let (n, m) = (rows.get(), cols.get());
        let total = n * m;
        let mut board: Board<S, R, C> = Board::new(rows, cols);
        let mut set: Vec<(usize, usize)> = Vec::new();
        for &i in bits {
            let i = i % total;
            let (r, c) = (i / m, i % m);
            board.set(r, c);
            set.push((r, c));
        }

        let check = |shifted: Board<S, R, C>, delta: (i64, i64), label: &str| {
            let expected: Vec<(usize, usize)> = set
                .iter()
                .filter_map(|&(r, c)| {
                    let nr = r as i64 + delta.0;
                    let nc = c as i64 + delta.1;
                    if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < m {
                        Some((nr as usize, nc as usize))
                    } else {
                        None
                    }
                })
                .collect();
            for row in 0..n {
                for col in 0..m {
                    let expect = expected.contains(&(row, col));
                    assert_eq!(
                        shifted.get(row, col),
                        expect,
                        "{label}: mismatch at ({row},{col})"
                    );
                }
            }
        };

        check(board.shift_north(), (1, 0), "shift_north");
        check(board.shift_south(), (-1, 0), "shift_south");
        check(board.shift_east(), (0, 1), "shift_east");
        check(board.shift_west(), (0, -1), "shift_west");
    }

    /// Independent BFS oracle for `flood4`.
    fn check_flood4_against_oracle<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
        rows: R,
        cols: C,
        bits: &[usize],
        start_row: usize,
        start_col: usize,
    ) {
        let (n, m) = (rows.get(), cols.get());
        let total = n * m;
        let mut board: Board<S, R, C> = Board::new(rows, cols);
        for &i in bits {
            let i = i % total;
            board.set(i / m, i % m);
        }
        let start_row = start_row % n;
        let start_col = start_col % m;
        let start = start_row * m + start_col;

        let result = board.flood4(start);

        let mut visited: Board<S, R, C> = Board::new(rows, cols);
        let mut stack = vec![(start_row, start_col)];
        let ns: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
        while let Some((row, col)) = stack.pop() {
            if !visited.get(row, col) && board.get(row, col) {
                visited.set(row, col);
                for &(dr, dc) in &ns {
                    let nr = row as i64 + dr;
                    let nc = col as i64 + dc;
                    if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < m {
                        stack.push((nr as usize, nc as usize));
                    }
                }
            }
        }

        assert_eq!(result, visited, "flood4 mismatch vs BFS oracle");
    }

    /// Independent BFS oracle for `flood6`, seeded from every set bit of
    /// `seed_bits` -- mirrors `flood6`'s own multi-seed contract. The six
    /// neighbor deltas are the four cardinals plus the northeast/southwest
    /// diagonal (not northwest/southeast) -- see `shift_northeast`'s doc
    /// comment.
    fn check_flood6_against_oracle<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
        rows: R,
        cols: C,
        board_bits: &[usize],
        seed_bits: &[usize],
    ) {
        let (n, m) = (rows.get(), cols.get());
        let total = n * m;
        let mut board: Board<S, R, C> = Board::new(rows, cols);
        for &i in board_bits {
            let i = i % total;
            board.set(i / m, i % m);
        }
        let mut seed: Board<S, R, C> = Board::new(rows, cols);
        for &i in seed_bits {
            let i = i % total;
            seed.set(i / m, i % m);
        }

        let result = board.flood6(seed);

        let mut visited: Board<S, R, C> = Board::new(rows, cols);
        let mut stack: Vec<(usize, usize)> = seed.iter_set().map(|i| (i / m, i % m)).collect();
        let ns: [(i64, i64); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)];
        while let Some((row, col)) = stack.pop() {
            if !visited.get(row, col) && board.get(row, col) {
                visited.set(row, col);
                for &(dr, dc) in &ns {
                    let nr = row as i64 + dr;
                    let nc = col as i64 + dc;
                    if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < m {
                        stack.push((nr as usize, nc as usize));
                    }
                }
            }
        }

        assert_eq!(result, visited, "flood6 mismatch vs BFS oracle");
    }

    macro_rules! shift_flood_oracle_tests {
        ($mod_name:ident, $n:expr, $m:expr, $storage:ty, $max_index:expr) => {
            mod $mod_name {
                use super::*;

                proptest! {
                    #[test]
                    fn const_shifts(bits in proptest::collection::vec(0usize..$max_index, 0..200)) {
                        check_shifts_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &bits);
                    }

                    #[test]
                    fn dyn_shifts(bits in proptest::collection::vec(0usize..$max_index, 0..200)) {
                        check_shifts_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &bits);
                    }

                    #[test]
                    fn const_flood4(
                        bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        start_row in 0usize..$n,
                        start_col in 0usize..$m,
                    ) {
                        check_flood4_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &bits, start_row, start_col);
                    }

                    #[test]
                    fn dyn_flood4(
                        bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        start_row in 0usize..$n,
                        start_col in 0usize..$m,
                    ) {
                        check_flood4_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &bits, start_row, start_col);
                    }

                    #[test]
                    fn const_flood6(
                        board_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        seed_bits in proptest::collection::vec(0usize..$max_index, 0..10),
                    ) {
                        check_flood6_against_oracle::<$storage, Const<$n>, Const<$m>>(Const, Const, &board_bits, &seed_bits);
                    }

                    #[test]
                    fn dyn_flood6(
                        board_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        seed_bits in proptest::collection::vec(0usize..$max_index, 0..10),
                    ) {
                        check_flood6_against_oracle::<$storage, Dyn, Dyn>(Dyn($n), Dyn($m), &board_bits, &seed_bits);
                    }
                }
            }
        };
    }

    // Sub-word board.
    shift_flood_oracle_tests!(shift_flood_3x3, 3, 3, u64, 9);
    // Exact single-word boundary (64 bits, remainder == 0).
    shift_flood_oracle_tests!(shift_flood_8x8, 8, 8, u64, 64);
    // Multi-word sizes.
    shift_flood_oracle_tests!(shift_flood_9x9, 9, 9, [u64; 2], 81);
    shift_flood_oracle_tests!(shift_flood_11x11, 11, 11, [u64; 2], 121);
    shift_flood_oracle_tests!(shift_flood_13x13, 13, 13, [u64; 3], 169);
    shift_flood_oracle_tests!(shift_flood_19x19, 19, 19, [u64; 6], 361);

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Hand-verified regressions for wall masks, word-boundary shift carry,
    // and the hex diagonal -- mirroring `bigbitboard.rs`'s equivalents.

    #[test]
    fn wall_masks_agree_between_const_and_dyn() {
        for direction in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            let const_wall: Board<u64, Const<8>, Const<8>> = Board::new_const().wall(direction);
            let dyn_wall: Board<u64, Dyn, Dyn> = Board::new(Dyn(8), Dyn(8)).wall(direction);
            assert_eq!(
                const_wall.iter_set().collect::<Vec<_>>(),
                dyn_wall.iter_set().collect::<Vec<_>>(),
                "{direction:?} wall mismatch between Const and Dyn"
            );
        }
    }

    #[test]
    fn shift_carries_across_word_boundary() {
        // 9x9: bit 63 is the last bit of word 0, bit 64 the first bit of
        // word 1. A horizontal chain straddling that boundary must shift
        // east/west as a unit, not lose the part that crosses words.
        let mut board: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
        board.set(7, 0); // index 63
        board.set(7, 1); // index 64

        let east = board.shift_east();
        assert!(east.get(7, 1));
        assert!(east.get(7, 2));
        assert_eq!(east.count_ones(), 2);

        let west = board.shift_west();
        assert_eq!(west.count_ones(), 1);
        assert!(west.get(7, 0));
    }

    #[test]
    fn flood6_carries_the_hex_diagonal_across_a_word_boundary() {
        let mut board: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
        board.set(7, 0); // index 63, last bit of word 0
        board.set(8, 1); // index 73, in word 1
        let mut seed: Board<[u64; 2], Const<9>, Const<9>> = Board::new_const();
        seed.set(7, 0);

        let flood = board.flood6(seed);
        assert!(flood.get(8, 1));
        assert_eq!(flood.count_ones(), 2);
    }

    #[test]
    fn flood6_uses_northeast_southwest_diagonal_only() {
        let board: Board<u64, Const<3>, Const<3>> = !Board::new_const();
        let mut seed: Board<u64, Const<3>, Const<3>> = Board::new_const();
        seed.set(0, 0);
        let flood = board.flood6(seed);
        assert!(flood.get(1, 1));

        let mut isolated: Board<u64, Const<3>, Const<3>> = Board::new_const();
        isolated.set(0, 1);
        isolated.set(1, 0);
        let mut isolated_seed: Board<u64, Const<3>, Const<3>> = Board::new_const();
        isolated_seed.set(0, 1);
        let flood = isolated.flood6(isolated_seed);
        assert_eq!(flood, isolated_seed);
    }

    #[test]
    fn adjacency_mask_excludes_self_and_off_board() {
        let mut board: Board<u64, Const<3>, Const<3>> = Board::new_const();
        board.set(1, 1);
        let adjacent = board.adjacency_mask();
        assert_eq!(adjacent.count_ones(), 4);
        assert!(adjacent.get(0, 1));
        assert!(adjacent.get(2, 1));
        assert!(adjacent.get(1, 0));
        assert!(adjacent.get(1, 2));
        assert!(!adjacent.get(1, 1));
    }

    #[test]
    fn has_opposite_connection4_detects_a_spanning_group() {
        let mut spanning: Board<u64, Const<4>, Const<4>> = Board::new_const();
        for row in 0..4 {
            spanning.set(row, 1);
        }
        assert!(spanning.has_opposite_connection4(spanning.index_of(0, 1)));

        let mut isolated: Board<u64, Const<4>, Const<4>> = Board::new_const();
        isolated.set(1, 1);
        assert!(!isolated.has_opposite_connection4(isolated.index_of(1, 1)));
    }

    #[test]
    fn has_opposite_connection8_matches_connection4_for_cardinal_only_groups() {
        let mut spanning: Board<u64, Const<4>, Const<4>> = Board::new_const();
        for col in 0..4 {
            spanning.set(2, col);
        }
        let start = spanning.index_of(2, 0);
        assert!(spanning.has_opposite_connection8(start));
        assert_eq!(
            spanning.has_opposite_connection8(start),
            spanning.has_opposite_connection4(start)
        );
    }

    #[test]
    fn flood8_reaches_diagonal_only_neighbors() {
        // Unlike `flood4`, `flood8` reaches a purely diagonal neighbor with
        // no cardinal bridge between it and the seed -- see `flood8`'s doc
        // comment for how splitting the shifts across two statements
        // achieves this without a real diagonal shift primitive.
        let mut board: Board<u64, Const<4>, Const<4>> = Board::new_const();
        board.set(0, 0);
        board.set(1, 1); // diagonal-only neighbor of (0, 0); no cardinal bridge set

        let via4 = board.flood4(board.index_of(0, 0));
        let via8 = board.flood8(board.index_of(0, 0));
        assert!(!via4.get(1, 1));
        assert!(via8.get(1, 1));
        assert_eq!(via8.count_ones(), 2);
    }

    /// Independent BFS oracle for `flood8`, using the real 8-way (including
    /// all four diagonals) neighbor set.
    fn check_flood8_against_oracle<S: Storage + std::fmt::Debug, R: Dim, C: Dim>(
        rows: R,
        cols: C,
        bits: &[usize],
        start_row: usize,
        start_col: usize,
    ) {
        let (n, m) = (rows.get(), cols.get());
        let total = n * m;
        let mut board: Board<S, R, C> = Board::new(rows, cols);
        for &i in bits {
            let i = i % total;
            board.set(i / m, i % m);
        }
        let start_row = start_row % n;
        let start_col = start_col % m;
        let start = start_row * m + start_col;

        let result = board.flood8(start);

        let mut visited: Board<S, R, C> = Board::new(rows, cols);
        let mut stack = vec![(start_row, start_col)];
        let ns: [(i64, i64); 8] = [
            (1, 1),
            (1, 0),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ];
        while let Some((row, col)) = stack.pop() {
            if !visited.get(row, col) && board.get(row, col) {
                visited.set(row, col);
                for &(dr, dc) in &ns {
                    let nr = row as i64 + dr;
                    let nc = col as i64 + dc;
                    if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < m {
                        stack.push((nr as usize, nc as usize));
                    }
                }
            }
        }

        assert_eq!(result, visited, "flood8 mismatch vs BFS oracle");
    }

    proptest! {
        #[test]
        fn flood8_connectivity_8x8(
            bits in proptest::collection::vec(0usize..64, 0..200),
            start_row in 0usize..8,
            start_col in 0usize..8,
        ) {
            check_flood8_against_oracle::<u64, Const<8>, Const<8>>(Const, Const, &bits, start_row, start_col);
        }

        #[test]
        fn flood8_connectivity_9x9_dyn(
            bits in proptest::collection::vec(0usize..81, 0..200),
            start_row in 0usize..9,
            start_col in 0usize..9,
        ) {
            check_flood8_against_oracle::<[u64; 2], Dyn, Dyn>(Dyn(9), Dyn(9), &bits, start_row, start_col);
        }
    }
}
