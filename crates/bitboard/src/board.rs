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

    /// An all-zero board with the same dims as `self` -- for a caller
    /// outside this module that needs a fresh same-shape board (e.g. a
    /// seed/accumulator mask) but only has an existing `Board` value to copy
    /// the dims from, not the bare `R`/`C` dim values `new` takes.
    #[inline(always)]
    pub fn empty_like(&self) -> Self {
        Self::new(self.rows, self.cols)
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

mod movement;
mod symmetry;
mod traits;
mod traversal;

#[cfg(test)]
mod tests;
