use crate::dim::Dim;
use crate::storage::Storage;

/// A generic `rows x cols` bitboard: `S` picks the storage backend (`u64`
/// for a single-word board, `[u64; WORDS]` for a multi-word one), `R`/`C`
/// pick whether the row/column counts are compile-time (`Const<N>`) or
/// runtime (`Dyn`) values. Indexing is row-major (`row * cols + col`),
/// matching `BitBoard`/`BigBitBoard`'s existing wire format.
///
/// Currently supports only `get`/`set`/`count_ones`/iteration over set
/// bits; shifts, flood fill, walls, and binary ops are not yet implemented.
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

    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> bool {
        let index = self.index_of(row, col);
        (self.bits.word(index / 64) >> (index % 64)) & 1 != 0
    }

    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize) {
        let index = self.index_of(row, col);
        *self.bits.word_mut(index / 64) |= 1u64 << (index % 64);
    }

    pub fn count_ones(&self) -> u32 {
        (0..S::CAPACITY_WORDS)
            .map(|w| self.bits.word(w).count_ones())
            .sum()
    }

    /// Iterates the row-major indices (`row * cols + col`) of set bits, in
    /// ascending order.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        (0..S::CAPACITY_WORDS).flat_map(move |w| {
            let word = self.bits.word(w);
            (0..64)
                .filter(move |b| (word >> b) & 1 != 0)
                .map(move |b| w * 64 + b)
        })
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
