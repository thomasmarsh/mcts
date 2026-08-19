use std::ops::{BitAnd, BitOr, BitXor, Not};

use serde::{Deserialize, Serialize};

use crate::dim::Dim;
use crate::storage::Storage;

/// A generic `rows x cols` bitboard: `S` picks the storage backend (`u64`
/// for a single-word board, `[u64; WORDS]` for a multi-word one), `R`/`C`
/// pick whether the row/column counts are compile-time (`Const<N>`) or
/// runtime (`Dyn`) values. Indexing is row-major (`row * cols + col`),
/// matching `BitBoard`/`BigBitBoard`'s existing wire format.
///
/// Currently supports `get`/`set`/`clear`/`count_ones`/iteration over set
/// bits, the `&`/`|`/`^`/`!` binary ops, and serde; shifts, flood fill, and
/// walls are not yet implemented.
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

    #[inline(always)]
    pub fn clear(&mut self, row: usize, col: usize) {
        let index = self.index_of(row, col);
        *self.bits.word_mut(index / 64) &= !(1u64 << (index % 64));
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

    /// True if no bits are set (independent of `is_empty`, which reports
    /// whether the board has zero cells at all).
    #[inline]
    fn bits_empty(&self) -> bool {
        (0..S::CAPACITY_WORDS).all(|w| self.bits.word(w) == 0)
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

impl<S: Storage, const N: usize, const M: usize>
    Board<S, crate::dim::Const<N>, crate::dim::Const<M>>
{
    #[inline(always)]
    pub fn new_const() -> Self {
        Self::new(crate::dim::Const, crate::dim::Const)
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
}
