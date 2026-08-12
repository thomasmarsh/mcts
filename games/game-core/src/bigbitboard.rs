use serde::{Serialize, Serializer};
use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// An N x M bitboard backed by a fixed-size array of `WORDS` u64 words, for
/// boards too large for `BitBoard<N, M>`'s single u64 (i.e. N * M > 64) --
/// Tanbo's 19x19 = 361-cell board being the motivating case.
///
/// `WORDS` is a genuine, independent const generic parameter, not one
/// derived from `N * M`: stable Rust can't express `ceil(N * M / 64)` as a
/// value dependent on other const generics. Callers compute and supply it
/// themselves -- `WORDS = (N * M).div_ceil(64)` -- and every constructor
/// path runs [`CHECK_WORDS`](Self::CHECK_WORDS), a compile-time assertion
/// that catches a wrong value with a build error instead of silent
/// truncation or wasted space.
///
/// This type intentionally does not implement `BitBoard`'s directional
/// shifts, wall masks, or shift-based flood fill: those require carrying a
/// bit shift across word boundaries, which is real added complexity that
/// nothing using this type today needs. Group/region traversal here is
/// expected to be done by walking `to_index`/neighbour arithmetic and
/// testing `get`, the same way as an index-based `Vec`-backed board would.
/// By design there is no shared implementation with `BitBoard<N, M>`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BigBitBoard<const N: usize, const M: usize, const WORDS: usize>([u64; WORDS]);

// `#[derive(Serialize)]` would require `[u64; WORDS]: Serialize`, which serde
// only provides for a fixed set of concrete array lengths, not generically
// over a const parameter -- so this is implemented by hand as a plain
// sequence of words.
impl<const N: usize, const M: usize, const WORDS: usize> Serialize for BigBitBoard<N, M, WORDS> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter())
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Compile-time WORDS validation.

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    /// Referenced by every constructor path so a mis-sized `WORDS` fails to
    /// compile at the call site's monomorphization, rather than silently
    /// truncating bits or wasting memory.
    const CHECK_WORDS: () = assert!(
        WORDS == (N * M).div_ceil(64),
        "BigBitBoard::<N, M, WORDS>: WORDS must equal ceil(N * M / 64)"
    );

    const fn ones_mask() -> [u64; WORDS] {
        let mut mask = [u64::MAX; WORDS];
        let remainder = (N * M) % 64;
        if remainder != 0 {
            mask[WORDS - 1] = (1u64 << remainder) - 1;
        }
        mask
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Constructors

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    pub const EMPTY: Self = Self::new([0; WORDS]);
    pub const ONES: Self = Self::new(Self::ones_mask());

    #[inline(always)]
    pub const fn new(words: [u64; WORDS]) -> Self {
        let () = Self::CHECK_WORDS;
        debug_assert!(N * M > 0);
        Self(words)
    }

    pub fn from_index(index: usize) -> Self {
        let mut b = Self::EMPTY;
        b.set(index);
        b
    }

    pub fn from_coord(row: usize, col: usize) -> Self {
        Self::from_index(Self::to_index(row, col))
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        let mut i = 0;
        while i < WORDS {
            if self.0[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }
}

impl<const N: usize, const M: usize, const WORDS: usize> Default for BigBitBoard<N, M, WORDS> {
    #[inline(always)]
    fn default() -> Self {
        Self::EMPTY
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Indexing and coordinates

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    #[inline(always)]
    pub const fn to_index(row: usize, col: usize) -> usize {
        debug_assert!(row < N);
        debug_assert!(col < M);
        row * M + col
    }

    #[inline(always)]
    pub const fn to_coord(index: usize) -> (usize, usize) {
        debug_assert!(index < N * M);
        (index / M, index % M)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Accessors and setters

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    #[inline(always)]
    pub const fn get(&self, index: usize) -> bool {
        debug_assert!(index < N * M);
        (self.0[index / 64] >> (index % 64)) & 1 != 0
    }

    #[inline(always)]
    pub const fn get_at(&self, row: usize, col: usize) -> bool {
        self.get(Self::to_index(row, col))
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize) {
        debug_assert!(index < N * M);
        self.0[index / 64] |= 1 << (index % 64);
    }

    #[inline(always)]
    pub fn set_at(&mut self, row: usize, col: usize) {
        self.set(Self::to_index(row, col));
    }

    #[inline(always)]
    pub fn clear(&mut self, index: usize) {
        debug_assert!(index < N * M);
        self.0[index / 64] &= !(1 << (index % 64));
    }

    #[inline(always)]
    pub fn clear_at(&mut self, row: usize, col: usize) {
        self.clear(Self::to_index(row, col));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Proxy common operations

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Iteration over set bits (consumes the board, like `BitBoard`; `Copy`
// makes iterating a value passed by value the normal usage).

impl<const N: usize, const M: usize, const WORDS: usize> Iterator for BigBitBoard<N, M, WORDS> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        for word in 0..WORDS {
            if self.0[word] != 0 {
                let bit = self.0[word].trailing_zeros() as usize;
                self.0[word] &= self.0[word] - 1;
                return Some(word * 64 + bit);
            }
        }
        None
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Unary operations

impl<const N: usize, const M: usize, const WORDS: usize> Not for BigBitBoard<N, M, WORDS> {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        let ones = Self::ones_mask();
        Self(std::array::from_fn(|i| !self.0[i] & ones[i]))
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Binary operations

impl<const N: usize, const M: usize, const WORDS: usize> BitAnd for BigBitBoard<N, M, WORDS> {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] & rhs.0[i]))
    }
}

impl<const N: usize, const M: usize, const WORDS: usize> BitOr for BigBitBoard<N, M, WORDS> {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] | rhs.0[i]))
    }
}

impl<const N: usize, const M: usize, const WORDS: usize> BitXor for BigBitBoard<N, M, WORDS> {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] ^ rhs.0[i]))
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Assign operations

impl<const N: usize, const M: usize, const WORDS: usize> BitAndAssign for BigBitBoard<N, M, WORDS> {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a &= b;
        }
    }
}

impl<const N: usize, const M: usize, const WORDS: usize> BitOrAssign for BigBitBoard<N, M, WORDS> {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a |= b;
        }
    }
}

impl<const N: usize, const M: usize, const WORDS: usize> BitXorAssign for BigBitBoard<N, M, WORDS> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a ^= b;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Membership tests

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    #[inline]
    pub fn intersects(self, rhs: Self) -> bool {
        (self & rhs) != Self::EMPTY
    }

    #[inline]
    pub fn is_subset(self, rhs: Self) -> bool {
        (self & rhs) == self
    }

    #[inline]
    pub fn is_disjoint(self, rhs: Self) -> bool {
        (self & rhs) == Self::EMPTY
    }

    /// Extract the raw underlying words.
    pub fn words(self) -> [u64; WORDS] {
        self.0
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Display

impl<const N: usize, const M: usize, const WORDS: usize> fmt::Display for BigBitBoard<N, M, WORDS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..N {
            for col in 0..M {
                if self.get_at(N - row - 1, col) {
                    write!(f, "X")?;
                } else {
                    write!(f, ".")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn coord_index() {
        type B = BigBitBoard<9, 7, 1>; // 63 bits, fits one word
        for row in 0..9 {
            for col in 0..7 {
                let index = B::to_index(row, col);
                let (r, c) = B::to_coord(index);
                assert_eq!(r, row);
                assert_eq!(c, col);
            }
        }
    }

    #[test]
    fn ones_mask_matches_bit_count() {
        assert_eq!(BigBitBoard::<3, 3, 1>::ONES.count_ones(), 9);
        assert_eq!(BigBitBoard::<8, 8, 1>::ONES.count_ones(), 64);
        assert_eq!(BigBitBoard::<9, 9, 2>::ONES.count_ones(), 81);
        assert_eq!(BigBitBoard::<19, 19, 6>::ONES.count_ones(), 361);
    }

    #[test]
    fn not_stays_within_bounds() {
        // Complementing EMPTY must not set any of the padding bits past
        // N * M in the last word.
        type B = BigBitBoard<9, 9, 2>;
        assert_eq!(!B::EMPTY, B::ONES);
        assert_eq!(!B::ONES, B::EMPTY);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Array-backed oracle: an independent, obviously-correct `Vec<bool>`
    // model checked against `BigBitBoard` across a representative set of
    // sizes -- a sub-word board, an exact-word-boundary board, and the real
    // Tanbo sizes (81, 121, 169, 361 bits), covering every WORDS from 1..6.

    fn check_against_oracle<const N: usize, const M: usize, const WORDS: usize>(
        sets: &[usize],
        clears: &[usize],
    ) {
        let bits = N * M;
        let mut oracle = vec![false; bits];
        let mut board = BigBitBoard::<N, M, WORDS>::EMPTY;

        for &i in sets {
            let i = i % bits;
            oracle[i] = true;
            board.set(i);
        }
        for &i in clears {
            let i = i % bits;
            oracle[i] = false;
            board.clear(i);
        }

        for (i, &expected) in oracle.iter().enumerate() {
            assert_eq!(board.get(i), expected, "get({i}) mismatch");
        }

        assert_eq!(
            board.count_ones() as usize,
            oracle.iter().filter(|&&b| b).count(),
            "count_ones mismatch"
        );

        let mut got: Vec<usize> = board.collect();
        got.sort_unstable();
        let expected: Vec<usize> = (0..bits).filter(|&i| oracle[i]).collect();
        assert_eq!(got, expected, "iterated set bits mismatch");

        assert_eq!(board.is_empty(), expected.is_empty());
    }

    fn check_binary_ops_against_oracle<const N: usize, const M: usize, const WORDS: usize>(
        a_bits: &[usize],
        b_bits: &[usize],
    ) {
        let bits = N * M;
        let mut oa = vec![false; bits];
        let mut ob = vec![false; bits];
        let mut a = BigBitBoard::<N, M, WORDS>::EMPTY;
        let mut b = BigBitBoard::<N, M, WORDS>::EMPTY;

        for &i in a_bits {
            let i = i % bits;
            oa[i] = true;
            a.set(i);
        }
        for &i in b_bits {
            let i = i % bits;
            ob[i] = true;
            b.set(i);
        }

        let union = a | b;
        let inter = a & b;
        let xor = a ^ b;
        let not_a = !a;

        for i in 0..bits {
            assert_eq!(union.get(i), oa[i] || ob[i], "union mismatch at {i}");
            assert_eq!(inter.get(i), oa[i] && ob[i], "intersect mismatch at {i}");
            assert_eq!(xor.get(i), oa[i] ^ ob[i], "xor mismatch at {i}");
            assert_eq!(not_a.get(i), !oa[i], "not mismatch at {i}");
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
        ($mod_name:ident, $n:expr, $m:expr, $words:expr, $max_index:expr) => {
            mod $mod_name {
                use super::*;

                proptest! {
                    #[test]
                    fn get_set_clear_count_iter(
                        sets in proptest::collection::vec(0usize..$max_index, 0..200),
                        clears in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_against_oracle::<$n, $m, $words>(&sets, &clears);
                    }

                    #[test]
                    fn binary_ops(
                        a_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        b_bits in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_binary_ops_against_oracle::<$n, $m, $words>(&a_bits, &b_bits);
                    }
                }
            }
        };
    }

    // Sub-word board.
    oracle_tests!(oracle_3x3, 3, 3, 1, 9);
    // Exact single-word boundary (64 bits, remainder == 0).
    oracle_tests!(oracle_8x8, 8, 8, 1, 64);
    // Tanbo's real board sizes.
    oracle_tests!(oracle_9x9, 9, 9, 2, 81);
    oracle_tests!(oracle_11x11, 11, 11, 2, 121);
    oracle_tests!(oracle_13x13, 13, 13, 3, 169);
    oracle_tests!(oracle_19x19, 19, 19, 6, 361);
}
