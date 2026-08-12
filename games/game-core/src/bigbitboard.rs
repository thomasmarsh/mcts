use crate::bitboard::Direction;
use serde::de::{self, Deserialize, Deserializer, SeqAccess, Visitor};
use serde::{Serialize, Serializer};
use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

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
/// This type implements `BitBoard`'s directional shifts, wall masks, and
/// shift-based `flood4`/4-way connectivity (used by go-variant capture
/// logic -- see [`check_go_move`]), carrying a bit shift across word
/// boundaries the same way a bignum shift does. It intentionally does *not*
/// implement the diagonal shifts or `flood8`/8-way connectivity: those
/// require masking two walls at once per direction, real added complexity
/// nothing using this type today needs. If a caller needs 8-way traversal
/// (or Tanbo-style group tracing that doesn't fit the shift model at all),
/// walking `to_index`/neighbour arithmetic and testing `get` remains the
/// expected approach. By design there is no shared implementation with
/// `BitBoard<N, M>`; the two types happen to expose parallel APIs.
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

// The `Deserialize` counterpart to the hand-written `Serialize` above, for
// the same reason: derive can't express `[u64; WORDS]: Deserialize`
// generically over a const parameter. A `Visitor` reading exactly `WORDS`
// sequence elements mirrors `collect_seq`'s output on the way back in.
impl<'de, const N: usize, const M: usize, const WORDS: usize> Deserialize<'de>
    for BigBitBoard<N, M, WORDS>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WordsVisitor<const WORDS: usize>;

        impl<'de, const WORDS: usize> Visitor<'de> for WordsVisitor<WORDS> {
            type Value = [u64; WORDS];

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a sequence of {WORDS} u64 words")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut words = [0u64; WORDS];
                for (i, w) in words.iter_mut().enumerate() {
                    *w = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                }
                Ok(words)
            }
        }

        deserializer
            .deserialize_seq(WordsVisitor::<WORDS>)
            .map(Self)
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

    /// Mask off any padding bits past `N * M` in the last word. Shifts can
    /// leave garbage there (the same tradeoff `BitBoard::sanitize` makes);
    /// operations that depend on those bits being clear (`flood4`) assert
    /// their input is already sanitized instead of paying for it on every
    /// call.
    #[inline]
    pub fn sanitize(self) -> Self {
        let ones = Self::ones_mask();
        Self(std::array::from_fn(|i| self.0[i] & ones[i]))
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

// Shifts (carry a bit shift across word boundaries, bignum-style). Only
// fixed offsets less than 64 are supported -- shift_north/south use `M` and
// shift_east/west use `1`, both always well under 64 for any board size
// this type is used for.

impl<const N: usize, const M: usize, const WORDS: usize> Shl<usize> for BigBitBoard<N, M, WORDS> {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        debug_assert!(rhs < 64);
        if rhs == 0 {
            return self;
        }
        let mut words = [0u64; WORDS];
        for i in (0..WORDS).rev() {
            words[i] = self.0[i] << rhs;
            if i > 0 {
                words[i] |= self.0[i - 1] >> (64 - rhs);
            }
        }
        Self(words)
    }
}

impl<const N: usize, const M: usize, const WORDS: usize> Shr<usize> for BigBitBoard<N, M, WORDS> {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        debug_assert!(rhs < 64);
        if rhs == 0 {
            return self;
        }
        let mut words = [0u64; WORDS];
        for (i, w) in words.iter_mut().enumerate() {
            *w = self.0[i] >> rhs;
            if i + 1 < WORDS {
                *w |= self.0[i + 1] << (64 - rhs);
            }
        }
        Self(words)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Wall masks

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    const fn wall_words(direction: Direction) -> [u64; WORDS] {
        let mut words = [0u64; WORDS];
        let limit = match direction {
            Direction::North | Direction::South => M,
            Direction::East | Direction::West => N,
        };
        let mut i = 0;
        while i < limit {
            let k = match direction {
                Direction::North => (N - 1) * M + i,
                Direction::East => (i + 1) * M - 1,
                Direction::South => i,
                Direction::West => i * M,
            };
            words[k / 64] |= 1 << (k % 64);
            i += 1;
        }
        words
    }

    // We define this because `wall` may be called in non-const contexts. We
    // would still like to remain branch free at the very least.
    const WALL_LUT: [Self; 4] = [
        Self::new(Self::wall_words(Direction::North)),
        Self::new(Self::wall_words(Direction::East)),
        Self::new(Self::wall_words(Direction::South)),
        Self::new(Self::wall_words(Direction::West)),
    ];

    pub const fn wall(direction: Direction) -> Self {
        Self::WALL_LUT[direction as usize]
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Board displacement

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    #[inline(always)]
    pub fn shift_north(self) -> Self {
        (self & !Self::wall(Direction::North)) << M
    }

    #[inline(always)]
    pub fn shift_east(self) -> Self {
        (self & !Self::wall(Direction::East)) << 1
    }

    #[inline(always)]
    pub fn shift_south(self) -> Self {
        self >> M
    }

    #[inline(always)]
    pub fn shift_west(self) -> Self {
        (self & !Self::wall(Direction::West)) >> 1
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

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    #[inline]
    pub fn adjacency_mask(self) -> Self {
        (self.shift_north() | self.shift_east() | self.shift_south() | self.shift_west()) & !self
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Flood fill

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    /// Performs a four-way floodfill traversing set bits. It might seem more
    /// natural to fill unset bits, but that requires one additional
    /// operation in this function, so that decision is up to the client.
    pub fn flood4(self, start: usize) -> Self {
        debug_assert!(start < N * M);
        debug_assert!(self == self.sanitize());
        let mut flood = Self::from_index(start) & self;

        if flood.is_empty() {
            return flood;
        }

        loop {
            let temp = flood;
            flood |=
                flood.shift_north() | flood.shift_east() | flood.shift_south() | flood.shift_west();
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

impl<const N: usize, const M: usize, const WORDS: usize> BigBitBoard<N, M, WORDS> {
    #[inline]
    pub fn has_opposite_connection4(self, start: usize) -> bool {
        let n = Self::wall(Direction::North);
        let e = Self::wall(Direction::East);
        let s = Self::wall(Direction::South);
        let w = Self::wall(Direction::West);

        let mut flood = Self::from_index(start) & self;

        if flood.is_empty() {
            return false;
        }

        loop {
            let temp = flood;
            flood |=
                flood.shift_north() | flood.shift_east() | flood.shift_south() | flood.shift_west();
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

//////////////////////////////////////////////////////////////////////////////////////////////////

// Go capture logic

/// Checks whether a move is valid for a game with go capture rules. Mirrors
/// `bitboard::check_go_move` for boards too large for a single `u64`.
pub fn check_go_move<const N: usize, const WORDS: usize>(
    player: BigBitBoard<N, N, WORDS>,
    opponent: BigBitBoard<N, N, WORDS>,
    index: usize,
) -> (bool, BigBitBoard<N, N, WORDS>) {
    debug_assert!(!player.intersects(opponent));
    let occupied = player | opponent;
    debug_assert!(!occupied.get(index));
    let player = player | BigBitBoard::from_index(index);
    let occupied = player | opponent;
    let group = player.flood4(index);
    let adjacent = group.adjacency_mask();
    let occupied_adjacent = occupied & adjacent;
    let empty_adjacent = !occupied & adjacent;

    // If we have adjacent empty positions we still have liberties.
    let safe = !empty_adjacent.is_empty();

    let mut seen = BigBitBoard::EMPTY;
    let mut will_capture = BigBitBoard::EMPTY;
    for point in occupied_adjacent {
        // By definition, adjacent non-empty points must be the opponent.
        debug_assert!(occupied.get(point));
        debug_assert!(opponent.get(point));
        if !seen.get(point) {
            let group = opponent.flood4(point);
            let adjacent = group.adjacency_mask();
            let empty_adjacent = !occupied & adjacent;
            if empty_adjacent.is_empty() {
                will_capture |= group;
            }
            seen |= group;
        }
    }

    (safe || !will_capture.is_empty(), will_capture)
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

    /// Independent row/col-arithmetic oracle for the four cardinal shifts,
    /// checked against `BigBitBoard`'s bignum-style carry shift.
    fn check_shifts_against_oracle<const N: usize, const M: usize, const WORDS: usize>(
        bits: &[usize],
    ) {
        let n = N * M;
        let mut board = BigBitBoard::<N, M, WORDS>::EMPTY;
        let mut set: Vec<(usize, usize)> = Vec::new();
        for &i in bits {
            let i = i % n;
            board.set(i);
            set.push(BigBitBoard::<N, M, WORDS>::to_coord(i));
        }

        let check = |shifted: BigBitBoard<N, M, WORDS>, delta: (i64, i64), label: &str| {
            let expected: Vec<(usize, usize)> = set
                .iter()
                .filter_map(|&(r, c)| {
                    let nr = r as i64 + delta.0;
                    let nc = c as i64 + delta.1;
                    if nr >= 0 && nc >= 0 && (nr as usize) < N && (nc as usize) < M {
                        Some((nr as usize, nc as usize))
                    } else {
                        None
                    }
                })
                .collect();
            for row in 0..N {
                for col in 0..M {
                    let expect = expected.contains(&(row, col));
                    assert_eq!(
                        shifted.get_at(row, col),
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

    /// Independent BFS oracle for `flood4`, mirroring `bitboard.rs`'s
    /// `check_connectivity` helper.
    fn check_flood4_against_oracle<const N: usize, const M: usize, const WORDS: usize>(
        bits: &[usize],
        start_row: usize,
        start_col: usize,
    ) {
        let n = N * M;
        let mut board = BigBitBoard::<N, M, WORDS>::EMPTY;
        for &i in bits {
            board.set(i % n);
        }
        let start_row = start_row % N;
        let start_col = start_col % M;
        let start = BigBitBoard::<N, M, WORDS>::to_index(start_row, start_col);

        let result = board.flood4(start);

        let mut visited = BigBitBoard::<N, M, WORDS>::EMPTY;
        let mut stack = vec![(start_row, start_col)];
        let ns: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
        while let Some((row, col)) = stack.pop() {
            if !visited.get_at(row, col) && board.get_at(row, col) {
                visited.set_at(row, col);
                for &(dr, dc) in &ns {
                    let nr = row as i64 + dr;
                    let nc = col as i64 + dc;
                    if nr >= 0 && nc >= 0 && (nr as usize) < N && (nc as usize) < M {
                        stack.push((nr as usize, nc as usize));
                    }
                }
            }
        }

        assert_eq!(result, visited, "flood4 mismatch vs BFS oracle");
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
                    fn shifts(
                        bits in proptest::collection::vec(0usize..$max_index, 0..200),
                    ) {
                        check_shifts_against_oracle::<$n, $m, $words>(&bits);
                    }

                    #[test]
                    fn flood4_connectivity(
                        bits in proptest::collection::vec(0usize..$max_index, 0..200),
                        start_row in 0usize..$n,
                        start_col in 0usize..$m,
                    ) {
                        check_flood4_against_oracle::<$n, $m, $words>(&bits, start_row, start_col);
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

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Cross-check against `BitBoard` (single-word, independently implemented
    // wall/shift logic) at a size both types support -- since `BigBitBoard`'s
    // wall/shift code is a from-scratch generalization rather than shared
    // code, this catches a divergence the same-crate oracle tests above
    // can't (they only check `BigBitBoard` against itself).

    #[test]
    fn serde_round_trips_across_word_boundary() {
        type B = BigBitBoard<9, 9, 2>;
        let mut board = B::EMPTY;
        board.set(0);
        board.set(63);
        board.set(64);
        board.set(80);

        let json = serde_json::to_string(&board).unwrap();
        let words: Vec<u64> = serde_json::from_str(&json).unwrap();
        assert_eq!(words, board.words().to_vec());

        let round_tripped: B = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, board);
    }

    #[test]
    fn wall_masks_match_bitboard_8x8() {
        use crate::bitboard::BitBoard;
        for direction in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            let big = BigBitBoard::<8, 8, 1>::wall(direction);
            let small = BitBoard::<8, 8>::wall(direction);
            assert_eq!(
                big.words()[0],
                small.get_raw(),
                "{direction:?} wall mismatch"
            );
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Hand-verified regressions for the word-boundary carry logic itself --
    // the oracle proptests above exercise it incidentally via random boards,
    // but these pin down the exact bit-63/bit-64 crossing that motivated
    // adding shifts to this type in the first place.

    #[test]
    fn shift_carries_across_word_boundary() {
        // 9x9: bit 63 is the last bit of word 0, bit 64 the first bit of
        // word 1. A horizontal chain straddling that boundary must shift
        // east/west as a unit, not lose the part that crosses words.
        type B = BigBitBoard<9, 9, 2>;
        // Row 7 spans indices 63..=71 (7*9=63); put stones at cols 0 and 1
        // (indices 63, 64) straddling the word boundary.
        let mut board = B::EMPTY;
        board.set(63);
        board.set(64);

        let east = board.shift_east();
        assert!(east.get_at(7, 1)); // was (7,0) -> now (7,1) = index 64
        assert!(east.get_at(7, 2)); // was (7,1) -> now (7,2) = index 65
        assert_eq!(east.count_ones(), 2);

        let west = board.shift_west();
        // (7,0) is at the west wall so it's dropped; (7,1) -> (7,0) = index 63.
        assert_eq!(west.count_ones(), 1);
        assert!(west.get_at(7, 0));
    }

    #[test]
    fn check_go_move_matches_bitboard_2x2() {
        // Mirrors `bitboard::tests::test_capture` exactly (single word, so
        // this only proves the two implementations agree, not that carrying
        // across words works -- see the boundary test below for that).
        type B = BigBitBoard<2, 2, 1>;
        let white = B::new([0b1001]);
        let black = B::EMPTY;
        let (safe, will_capture) = check_go_move::<2, 1>(black, white, 2);
        assert!(!safe);
        assert_eq!(will_capture, B::EMPTY);
    }

    #[test]
    fn check_go_move_captures_across_word_boundary() {
        // 9x9 board: opponent's lone stone at index 64 (row 7, col 1) --
        // the first bit of word 1 -- surrounded by the player's stones at
        // its north/south/east neighbours (indices 73, 55, 65), all in word
        // 1. Playing the west neighbour, index 63 (row 7, col 0), the last
        // bit of word 0, completes the capture: this only comes out right
        // if flood4/adjacency_mask correctly carry the group and its
        // liberties across the word-0/word-1 boundary.
        type B = BigBitBoard<9, 9, 2>;
        let mut player = B::EMPTY;
        player.set(73); // (8,1) north of (7,1)
        player.set(55); // (6,1) south of (7,1)
        player.set(65); // (7,2) east of (7,1)
        let mut opponent = B::EMPTY;
        opponent.set(64); // (7,1)

        let (safe, will_capture) = check_go_move::<9, 2>(player, opponent, 63);
        assert!(safe);
        assert_eq!(will_capture, B::from_index(64));
    }
}
