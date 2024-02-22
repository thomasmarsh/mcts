use log::debug;
use serde::Serialize;
use std::fmt;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index, Not, Shl, ShlAssign,
    Shr, ShrAssign,
};

#[derive(Clone, Copy, Serialize, Debug, PartialEq)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

/// Defines an N x M bitboard with u64 as underlying storage. |N x M| must be
/// 64 bits or less. By convention, N refers to the number of rows, and M the
/// number of columns. The origin of the bitboard is at the bottom left and
/// indexing moves left to right, bottom to top. For accessing a coordinate,
/// coordinates are indexed by (col, row).
///
/// Note that more care must be taken with sizes that utilize fewer than 64
/// bits as some operations may leave garbage outside the bounds. For example,
/// with an 8x8 bitboards it is often useful to take !0 to be all bits. With a
/// smmaller bitboard, you will likely need to mask off the areas outside the
/// play area. For such concerns, the `ones`, `unused`, and `sanitize` functions
/// can be used.
#[derive(Clone, Copy, Serialize, PartialEq, Hash)]
pub struct BitBoard<const N: usize = 8, const M: usize = 8>(u64);

//////////////////////////////////////////////////////////////////////////////////////////////////

// Constructors

impl<const N: usize, const M: usize> BitBoard<N, M> {
    #[inline(always)]
    pub const fn new(value: u64) -> Self {
        debug_assert!((N * M) > 0);
        debug_assert!((N * M) <= 64);
        debug_assert!(value < (1 << (N * M)));
        Self(value)
    }

    #[inline(always)]
    pub const fn from_index(index: usize) -> Self {
        debug_assert!((N * M) > 0);
        debug_assert!((N * M) <= 64);
        debug_assert!(index < N * M);
        Self(1 << index)
    }

    #[inline(always)]
    pub fn from_coord(row: usize, col: usize) -> Self {
        debug_assert!(row < N);
        debug_assert!(col < M);
        Self::from_index(Self::to_index(row, col))
    }

    #[inline(always)]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn ones() -> Self {
        Self((1 << (N * M)) - 1)
    }

    #[inline(always)]
    pub const fn unused() -> Self {
        Self(!Self::ones().0)
    }

    #[inline(always)]
    pub const fn sanitize(self) -> Self {
        Self(self.0 & Self::ones().0)
    }
}

impl<const N: usize, const M: usize> Default for BitBoard<N, M> {
    #[inline(always)]
    fn default() -> Self {
        Self::empty()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Indexing and coordinates

impl<const N: usize, const M: usize> BitBoard<N, M> {
    /// Converts row and column coordinates into an index.
    #[inline(always)]
    pub const fn to_index(row: usize, col: usize) -> usize {
        debug_assert!(row < N);
        debug_assert!(col < M);
        row * M + col
    }

    /// Converts an index into a row and column.
    #[inline(always)]
    pub const fn to_coord(index: usize) -> (usize, usize) {
        debug_assert!(index < N * M);
        (index / M, index % M)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Accessors

impl<const N: usize, const M: usize> BitBoard<N, M> {
    /// Check if the bit at the specified linear index is set.
    #[inline(always)]
    pub fn get(self, index: usize) -> bool {
        debug_assert!(index < N * M);
        self & Self::from_index(index) != Self::empty()
    }

    /// Check if the bit at the specified 2D coordinate is set.
    #[inline(always)]
    fn get_at(&self, row: usize, col: usize) -> bool {
        self.get(row * M + col)
    }
}

impl<const N: usize, const M: usize> Index<usize> for BitBoard<N, M> {
    type Output = bool;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < N * M);
        if self.get(index) {
            &true
        } else {
            &false
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Setters

impl<const N: usize, const M: usize> BitBoard<N, M> {
    /// Check if the bit at the specified linear index is set.
    #[inline(always)]
    pub fn set(&mut self, index: usize) {
        debug_assert!(index < N * M);
        *self |= Self::from_index(index);
    }

    /// Check if the bit at the specified 2D coordinate is set.
    #[inline(always)]
    pub fn set_at(&mut self, row: usize, col: usize) {
        debug_assert!(row < N);
        debug_assert!(col < M);
        self.set(row * M + col)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Proxy common operations

impl<const N: usize, const M: usize> BitBoard<N, M> {
    #[inline(always)]
    pub fn count_ones(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub fn leading_ones(self) -> u32 {
        self.0.leading_ones()
    }

    #[inline(always)]
    pub fn trailing_ones(self) -> u32 {
        self.0.trailing_ones()
    }

    #[inline(always)]
    pub fn leading_zeros(self) -> u32 {
        self.0.leading_zeros()
    }

    #[inline(always)]
    pub fn trailing_zeros(self) -> u32 {
        self.0.trailing_zeros()
    }

    #[inline(always)]
    pub fn reverse_bits(self) -> Self {
        Self(self.0.reverse_bits())
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Unary operations

impl<const N: usize, const M: usize> Not for BitBoard<N, M> {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(!self.0 & Self::ones().0)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Binary operations

impl<const N: usize, const M: usize> BitAnd for BitBoard<N, M> {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl<const N: usize, const M: usize> BitOr for BitBoard<N, M> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl<const N: usize, const M: usize> BitXor for BitBoard<N, M> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl<const N: usize, const M: usize> Shl<usize> for BitBoard<N, M> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: usize) -> Self::Output {
        Self(self.0 << rhs)
    }
}

impl<const N: usize, const M: usize> Shr<usize> for BitBoard<N, M> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self::Output {
        Self(self.0 >> rhs)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Assign operations

impl<const N: usize, const M: usize> BitAndAssign for BitBoard<N, M> {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0
    }
}

impl<const N: usize, const M: usize> BitOrAssign for BitBoard<N, M> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0
    }
}

impl<const N: usize, const M: usize> BitXorAssign for BitBoard<N, M> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0
    }
}

impl<const N: usize, const M: usize> ShlAssign<usize> for BitBoard<N, M> {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: usize) {
        self.0 <<= rhs;
    }
}

impl<const N: usize, const M: usize> ShrAssign<usize> for BitBoard<N, M> {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: usize) {
        self.0 >>= rhs;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Wall masks

impl<const N: usize, const M: usize> BitBoard<N, M> {
    const fn wall_mask(direction: Direction, i: usize, mask: u64) -> u64 {
        let (limit, k) = match direction {
            Direction::North => (M, (N - 1) * M + i),
            Direction::East => (N, (i + 1) * M - 1),
            Direction::South => (M, i),
            Direction::West => (N, i * M),
        };
        if i >= limit {
            mask
        } else {
            Self::wall_mask(direction, i + 1, mask | (1 << k))
        }
    }

    // We define this because `wall` may be called in non-const contexts. We
    // would still like to remain branch free at the very least.
    const WALL_LUT: [Self; 4] = [
        Self(Self::wall_mask(Direction::North, 0, 0)),
        Self(Self::wall_mask(Direction::East, 0, 0)),
        Self(Self::wall_mask(Direction::South, 0, 0)),
        Self(Self::wall_mask(Direction::West, 0, 0)),
    ];

    pub const fn wall(direction: Direction) -> Self {
        Self::WALL_LUT[direction as usize]
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Board displacement

impl<const N: usize, const M: usize> BitBoard<N, M> {
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
        (self & !Self::wall(Direction::South)) >> M
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

/////////////////////////////////////////////////////////////////////////////////////////////////

// Adjacency

impl<const N: usize, const M: usize> BitBoard<N, M> {
    #[inline]
    pub fn adjacency_mask(self) -> Self {
        (self.shift_north() | self.shift_east() | self.shift_south() | self.shift_west()) ^ self
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Flood fill

impl<const N: usize, const M: usize> BitBoard<N, M> {
    /// Performs a four-way floodfill traversing set bits. It might seem more
    /// natural to fill unset bits, but that requires one additional operation
    /// in this function, so that decision is up to the client.
    pub fn flood4(self, start: usize) -> Self {
        debug_assert!(start < N * M);
        debug_assert!(self == self.sanitize());
        let mut flood = Self::from_index(start) & self;

        if flood.is_empty() {
            return flood;
        }

        while !flood.is_empty() {
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

    /// Performs a eight-way floodfill traversing set bits. It might seem more
    /// natural to fill unset bits, but that requires one additional operation
    /// in this function, so that decision is up to the client.
    pub fn flood8(self, start: usize) -> Self {
        debug_assert!(start < N * M);
        debug_assert!(self == self.sanitize());
        let mut flood = Self::from_index(start) & self;

        if flood.is_empty() {
            return flood;
        }

        while !flood.is_empty() {
            let temp = flood;
            flood |= flood.shift_north() | flood.shift_south();
            flood |= flood.shift_east() | flood.shift_west();
            flood &= self;
            if flood == temp {
                break;
            }
        }
        flood
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// For the `BitBoard`, iterate over every positition set.

impl Iterator for BitBoard {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            let result = self.trailing_zeros() as usize;
            *self ^= Self::from_index(result);
            Some(result)
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Display

impl<const N: usize, const M: usize> fmt::Display for BitBoard<N, M> {
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

impl<const N: usize, const M: usize> fmt::Debug for BitBoard<N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in (0..8).rev() {
            for col in 0..8 {
                let index = row * 8 + col;
                let bit = self.0 & (1 << index) != 0;
                let c = if index < N * M {
                    if bit {
                        'X'
                    } else {
                        '.'
                    }
                } else if bit {
                    '%'
                } else {
                    '#'
                };
                write!(f, " {}", c)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shift_properties_1x1() {
        type B = BitBoard<1, 1>;
        let init = B::new(1);

        assert_eq!(init.shift_north(), B::empty());
        assert_eq!(init.shift_east(), B::empty());
        assert_eq!(init.shift_south(), B::empty());
        assert_eq!(init.shift_west(), B::empty());
    }

    #[test]
    fn test_shift_properties_1x2() {
        type B = BitBoard<1, 2>;
        let init = B::new(0b11);
        let n = B::new(0b00);
        let e = B::new(0b10);
        let s = B::new(0b00);
        let w = B::new(0b01);

        assert_eq!(init.shift_north(), n);
        assert_eq!(init.shift_east(), e);
        assert_eq!(init.shift_south(), s);
        assert_eq!(init.shift_west(), w);
    }

    #[test]
    fn test_shifts_off_board() {
        use Direction::*;
        for direction in [North, East, South, West] {
            let mut b = BitBoard::<4, 4>::wall(direction);
            b = b.shift(direction);
            assert_eq!(b, BitBoard::empty());
        }
    }

    #[test]
    fn test_shifts_across_board() {
        use Direction::*;
        for (direction, opposite) in [(North, South), (East, West), (South, North), (West, East)] {
            let mut b = BitBoard::<4, 4>::wall(opposite);
            b = b.shift(direction);
            b = b.shift(direction);
            b = b.shift(direction);
            assert_eq!(b, BitBoard::wall(direction));
        }
    }

    #[test]
    fn test_flood4() {
        type B = BitBoard<3, 3>;
        let init = B::new(0b000_010_001);
        let flood = (!init).flood4(B::to_index(2, 1));
        let expected = B::ones() ^ init;
        assert_eq!(flood, expected);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    use proptest::prelude::*;

    #[derive(Debug)]
    struct RuntimeBitBoard {
        n: usize,
        m: usize,
        row: usize,
        col: usize,
        value: u64,
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    impl Arbitrary for RuntimeBitBoard {
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            ((1usize..8), (1usize..8))
                .prop_flat_map(|(n, m)| {
                    (0..n, 0..m, 0u64..(1 << (n * m))).prop_map(move |(row, col, value)| {
                        RuntimeBitBoard {
                            n,
                            m,
                            row,
                            col,
                            value,
                        }
                    })
                })
                .boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    // There is probably a better way to do this.
    macro_rules! match_bitboard {
        ($b:expr, $func:expr) => {
            type _B<const N: usize, const M: usize> = BitBoard<N, M>;
            match ($b.n, $b.m) {
                (1, 1) => $func(_B::<1, 1>::new($b.value), $b.row, $b.col),
                (1, 2) => $func(_B::<1, 2>::new($b.value), $b.row, $b.col),
                (1, 3) => $func(_B::<1, 3>::new($b.value), $b.row, $b.col),
                (1, 4) => $func(_B::<1, 4>::new($b.value), $b.row, $b.col),
                (1, 5) => $func(_B::<1, 5>::new($b.value), $b.row, $b.col),
                (1, 6) => $func(_B::<1, 6>::new($b.value), $b.row, $b.col),
                (1, 7) => $func(_B::<1, 7>::new($b.value), $b.row, $b.col),
                (1, 8) => $func(_B::<1, 8>::new($b.value), $b.row, $b.col),
                (2, 1) => $func(_B::<2, 1>::new($b.value), $b.row, $b.col),
                (2, 2) => $func(_B::<2, 2>::new($b.value), $b.row, $b.col),
                (2, 3) => $func(_B::<2, 3>::new($b.value), $b.row, $b.col),
                (2, 4) => $func(_B::<2, 4>::new($b.value), $b.row, $b.col),
                (2, 5) => $func(_B::<2, 5>::new($b.value), $b.row, $b.col),
                (2, 6) => $func(_B::<2, 6>::new($b.value), $b.row, $b.col),
                (2, 7) => $func(_B::<2, 7>::new($b.value), $b.row, $b.col),
                (2, 8) => $func(_B::<2, 8>::new($b.value), $b.row, $b.col),
                (3, 1) => $func(_B::<3, 1>::new($b.value), $b.row, $b.col),
                (3, 2) => $func(_B::<3, 2>::new($b.value), $b.row, $b.col),
                (3, 3) => $func(_B::<3, 3>::new($b.value), $b.row, $b.col),
                (3, 4) => $func(_B::<3, 4>::new($b.value), $b.row, $b.col),
                (3, 5) => $func(_B::<3, 5>::new($b.value), $b.row, $b.col),
                (3, 6) => $func(_B::<3, 6>::new($b.value), $b.row, $b.col),
                (3, 7) => $func(_B::<3, 7>::new($b.value), $b.row, $b.col),
                (3, 8) => $func(_B::<3, 8>::new($b.value), $b.row, $b.col),
                (4, 1) => $func(_B::<4, 1>::new($b.value), $b.row, $b.col),
                (4, 2) => $func(_B::<4, 2>::new($b.value), $b.row, $b.col),
                (4, 3) => $func(_B::<4, 3>::new($b.value), $b.row, $b.col),
                (4, 4) => $func(_B::<4, 4>::new($b.value), $b.row, $b.col),
                (4, 5) => $func(_B::<4, 5>::new($b.value), $b.row, $b.col),
                (4, 6) => $func(_B::<4, 6>::new($b.value), $b.row, $b.col),
                (4, 7) => $func(_B::<4, 7>::new($b.value), $b.row, $b.col),
                (4, 8) => $func(_B::<4, 8>::new($b.value), $b.row, $b.col),
                (5, 1) => $func(_B::<5, 1>::new($b.value), $b.row, $b.col),
                (5, 2) => $func(_B::<5, 2>::new($b.value), $b.row, $b.col),
                (5, 3) => $func(_B::<5, 3>::new($b.value), $b.row, $b.col),
                (5, 4) => $func(_B::<5, 4>::new($b.value), $b.row, $b.col),
                (5, 5) => $func(_B::<5, 5>::new($b.value), $b.row, $b.col),
                (5, 6) => $func(_B::<5, 6>::new($b.value), $b.row, $b.col),
                (5, 7) => $func(_B::<5, 7>::new($b.value), $b.row, $b.col),
                (5, 8) => $func(_B::<5, 8>::new($b.value), $b.row, $b.col),
                (6, 1) => $func(_B::<6, 1>::new($b.value), $b.row, $b.col),
                (6, 2) => $func(_B::<6, 2>::new($b.value), $b.row, $b.col),
                (6, 3) => $func(_B::<6, 3>::new($b.value), $b.row, $b.col),
                (6, 4) => $func(_B::<6, 4>::new($b.value), $b.row, $b.col),
                (6, 5) => $func(_B::<6, 5>::new($b.value), $b.row, $b.col),
                (6, 6) => $func(_B::<6, 6>::new($b.value), $b.row, $b.col),
                (6, 7) => $func(_B::<6, 7>::new($b.value), $b.row, $b.col),
                (6, 8) => $func(_B::<6, 8>::new($b.value), $b.row, $b.col),
                (7, 1) => $func(_B::<7, 1>::new($b.value), $b.row, $b.col),
                (7, 2) => $func(_B::<7, 2>::new($b.value), $b.row, $b.col),
                (7, 3) => $func(_B::<7, 3>::new($b.value), $b.row, $b.col),
                (7, 4) => $func(_B::<7, 4>::new($b.value), $b.row, $b.col),
                (7, 5) => $func(_B::<7, 5>::new($b.value), $b.row, $b.col),
                (7, 6) => $func(_B::<7, 6>::new($b.value), $b.row, $b.col),
                (7, 7) => $func(_B::<7, 7>::new($b.value), $b.row, $b.col),
                (7, 8) => $func(_B::<7, 8>::new($b.value), $b.row, $b.col),
                (8, 1) => $func(_B::<8, 1>::new($b.value), $b.row, $b.col),
                (8, 2) => $func(_B::<8, 2>::new($b.value), $b.row, $b.col),
                (8, 3) => $func(_B::<8, 3>::new($b.value), $b.row, $b.col),
                (8, 4) => $func(_B::<8, 4>::new($b.value), $b.row, $b.col),
                (8, 5) => $func(_B::<8, 5>::new($b.value), $b.row, $b.col),
                (8, 6) => $func(_B::<8, 6>::new($b.value), $b.row, $b.col),
                (8, 7) => $func(_B::<8, 7>::new($b.value), $b.row, $b.col),
                (8, 8) => $func(_B::<8, 8>::new($b.value), $b.row, $b.col),
                _ => panic!("Unsupported dimensions: ({}, {})", $b.n, $b.m),
            }
        };
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Idempotency: running flood twice should produce the same result.

    proptest! {
        #[test]
        fn idempotence4(input: RuntimeBitBoard) {
            match_bitboard!(input, idempotence4_impl);
        }

        #[test]
        fn idempotence8(input: RuntimeBitBoard) {
            match_bitboard!(input, idempotence8_impl);
        }
    }

    fn idempotence4_impl<const N: usize, const M: usize>(
        input: BitBoard<N, M>,
        row: usize,
        col: usize,
    ) {
        let start = BitBoard::<N, M>::to_index(row, col);

        let result1 = input.flood4(start);
        let result2 = result1.flood4(start);
        assert_eq!(result1, result2);
    }

    fn idempotence8_impl<const N: usize, const M: usize>(
        input: BitBoard<N, M>,
        row: usize,
        col: usize,
    ) {
        let start = BitBoard::<N, M>::to_index(row, col);

        let result1 = input.flood8(start);
        let result2 = result1.flood8(start);
        assert_eq!(result1, result2);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Monotonicity: If a bit is set in the original bit board, it should remain
    // set or be set in the result after flood fill.

    proptest! {
        #[test]
        fn monotonicity4(input: RuntimeBitBoard) {
            match_bitboard!(input, monotonicity4_impl);
        }

        #[test]
        fn monotonicity8(input: RuntimeBitBoard) {
            match_bitboard!(input, monotonicity8_impl);
        }
    }

    fn monotonicity4_impl<const N: usize, const M: usize>(
        input: BitBoard<N, M>,
        row: usize,
        col: usize,
    ) {
        let start = BitBoard::<N, M>::to_index(row, col);

        let result = input.flood4(start);
        assert!(result & !input == BitBoard::empty() || result & input == result);
    }

    fn monotonicity8_impl<const N: usize, const M: usize>(
        input: BitBoard<N, M>,
        row: usize,
        col: usize,
    ) {
        let start = BitBoard::<N, M>::to_index(row, col);

        let result = input.flood8(start);
        assert!(result & !input == BitBoard::empty() || result & input == result);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////

    // Connectivity tests: validate flood fill using alternate BFS implementation.
    proptest! {
        #[test]
        fn connectivity4(input: RuntimeBitBoard) {
            match_bitboard!(input, connectivity4_impl);
        }

        #[test]
        fn connectivity8(input: RuntimeBitBoard) {
            match_bitboard!(input, connectivity8_impl);
        }
    }

    fn connectivity4_impl<const N: usize, const M: usize>(
        input: BitBoard<N, M>,
        row: usize,
        col: usize,
    ) {
        assert!(row < N);
        assert!(col < M);
        let start = BitBoard::<N, M>::to_index(row, col);

        let result = input.flood4(start);
        let ns = [(1, 0), (0, 1), (-1, 0), (0, -1)];
        assert!(check_connectivity(input, result, start, &ns));
    }

    fn connectivity8_impl<const N: usize, const M: usize>(
        input: BitBoard<N, M>,
        row: usize,
        col: usize,
    ) {
        assert!(row < N);
        assert!(col < M);
        let start = BitBoard::<N, M>::to_index(row, col);

        let result = input.flood8(start);
        let ns = [
            (1, 1),
            (1, 0),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ];
        assert!(check_connectivity(input, result, start, &ns));
    }

    // Helper function to check connectivity
    fn check_connectivity<const N: usize, const M: usize>(
        bitboard: BitBoard<N, M>,
        filled: BitBoard<N, M>,
        start: usize,
        ns: &[(i64, i64)],
    ) -> bool {
        let mut visited = BitBoard::empty();
        let mut stack = vec![BitBoard::<N, M>::to_coord(start)];

        while let Some((row, col)) = stack.pop() {
            if !visited.get_at(row, col) && bitboard.get_at(row, col) {
                visited.set_at(row, col);
                for &(dr, dc) in ns {
                    let next_row = (row as i64).wrapping_add(dr) as usize;
                    let next_col = (col as i64).wrapping_add(dc) as usize;
                    if next_row < N && next_col < M {
                        stack.push((next_row, next_col));
                    }
                }
            }
        }

        visited == filled
    }
}