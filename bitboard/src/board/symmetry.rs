use super::Board;
use crate::dim::Dim;
use crate::storage::Storage;

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

//////////////////////////////////////////////////////////////////////////////////////////////////

// O(1) word-parallel symmetry transforms. Unlike `D4Symmetry::apply_to_bits`
// (`games/game-core/src/symmetry.rs`), which loops one set bit at a time via
// `trailing_zeros` (O(popcount)), these permute every cell of the word at
// once via a handful of masked shift/xor steps -- the classic SWAR
// "butterfly network" technique `nego`'s `bitboard.rs` uses for
// `rot90`/`flip_diag_a1h8`/etc. Cost is O(1) regardless of how full the
// board is, which matters once canonicalization runs on every visited node
// (`Game::canonical_representation`) rather than once per expansion.

impl<R: Dim, C: Dim> Board<u64, R, C> {
    /// Reverses the row-major order of every valid cell -- equivalent to
    /// applying `flip_rows` then `flip_cols` (in either order), but done in
    /// one step: reversing a contiguous row-major bit sequence end-to-end is
    /// exactly a 180-degree rotation, independent of how the sequence is
    /// partitioned into rows/cols. Works for any single-word board shape
    /// (not just square), via `u64::reverse_bits` plus a realigning shift.
    #[inline]
    pub fn rot180(&self) -> Self {
        let total = self.len();
        let bits = if total == 0 {
            0
        } else {
            self.bits.reverse_bits() >> (64 - total)
        };
        Self {
            bits,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Board<u64, crate::dim::Const<8>, crate::dim::Const<8>> {
    /// Reverses column order within each row (col -> 7-col), rows
    /// unchanged. An 8x8 board is byte-aligned (each row is exactly one
    /// byte, since `index = row * 8 + col`), so this is "reverse the bit
    /// order within each byte, for all 8 bytes at once" -- the standard
    /// three-step SWAR delta-swap, done word-parallel across every row
    /// simultaneously rather than one row (or one bit) at a time.
    #[inline]
    pub fn flip_cols(&self) -> Self {
        let mut x = self.bits;
        x = ((x & 0x5555_5555_5555_5555) << 1) | ((x >> 1) & 0x5555_5555_5555_5555);
        x = ((x & 0x3333_3333_3333_3333) << 2) | ((x >> 2) & 0x3333_3333_3333_3333);
        x = ((x & 0x0f0f_0f0f_0f0f_0f0f) << 4) | ((x >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
        Self {
            bits: x,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Reverses row order (row -> 7-row), columns unchanged. Each row is
    /// exactly one byte on a byte-aligned 8x8 board (see `flip_cols`), so
    /// reversing row order is exactly reversing byte order -- a single
    /// `swap_bytes`.
    #[inline]
    pub fn flip_rows(&self) -> Self {
        Self {
            bits: self.bits.swap_bytes(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Transposes across the main diagonal: (row, col) -> (col, row). The
    /// classic Hacker's Delight / chessprogramming-wiki `flipDiagA1H8`
    /// delta-swap (three masked shift-xor steps), which assumes the same
    /// `index = row * 8 + col` bit layout this crate already uses.
    #[inline]
    pub fn transpose(&self) -> Self {
        const K1: u64 = 0x5500_5500_5500_5500;
        const K2: u64 = 0x3333_0000_3333_0000;
        const K4: u64 = 0x0f0f_0f0f_0000_0000;
        let mut x = self.bits;
        let mut t = K4 & (x ^ (x << 28));
        x ^= t ^ (t >> 28);
        t = K2 & (x ^ (x << 14));
        x ^= t ^ (t >> 14);
        t = K1 & (x ^ (x << 7));
        x ^= t ^ (t >> 7);
        Self {
            bits: x,
            rows: self.rows,
            cols: self.cols,
        }
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
