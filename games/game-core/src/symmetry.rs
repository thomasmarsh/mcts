//! Board symmetry groups: which rigid transformations of a board leave its
//! adjacency structure unchanged.
//!
//! `D4Symmetry<S>` (8 elements: identity, column flip, row flip,
//! main-diagonal transpose, and their compositions) applies to square
//! boards. `KleinFour<ROWS, COLS>` (4 elements: identity, column flip,
//! row flip, and their composition) applies to any rectangular board,
//! square or not — a non-square board has no diagonal transpose to include
//! since that would map it onto a differently-shaped board. `ColMirror<COLS>`
//! (2 elements: identity, column flip) is `KleinFour`'s subgroup for boards
//! where a row flip specifically *isn't* a valid symmetry — e.g. a
//! gravity-based game, where row 0 is a fixed floor and flipping rows would
//! swap which end gravity pulls toward.
//!
//! All three implement the shared [`SymmetryGroup`] trait, so callers that
//! only need "map a cell index through symmetry element `sym`" / "which
//! element inverts `sym`" can be generic over which group a given board's
//! shape admits, without hardcoding D4.
//!
//! Rather than pre-computing permutation tables (which would require
//! `[usize; S * S]` in a const-generic context, unstable on stable Rust),
//! all transformations are computed inline from the cell index using simple
//! arithmetic.  For any concrete `S`/`ROWS`/`COLS` the compiler will
//! constant-fold these into the same machine code a table lookup would
//! produce.

/// Reverse column order on an `rows`×`cols` grid — col → cols-1-col, row
/// unchanged. Named for its effect (left becomes right), not an axis, since
/// "horizontal flip" is read both ways in the wild.
#[inline]
fn flip_cols_index(i: usize, cols: usize) -> usize {
    let row = i / cols;
    let col = i % cols;
    row * cols + (cols - 1 - col)
}

/// Reverse row order on an `rows`×`cols` grid — row → rows-1-row, col
/// unchanged. Named for its effect (top becomes bottom), not an axis.
#[inline]
fn flip_rows_index(i: usize, rows: usize, cols: usize) -> usize {
    let row = i / cols;
    let col = i % cols;
    (rows - 1 - row) * cols + col
}

/// A finite group of permutations of a board's cell indices — the
/// transformations that leave the board's adjacency structure unchanged.
///
/// Element 0 is always the identity. Code that needs to canonicalize a
/// board or transform an action through a symmetry can be generic over
/// which concrete group a game's board shape admits ([`D4Symmetry`] for
/// square boards, [`KleinFour`] for rectangular ones), rather than
/// hardcoding D4.
pub trait SymmetryGroup {
    /// Number of elements in the group, including the identity at index 0.
    const ORDER: usize;

    /// Map a cell index through group element `sym`.
    fn apply_index(i: usize, sym: usize) -> usize;

    /// Index of the group element that inverts `sym`: applying `sym` and
    /// then `invert(sym)` (via `apply_index` twice) is the identity.
    fn invert(sym: usize) -> usize;

    /// Map an index back through the inverse of `sym` in one step.
    #[inline]
    fn invert_index(i: usize, sym: usize) -> usize {
        Self::apply_index(i, Self::invert(sym))
    }
}

/// Board-side-length parameterised D4 symmetry group.
///
/// # Example
///
/// ```
/// use game_core::symmetry::D4Symmetry;
///
/// // 8×8 board (Othello)
/// type Sym8 = D4Symmetry<8>;
/// let syms = Sym8::index_symmetries(27);  // all 8 images of cell 27
/// assert_eq!(syms[0], 27);                // identity
/// assert_eq!(syms[1], 28);                // column flip
/// ```
pub struct D4Symmetry<const S: usize>;

impl<const S: usize> D4Symmetry<S> {
    /// Reverse column order — col → S-1-col.
    #[inline]
    fn flip_cols(i: usize) -> usize {
        flip_cols_index(i, S)
    }

    /// Reverse row order — row → S-1-row.
    #[inline]
    fn flip_rows(i: usize) -> usize {
        flip_rows_index(i, S, S)
    }

    /// Transpose across the main diagonal — (row, col) → (col, row).
    #[inline]
    fn transpose(i: usize) -> usize {
        let row = i / S;
        let col = i % S;
        col * S + row
    }

    /// Produce all 8 symmetric images of a cell index.
    ///
    /// Order: identity, flip_cols, flip_rows, transpose, flip_rows∘flip_cols,
    /// transpose∘flip_cols, transpose∘flip_rows, transpose∘flip_rows∘flip_cols.
    #[inline]
    pub fn index_symmetries(i: usize) -> [usize; 8] {
        let fc = Self::flip_cols(i);
        let fr = Self::flip_rows(i);
        let t = Self::transpose(i);
        [
            i,
            fc,
            fr,
            t,
            Self::flip_rows(fc),
            Self::transpose(fc),
            Self::transpose(fr),
            Self::transpose(Self::flip_rows(fc)),
        ]
    }

    /// Map an index back through the inverse of a symmetry.
    ///
    /// For an involution (flip_cols, flip_rows, transpose) the inverse is
    /// the same permutation. For a composition the inverse is the reverse
    /// composition.
    #[inline]
    pub fn invert_symmetry(i: usize, sym_idx: usize) -> usize {
        match sym_idx {
            0 => i,
            1 => Self::flip_cols(i),
            2 => Self::flip_rows(i),
            3 => Self::transpose(i),
            4 => Self::flip_cols(Self::flip_rows(i)), // (flip_rows∘flip_cols)⁻¹ = flip_cols∘flip_rows
            5 => Self::flip_cols(Self::transpose(i)), // (transpose∘flip_cols)⁻¹ = flip_cols∘transpose
            6 => Self::flip_rows(Self::transpose(i)), // (transpose∘flip_rows)⁻¹ = flip_rows∘transpose
            7 => Self::flip_cols(Self::flip_rows(Self::transpose(i))), // (transpose∘flip_rows∘flip_cols)⁻¹ = flip_cols∘flip_rows∘transpose
            _ => unreachable!(),
        }
    }

    /// Apply a symmetry permutation to a raw u64 bitboard.
    ///
    /// Iterates each set bit, maps it through the symmetry, and sets the
    /// result bit in the output.  Works for any board size ≤ 64 cells.
    #[inline]
    pub fn apply_to_bits(board: u64, sym_idx: usize) -> u64 {
        let mut result = 0u64;
        let mut bits = board;
        while bits != 0 {
            let lsb = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let dst = Self::index_symmetries(lsb)[sym_idx];
            result |= 1u64 << dst;
        }
        result
    }

    /// Compute all 8 symmetric images of a packed-`u32` board encoding (2 bits per cell).
    ///
    /// Each cell `i` occupies bits `(i * 2) .. (i * 2 + 1)`, with value `0b00` = empty,
    /// `0b01` = player 0, `0b10` = player 1.  The output `symmetries` array is filled
    /// with the 8 symmetric versions of the board.
    #[inline]
    pub fn packed_board_symmetries(board: u32, symmetries: &mut [u32; 8]) {
        debug_assert!(symmetries.iter().all(|x| *x == 0));
        for i in 0..(S * S) {
            let p = (board >> (i << 1)) & 0b11;
            for (s, &dst) in Self::index_symmetries(i).iter().enumerate() {
                symmetries[s] |= p << (dst << 1);
            }
        }
    }

    /// Index of the symmetry whose image of a packed-`u32` board is minimal.
    ///
    /// This is the canonical symmetry for the board: all 8 symmetric board
    /// values are compared and the smallest one's symmetry index is returned.
    #[inline]
    pub fn packed_canonical_symmetry(board: u32) -> usize {
        let mut sym = [0; 8];
        Self::packed_board_symmetries(board, &mut sym);
        sym.iter().enumerate().min_by_key(|(_, &v)| v).unwrap().0
    }
}

impl<const S: usize> SymmetryGroup for D4Symmetry<S> {
    const ORDER: usize = 8;

    #[inline]
    fn apply_index(i: usize, sym: usize) -> usize {
        Self::index_symmetries(i)[sym]
    }

    #[inline]
    fn invert(sym: usize) -> usize {
        // Same table as `invert_symmetry`'s match arms, indexed directly:
        // each of D4's 8 elements is its own inverse except the two
        // 4-cycles (transpose∘flip_cols and transpose∘flip_rows), which
        // are inverses of each other.
        const INV: [usize; 8] = [0, 1, 2, 3, 4, 6, 5, 7];
        INV[sym]
    }
}

/// Runtime-sized D4 symmetry group for a square board whose side length
/// isn't fixed at compile time -- the `Dyn`-dims counterpart of
/// [`D4Symmetry`], for games like Gonnect/AtariGo whose board size varies
/// per instance (3x3 through 19x19) rather than being one fixed constant.
/// Same 8 elements, same index arithmetic, just parameterised by a runtime
/// `size` field instead of a const generic.
#[derive(Clone, Copy, Debug)]
pub struct D4Dyn {
    size: usize,
}

impl D4Dyn {
    #[inline]
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    #[inline]
    fn flip_cols(&self, i: usize) -> usize {
        flip_cols_index(i, self.size)
    }

    #[inline]
    fn flip_rows(&self, i: usize) -> usize {
        flip_rows_index(i, self.size, self.size)
    }

    #[inline]
    fn transpose(&self, i: usize) -> usize {
        let row = i / self.size;
        let col = i % self.size;
        col * self.size + row
    }

    /// Produce all 8 symmetric images of a cell index -- see
    /// `D4Symmetry::index_symmetries` for the element ordering.
    #[inline]
    pub fn index_symmetries(&self, i: usize) -> [usize; 8] {
        let fc = self.flip_cols(i);
        let fr = self.flip_rows(i);
        let t = self.transpose(i);
        [
            i,
            fc,
            fr,
            t,
            self.flip_rows(fc),
            self.transpose(fc),
            self.transpose(fr),
            self.transpose(self.flip_rows(fc)),
        ]
    }

    /// Map an index back through the inverse of a symmetry -- see
    /// `D4Symmetry::invert_symmetry`.
    #[inline]
    pub fn invert_symmetry(&self, i: usize, sym_idx: usize) -> usize {
        match sym_idx {
            0 => i,
            1 => self.flip_cols(i),
            2 => self.flip_rows(i),
            3 => self.transpose(i),
            4 => self.flip_cols(self.flip_rows(i)),
            5 => self.flip_cols(self.transpose(i)),
            6 => self.flip_rows(self.transpose(i)),
            7 => self.flip_cols(self.flip_rows(self.transpose(i))),
            _ => unreachable!(),
        }
    }
}

/// Applies symmetry element `sym_idx` (of `sym`'s D4 group) to every set bit
/// of `board`, producing the transformed board -- the `Dyn`-dims/multi-word
/// counterpart of `D4Symmetry::apply_to_bits`, generic over any
/// `bitboard::Board` storage/dim combination since `iter_set`/`set_index`/
/// `empty_like` don't depend on either. Shared by every square-board `Dyn`
/// game (Gonnect, AtariGo) that needs to rotate/reflect a whole board rather
/// than a bare bit pattern.
pub fn transform_board<S: bitboard::Storage, R: bitboard::Dim, C: bitboard::Dim>(
    board: bitboard::Board<S, R, C>,
    sym: &D4Dyn,
    sym_idx: usize,
) -> bitboard::Board<S, R, C> {
    let mut out = board.empty_like();
    for idx in board.iter_set() {
        out.set_index(sym.index_symmetries(idx)[sym_idx]);
    }
    out
}

/// Applies symmetry element `sym_idx` to a raw `WORDS`-word bitmask --
/// e.g. a `Move`'s wire-format capture mask, which (see `Move`'s own doc
/// comment in Gonnect/AtariGo) is deliberately not a dims-carrying `Board`
/// since a `Move` is deserialized before its target `State`'s size is known.
pub fn transform_words<const WORDS: usize>(
    words: [u64; WORDS],
    d4: &D4Dyn,
    sym_idx: usize,
) -> [u64; WORDS] {
    let mut out = [0u64; WORDS];
    for (w, &word) in words.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            word &= word - 1;
            let dst = d4.index_symmetries(w * 64 + bit)[sym_idx];
            out[dst / 64] |= 1u64 << (dst % 64);
        }
    }
    out
}

/// The inverse of `transform_words`: applies the inverse of symmetry element
/// `sym_idx` to a raw `WORDS`-word bitmask.
pub fn invert_words<const WORDS: usize>(
    words: [u64; WORDS],
    d4: &D4Dyn,
    sym_idx: usize,
) -> [u64; WORDS] {
    let mut out = [0u64; WORDS];
    for (w, &word) in words.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            word &= word - 1;
            let dst = d4.invert_symmetry(w * 64 + bit, sym_idx);
            out[dst / 64] |= 1u64 << (dst % 64);
        }
    }
    out
}

/// O(1) word-parallel counterpart to calling `D4Symmetry::<8>::apply_to_bits`
/// eight times (once per symmetry index): produces all 8 symmetric images of
/// a single-word 8x8 `bitboard::Board` via `Board::flip_cols`/`flip_rows`/
/// `transpose` (a handful of masked shift/xor steps each, independent of how
/// many bits are set), in the same element order as
/// `D4Symmetry::index_symmetries`/`apply_to_bits` (identity, flip_cols,
/// flip_rows, transpose, flip_rows∘flip_cols, transpose∘flip_cols,
/// transpose∘flip_rows, transpose∘flip_rows∘flip_cols) -- see
/// `test_board_symmetries_8x8_matches_apply_to_bits` for the cross-check.
/// The `Board`-level `flip_cols`/`flip_rows`/`transpose` primitives this
/// composes only exist for `Const<8>, Const<8>`, so this helper is likewise
/// 8x8-only; other board sizes still go through `D4Symmetry::apply_to_bits`.
pub fn board_symmetries_8x8(
    board: bitboard::Board<u64, bitboard::Const<8>, bitboard::Const<8>>,
) -> [bitboard::Board<u64, bitboard::Const<8>, bitboard::Const<8>>; 8] {
    let flip_cols = board.flip_cols();
    let flip_rows = board.flip_rows();
    let transpose = board.transpose();
    let rot180 = flip_cols.flip_rows();
    [
        board,
        flip_cols,
        flip_rows,
        transpose,
        rot180,
        flip_cols.transpose(),
        flip_rows.transpose(),
        rot180.transpose(),
    ]
}

/// Klein four-group symmetries (identity, column flip, row flip, and their
/// composition) for a `ROWS`×`COLS` board of any aspect ratio.
///
/// A non-square board has no diagonal transpose available — that would map
/// it onto a `COLS`×`ROWS` board, a different shape, not a symmetry of the
/// board it started as. Klein four is the largest group of rigid
/// transformations every rectangular board (square ones included) admits;
/// [`D4Symmetry`] is a strict refinement available only when `ROWS == COLS`.
pub struct KleinFour<const ROWS: usize, const COLS: usize>;

impl<const ROWS: usize, const COLS: usize> KleinFour<ROWS, COLS> {
    #[inline]
    fn flip_cols(i: usize) -> usize {
        flip_cols_index(i, COLS)
    }

    #[inline]
    fn flip_rows(i: usize) -> usize {
        flip_rows_index(i, ROWS, COLS)
    }

    /// Produce all 4 symmetric images of a cell index.
    ///
    /// Order: identity, flip_cols, flip_rows, flip_rows∘flip_cols.
    #[inline]
    pub fn index_symmetries(i: usize) -> [usize; 4] {
        let fc = Self::flip_cols(i);
        let fr = Self::flip_rows(i);
        [i, fc, fr, Self::flip_rows(fc)]
    }

    /// Map an index back through the inverse of a symmetry.
    ///
    /// Every element of Klein four is its own inverse (the group is
    /// isomorphic to Z/2 × Z/2), so this is just `index_symmetries(i)[sym_idx]`
    /// spelled out for parity with `D4Symmetry::invert_symmetry`.
    #[inline]
    pub fn invert_symmetry(i: usize, sym_idx: usize) -> usize {
        match sym_idx {
            0 => i,
            1 => Self::flip_cols(i),
            2 => Self::flip_rows(i),
            3 => Self::flip_rows(Self::flip_cols(i)),
            _ => unreachable!(),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> SymmetryGroup for KleinFour<ROWS, COLS> {
    const ORDER: usize = 4;

    #[inline]
    fn apply_index(i: usize, sym: usize) -> usize {
        Self::index_symmetries(i)[sym]
    }

    #[inline]
    fn invert(sym: usize) -> usize {
        // Every element is its own inverse.
        sym
    }
}

/// Column-mirror symmetry group (2 elements: identity, column flip) for
/// boards where a row flip is *not* a valid symmetry -- e.g. a gravity-based
/// game (Connect Four) where row 0 is a fixed floor and flipping rows would
/// swap which end gravity pulls toward. `KleinFour` is the right choice
/// whenever both flips are valid; reach for `ColMirror` only when a row flip
/// specifically isn't. (No `RowMirror` exists yet since nothing in this repo
/// needs the row-only case -- add one the same way if that changes.)
pub struct ColMirror<const COLS: usize>;

impl<const COLS: usize> ColMirror<COLS> {
    #[inline]
    fn flip_cols(i: usize) -> usize {
        flip_cols_index(i, COLS)
    }

    /// Produce both symmetric images of a cell index: `[identity, mirror]`.
    #[inline]
    pub fn index_symmetries(i: usize) -> [usize; 2] {
        [i, Self::flip_cols(i)]
    }

    /// Map an index back through the inverse of a symmetry. The column flip
    /// is its own inverse, so this is `index_symmetries(i)[sym_idx]` spelled
    /// out for parity with `KleinFour::invert_symmetry`.
    #[inline]
    pub fn invert_symmetry(i: usize, sym_idx: usize) -> usize {
        match sym_idx {
            0 => i,
            1 => Self::flip_cols(i),
            _ => unreachable!(),
        }
    }
}

impl<const COLS: usize> SymmetryGroup for ColMirror<COLS> {
    const ORDER: usize = 2;

    #[inline]
    fn apply_index(i: usize, sym: usize) -> usize {
        Self::index_symmetries(i)[sym]
    }

    #[inline]
    fn invert(sym: usize) -> usize {
        // Both elements are their own inverse.
        sym
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that flip_cols, flip_rows, transpose are bijections on [0, S*S).
    fn check_permutations<const S: usize>() {
        let n = S * S;
        for (name, perm) in [
            (
                "flip_cols",
                D4Symmetry::<S>::flip_cols as fn(usize) -> usize,
            ),
            (
                "flip_rows",
                D4Symmetry::<S>::flip_rows as fn(usize) -> usize,
            ),
            (
                "transpose",
                D4Symmetry::<S>::transpose as fn(usize) -> usize,
            ),
        ] {
            let mut seen = vec![false; n];
            for i in 0..n {
                let v = perm(i);
                assert!(v < n, "{name}[{i}] = {v} out of range");
                assert!(!seen[v], "{name} is not injective (dupe at {v})");
                seen[v] = true;
            }
            assert!(seen.iter().all(|&x| x), "{name} is not surjective");
        }
    }

    #[test]
    fn test_permutations_3x3() {
        check_permutations::<3>();
    }

    #[test]
    fn test_permutations_8x8() {
        check_permutations::<8>();
    }

    /// Verify that index_symmetries produces valid images (within bounds).
    #[test]
    fn test_index_symmetries_in_range() {
        for i in 0..9 {
            let s = D4Symmetry::<3>::index_symmetries(i);
            for &v in &s {
                assert!(v < 9, "index_symmetries({i}) has out-of-range {v}");
            }
        }
        for i in 0..64 {
            let s = D4Symmetry::<8>::index_symmetries(i);
            for &v in &s {
                assert!(v < 64, "index_symmetries({i}) has out-of-range {v}");
            }
        }
    }

    /// Cross-checks `board_symmetries_8x8` (the O(1) word-parallel path)
    /// against `D4Symmetry::<8>::apply_to_bits` (the O(popcount) per-bit
    /// path already proven correct by `test_permutations_8x8`/
    /// `test_index_symmetries_in_range` above) -- exhaustive over every
    /// single-bit board (sufficient since both sides are bitwise-linear
    /// permutations: correct on every basis vector implies correct on every
    /// board), plus a handful of multi-bit boards for combination
    /// confidence.
    #[test]
    fn test_board_symmetries_8x8_matches_apply_to_bits() {
        let check = |bits: u64| {
            let board: bitboard::Board<u64, bitboard::Const<8>, bitboard::Const<8>> =
                bitboard::Board::from_bits(bits);
            let got = board_symmetries_8x8(board);
            for (sym_idx, image) in got.iter().enumerate() {
                let expected = D4Symmetry::<8>::apply_to_bits(bits, sym_idx);
                assert_eq!(
                    image.bits(),
                    expected,
                    "sym {sym_idx} mismatch for board {bits:#x}"
                );
            }
        };

        for i in 0..64 {
            check(1u64 << i);
        }

        check(0);
        check(u64::MAX);
        check(0x0000_0018_1800_0000); // the initial Othello position
        check(0xDEAD_BEEF_1234_5678);
        check(0xAAAA_AAAA_5555_5555);
    }

    /// Verify that invert_symmetry is the true inverse of index_symmetries.
    #[test]
    fn test_invert_symmetry_is_inverse() {
        for i in 0..9 {
            let s = D4Symmetry::<3>::index_symmetries(i);
            for (sym_idx, &s_i) in s.iter().enumerate() {
                let back = D4Symmetry::<3>::invert_symmetry(s_i, sym_idx);
                assert_eq!(
                    back, i,
                    "invert_symmetry({s_i}, {sym_idx}) = {back}, expected {i}"
                );
            }
        }
        for i in 0..64 {
            let s = D4Symmetry::<8>::index_symmetries(i);
            for (sym_idx, &s_i) in s.iter().enumerate() {
                let back = D4Symmetry::<8>::invert_symmetry(s_i, sym_idx);
                assert_eq!(
                    back, i,
                    "invert_symmetry({s_i}, {sym_idx}) = {back}, expected {i}"
                );
            }
        }
    }

    /// Applying then inverting a symmetry should yield the original board.
    #[test]
    fn test_apply_to_bits_inverse() {
        // Pre-computed inverse of each symmetry index.
        const INV: [usize; 8] = [0, 1, 2, 3, 4, 6, 5, 7];

        // 3×3 board: test a few representative bit patterns.
        for board in [0b_100_000_001u64, 0b_010_010_010, 0b_001_000_100] {
            for (sym_idx, &inv) in INV.iter().enumerate() {
                let transformed = D4Symmetry::<3>::apply_to_bits(board, sym_idx);
                let back = D4Symmetry::<3>::apply_to_bits(transformed, inv);
                assert_eq!(
                    back, board,
                    "sym {sym_idx} then inv {inv} on {board:#x} gave {back:#x}"
                );
            }
        }

        // 8×8 board: Othello's initial position.
        let board = (1 << 28) | (1 << 35);
        for (sym_idx, &inv) in INV.iter().enumerate() {
            let transformed = D4Symmetry::<8>::apply_to_bits(board, sym_idx);
            let back = D4Symmetry::<8>::apply_to_bits(transformed, inv);
            assert_eq!(
                back, board,
                "sym {sym_idx} then inv {inv} on {board:#x} gave {back:#x}"
            );
        }
    }

    /// Generic round-trip check usable by any `SymmetryGroup`: applying an
    /// element and then its inverse (via `invert_index`) must be the
    /// identity, for every cell and every group element.
    fn check_symmetry_group_round_trip<G: SymmetryGroup>(cells: usize) {
        for i in 0..cells {
            for sym in 0..G::ORDER {
                let image = G::apply_index(i, sym);
                assert!(
                    image < cells,
                    "apply_index({i}, {sym}) = {image} out of range"
                );
                let back = G::invert_index(image, sym);
                assert_eq!(
                    back, i,
                    "invert_index(apply_index({i}, {sym}), {sym}) = {back}"
                );
            }
        }
    }

    #[test]
    fn test_d4_symmetry_group_round_trip() {
        check_symmetry_group_round_trip::<D4Symmetry<3>>(9);
        check_symmetry_group_round_trip::<D4Symmetry<8>>(64);
    }

    #[test]
    fn test_klein_four_symmetry_group_round_trip() {
        check_symmetry_group_round_trip::<KleinFour<3, 3>>(9);
        check_symmetry_group_round_trip::<KleinFour<3, 5>>(15);
        check_symmetry_group_round_trip::<KleinFour<8, 8>>(64);
    }

    /// Verify flip_cols/flip_rows are bijections on a rectangular
    /// (non-square) grid too -- `check_permutations` above only ever
    /// exercised square `D4Symmetry` boards, so this is the first coverage
    /// of the `ROWS != COLS` case.
    #[test]
    fn test_klein_four_permutations_rectangular() {
        let (rows, cols) = (3usize, 5usize);
        let n = rows * cols;
        for (name, perm) in [
            (
                "flip_cols",
                KleinFour::<3, 5>::flip_cols as fn(usize) -> usize,
            ),
            (
                "flip_rows",
                KleinFour::<3, 5>::flip_rows as fn(usize) -> usize,
            ),
        ] {
            let mut seen = vec![false; n];
            for i in 0..n {
                let v = perm(i);
                assert!(v < n, "{name}[{i}] = {v} out of range");
                assert!(!seen[v], "{name} is not injective (dupe at {v})");
                seen[v] = true;
            }
            assert!(seen.iter().all(|&x| x), "{name} is not surjective");
        }
    }

    /// Verify invert_symmetry is the true inverse of index_symmetries for
    /// KleinFour, mirroring `test_invert_symmetry_is_inverse` above.
    #[test]
    fn test_klein_four_invert_symmetry_is_inverse() {
        for i in 0..15 {
            let s = KleinFour::<3, 5>::index_symmetries(i);
            for (sym_idx, &s_i) in s.iter().enumerate() {
                let back = KleinFour::<3, 5>::invert_symmetry(s_i, sym_idx);
                assert_eq!(
                    back, i,
                    "invert_symmetry({s_i}, {sym_idx}) = {back}, expected {i}"
                );
            }
        }
    }

    /// `D4Dyn` must agree bit-for-bit with `D4Symmetry<S>` at the same size --
    /// it's the same group, just with the size moved from a const generic to
    /// a runtime field.
    #[test]
    fn test_d4_dyn_matches_d4_symmetry() {
        for size in [3usize, 8, 13, 19] {
            let dyn_sym = D4Dyn::new(size);
            for i in 0..(size * size) {
                let expected = match size {
                    3 => D4Symmetry::<3>::index_symmetries(i).to_vec(),
                    8 => D4Symmetry::<8>::index_symmetries(i).to_vec(),
                    13 => D4Symmetry::<13>::index_symmetries(i).to_vec(),
                    19 => D4Symmetry::<19>::index_symmetries(i).to_vec(),
                    _ => unreachable!(),
                };
                assert_eq!(dyn_sym.index_symmetries(i).to_vec(), expected);
            }
        }
    }

    /// `D4Dyn::invert_symmetry` must be the true inverse of
    /// `D4Dyn::index_symmetries`, mirroring `test_invert_symmetry_is_inverse`.
    #[test]
    fn test_d4_dyn_invert_symmetry_is_inverse() {
        for size in [3usize, 9, 19] {
            let sym = D4Dyn::new(size);
            for i in 0..(size * size) {
                let s = sym.index_symmetries(i);
                for (sym_idx, &s_i) in s.iter().enumerate() {
                    let back = sym.invert_symmetry(s_i, sym_idx);
                    assert_eq!(
                        back, i,
                        "invert_symmetry({s_i}, {sym_idx}) = {back}, expected {i}"
                    );
                }
            }
        }
    }

    /// `D4Dyn`'s permutations must be bijections on `[0, size*size)`,
    /// mirroring `check_permutations`.
    #[test]
    fn test_d4_dyn_permutations() {
        for size in [3usize, 9, 19] {
            let sym = D4Dyn::new(size);
            let n = size * size;
            for sym_idx in 0..8 {
                let mut seen = vec![false; n];
                for i in 0..n {
                    let v = sym.index_symmetries(i)[sym_idx];
                    assert!(v < n, "sym {sym_idx}[{i}] = {v} out of range");
                    assert!(!seen[v], "sym {sym_idx} is not injective (dupe at {v})");
                    seen[v] = true;
                }
                assert!(seen.iter().all(|&x| x), "sym {sym_idx} is not surjective");
            }
        }
    }

    #[test]
    fn test_col_mirror_symmetry_group_round_trip() {
        check_symmetry_group_round_trip::<ColMirror<7>>(42); // 6x7 Connect Four
        check_symmetry_group_round_trip::<ColMirror<5>>(20); // 4x5 Connect Four
    }

    /// Verify flip_cols is a bijection on a rectangular grid, mirroring
    /// `test_klein_four_permutations_rectangular`.
    #[test]
    fn test_col_mirror_permutations_rectangular() {
        let (rows, cols) = (6usize, 7usize);
        let n = rows * cols;
        let mut seen = vec![false; n];
        for i in 0..n {
            let v = ColMirror::<7>::flip_cols(i);
            assert!(v < n, "flip_cols[{i}] = {v} out of range");
            assert!(!seen[v], "flip_cols is not injective (dupe at {v})");
            seen[v] = true;
        }
        assert!(seen.iter().all(|&x| x), "flip_cols is not surjective");
    }

    /// `ColMirror`'s 2 elements must agree with `KleinFour`'s corresponding
    /// 2 (identity, flip_cols) -- `ColMirror` is a subgroup of `KleinFour`,
    /// not an independent definition that happens to look similar.
    #[test]
    fn test_col_mirror_is_klein_four_subgroup() {
        for i in 0..(6 * 7) {
            let k4 = KleinFour::<6, 7>::index_symmetries(i);
            let cm = ColMirror::<7>::index_symmetries(i);
            assert_eq!(cm, [k4[0], k4[1]]);
        }
    }

    /// On a square board, `KleinFour`'s 4 elements (identity, flip_cols,
    /// flip_rows, flip_rows∘flip_cols) must agree with the corresponding 4
    /// of `D4Symmetry`'s 8 -- Klein four is a subgroup of D4, not an
    /// independent definition that happens to look similar.
    #[test]
    fn test_klein_four_is_d4_subgroup_on_square_board() {
        // D4Symmetry::index_symmetries order is
        // [id, flip_cols, flip_rows, transpose, flip_rows∘flip_cols, ...];
        // KleinFour::index_symmetries order is [id, flip_cols, flip_rows,
        // flip_rows∘flip_cols] -- indices 0,1,2,4 of D4's.
        for i in 0..9 {
            let d4 = D4Symmetry::<3>::index_symmetries(i);
            let k4 = KleinFour::<3, 3>::index_symmetries(i);
            assert_eq!(k4, [d4[0], d4[1], d4[2], d4[4]]);
        }
    }
}
