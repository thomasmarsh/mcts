// Tak, designed by James Ernest and Patrick Rothfuss.
//
// Rules summary: two players race to build a "road" -- a path of their pieces
// (top of stacks, flats or capstones; standing stones never count) connecting
// two opposite edges of an N x N board. On a turn you either place a piece
// (flat / standing "wall" / capstone) on an empty cell, or move a stack you
// control (your piece on top): take up to N pieces off the top (the "carry
// limit" equals the board width), walk in a straight line dropping at least
// one piece per cell. Walls and capstones can never be covered, so every cell
// on the path must be empty or flat-topped -- except that a capstone moving
// *by itself* may land on a wall and flatten it to a flat. The first two
// plies are special: each player places one of their *opponent's* flats. The
// game ends when someone completes a road (if one move completes a road for
// both players, the mover wins), or when the board is full or either reserve
// is empty, in which case flat-topped stacks are counted (tie = draw).
//
// # Representation
//
// Boards are N x N with N in 3..=6 (5x5 is the standard tournament size; the
// "Classic Set" is 6x6; 8x8 is unsupported -- see the height bound below).
// `State` is a fixed-size, `Copy`, allocation-free struct (~304 bytes)
// designed for the MCTS hot path: cheap clone-per-apply, no indirection, and
// terminal/movegen work proportional to the 36 cell words.
//
// - `cells: [u64; 36]`: one word per cell encoding the whole stack:
//     * bits [0, 2): kind of the TOP piece (flat / wall / capstone). Pieces
//       below the top are always flat (nothing may be stacked on a wall or
//       capstone, so they can never end up mid-stack), so one kind field
//       per cell suffices.
//     * bits [2, 2+h): piece colors, one bit per piece, LSB = bottom of the
//       stack (0 = white, 1 = black).
//     * bit 2+h: a sentinel 1 bit, so height is recoverable in one
//       `leading_zeros`: h = 61 - lz(w) for w != 0 (0 = empty cell).
//   The per-cell ops `apply` needs -- pop k pieces off the top, push a group
//   of pieces onto a stack, flatten a wall -- are each a handful of shifts.
//
//   Height bound: pieces never leave the board and the game ends as soon as
//   a player places their *last* piece, so at most 2p - 1 pieces can be on
//   the board (p = stones + caps per player) and no stack can be taller.
//   The worst supported case is 6x6 (p = 30 + 1) -> h <= 61, and the
//   encoding fits exactly h <= 61 (2 kind bits + 61 color bits + 1 sentinel
//   = 64 bits). That is why 8x8 (which would need h <= 103) does not fit a
//   u64 cell.
//
// - `stones` / `caps`: per-player reserves. Placement is the only thing
//   that depletes them, and only onto empty cells.
// - `turn`, `opening`: side to move (White first, per Tak convention) and
//   the two-ply opening phase in which each player places an opponent flat.
// - `hash`: an incremental hash for the transposition table, updated on
//   apply by XORing out the old and in the new contributions of exactly the
//   cells a move touches (<= N + 1 cells), plus reserve / side-to-move /
//   opening-phase keys. A cell's contribution is a per-cell-salted 64-bit
//   mix of its word (bijective in the word, like a Zobrist table lookup
//   over a huge alphabet); the table stores and compares full states, so a
//   collision would only cost an extra bucket entry.
//
// # Moves
//
// `Move` is a u32:
//
//   bit 0:      0 = placement, 1 = stack movement ("spread")
//   bits 1..7:  square index (row * N + col; row 0 = south edge)
//   placement:  bits 7..9: piece kind (flat / wall / capstone)
//   spread:     bits 7..9:  direction (N/E/S/W, matching bitboard::Direction)
//               bits 9..13: take count (1..=N)
//               bits 13..19: drop schedule -- bit i set means "a boundary
//               between two consecutive drop squares comes right after the
//               i-th carried piece" (counting from the bottom of the carried
//               stack). The topmost bit (count - 1) is always set, so the
//               popcount is the number of squares visited and the distances
//               between successive set bits are the per-square drop counts.
//               E.g. with count 5, the rulebook's (2, 2, 1) drop is 0b11010.
//
//   (take, drops) pairs are unique per (src, dir): the drop counts sum to
//   `take`, so different takes or different schedules always describe
//   different moves, and generation never produces duplicates.
//
// # Win detection
//
// Only stack tops matter for roads, so `terminal_status` scans the cell
// words into two bitboards (one per color, walls excluded) and flood-fills
// from each pair of opposite edges using the shared `BitBoard` shift
// operations. The player who just moved is checked first and wins ties (the
// double-road rule). A druid-style incremental union-find is a poor fit for
// Tak: covering an opponent's stack *deletes* their connectivity every few
// plies, forcing constant rebuilds -- per-ply bitboard floods over <= 36
// cells are cheap and always correct.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::bitboard::{BitBoard, Direction};
use crate::display::{RectangularBoard, RectangularBoardDisplay};
use crate::game::{Game, PlayerIndex, TerminalStatus};
use crate::zobrist::LazyZobristTable;

/// Largest supported board dimension. The cell encoding (see above) bounds
/// the stack height to 61 pieces, which exactly covers the 6x6 worst case.
pub const MAX_SIZE: usize = 6;
const MAX_CELLS: usize = MAX_SIZE * MAX_SIZE;

/// Kind of the piece on top of a stack, stored in bits [0, 2) of a cell
/// word. `FLAT` is also the kind of every piece below the top.
pub const FLAT: u8 = 0;
pub const WALL: u8 = 1;
pub const CAP: u8 = 2;

// Direction deltas, indexed to match `bitboard::Direction` (N/E/S/W).
const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq, Hash)]
pub enum Player {
    #[default]
    White,
    Black,
}

impl Player {
    #[inline(always)]
    fn next(self) -> Player {
        match self {
            Player::White => Player::Black,
            Player::Black => Player::White,
        }
    }

    /// The color bit this player contributes to a cell word.
    #[inline(always)]
    fn bit(self) -> u64 {
        self as u64
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Cell words
//////////////////////////////////////////////////////////////////////////////////////////////////

/// Pack a stack into its cell word: `colors` has one bit per piece (LSB =
/// bottom, 0 = white / 1 = black), `h` the stack height, `k` the kind of the
/// top piece.
#[inline(always)]
pub const fn make_cell(colors: u64, h: u32, k: u8) -> u64 {
    debug_assert!(h >= 1 && h <= 61);
    (1u64 << (2 + h)) | (colors << 2) | k as u64
}

/// Height of the stack in a nonempty cell word.
#[inline(always)]
pub fn cell_height(w: u64) -> u32 {
    debug_assert!(w != 0);
    61 - w.leading_zeros()
}

/// Kind (FLAT / WALL / CAP) of the piece on top of the stack.
#[inline(always)]
pub fn cell_kind(w: u64) -> u8 {
    (w & 3) as u8
}

/// Color bit (0 = white, 1 = black) of the piece on top of the stack.
#[inline(always)]
pub fn cell_top_color(w: u64) -> u8 {
    debug_assert!(w != 0);
    ((w >> (cell_height(w) + 1)) & 1) as u8
}

/// Color bit of the piece at height `j` (0 = bottom of the stack).
#[inline(always)]
pub fn cell_color_at(w: u64, j: u32) -> u8 {
    debug_assert!(w != 0 && j < cell_height(w));
    ((w >> (2 + j)) & 1) as u8
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Hashing
//////////////////////////////////////////////////////////////////////////////////////////////////

const HASH_RESERVE: usize = MAX_CELLS; // .. + player index
const HASH_TURN: usize = MAX_CELLS + 2;
const HASH_OPENING: usize = MAX_CELLS + 3;
const HASHES_LEN: usize = MAX_CELLS + 4;

static HASHES: LazyZobristTable<HASHES_LEN> = LazyZobristTable::new(0x7A6B);

/// splitmix64 finalizer: a cheap bijective 64-bit mix. Each cell's word is
/// salted with a per-cell random key and mixed, giving a Zobrist-style
/// contribution without a table indexed by the (astronomical) word space.
#[inline(always)]
const fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Hash contribution of one cell. Empty cells contribute 0, so the initial
/// (empty) board needs no per-cell work.
#[inline(always)]
fn cell_hash(i: usize, w: u64) -> u64 {
    if w == 0 {
        0
    } else {
        mix(w ^ HASHES.hash(i))
    }
}

#[inline(always)]
fn reserve_hash(p: usize, stones: u8, caps: u8) -> u64 {
    mix(((caps as u64) << 8 | stones as u64) ^ HASHES.hash(HASH_RESERVE + p))
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Moves
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Move(u32);

impl Move {
    /// Place a piece of the given kind on (empty) square `sq`.
    #[inline(always)]
    pub fn place(sq: usize, kind: u8) -> Move {
        debug_assert!(sq < 64 && kind <= CAP);
        Move(((sq as u32) << 1) | ((kind as u32) << 7))
    }

    /// Move `take` pieces off the top of the stack at `src` in direction
    /// `dir`, dropping them according to `drops` (see the file header: bit i
    /// set = square boundary after the i-th carried piece; bit `take - 1`
    /// must be set).
    #[inline(always)]
    pub fn spread(src: usize, dir: usize, take: u32, drops: u32) -> Move {
        debug_assert!(src < 64 && dir < 4);
        debug_assert!((1..=8).contains(&take));
        debug_assert!(drops < (1 << take) && drops & (1 << (take - 1)) != 0);
        Move(1 | ((src as u32) << 1) | ((dir as u32) << 7) | (take << 9) | (drops << 13))
    }

    #[inline(always)]
    pub fn is_spread(self) -> bool {
        self.0 & 1 != 0
    }

    /// Placement square, or the source square of a spread.
    #[inline(always)]
    pub fn square(self) -> usize {
        ((self.0 >> 1) & 63) as usize
    }

    /// Piece kind of a placement (FLAT / WALL / CAP).
    #[inline(always)]
    pub fn kind(self) -> u8 {
        ((self.0 >> 7) & 3) as u8
    }

    /// Direction index of a spread (0 = N, 1 = E, 2 = S, 3 = W).
    #[inline(always)]
    pub fn dir(self) -> usize {
        ((self.0 >> 7) & 3) as usize
    }

    #[inline(always)]
    pub fn direction(self) -> Direction {
        match self.dir() {
            0 => Direction::North,
            1 => Direction::East,
            2 => Direction::South,
            _ => Direction::West,
        }
    }

    /// Number of pieces a spread takes off the source stack (1..=N).
    #[inline(always)]
    pub fn count(self) -> u32 {
        (self.0 >> 9) & 15
    }

    /// Drop schedule of a spread (bit i set = square boundary after the i-th
    /// carried piece, counting from the bottom).
    #[inline(always)]
    pub fn drops(self) -> u32 {
        (self.0 >> 13) & 63
    }

    /// Per-square drop counts of a spread, in walk order.
    pub fn drop_sizes(self) -> Vec<u32> {
        let mut sizes = Vec::new();
        let mut bits = self.drops();
        let mut dropped = 0;
        while bits != 0 {
            let p = bits.trailing_zeros();
            bits &= bits - 1;
            sizes.push(p + 1 - dropped);
            dropped = p + 1;
        }
        debug_assert!(dropped == self.count());
        sizes
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_spread() {
            f.debug_struct("Spread")
                .field("src", &self.square())
                .field("dir", &self.direction())
                .field("count", &self.count())
                .field("drops", &self.drop_sizes())
                .finish()
        } else {
            let kind = ["flat", "wall", "cap"][self.kind() as usize];
            f.debug_struct("Place")
                .field("sq", &self.square())
                .field("kind", &kind)
                .finish()
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// State
//////////////////////////////////////////////////////////////////////////////////////////////////

// NOTE: no serde derives: this serde version only implements array traits
// up to 32 elements, and `cells` is [u64; 36]. The `Game` trait does not
// require `S: Serialize` (only actions are serialized), so this costs
// nothing; add a manual impl if a serialization consumer ever needs one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State<const N: usize> {
    /// One packed stack word per cell; only the first N * N entries are
    /// used, the rest stay 0 (so `Eq` over the full array is canonical).
    pub cells: [u64; MAX_CELLS],
    /// Incremental hash (see the file header). Kept in sync by `apply`.
    pub hash: u64,
    /// Remaining flat/wall stones per player (shared pool).
    pub stones: [u8; 2],
    /// Remaining capstones per player.
    pub caps: [u8; 2],
    /// Side to move; White moves first.
    pub turn: Player,
    /// True for the first two plies, in which each player places one of
    /// their *opponent's* stones, flat, on any empty cell.
    pub opening: bool,
}

/// Stones and capstones per player for a board of size `n`.
fn piece_counts(n: usize) -> (u8, u8) {
    match n {
        3 => (10, 0),
        4 => (15, 0),
        5 => (21, 1),
        6 => (30, 1),
        _ => panic!("unsupported Tak board size {n}; supported sizes are 3..=6"),
    }
}

impl<const N: usize> Default for State<N> {
    fn default() -> Self {
        let (stones, caps) = piece_counts(N);
        let hash = reserve_hash(0, stones, caps)
            ^ reserve_hash(1, stones, caps)
            ^ HASHES.hash(HASH_OPENING); // opening = true; White to move contributes 0
        State {
            cells: [0; MAX_CELLS],
            hash,
            stones: [stones, stones],
            caps: [caps, caps],
            turn: Player::White,
            opening: true,
        }
    }
}

impl<const N: usize> State<N> {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn idx(col: i32, row: i32) -> usize {
        row as usize * N + col as usize
    }

    /// Build a stack word directly. Test/display helper; prefer `apply`.
    pub fn set_cell(&mut self, i: usize, w: u64) {
        self.hash ^= cell_hash(i, self.cells[i]) ^ cell_hash(i, w);
        self.cells[i] = w;
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.cells[..N * N].iter().all(|&w| w != 0)
    }

    /// Flat-topped stacks per player (the "flat win" score). Standing
    /// stones, capstones, and covered flats are not counted.
    fn flat_counts(&self) -> [u8; 2] {
        let mut counts = [0u8; 2];
        for &w in &self.cells[..N * N] {
            if w != 0 && cell_kind(w) == FLAT {
                counts[cell_top_color(w) as usize] += 1;
            }
        }
        counts
    }

    /// Bitboard of `color`'s road pieces: cells whose top piece is `color`'s
    /// and not a standing stone (capstones count).
    fn road_mask(&self, color: Player) -> BitBoard<N, N> {
        let mut bits = 0u64;
        for (i, &w) in self.cells[..N * N].iter().enumerate() {
            if w != 0 && cell_kind(w) != WALL && cell_top_color(w) == color as u8 {
                bits |= 1 << i;
            }
        }
        BitBoard::new(bits)
    }

    /// Whether `color` has a road (a top-piece path, flats and capstones
    /// only) connecting either pair of opposite edges.
    pub fn has_road(&self, color: Player) -> bool {
        let mask = self.road_mask(color);
        connects(mask, BitBoard::wall(Direction::North), BitBoard::wall(Direction::South))
            || connects(mask, BitBoard::wall(Direction::East), BitBoard::wall(Direction::West))
    }

    pub fn terminal_status(&self) -> TerminalStatus<Player> {
        // Road check, most recent mover first: if one move completes a road
        // for both players (a "double road" -- e.g. a spread reveals an
        // opponent flat, or a capstone flattens an opponent wall, completing
        // their road), the mover wins. A road for the non-mover can only
        // have been created by the last move, otherwise the game would
        // already have ended, so checking the mover first is exactly the
        // double-road rule.
        let mover = self.turn.next();
        if self.has_road(mover) {
            return TerminalStatus::Winner(mover);
        }
        if self.has_road(self.turn) {
            return TerminalStatus::Winner(self.turn);
        }

        // Flat win: the game ends immediately when the board is full or when
        // either player has played their last piece (stones *and* caps).
        let depleted = (0..2).any(|p| self.stones[p] == 0 && self.caps[p] == 0);
        if depleted || self.is_full() {
            let flats = self.flat_counts();
            return match flats[0].cmp(&flats[1]) {
                std::cmp::Ordering::Greater => TerminalStatus::Winner(Player::White),
                std::cmp::Ordering::Less => TerminalStatus::Winner(Player::Black),
                std::cmp::Ordering::Equal => TerminalStatus::Draw,
            };
        }
        TerminalStatus::NotTerminal
    }

    pub fn moves(&self, out: &mut Vec<Move>) {
        if self.opening {
            // Each player places one of their opponent's stones, flat. (The
            // color is implicit in `opening` + `turn`; see `apply_place`.)
            for i in 0..N * N {
                if self.cells[i] == 0 {
                    out.push(Move::place(i, FLAT));
                }
            }
            return;
        }

        let p = self.turn as usize;
        let has_stones = self.stones[p] > 0;
        let has_caps = self.caps[p] > 0;
        if has_stones || has_caps {
            for (i, &w) in self.cells[..N * N].iter().enumerate() {
                if w == 0 {
                    if has_stones {
                        out.push(Move::place(i, FLAT));
                        out.push(Move::place(i, WALL));
                    }
                    if has_caps {
                        out.push(Move::place(i, CAP));
                    }
                }
            }
        }

        // Spreads: move up to N pieces off the top of any controlled stack.
        let color = self.turn as u8;
        for src in 0..N * N {
            let w = self.cells[src];
            if w == 0 || cell_top_color(w) != color {
                continue;
            }
            let cap_top = cell_kind(w) == CAP;
            let take_max = (cell_height(w) as usize).min(N) as u32;
            let (col, row) = ((src % N) as i32, (src / N) as i32);
            for dir in 0..4 {
                let steps = match dir {
                    0 => N as i32 - 1 - row,
                    1 => N as i32 - 1 - col,
                    2 => row,
                    _ => col,
                };
                if steps <= 0 {
                    continue;
                }
                for take in 1..=take_max {
                    self.gen_drops(out, src, dir, take, cap_top, col, row, steps as usize, 0, 0);
                }
            }
        }
    }

    /// Enumerate the drop schedules for carrying `take` pieces from `src`
    /// along one ray, recursing square by square. `dropped` pieces are
    /// already placed; `mask` accumulates the boundary bits. Path legality:
    /// capstone-topped cells are never enterable; a wall is enterable only
    /// by the lone capstone (which flattens it, ending the move); anything
    /// else (empty or flat-topped, of either color) accepts any drop.
    #[allow(clippy::too_many_arguments)]
    fn gen_drops(
        &self,
        out: &mut Vec<Move>,
        src: usize,
        dir: usize,
        take: u32,
        cap_top: bool,
        col: i32,
        row: i32,
        steps_left: usize,
        dropped: u32,
        mask: u32,
    ) {
        let (dc, dr) = DIRS[dir];
        let (col, row) = (col + dc, row + dr);
        let w = self.cells[Self::idx(col, row)];
        let remaining = take - dropped;
        if w != 0 {
            match cell_kind(w) {
                CAP => return,
                WALL => {
                    if remaining == 1 && cap_top {
                        // The carried stack's top piece is the capstone, and
                        // it is the only piece left: flatten the wall.
                        out.push(Move::spread(src, dir, take, mask | (1 << (take - 1))));
                    }
                    return;
                }
                _ => {}
            }
        }
        for d in 1..=remaining {
            let dropped = dropped + d;
            let mask = mask | (1 << (dropped - 1));
            if dropped == take {
                out.push(Move::spread(src, dir, take, mask));
            } else if steps_left > 1 {
                self.gen_drops(out, src, dir, take, cap_top, col, row, steps_left - 1, dropped, mask);
            }
        }
    }

    #[inline]
    pub fn apply(&mut self, m: &Move) -> Self {
        if m.is_spread() {
            self.apply_spread(m);
        } else {
            self.apply_place(m);
        }
        self.hash ^= HASHES.hash(HASH_TURN);
        self.turn = self.turn.next();
        // The opening phase ends once Black has placed White's stone, i.e.
        // when the turn returns to White.
        if self.opening && self.turn == Player::White {
            self.opening = false;
            self.hash ^= HASHES.hash(HASH_OPENING);
        }
        *self
    }

    fn apply_place(&mut self, m: &Move) {
        let idx = m.square();
        debug_assert!(idx < N * N && self.cells[idx] == 0, "placement on an occupied cell");
        // During the opening each player places an *opponent's* flat.
        let color = if self.opening { self.turn.next() } else { self.turn };
        let k = if self.opening { FLAT } else { m.kind() };
        let p = color as usize;

        let (old_stones, old_caps) = (self.stones[p], self.caps[p]);
        if k == CAP {
            debug_assert!(self.caps[p] > 0);
            self.caps[p] -= 1;
        } else {
            debug_assert!(self.stones[p] > 0);
            self.stones[p] -= 1;
        }

        let w = make_cell(color.bit(), 1, k);
        self.hash ^= cell_hash(idx, w)
            ^ reserve_hash(p, old_stones, old_caps)
            ^ reserve_hash(p, self.stones[p], self.caps[p]);
        self.cells[idx] = w;
    }

    fn apply_spread(&mut self, m: &Move) {
        let src = m.square();
        let take = m.count();
        let w = self.cells[src];
        let h = cell_height(w);
        debug_assert!(w != 0 && cell_top_color(w) == self.turn as u8);
        debug_assert!(take >= 1 && take as usize <= N.min(h as usize));

        // Pop the top `take` pieces: their color bits become the carried
        // stack (LSB = bottom = the piece that lands first). The carried
        // top piece keeps its kind; every other carried piece is flat
        // (walls and capstones can only ever sit on top of a stack).
        let carried_kind = cell_kind(w);
        let carried = (w >> (2 + h - take)) & ((1u64 << take) - 1);
        let new_src = if h == take {
            0
        } else {
            // Lower the sentinel, keep the lower color bits, and mark the
            // newly exposed top piece flat.
            (w & ((1u64 << (2 + h - take)) - 1) & !3u64) | (1u64 << (2 + h - take))
        };
        self.hash ^= cell_hash(src, w) ^ cell_hash(src, new_src);
        self.cells[src] = new_src;

        // Walk the drop schedule: each set bit of `drops` ends one square's
        // drop; group sizes are the gaps between successive set bits.
        let (dc, dr) = DIRS[m.dir()];
        let (mut col, mut row) = ((src % N) as i32, (src / N) as i32);
        let mut bits = m.drops();
        let mut dropped = 0;
        while bits != 0 {
            let p = bits.trailing_zeros();
            bits &= bits - 1;
            let gh = p + 1 - dropped;
            let gcolors = (carried >> dropped) & ((1u64 << gh) - 1);
            let gkind = if p == take - 1 { carried_kind } else { FLAT };
            dropped = p + 1;

            col += dc;
            row += dr;
            let idx = Self::idx(col, row);
            let old = self.cells[idx];
            let mut base = old;
            if gkind == CAP && base != 0 && cell_kind(base) == WALL {
                debug_assert!(gh == 1, "a capstone flattens only by itself");
                base &= !3u64; // flatten: the wall becomes a flat stone
            }
            let dh = if base == 0 { 0 } else { cell_height(base) };
            let dcolors = if base == 0 { 0 } else { (base >> 2) & ((1u64 << dh) - 1) };
            debug_assert!(dh + gh <= 61, "stack height exceeds the cell encoding");
            let new = (1u64 << (2 + dh + gh)) | ((dcolors | (gcolors << dh)) << 2) | gkind as u64;
            self.hash ^= cell_hash(idx, old) ^ cell_hash(idx, new);
            self.cells[idx] = new;
        }
        debug_assert!(dropped == take);
    }

    /// Full from-scratch hash, and a way to re-sync after poking `cells`
    /// directly (only test code that hand-constructs positions should do
    /// this; `set_cell` keeps the hash in sync on its own).
    #[cfg(test)]
    fn recompute_hash(&self) -> u64 {
        let mut h =
            reserve_hash(0, self.stones[0], self.caps[0]) ^ reserve_hash(1, self.stones[1], self.caps[1]);
        if self.turn == Player::Black {
            h ^= HASHES.hash(HASH_TURN);
        }
        if self.opening {
            h ^= HASHES.hash(HASH_OPENING);
        }
        for (i, &w) in self.cells[..N * N].iter().enumerate() {
            h ^= cell_hash(i, w);
        }
        h
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn resync_hash(&mut self) {
        self.hash = self.recompute_hash();
    }
}

/// Does `mask` contain a 4-connected path between border sets `a` and `b`?
/// Flood fill from `a` with early exit on touching `b`.
fn connects<const N: usize>(mask: BitBoard<N, N>, a: BitBoard<N, N>, b: BitBoard<N, N>) -> bool {
    let mut flood = mask & a;
    if flood.is_empty() {
        return false;
    }
    loop {
        if flood.intersects(b) {
            return true;
        }
        let grown = (flood
            | flood.shift_north()
            | flood.shift_east()
            | flood.shift_south()
            | flood.shift_west())
            & mask;
        if grown == flood {
            return false;
        }
        flood = grown;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Game
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct Tak<const N: usize>;

impl<const N: usize> Game for Tak<N> {
    type S = State<N>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<N>, m: &Move) -> State<N> {
        state.apply(m)
    }

    fn generate_actions(state: &State<N>, actions: &mut Vec<Move>) {
        state.moves(actions);
    }

    fn is_terminal(state: &State<N>) -> bool {
        !matches!(Self::terminal_status(state), TerminalStatus::NotTerminal)
    }

    /// Both the road check and the flat-win trigger are answered by one
    /// scan here, so callers that need `is_terminal` + `winner` (e.g. the
    /// end of every rollout) get them for the price of one.
    fn terminal_status(state: &State<N>) -> TerminalStatus<Player> {
        state.terminal_status()
    }

    fn winner(state: &State<N>) -> Option<Player> {
        match Self::terminal_status(state) {
            TerminalStatus::Winner(p) => Some(p),
            _ => None,
        }
    }

    fn player_to_move(state: &State<N>) -> Player {
        state.turn
    }

    fn zobrist_hash(state: &State<N>) -> u64 {
        state.hash
    }

    /// PTN-style notation: placements are e.g. `a1`, `Sa1` (wall), `Ca1`
    /// (capstone); spreads are e.g. `a1>`, `3c3>12` (take 3 from c3 moving
    /// east, dropping 1 then 2). Direction glyphs: `+` north, `-` south,
    /// `>` east, `<` west.
    fn notation(_state: &Self::S, m: &Self::A) -> String {
        let sq = m.square();
        let (col, row) = (sq % N, sq / N);
        let at = format!("{}{}", (b'a' + col as u8) as char, row + 1);
        if !m.is_spread() {
            return format!("{}{}", ["", "S", "C"][m.kind() as usize], at);
        }
        let take = m.count();
        let mut s = String::new();
        if take > 1 {
            s.push_str(&take.to_string());
        }
        s.push_str(&at);
        s.push(['+', '>', '-', '<'][m.dir()]);
        if take > 1 {
            for d in m.drop_sizes() {
                s.push(char::from_digit(d, 10).unwrap());
            }
        }
        s
    }

    fn num_players() -> usize {
        2
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Display
//////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> RectangularBoard for State<N> {
    const NUM_DISPLAY_ROWS: usize = N;
    const NUM_DISPLAY_COLS: usize = N;

    /// Top-down view of stack tops: flats are `o` (white) / `x` (black),
    /// walls `s` / `S`, capstones `c` / `C` (lowercase = white).
    fn display_char_at(&self, row: usize, col: usize) -> char {
        let w = self.cells[Self::idx(col as i32, row as i32)];
        if w == 0 {
            return '.';
        }
        let white = cell_top_color(w) == 0;
        match (cell_kind(w), white) {
            (FLAT, true) => 'o',
            (FLAT, false) => 'x',
            (WALL, true) => 's',
            (WALL, false) => 'S',
            (_, true) => 'c',
            (_, false) => 'C',
        }
    }
}

impl<const N: usize> fmt::Display for State<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)?;
        writeln!(
            f,
            "{:?} to move{} | stones W:{} B:{} | caps W:{} B:{}",
            self.turn,
            if self.opening { " (opening)" } else { "" },
            self.stones[0],
            self.stones[1],
            self.caps[0],
            self.caps[1],
        )
    }
}

#[cfg(test)]
impl<const N: usize> crate::strategies::mcts::render::NodeRender for State<N> {}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn at(col: usize, row: usize, n: usize) -> usize {
        row * n + col
    }

    /// A game position after a standard opening: White placed a black flat
    /// at a1, Black placed a white flat at c5 (5x5 indices). White to move.
    fn opened_state() -> State<5> {
        let mut s = State::<5>::default();
        s = s.apply(&Move::place(at(0, 0, 5), FLAT));
        s = s.apply(&Move::place(at(2, 4, 5), FLAT));
        s
    }

    #[test]
    fn test_initial_state_and_opening_phase() {
        let s = State::<5>::default();
        assert_eq!(s.stones, [21, 21]);
        assert_eq!(s.caps, [1, 1]);
        assert_eq!(s.turn, Player::White);
        assert!(s.opening);
        assert!(matches!(s.terminal_status(), TerminalStatus::NotTerminal));

        // Opening: one flat placement per empty cell, nothing else.
        let mut moves = Vec::new();
        s.moves(&mut moves);
        assert_eq!(moves.len(), 25);
        assert!(moves.iter().all(|m| !m.is_spread() && m.kind() == FLAT));
    }

    #[test]
    fn test_opening_places_opponent_stones() {
        let s = opened_state();
        // White placed a *black* flat at a1, then Black a *white* flat at c5.
        let a1 = s.cells[at(0, 0, 5)];
        assert_eq!(cell_height(a1), 1);
        assert_eq!(cell_kind(a1), FLAT);
        assert_eq!(cell_top_color(a1), Player::Black as u8);
        let c5 = s.cells[at(2, 4, 5)];
        assert_eq!(cell_top_color(c5), Player::White as u8);
        // Both reserves paid for their own stone.
        assert_eq!(s.stones, [20, 20]);
        assert_eq!(s.caps, [1, 1]);
        assert_eq!(s.turn, Player::White);
        assert!(!s.opening);
    }

    #[test]
    fn test_move_count_after_opening() {
        let s = opened_state();
        let mut moves = Vec::new();
        s.moves(&mut moves);
        // 23 empty cells x (flat, wall, cap) + 3 spreads of the single white
        // flat at c5 (N edge is blocked: c5 is on the north edge).
        assert_eq!(moves.len(), 23 * 3 + 3);
    }

    #[test]
    fn test_no_capstones_on_4x4() {
        let mut s = State::<4>::default();
        assert_eq!(s.caps, [0, 0]);
        s.apply(&Move::place(0, FLAT));
        s.apply(&Move::place(1, FLAT));
        let mut moves = Vec::new();
        s.moves(&mut moves);
        assert!(moves.iter().all(|m| m.is_spread() || m.kind() != CAP));
    }

    #[test]
    fn test_single_piece_spreads() {
        let mut s = State::<5>::default();
        s.opening = false;
        s.set_cell(at(2, 2, 5), make_cell(0, 1, FLAT)); // white flat at c3
        let mut moves = Vec::new();
        s.moves(&mut moves);
        let spreads: Vec<_> = moves.iter().filter(|m| m.is_spread()).collect();
        assert_eq!(spreads.len(), 4); // one per direction
        assert!(spreads
            .iter()
            .all(|m| m.count() == 1 && m.drops() == 0b1 && m.square() == at(2, 2, 5)));
    }

    #[test]
    fn test_spread_compositions_and_carry_limit() {
        // A height-7 white stack in the center of a 5x5 board. The carry
        // limit is 5, and from c3 every ray has length 2, so the number of
        // drop schedules for taking c pieces is C(c-1,0) + C(c-1,1) = c
        // (compositions into at most 2 parts): 1+2+3+4+5 = 15 per direction.
        let mut s = State::<5>::default();
        s.opening = false;
        s.set_cell(at(2, 2, 5), make_cell(0, 7, FLAT));
        let mut moves = Vec::new();
        s.moves(&mut moves);
        let spreads: Vec<_> = moves.iter().filter(|m| m.is_spread()).collect();
        assert_eq!(spreads.len(), 4 * 15);
        assert!(spreads.iter().all(|m| m.count() <= 5));

        // A height-3 stack in the corner (a1): rays go north and east, both
        // length 4, so all compositions are available: 1 + 2 + 4 = 7 per
        // direction, 14 total.
        let mut s = State::<5>::default();
        s.opening = false;
        s.set_cell(at(0, 0, 5), make_cell(0, 3, FLAT));
        let mut moves = Vec::new();
        s.moves(&mut moves);
        let spreads: Vec<_> = moves.iter().filter(|m| m.is_spread()).collect();
        assert_eq!(spreads.len(), 2 * 7);
    }

    #[test]
    fn test_wall_blocks_capstone_flattens() {
        // A wall cannot be covered or moved onto, except by a lone capstone.
        let mut s = State::<5>::default();
        s.opening = false;
        s.set_cell(at(0, 0, 5), make_cell(0, 1, FLAT)); // white flat a1
        s.set_cell(at(1, 0, 5), make_cell(1, 1, WALL)); // black wall b1
        let mut moves = Vec::new();
        s.moves(&mut moves);
        let spreads: Vec<_> = moves.iter().filter(|m| m.is_spread()).collect();
        assert_eq!(spreads.len(), 1); // only north; east is walled off
        assert_eq!(spreads[0].direction(), Direction::North);

        // Same setup but with the white capstone at a1: it may move east by
        // itself to flatten the wall.
        let mut s = State::<5>::default();
        s.opening = false;
        s.set_cell(at(0, 0, 5), make_cell(0, 1, CAP));
        s.set_cell(at(1, 0, 5), make_cell(1, 1, WALL));
        let mut moves = Vec::new();
        s.moves(&mut moves);
        let spreads: Vec<_> = moves.iter().copied().filter(|m| m.is_spread()).collect();
        assert_eq!(spreads.len(), 2);
        let flatten = spreads
            .iter()
            .find(|m| m.direction() == Direction::East)
            .expect("a flattening move east should exist");
        s.apply(flatten);
        let b1 = s.cells[at(1, 0, 5)];
        assert_eq!(cell_height(b1), 2);
        assert_eq!(cell_kind(b1), CAP);
        assert_eq!(cell_top_color(b1), Player::White as u8);
        // The flattened wall remains, as a black flat, under the capstone.
        assert_eq!(cell_color_at(b1, 0), Player::Black as u8);
    }

    #[test]
    fn test_capstone_needs_to_be_alone_to_flatten() {
        // Capstone on top of a flat: carrying both onto a wall is illegal.
        let mut s = State::<5>::default();
        s.opening = false;
        // a1: white flat under white capstone (colors bottom-to-top: w, w).
        s.set_cell(at(0, 0, 5), make_cell(0b00, 2, CAP));
        s.set_cell(at(1, 0, 5), make_cell(1, 1, WALL)); // black wall b1
        let mut moves = Vec::new();
        s.moves(&mut moves);
        let spreads: Vec<_> = moves.iter().copied().filter(|m| m.is_spread()).collect();
        // take=1 can flatten (cap alone); take=2 may only go north.
        assert!(spreads
            .iter()
            .any(|m| m.direction() == Direction::East && m.count() == 1));
        assert!(!spreads
            .iter()
            .any(|m| m.direction() == Direction::East && m.count() == 2));

        // But take=2 may drop the flat on b1's... no: the wall can only be
        // entered by the lone cap. take=2 drop (1,1) east means the cap
        // arrives at b1 with one piece still under it -- illegal. take=2 can
        // drop (1,1) only if the second square is beyond the wall: it is
        // not. Verify by applying take=1 east and checking the result.
        let flatten = spreads
            .iter()
            .find(|m| m.direction() == Direction::East && m.count() == 1)
            .unwrap();
        s.apply(flatten);
        // a1 keeps its flat (white), b1 is flattened wall + cap.
        assert_eq!(cell_height(s.cells[at(0, 0, 5)]), 1);
        let b1 = s.cells[at(1, 0, 5)];
        assert_eq!(cell_height(b1), 2);
        assert_eq!(cell_kind(b1), CAP);
    }

    #[test]
    fn test_spread_drop_schedule_from_rulebook() {
        // The rulebook's "Moving a Taller Stack" example: a height-5 stack
        // (white on top, a wall) moves east dropping 2, 2, 1, giving White
        // control of all three landing squares.
        let mut s = State::<5>::default();
        s.opening = false;
        // a2: colors bottom-to-top = black, white, white, white, white-wall.
        s.set_cell(at(0, 1, 5), make_cell(0b00001, 5, WALL));
        let m = Move::spread(at(0, 1, 5), Direction::East as usize, 5, 0b11010);
        assert_eq!(m.drop_sizes(), vec![2, 2, 1]);
        s.apply(&m);

        assert_eq!(s.cells[at(0, 1, 5)], 0, "the whole stack moved");
        for (col, h) in [(1, 2), (2, 2), (3, 1)] {
            let w = s.cells[at(col, 1, 5)];
            assert_eq!(cell_height(w), h);
            assert_eq!(cell_top_color(w), Player::White as u8);
        }
        // The last square received the standing stone, still standing.
        assert_eq!(cell_kind(s.cells[at(3, 1, 5)]), WALL);
        assert_eq!(cell_kind(s.cells[at(1, 1, 5)]), FLAT);
    }

    #[test]
    fn test_road_wins() {
        // White flats along the south edge: an east-west road.
        let mut s = State::<5>::default();
        s.opening = false;
        s.turn = Player::Black; // pretend White just moved
        for col in 0..5 {
            s.set_cell(at(col, 0, 5), make_cell(0, 1, FLAT));
        }
        assert_eq!(s.terminal_status(), TerminalStatus::Winner(Player::White));

        // Black flats along the west edge: a north-south road.
        let mut s = State::<5>::default();
        s.opening = false;
        s.turn = Player::White;
        for row in 0..5 {
            s.set_cell(at(0, row, 5), make_cell(1, 1, FLAT));
        }
        assert_eq!(s.terminal_status(), TerminalStatus::Winner(Player::Black));
    }

    #[test]
    fn test_walls_break_roads_capstones_dont() {
        // A road with one wall in it is not a road.
        let mut s = State::<5>::default();
        s.opening = false;
        s.turn = Player::Black;
        for col in 0..4 {
            s.set_cell(at(col, 0, 5), make_cell(0, 1, FLAT));
        }
        s.set_cell(at(4, 0, 5), make_cell(0, 1, WALL));
        assert!(!s.has_road(Player::White));
        assert_eq!(s.terminal_status(), TerminalStatus::NotTerminal);

        // Replacing the wall with a capstone completes the road.
        s.set_cell(at(4, 0, 5), make_cell(0, 1, CAP));
        assert!(s.has_road(Player::White));
        assert_eq!(s.terminal_status(), TerminalStatus::Winner(Player::White));
    }

    #[test]
    fn test_double_road_mover_wins() {
        // Both players have a road (parallel east-west roads); the player
        // who just moved wins.
        let mut s = State::<5>::default();
        s.opening = false;
        for col in 0..5 {
            s.set_cell(at(col, 0, 5), make_cell(1, 1, FLAT)); // black road, row 1
            s.set_cell(at(col, 1, 5), make_cell(0, 1, FLAT)); // white road, row 2
        }
        s.turn = Player::White; // Black just moved
        assert_eq!(s.terminal_status(), TerminalStatus::Winner(Player::Black));
        s.turn = Player::Black; // White just moved
        assert_eq!(s.terminal_status(), TerminalStatus::Winner(Player::White));
    }

    #[test]
    fn test_flat_win_scoring() {
        // White's reserve is empty: count flat tops. Black has more.
        let mut s = State::<5>::default();
        s.opening = false;
        s.stones[0] = 0;
        s.caps[0] = 0;
        s.set_cell(at(0, 0, 5), make_cell(0, 1, FLAT)); // white flat
        s.set_cell(at(1, 0, 5), make_cell(0, 1, FLAT)); // white flat
        s.set_cell(at(2, 0, 5), make_cell(1, 1, FLAT)); // black flat
        s.set_cell(at(3, 0, 5), make_cell(1, 1, FLAT)); // black flat
        s.set_cell(at(4, 0, 5), make_cell(1, 1, FLAT)); // black flat
        assert_eq!(s.terminal_status(), TerminalStatus::Winner(Player::Black));

        // Equal flat counts are a draw. Walls and capstones don't count.
        s.set_cell(at(4, 0, 5), make_cell(1, 1, WALL)); // black wall: not counted
        assert_eq!(s.terminal_status(), TerminalStatus::Draw);

        // Stones exhausted but a capstone remains: not a flat-win trigger.
        let mut s = State::<5>::default();
        s.opening = false;
        s.stones[0] = 0;
        assert_eq!(s.caps[0], 1);
        assert_eq!(s.terminal_status(), TerminalStatus::NotTerminal);
    }

    #[test]
    fn test_full_board_is_a_flat_win() {
        // Fill the board entirely with standing stones: no flats at all,
        // so 0 - 0, a draw.
        let mut s = State::<3>::default();
        s.opening = false;
        for i in 0..9 {
            s.set_cell(i, make_cell((i % 2) as u64, 1, WALL));
        }
        assert_eq!(s.terminal_status(), TerminalStatus::Draw);
    }

    #[test]
    fn test_flat_win_stacks_count_only_tops() {
        // A tall stack counts once, by its top piece.
        let mut s = State::<5>::default();
        s.opening = false;
        s.stones[1] = 0;
        s.caps[1] = 0; // Black's reserve is empty.
        // One white flat buried under two black flats: 1 point for Black.
        s.set_cell(at(0, 0, 5), make_cell(0b110, 3, FLAT));
        // One white flat elsewhere: 1 point for White. Draw.
        s.set_cell(at(1, 0, 5), make_cell(0, 1, FLAT));
        assert_eq!(s.terminal_status(), TerminalStatus::Draw);
    }

    #[test]
    fn test_placement_ends_game_when_reserve_empties() {
        let mut s = State::<5>::default();
        s.opening = false;
        s.turn = Player::White;
        s.stones[0] = 1;
        s.caps[0] = 0; // one piece left in White's reserve, total
        s.set_cell(at(0, 0, 5), make_cell(1, 1, FLAT)); // one black flat
        let s = s.apply(&Move::place(at(1, 0, 5), FLAT)); // White's last piece
        assert_eq!(s.stones, [0, 21]);
        // 1 flat each: draw.
        assert_eq!(s.terminal_status(), TerminalStatus::Draw);
    }

    #[test]
    #[should_panic]
    fn test_unsupported_size() {
        let _ = State::<7>::default();
    }

    #[test]
    fn test_incremental_hash_and_move_uniqueness() {
        fn check<const N: usize>(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut moves = Vec::new();
            let mut terminals = 0;
            for _game in 0..10 {
                let mut s = State::<N>::default();
                for _ply in 0..300 {
                    if !matches!(s.terminal_status(), TerminalStatus::NotTerminal) {
                        terminals += 1;
                        break;
                    }
                    s.moves(&mut moves);
                    assert!(!moves.is_empty());
                    let unique: HashSet<_> = moves.iter().collect();
                    assert_eq!(unique.len(), moves.len(), "duplicate moves generated");
                    let m = moves[rng.gen_range(0..moves.len())];
                    s.apply(&m);
                    moves.clear();
                    assert_eq!(s.hash, s.recompute_hash(), "incremental hash diverged");
                }
            }
            assert!(terminals > 0, "random games should reach terminal states");
        }
        check::<3>(0xC0FFEE);
        check::<4>(0xDECADE);
        check::<5>(0xF00D);
    }

    #[test]
    fn test_mcts_smoke() {
        use crate::strategies::{
            mcts::{node::QInit, render, strategy, SearchConfig, TreeSearch},
            Search,
        };
        let mut search = TreeSearch::<Tak<3>, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .expand_threshold(1)
                .q_init(QInit::Infinity)
                .use_transpositions(true)
                .max_iterations(20),
        );
        _ = search.choose_action(&State::<3>::default());
        render::render(&search);
    }

    #[test]
    fn test_random_play() {
        use crate::util::random_play;
        random_play::<Tak<3>>();
    }
}
