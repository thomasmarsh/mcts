//! Focus (a.k.a. Domination), designed by Sid Sackson.
//!
//! Played on an 8x8 board with the 3 squares in each corner notched off (52
//! playable cells), by 2, 3, or 4 players. On a turn a player either:
//!
//! - Slides a stack they control (their color on top) orthogonally exactly
//!   as many squares as the stack is tall -- or takes just the top part of
//!   the stack (a "split") and moves that k-piece sub-stack exactly k
//!   squares. The straight-line path (not just the landing cell) must stay
//!   within the 52 valid cells the whole way; nothing about the path being
//!   occupied by other stacks matters -- landing on a stack merges with it.
//! - Or, if they hold reserve pieces (see below), places one of them on any
//!   cell on the board, occupied or not, which merges with whatever is
//!   there exactly like a landing slide would.
//!
//! A merge stacks the mover's pieces on top, preserving each side's
//! internal order. If the resulting stack exceeds 5 pieces, the bottom
//! `n - 5` are removed: pieces matching the *mover's own* color go to that
//! player's reserve (replayable later); any other color is captured --
//! permanently removed from the game.
//!
//! A player with no reserve and no cell where their color is on top has no
//! legal move; they are simply skipped in turn rotation rather than
//! eliminated (merges only ever bury a piece deeper, never expose it, so
//! this is a de facto permanent lockout). The game ends when at most one
//! player still has a legal move; that player, if any, wins.
//!
//! Full board symmetry (D4) exists but isn't implemented here --
//! `canonical_representation`/`apply_to_action`/`invert_action` are left at
//! the `Game` trait default (identity).
//!
//! # Representation
//!
//! - Cells: one `u16` per board index (`row * 8 + col`, row 0 = top), of
//!   which only the 52 valid indices are ever nonzero. Two bits per piece
//!   (player id, up to 4 players), LSB pair = bottom of the stack, packed
//!   from bit 0 up; a sentinel bit at `2 * height` marks the top, so height
//!   is recoverable via `leading_zeros` (mirrors `games::tak`'s cell-word
//!   encoding, scaled down: no "kind" field, max height 5 instead of 61).
//! - `reserves: [u8; P]`: off-board pieces available to place, per player.
//! - `hash`: incremental Zobrist-style hash, kept in sync by `apply`. A
//!   slide touches 2 cells + the mover's reserve (only if it overflows) +
//!   the turn field; a placement touches 1 cell + the placer's reserve +
//!   the turn field.
//!
//! # Moves
//!
//! `Move` is a `u16`: bit 0 = is-slide, bits `[1, 7)` = cell index (0..64),
//! and for a slide only, bits `[7, 9)` = direction (0=N, 1=E, 2=S, 3=W) and
//! bits `[9, 12)` = count (1..=5, the number of pieces taken off the top and
//! the number of squares moved -- always equal, per the split-move rule
//! above).

use mcts::game::{Game, PlayerIndex, TerminalStatus};
use mcts::zobrist::LazyZobristTable;
use serde::{Deserialize, Serialize};
use std::fmt;

pub mod adapter;

/// Maximum stack height (a merge exceeding this immediately overflows).
const MAX_HEIGHT: u32 = 5;

// Direction deltas, indexed 0=N, 1=E, 2=S, 3=W (row 0 = top of the board).
const DIRS: [(i32, i32); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];

//////////////////////////////////////////////////////////////////////////////////////////////////
// Board shape
//////////////////////////////////////////////////////////////////////////////////////////////////

/// Valid column range `[lo, hi]` for a board row, 0-indexed.
const fn row_range(row: usize) -> (usize, usize) {
    match row {
        0 | 7 => (2, 5),
        1 | 6 => (1, 6),
        _ => (0, 7),
    }
}

const fn is_valid_cell(idx: usize) -> bool {
    let row = idx / 8;
    let col = idx % 8;
    let (lo, hi) = row_range(row);
    col >= lo && col <= hi
}

const fn build_valid() -> [bool; 64] {
    let mut v = [false; 64];
    let mut i = 0;
    while i < 64 {
        v[i] = is_valid_cell(i);
        i += 1;
    }
    v
}

/// Which of the 64 `row * 8 + col` indices are playable (52 of them).
const VALID: [bool; 64] = build_valid();

//////////////////////////////////////////////////////////////////////////////////////////////////
// Player
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Player(pub u8);

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        self.0 as usize
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Cell words
//////////////////////////////////////////////////////////////////////////////////////////////////

/// Pack a single-piece stack (or the seed of one): `colors` has one 2-bit
/// field per piece already placed (LSB pair = bottom), `h` the height.
#[inline(always)]
pub const fn make_cell(colors: u16, h: u32) -> u16 {
    debug_assert!(h >= 1 && h <= MAX_HEIGHT);
    (1u16 << (2 * h)) | colors
}

/// Height of the stack in a nonempty cell word.
#[inline(always)]
pub fn cell_height(w: u16) -> u32 {
    debug_assert!(w != 0);
    (15 - w.leading_zeros()) / 2
}

/// Color (player id) of the piece on top of the stack.
#[inline(always)]
pub fn cell_top_color(w: u16) -> u8 {
    let h = cell_height(w);
    ((w >> (2 * (h - 1))) & 3) as u8
}

/// Color of the piece at height `j` (0 = bottom of the stack).
#[inline(always)]
pub fn cell_color_at(w: u16, j: u32) -> u8 {
    ((w >> (2 * j)) & 3) as u8
}

/// Unpacks a cell word's colors, bottom to top, into `buf`. Returns the
/// height. `buf` is sized 10 (not 5) so it can also hold a merge's
/// pre-overflow total (dest height + moving-stack length, each up to 5).
fn stack_into(w: u16, buf: &mut [u8; 10]) -> usize {
    if w == 0 {
        return 0;
    }
    let h = cell_height(w) as usize;
    for (j, slot) in buf.iter_mut().enumerate().take(h) {
        *slot = cell_color_at(w, j as u32);
    }
    h
}

/// Stacks `moving` (bottom to top) onto `dest`, mover's pieces on top, then
/// resolves any overflow past `MAX_HEIGHT`: bottom pieces matching `mover`
/// go back to `*mover_reserve`, any other color is captured (dropped).
/// Returns the resulting (<= `MAX_HEIGHT`-tall) cell word.
fn merge(dest: u16, moving: &[u8], mover: u8, mover_reserve: &mut u8) -> u16 {
    let mut buf = [0u8; 10];
    let dest_h = stack_into(dest, &mut buf);
    let n = moving.len();
    buf[dest_h..dest_h + n].copy_from_slice(moving);
    let total = dest_h + n;
    let overflow = total.saturating_sub(MAX_HEIGHT as usize);
    for &c in &buf[..overflow] {
        if c == mover {
            *mover_reserve += 1;
        }
    }
    let new_h = total - overflow;
    let mut w: u16 = 0;
    for (j, &c) in buf[overflow..overflow + new_h].iter().enumerate() {
        w |= (c as u16) << (2 * j);
    }
    w | (1u16 << (2 * new_h))
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Hashing
//////////////////////////////////////////////////////////////////////////////////////////////////

const HASH_RESERVE: usize = 64; // .. + player index (reserves up to 4 players)
const HASH_TURN: usize = 68; // .. + player index
const HASHES_LEN: usize = 72;

static HASHES: LazyZobristTable<HASHES_LEN> = LazyZobristTable::new(0xF0C5);

/// splitmix64 finalizer: a cheap bijective 64-bit mix, salted per cell/
/// reserve slot so a value change only needs an XOR-out/XOR-in of two mixes.
#[inline(always)]
const fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[inline(always)]
fn cell_hash(i: usize, w: u16) -> u64 {
    if w == 0 {
        0
    } else {
        mix((w as u64) ^ HASHES.hash(i))
    }
}

#[inline(always)]
fn reserve_hash(p: usize, count: u8) -> u64 {
    mix((count as u64) ^ HASHES.hash(HASH_RESERVE + p))
}

#[inline(always)]
fn turn_hash(p: usize) -> u64 {
    HASHES.hash(HASH_TURN + p)
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Moves
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Move(u16);

impl Move {
    /// Place a reserve piece on (any) cell `c`.
    #[inline(always)]
    pub fn place(c: usize) -> Move {
        debug_assert!(c < 64);
        Move((c as u16) << 1)
    }

    /// Slide the top `count` pieces of the stack at `c` exactly `count`
    /// squares in direction `dir` (0=N, 1=E, 2=S, 3=W).
    #[inline(always)]
    pub fn slide(c: usize, dir: usize, count: u32) -> Move {
        debug_assert!(c < 64 && dir < 4 && (1..=MAX_HEIGHT).contains(&count));
        Move(1 | ((c as u16) << 1) | ((dir as u16) << 7) | ((count as u16) << 9))
    }

    #[inline(always)]
    pub fn is_slide(self) -> bool {
        self.0 & 1 != 0
    }

    /// Placement cell, or the source cell of a slide.
    #[inline(always)]
    pub fn cell(self) -> usize {
        ((self.0 >> 1) & 63) as usize
    }

    #[inline(always)]
    pub fn dir(self) -> usize {
        ((self.0 >> 7) & 3) as usize
    }

    #[inline(always)]
    pub fn count(self) -> u32 {
        ((self.0 >> 9) & 7) as u32
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Starting layouts (row, col, player), pixel-extracted from the primary
// source and verified by per-player piece-count totals.
//////////////////////////////////////////////////////////////////////////////////////////////////

const LAYOUT2: [(usize, usize, u8); 36] = [
    (1, 1, 0),
    (1, 2, 0),
    (1, 3, 1),
    (1, 4, 1),
    (1, 5, 0),
    (1, 6, 0),
    (2, 1, 1),
    (2, 2, 1),
    (2, 3, 0),
    (2, 4, 0),
    (2, 5, 1),
    (2, 6, 1),
    (3, 1, 0),
    (3, 2, 0),
    (3, 3, 1),
    (3, 4, 1),
    (3, 5, 0),
    (3, 6, 0),
    (4, 1, 1),
    (4, 2, 1),
    (4, 3, 0),
    (4, 4, 0),
    (4, 5, 1),
    (4, 6, 1),
    (5, 1, 0),
    (5, 2, 0),
    (5, 3, 1),
    (5, 4, 1),
    (5, 5, 0),
    (5, 6, 0),
    (6, 1, 1),
    (6, 2, 1),
    (6, 3, 0),
    (6, 4, 0),
    (6, 5, 1),
    (6, 6, 1),
];

const LAYOUT3: [(usize, usize, u8); 36] = [
    (1, 1, 0),
    (1, 2, 0),
    (1, 3, 1),
    (1, 4, 1),
    (1, 5, 2),
    (1, 6, 2),
    (2, 1, 2),
    (2, 2, 2),
    (2, 3, 0),
    (2, 4, 0),
    (2, 5, 1),
    (2, 6, 1),
    (3, 1, 1),
    (3, 2, 1),
    (3, 3, 2),
    (3, 4, 2),
    (3, 5, 0),
    (3, 6, 0),
    (4, 1, 0),
    (4, 2, 0),
    (4, 3, 1),
    (4, 4, 1),
    (4, 5, 2),
    (4, 6, 2),
    (5, 1, 2),
    (5, 2, 2),
    (5, 3, 0),
    (5, 4, 0),
    (5, 5, 1),
    (5, 6, 1),
    (6, 1, 1),
    (6, 2, 1),
    (6, 3, 2),
    (6, 4, 2),
    (6, 5, 0),
    (6, 6, 0),
];

const LAYOUT4: [(usize, usize, u8); 52] = [
    (0, 2, 0),
    (0, 3, 0),
    (0, 4, 2),
    (0, 5, 3),
    (1, 1, 3),
    (1, 2, 3),
    (1, 3, 3),
    (1, 4, 2),
    (1, 5, 3),
    (1, 6, 2),
    (2, 0, 0),
    (2, 1, 0),
    (2, 2, 0),
    (2, 3, 0),
    (2, 4, 2),
    (2, 5, 3),
    (2, 6, 2),
    (2, 7, 3),
    (3, 0, 3),
    (3, 1, 3),
    (3, 2, 3),
    (3, 3, 3),
    (3, 4, 2),
    (3, 5, 3),
    (3, 6, 2),
    (3, 7, 3),
    (4, 0, 1),
    (4, 1, 0),
    (4, 2, 1),
    (4, 3, 0),
    (4, 4, 1),
    (4, 5, 1),
    (4, 6, 1),
    (4, 7, 1),
    (5, 0, 1),
    (5, 1, 0),
    (5, 2, 1),
    (5, 3, 0),
    (5, 4, 2),
    (5, 5, 2),
    (5, 6, 2),
    (5, 7, 2),
    (6, 1, 0),
    (6, 2, 1),
    (6, 3, 0),
    (6, 4, 1),
    (6, 5, 1),
    (6, 6, 1),
    (7, 2, 1),
    (7, 3, 0),
    (7, 4, 2),
    (7, 5, 2),
];

//////////////////////////////////////////////////////////////////////////////////////////////////
// State
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State<const P: usize> {
    /// One packed stack word per board index (`row * 8 + col`); only the 52
    /// valid indices are ever nonzero.
    pub cells: [u16; 64],
    /// Off-board pieces available to place, per player.
    pub reserves: [u8; P],
    /// Player to move. Never a player with no legal move (see `apply`).
    pub turn: Player,
    /// Incremental hash (see the file header). Kept in sync by `apply`.
    pub hash: u64,
}

impl<const P: usize> Default for State<P> {
    fn default() -> Self {
        const {
            assert!(
                P == 2 || P == 3 || P == 4,
                "Focus supports 2, 3, or 4 players"
            )
        };
        let mut cells = [0u16; 64];
        match P {
            2 => {
                for &(row, col, pl) in LAYOUT2.iter() {
                    cells[row * 8 + col] = make_cell(pl as u16, 1);
                }
            }
            3 => {
                for &(row, col, pl) in LAYOUT3.iter() {
                    cells[row * 8 + col] = make_cell(pl as u16, 1);
                }
            }
            4 => {
                for &(row, col, pl) in LAYOUT4.iter() {
                    cells[row * 8 + col] = make_cell(pl as u16, 1);
                }
            }
            _ => unreachable!(),
        }
        let mut state = State {
            cells,
            reserves: [0; P],
            turn: Player(0),
            hash: 0,
        };
        state.hash = state.recompute_hash();
        state
    }
}

impl<const P: usize> State<P> {
    /// Sets a raw cell word directly, keeping the hash in sync. Test/setup
    /// helper for hand-building positions; prefer `apply` otherwise.
    pub fn set_cell(&mut self, i: usize, w: u16) {
        self.hash ^= cell_hash(i, self.cells[i]) ^ cell_hash(i, w);
        self.cells[i] = w;
    }

    /// Sets a player's reserve count directly, keeping the hash in sync.
    pub fn set_reserve(&mut self, p: usize, n: u8) {
        self.hash ^= reserve_hash(p, self.reserves[p]) ^ reserve_hash(p, n);
        self.reserves[p] = n;
    }

    /// Sets the player to move directly, keeping the hash in sync.
    pub fn set_turn(&mut self, p: usize) {
        self.hash ^= turn_hash(self.turn.0 as usize) ^ turn_hash(p);
        self.turn = Player(p as u8);
    }

    /// Whether player `p` has any legal move: a reserve piece (always
    /// placeable somewhere, since every valid cell accepts a placement), or
    /// a cell with `p`'s color on top and at least one in-bounds orthogonal
    /// neighbor to slide onto (checked directly rather than assumed, even
    /// though the 52-cell board happens to be fully connected).
    fn player_has_legal_move(&self, p: usize) -> bool {
        if self.reserves[p] > 0 {
            return true;
        }
        for i in 0..64 {
            let w = self.cells[i];
            if w == 0 || cell_top_color(w) != p as u8 {
                continue;
            }
            let (row, col) = (i / 8, i % 8);
            for &(dr, dc) in DIRS.iter() {
                let r = row as i32 + dr;
                let c = col as i32 + dc;
                if (0..8).contains(&r)
                    && (0..8).contains(&c)
                    && VALID[(r as usize) * 8 + c as usize]
                {
                    return true;
                }
            }
        }
        false
    }

    /// The outcome of the game: `NotTerminal` while 2+ players still have a
    /// legal move, `Winner` when exactly one does, `Draw` if (defensively)
    /// none do.
    pub fn terminal_status(&self) -> TerminalStatus<Player> {
        let mut capable = None;
        let mut count = 0;
        for p in 0..P {
            if self.player_has_legal_move(p) {
                count += 1;
                capable = Some(p);
            }
        }
        match count {
            0 => TerminalStatus::Draw,
            1 => TerminalStatus::Winner(Player(capable.unwrap() as u8)),
            _ => TerminalStatus::NotTerminal,
        }
    }

    pub fn moves(&self, out: &mut Vec<Move>) {
        let p = self.turn.0;
        if self.reserves[p as usize] > 0 {
            for (i, &valid) in VALID.iter().enumerate() {
                if valid {
                    out.push(Move::place(i));
                }
            }
        }
        for i in 0..64 {
            let w = self.cells[i];
            if w == 0 || cell_top_color(w) != p {
                continue;
            }
            let h = cell_height(w);
            let (row, col) = (i / 8, i % 8);
            for (dir, &(dr, dc)) in DIRS.iter().enumerate() {
                for count in 1..=h {
                    let r = row as i32 + dr * count as i32;
                    let c = col as i32 + dc * count as i32;
                    if !(0..8).contains(&r)
                        || !(0..8).contains(&c)
                        || !VALID[(r as usize) * 8 + c as usize]
                    {
                        break; // longer slides in this direction are also off-board
                    }
                    out.push(Move::slide(i, dir, count));
                }
            }
        }
    }

    fn apply_place(&mut self, m: Move) {
        let dest = m.cell();
        let p = self.turn.0 as usize;
        debug_assert!(self.reserves[p] > 0);
        let old_reserve = self.reserves[p];
        self.reserves[p] -= 1;
        let old_dest = self.cells[dest];
        let new_dest = merge(old_dest, &[self.turn.0], self.turn.0, &mut self.reserves[p]);
        self.hash ^= cell_hash(dest, old_dest)
            ^ cell_hash(dest, new_dest)
            ^ reserve_hash(p, old_reserve)
            ^ reserve_hash(p, self.reserves[p]);
        self.cells[dest] = new_dest;
    }

    fn apply_slide(&mut self, m: Move) {
        let src = m.cell();
        let count = m.count() as usize;
        let dir = m.dir();
        let w = self.cells[src];
        let h = cell_height(w) as usize;
        debug_assert!(w != 0 && cell_top_color(w) == self.turn.0);
        debug_assert!(count >= 1 && count <= h);

        let mut buf = [0u8; 10];
        stack_into(w, &mut buf);
        let mut moving = [0u8; MAX_HEIGHT as usize];
        moving[..count].copy_from_slice(&buf[h - count..h]);

        let new_src_h = h - count;
        let new_src = if new_src_h == 0 {
            0
        } else {
            let mut nw: u16 = 0;
            for (j, &c) in buf[..new_src_h].iter().enumerate() {
                nw |= (c as u16) << (2 * j);
            }
            nw | (1u16 << (2 * new_src_h))
        };
        self.hash ^= cell_hash(src, w) ^ cell_hash(src, new_src);
        self.cells[src] = new_src;

        let (dr, dc) = DIRS[dir];
        let (row, col) = (src / 8, src % 8);
        let dest_row = (row as i32 + dr * count as i32) as usize;
        let dest_col = (col as i32 + dc * count as i32) as usize;
        let dest = dest_row * 8 + dest_col;

        let mover = self.turn.0;
        let p = mover as usize;
        let old_reserve = self.reserves[p];
        let old_dest = self.cells[dest];
        let new_dest = merge(old_dest, &moving[..count], mover, &mut self.reserves[p]);
        self.hash ^= cell_hash(dest, old_dest)
            ^ cell_hash(dest, new_dest)
            ^ reserve_hash(p, old_reserve)
            ^ reserve_hash(p, self.reserves[p]);
        self.cells[dest] = new_dest;
    }

    fn advance_turn(&mut self) {
        let old = self.turn.0 as usize;
        let mut next = (old + 1) % P;
        while !self.player_has_legal_move(next) {
            next = (next + 1) % P;
        }
        self.hash ^= turn_hash(old) ^ turn_hash(next);
        self.turn = Player(next as u8);
    }

    #[inline]
    pub fn apply(&mut self, m: &Move) -> Self {
        if m.is_slide() {
            self.apply_slide(*m);
        } else {
            self.apply_place(*m);
        }
        self.advance_turn();
        *self
    }

    /// Full from-scratch hash, and a way to re-sync after poking `cells`/
    /// `reserves`/`turn` directly instead of through `set_cell`/etc.
    pub fn recompute_hash(&self) -> u64 {
        let mut h = 0u64;
        for (i, &w) in self.cells.iter().enumerate() {
            h ^= cell_hash(i, w);
        }
        for p in 0..P {
            h ^= reserve_hash(p, self.reserves[p]);
        }
        h ^= turn_hash(self.turn.0 as usize);
        h
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Display
//////////////////////////////////////////////////////////////////////////////////////////////////

const PLAYER_CHARS: [char; 4] = ['A', 'B', 'C', 'D'];

impl<const P: usize> fmt::Display for State<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..8 {
            for col in 0..8 {
                let i = row * 8 + col;
                if !VALID[i] {
                    write!(f, "   ")?;
                    continue;
                }
                let w = self.cells[i];
                if w == 0 {
                    write!(f, " . ")?;
                } else {
                    write!(
                        f,
                        "{}{} ",
                        PLAYER_CHARS[cell_top_color(w) as usize],
                        cell_height(w)
                    )?;
                }
            }
            writeln!(f)?;
        }
        write!(f, "P{} to move | reserves: ", self.turn.0)?;
        for (&ch, &r) in PLAYER_CHARS.iter().zip(self.reserves.iter()) {
            write!(f, "{}={} ", ch, r)?;
        }
        writeln!(f)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Game
//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct Focus<const P: usize>;

pub type Focus2 = Focus<2>;
pub type Focus3 = Focus<3>;
pub type Focus4 = Focus<4>;

impl<const P: usize> Game for Focus<P> {
    type S = State<P>;
    type A = Move;
    type P = Player;

    fn apply(mut state: State<P>, m: &Move) -> State<P> {
        state.apply(m)
    }

    fn generate_actions(state: &State<P>, actions: &mut Vec<Move>) {
        state.moves(actions);
    }

    fn is_terminal(state: &State<P>) -> bool {
        !matches!(Self::terminal_status(state), TerminalStatus::NotTerminal)
    }

    /// Both `is_terminal` and `winner` are answered by one capable-player
    /// scan here, so callers that need both (e.g. the end of every rollout)
    /// get them for the price of one.
    fn terminal_status(state: &State<P>) -> TerminalStatus<Player> {
        state.terminal_status()
    }

    fn winner(state: &State<P>) -> Option<Player> {
        match Self::terminal_status(state) {
            TerminalStatus::Winner(p) => Some(p),
            _ => None,
        }
    }

    fn player_to_move(state: &State<P>) -> Player {
        state.turn
    }

    fn zobrist_hash(state: &State<P>) -> u64 {
        state.hash
    }

    fn num_players() -> usize {
        P
    }

    fn notation(_state: &State<P>, m: &Move) -> String {
        let cell = m.cell();
        let (row, col) = (cell / 8, cell % 8);
        let at = format!("{}{}", (b'a' + col as u8) as char, row + 1);
        if m.is_slide() {
            format!("{}{}{}", at, ["N", "E", "S", "W"][m.dir()], m.count())
        } else {
            format!("place@{}", at)
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    fn piece_counts<const P: usize>() -> [u32; P] {
        let s = State::<P>::default();
        let mut counts = [0u32; P];
        for &w in s.cells.iter() {
            if w != 0 {
                counts[cell_top_color(w) as usize] += 1;
            }
        }
        counts
    }

    #[test]
    fn starting_layout_piece_counts() {
        assert_eq!(piece_counts::<2>(), [18, 18]);
        assert_eq!(piece_counts::<3>(), [12, 12, 12]);
        assert_eq!(piece_counts::<4>(), [13, 13, 13, 13]);
    }

    #[test]
    fn valid_cell_count_is_52() {
        assert_eq!(VALID.iter().filter(|&&v| v).count(), 52);
    }

    #[test]
    fn bounded_random_playouts_2_3_4p() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        fn check<const P: usize>(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut state = State::<P>::default();
            let mut actions = Vec::new();
            let mut terminated = false;
            for _ in 0..400 {
                if !matches!(state.terminal_status(), TerminalStatus::NotTerminal) {
                    terminated = true;
                    break;
                }
                actions.clear();
                state.moves(&mut actions);
                assert!(!actions.is_empty(), "current player must have a legal move");
                let m = actions[rng.gen_range(0..actions.len())];
                state.apply(&m);
                assert_eq!(
                    state.hash,
                    state.recompute_hash(),
                    "incremental hash diverged"
                );
            }
            // Not asserting `terminated`: Focus games can run long, and this is a
            // smoke test for panics/invariants, not a termination proof.
            let _ = terminated;
        }
        check::<2>(1);
        check::<3>(2);
        check::<4>(3);
    }

    #[test]
    fn slide_moves_stack_and_shortens_source() {
        let mut s = State::<2>::default();
        // Clear the board and hand-place a height-2 white (P0) stack: a
        // stack of height h must move exactly h squares (the non-split
        // case), so this moves the whole stack 2 squares east.
        for i in 0..64 {
            s.set_cell(i, 0);
        }
        let src = 3 * 8 + 3; // (row 3, col 3), interior cell
        s.set_cell(src, make_cell(0, 2));
        let dest = 3 * 8 + 5; // 2 squares east
        let m = Move::slide(src, 1 /* E */, 2);
        s.apply(&m);
        assert_eq!(s.cells[src], 0, "source is now empty");
        let w = s.cells[dest];
        assert_eq!(cell_height(w), 2);
        assert_eq!(cell_top_color(w), 0);
    }

    #[test]
    fn split_move_leaves_remainder_at_source() {
        let mut s = State::<2>::default();
        for i in 0..64 {
            s.set_cell(i, 0);
        }
        let src = 3 * 8 + 3;
        // Height-3 white stack: bottom to top colors [0, 0, 0].
        s.set_cell(src, make_cell(0b000000, 3));
        let m = Move::slide(src, 1 /* E */, 2); // split: move top 2, leave 1
        s.apply(&m);
        let remainder = s.cells[src];
        assert_eq!(cell_height(remainder), 1);
        let dest = 3 * 8 + 5;
        assert_eq!(cell_height(s.cells[dest]), 2);
    }

    #[test]
    fn overflow_returns_own_color_and_captures_other_colors() {
        let mut reserve = 0u8;
        // Dest already 5 tall: bottom to top = [1, 1, 0, 0, 0] (colors 1,1,0,0,0).
        let dest = make_cell(0b00_00_00_01_01, 5);
        // Mover (color 0) merges 2 more of their own pieces on top.
        let result = merge(dest, &[0, 0], 0, &mut reserve);
        // Total height 7, overflow 2: bottom two removed are colors [1, 1]
        // (mover is 0), so both are captured, not returned to any reserve.
        assert_eq!(reserve, 0);
        assert_eq!(cell_height(result), 5);
        // Surviving stack, bottom to top: [0, 0, 0, 0, 0] (all mover's).
        for j in 0..5 {
            assert_eq!(cell_color_at(result, j), 0);
        }

        // Same setup, but the mover's own color is what gets buried: a
        // height-5 stack of all-mover pieces overflowed by an opponent's
        // 2-piece merge returns the buried mover pieces to *the opponent's*
        // reserve only if they match the opponent -- here mover is 1 and the
        // buried pieces are 0s, so they are captured, not returned.
        let mut reserve1 = 0u8;
        let dest2 = make_cell(0, 5); // all-zero stack, height 5
        let result2 = merge(dest2, &[1, 1], 1, &mut reserve1);
        assert_eq!(
            reserve1, 0,
            "buried pieces are color 0, not mover's color 1"
        );
        assert_eq!(cell_height(result2), 5);

        // Overflow of the mover's own buried pieces does return to reserve.
        let mut reserve2 = 0u8;
        let dest3 = make_cell(0, 5); // all mover-colored (0) stack, height 5
        let result3 = merge(dest3, &[0, 0], 0, &mut reserve2);
        assert_eq!(reserve2, 2, "both buried pieces are the mover's own color");
        assert_eq!(cell_height(result3), 5);
    }

    #[test]
    fn reserve_placement_on_empty_and_overflowing_stack() {
        let mut s = State::<2>::default();
        for i in 0..64 {
            s.set_cell(i, 0);
        }
        s.set_reserve(0, 2);
        s.set_turn(0);

        let empty_cell = 3 * 8 + 3;
        s.apply(&Move::place(empty_cell));
        assert_eq!(cell_height(s.cells[empty_cell]), 1);
        assert_eq!(s.reserves[0], 1);

        // Now overflow: place onto an already-5-tall stack of a different color.
        s.set_turn(0);
        let tall_cell = 3 * 8 + 4;
        s.set_cell(tall_cell, make_cell(0b01_01_01_01_01, 5)); // all color 1
        s.apply(&Move::place(tall_cell));
        assert_eq!(s.reserves[0], 0, "placement spent the last reserve piece");
        let w = s.cells[tall_cell];
        assert_eq!(cell_height(w), 5);
        assert_eq!(cell_top_color(w), 0, "mover's piece is on top");
    }

    #[test]
    fn path_through_a_corner_notch_is_excluded() {
        let mut s = State::<2>::default();
        for i in 0..64 {
            s.set_cell(i, 0);
        }
        // (row 1, col 1) is valid; straight north 1 square lands at (row 0,
        // col 1), which is *not* valid (row 0 only has cols 2..=5) -- so
        // that slide must not be generated.
        let src = 8 + 1;
        s.set_cell(src, make_cell(0, 1));
        s.set_turn(0);
        let mut actions = Vec::new();
        s.moves(&mut actions);
        assert!(
            !actions.iter().any(|m| m.is_slide() && m.dir() == 0),
            "north slide off the notched corner should be illegal"
        );
        // East (into the board interior) should still be legal.
        assert!(actions.iter().any(|m| m.is_slide() && m.dir() == 1));
    }

    #[test]
    fn stuck_player_is_skipped_and_lone_capable_player_wins() {
        let mut s = State::<3>::default();
        for i in 0..64 {
            s.set_cell(i, 0);
        }
        s.set_reserve(0, 0);
        s.set_reserve(1, 0);
        s.set_reserve(2, 1);
        // Only player 2 has anything on the board or in reserve.
        s.set_turn(0);
        assert_eq!(s.terminal_status(), TerminalStatus::Winner(Player(2)));

        // Give player 1 a lone piece too: no longer terminal, and applying a
        // move from player 2 must skip stuck player 0 and land on player 1.
        let cell = 3 * 8 + 3;
        s.set_cell(cell, make_cell(1, 1));
        assert_eq!(s.terminal_status(), TerminalStatus::NotTerminal);
        s.set_turn(2);
        s.apply(&Move::place(4 * 8 + 4));
        assert_eq!(s.turn, Player(1), "player 0 has no move and is skipped");
    }

    #[test]
    fn incremental_hash_matches_recompute_across_move_sequences() {
        let mut a = State::<2>::default();
        let mut b = State::<2>::default();
        // Two different orderings of the same two moves should reach an
        // identical hash (same final board/reserves/turn).
        let m1 = {
            let mut actions = Vec::new();
            a.moves(&mut actions);
            actions[0]
        };
        a.apply(&m1);
        let m2 = {
            let mut actions = Vec::new();
            a.moves(&mut actions);
            actions[0]
        };
        a.apply(&m2);
        assert_eq!(a.hash, a.recompute_hash());

        b.apply(&m1);
        b.apply(&m2);
        assert_eq!(a.hash, b.hash);
        assert_eq!(a, b);
    }

    // MCTS-Solver on a real 3-player game (Nijssen & Winands, CG 2010's own
    // benchmark game): direct regression coverage that
    // `use_mcts_solver(true)` is actually usable for `Focus3` now that
    // `SearchConfig::validate()` no longer rejects `num_players() > 2`, and
    // that the search integrates end to end (finds the forced win, stops
    // early) rather than just type-checking. Mirrors `games/ttt`'s
    // `test_mcts_solver_finds_forced_block_and_terminates_early`.
    //
    // Hand-built position, player 0 to move: sliding the piece at (3,2)
    // one square east onto (3,3) buries player 1's only piece (0 reserve,
    // no other cell) -- player 1 becomes immobile, player 2 was already
    // immobile (no pieces, no reserve, rigged that way from the start), and
    // player 0 keeps (3,3) itself mobile (a slide always leaves the mover
    // on top of its destination -- see `backprop::derive_proven`'s doc
    // comment), so this one move is an immediate win for player 0. Player
    // 0's *other* piece at (0,3) has three harmless alternative slides that
    // don't end the game, so the solver actually has to distinguish the
    // winning move from real alternatives, not just play the only legal one.
    #[test]
    fn mcts_solver_finds_forced_win_on_three_players_and_terminates_early() {
        use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
        use mcts::strategies::Search;

        let mut state = State::<3>::default();
        for i in 0..64 {
            state.set_cell(i, 0);
        }
        for p in 0..3 {
            state.set_reserve(p, 0);
        }
        state.set_cell(3 * 8 + 2, make_cell(0, 1)); // player 0, about to slide east
        state.set_cell(3, make_cell(0, 1)); // player 0, harmless alternative piece (row0, col3)
        state.set_cell(3 * 8 + 3, make_cell(1, 1)); // player 1's only piece -- about to be buried
        state.set_turn(0);
        // Player 2 has zero pieces and zero reserve: already permanently
        // immobile, exactly as if eliminated earlier in a real game.
        assert_eq!(state.terminal_status(), TerminalStatus::NotTerminal);

        let winning_move = Move::slide(3 * 8 + 2, 1, 1); // dir 1 = east, per `DIRS`
        let mut actions = Vec::new();
        state.moves(&mut actions);
        assert!(
            actions.contains(&winning_move),
            "the rigged position must actually offer the intended winning move"
        );
        assert_eq!(actions.len(), 7, "4 slides from (3,2) + 3 from (0,3)");

        type TS = TreeSearch<Focus3, strategy::Ucb1>;

        let mut solved = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(2000)
                .q_init(QInit::Loss)
                .use_mcts_solver(true)
                .seed(42),
        );
        let action = solved.choose_action(&state);
        assert_eq!(action, winning_move);
        let mut after = state;
        after.apply(&action);
        assert_eq!(after.terminal_status(), TerminalStatus::Winner(Player(0)));
        let solved_iters = solved
            .stats
            .iter_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            solved_iters < 2000,
            "solver should stop once the root is proven, used {solved_iters} iterations"
        );

        let mut unsolved = TS::default().config(
            SearchConfig::default()
                .expand_threshold(0)
                .max_iterations(2000)
                .q_init(QInit::Loss)
                .seed(42),
        );
        let action = unsolved.choose_action(&state);
        assert_eq!(action, winning_move);
        let unsolved_iters = unsolved
            .stats
            .iter_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            unsolved_iters, 2000,
            "without the solver, the full iteration budget should still run"
        );
    }
}
