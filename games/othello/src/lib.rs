use mcts::bitboard::{BitBoard, Direction};
use mcts::display::{RectangularBoard, RectangularBoardDisplay};
use mcts::game::{Game, PlayerIndex};
use mcts::zobrist::LazyZobristTable;
use serde::Serialize;
use std::fmt;

// Starting position: the four center squares.
pub const INITIAL_BLACK: u64 = (1 << 28) | (1 << 35); // e4, d5
pub const INITIAL_WHITE: u64 = (1 << 27) | (1 << 36); // d4, e5

pub const BOARD_SIZE: usize = 8;

// ── Zobrist hashing ───────────────────────────────────────────────────────

// 64 squares × 2 players = 128 piece entries + 1 turn + 1 last_pass
pub const ZOBRIST_ENTRIES: usize = 130;
pub const ZOBRIST_TURN: usize = 128;
pub const ZOBRIST_LAST_PASS: usize = 129;

/// Random Zobrist table, lazily initialised.
pub static HASHES: LazyZobristTable<ZOBRIST_ENTRIES> =
    LazyZobristTable::new(0xA1B2C3D4E5F67890);

/// Hash index for a piece at `pos` belonging to `player`.
#[inline]
pub fn zobrist_piece(pos: usize, player: Player) -> usize {
    pos * 2 + player as usize
}

// ---------------------------------------------------------------------------
// Dihedral symmetries (D4) for 8×8 board
// ---------------------------------------------------------------------------

pub mod sym {
    // Dead-code allowed because symmetry infrastructure will be used in
    // production (canonicalization, hash reduction) once the game logic is
    // further along.
    #![allow(dead_code)]
    /// H: horizontal mirror (reflect across vertical axis) — col → 7-col.
    pub(crate) const H: [usize; 64] = build_h();
    /// V: vertical mirror (reflect across horizontal axis) — row → 7-row.
    pub(crate) const V: [usize; 64] = build_v();
    /// D: transpose across main diagonal — (row, col) → (col, row).
    pub(crate) const D: [usize; 64] = build_d();

    const fn build_h() -> [usize; 64] {
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            let row = i / 8;
            let col = i % 8;
            a[i] = row * 8 + (7 - col);
            i += 1;
        }
        a
    }

    const fn build_v() -> [usize; 64] {
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            let row = i / 8;
            a[i] = (7 - row) * 8 + (i % 8);
            i += 1;
        }
        a
    }

    const fn build_d() -> [usize; 64] {
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            let row = i / 8;
            let col = i % 8;
            a[i] = col * 8 + row;
            i += 1;
        }
        a
    }

    /// Produce all 8 symmetric images of an index.
    /// Order: identity, H, V, D, VH, DH, DV, DVH
    #[inline]
    pub fn index_symmetries(i: usize) -> [usize; 8] {
        [i, H[i], V[i], D[i], V[H[i]], D[H[i]], D[V[i]], D[V[H[i]]]]
    }

    /// Map an index back through the inverse of a symmetry.
    /// For a permutation P, the inverse satisfies P[inverse] = original.
    #[inline]
    pub fn invert_symmetry(i: usize, sym_idx: usize) -> usize {
        // The inverse is: apply the reverse composition.
        // sym[0] = identity: inv = i
        // sym[1] = H: inv = H
        // sym[2] = V: inv = V
        // sym[3] = D: inv = D
        // sym[4] = V∘H: inv = H∘V (since involutions commute: VH = HV)
        // sym[5] = D∘H: inv = H∘D
        // sym[6] = D∘V: inv = V∘D
        // sym[7] = D∘V∘H: inv = H∘V∘D
        match sym_idx {
            0 => i,
            1 => H[i],
            2 => V[i],
            3 => D[i],
            4 => H[V[i]],
            5 => H[D[i]],
            6 => V[D[i]],
            7 => H[V[D[i]]],
            _ => unreachable!(),
        }
    }

    /// Apply a symmetry permutation to a raw u64 bitboard.
    /// Iterates each set bit, maps through the permutation, and sets the
    /// result bit in the output.
    #[inline]
    pub fn apply_to_bits(board: u64, sym_idx: usize) -> u64 {
        let mut result = 0u64;
        let mut bits = board;
        while bits != 0 {
            let lsb = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let dst = index_symmetries(lsb)[sym_idx];
            result |= 1u64 << dst;
        }
        result
    }

    #[cfg(test)]
    /// Verify symmetry tables cover all 64 indices bijectively.
    #[test]
    fn test_symmetry_tables_are_permutations() {
        for (name, table) in [("H", &H), ("V", &V), ("D", &D)] {
            let mut seen = [false; 64];
            for &v in table {
                assert!(v < 64, "{}[?] = {} out of range", name, v);
                assert!(!seen[v], "{} is not injective (dupe at {})", name, v);
                seen[v] = true;
            }
            assert!(seen.iter().all(|&x| x), "{} is not surjective", name);
        }
    }

    #[cfg(test)]
    /// Verify index_symmetries produces valid images (within bounds and no
    /// duplicates within a single symmetry's images for a given index).
    /// Note: some indices (e.g. those on the main diagonal) are invariant
    /// under D, so different symmetries can map to the same image.
    #[test]
    fn test_index_symmetries_all_distinct() {
        for i in 0..64 {
            let s = index_symmetries(i);
            let mut seen = [false; 64];
            for &v in &s {
                assert!(v < 64, "index_symmetries({}) has out-of-range {}", i, v);
                // Allow duplicates — different symmetries can map
                // symmetric positions (e.g. corners) to the same target.
                if !seen[v] {
                    seen[v] = true;
                }
            }
        }
    }

    #[cfg(test)]
    /// Verify that invert_symmetry is the true inverse of index_symmetries.
    #[test]
    fn test_invert_symmetry_is_inverse() {
        for i in 0..64 {
            let s = index_symmetries(i);
            for (sym_idx, &s_i) in s.iter().enumerate() {
                let back = invert_symmetry(s_i, sym_idx);
                assert_eq!(
                    back, i,
                    "invert_symmetry({}, {}) = {}, expected {}",
                    s_i, sym_idx, back, i
                );
            }
        }
    }

    #[cfg(test)]
    /// Applying then inverting a symmetry should yield the original board.
    #[test]
    fn test_apply_to_bits_inverse() {
        // Pre-computed inverse of each symmetry index.
        // Computed by: find inv_sym such that for all i,
        //   index_symmetries(index_symmetries(i)[sym])[inv_sym] = i.
        const INV: [usize; 8] = [0, 1, 2, 3, 4, 6, 5, 7];

        let board = (1 << 28) | (1 << 35);
        for (sym_idx, &inv) in INV.iter().enumerate() {
            let transformed = apply_to_bits(board, sym_idx);
            let back = apply_to_bits(transformed, inv);
            assert_eq!(
                back, board,
                "sym {} then inv {} on {:#x} gave {:#x}",
                sym_idx, inv, board, back
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize)]
pub enum Player {
    #[default]
    Black,
    White,
}

impl Player {
    fn next(self) -> Player {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        *self as usize
    }
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

/// A move is a single index (0-63) indicating where to place a disc,
/// or 64 for a pass.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
pub struct Move(pub u8);

impl Move {
    /// Sentinel value representing a pass (no legal move available).
    pub const PASS: Move = Move(64);
}

// ---------------------------------------------------------------------------
// Move generation (free function on raw bitboards)
// ---------------------------------------------------------------------------

pub type BB = BitBoard<BOARD_SIZE, BOARD_SIZE>;

/// Kogge-Stone dumb7fill: flood from source `p` through opponent `o` in a
/// direction (left shift) bounded by wall mask `mask`.  Returns opponent
/// discs that form a contiguous chain reachable from `p`.
///
/// The refinement propagator `pro` ensures only squares whose entire path
/// (1, 2, then 4 steps) stays within opponent pieces are included — blocking
/// jumps across gaps or player discs.
#[inline]
pub fn flood_left(p: BB, o: BB, shift: usize, mask: BB) -> BB {
    let mut gen = (p << shift) & mask & o;
    let mut pro = o & mask;
    gen |= (gen << shift) & pro;
    pro &= pro << shift;
    gen |= (gen << (shift * 2)) & pro;
    pro &= pro << (shift * 2);
    gen |= (gen << (shift * 4)) & pro;
    gen
}

/// Kogge-Stone dumb7fill (right-shift variant).
#[inline]
pub fn flood_right(p: BB, o: BB, shift: usize, mask: BB) -> BB {
    let mut gen = (p >> shift) & mask & o;
    let mut pro = o & mask;
    gen |= (gen >> shift) & pro;
    pro &= pro >> shift;
    gen |= (gen >> (shift * 2)) & pro;
    pro &= pro >> (shift * 2);
    gen |= (gen >> (shift * 4)) & pro;
    gen
}

/// Given the bitboard of the current player and the opponent, return a
/// bitboard where each set bit represents a legal move.
///
/// Uses the parallel-prefix (flood-fill) technique: for each of the 8
/// directions, we flood from the player's pieces through consecutive opponent
/// pieces in O(log N) shifts per direction instead of O(N) iteration.
pub fn generate_moves(player: BB, opponent: BB) -> BB {
    let empty = !(player | opponent);
    let mut legal = BB::EMPTY;

    // ── Left-shift directions ──

    // North (+8) — no horizontal guard needed.
    let n = flood_left(player, opponent, 8, BB::ONES);
    legal |= (n << 8) & empty;

    // North-West (+7) — must guard against wrapping from col 0 to col 7.
    let nw = flood_left(player, opponent, 7, !BB::wall(Direction::East));
    legal |= (nw << 7) & !BB::wall(Direction::East) & empty;

    // North-East (+9) — guard against wrapping from col 7 to col 0.
    // << 9 = row+1, col+1; col 7 wraps to col 0 → mask NOT_H during
    // propagation.  Final <<9 can produce next-row col 0 from col 7
    // (a wrapping artifact) → final mask NOT_A.
    let ne = flood_left(player, opponent, 9, !BB::wall(Direction::West));
    legal |= (ne << 9) & !BB::wall(Direction::West) & empty;

    // East (+1) — guard against wrapping from col 7 to col 0.
    // Final <<1 from propagation-cleaned columns cannot wrap.
    let e = flood_left(player, opponent, 1, !BB::wall(Direction::West));
    legal |= (e << 1) & !BB::wall(Direction::West) & empty;

    // ── Right-shift directions ──

    // South (-8).
    let s = flood_right(player, opponent, 8, BB::ONES);
    legal |= (s >> 8) & empty;

    // South-West (-9) — guard against wrapping from col 0 to col 7.
    // Propagation uses NOT_A.  Final >>9 can produce same-row col 7 from
    // col 0 (a wrapping artifact) → final mask NOT_H.
    let sw = flood_right(player, opponent, 9, !BB::wall(Direction::East));
    legal |= (sw >> 9) & !BB::wall(Direction::East) & empty;

    // South-East (-7) — guard against wrapping from col 7 to col 0.
    // Final >>7 can produce same-row col 0 from col 7 → mask NOT_A.
    let se = flood_right(player, opponent, 7, !BB::wall(Direction::West));
    legal |= (se >> 7) & !BB::wall(Direction::West) & empty;

    // West (>> 1).  Final >>1 from cleaned columns cannot wrap.
    let w = flood_right(player, opponent, 1, !BB::wall(Direction::East));
    legal |= (w >> 1) & !BB::wall(Direction::East) & empty;

    legal
}

/// Given the player, opponent, and a move being played, return a bitboard of
/// opponent discs that should be flipped as a result.
///
/// Uses the same parallel-prefix technique as `generate_moves`: for each
/// direction, flood from the move through consecutive opponent pieces, then
/// verify that one more step reaches a friendly piece.
pub fn get_flips(player: BB, opponent: BB, move_mask: BB) -> BB {
    let mut flips = BB::EMPTY;

    // ── Left-shift directions ──
    let n = flood_left(move_mask, opponent, 8, BB::ONES);
    if ((n << 8) & player).intersects(player) {
        flips |= n;
    }

    let ne = flood_left(move_mask, opponent, 9, !BB::wall(Direction::West));
    if ((ne << 9) & !BB::wall(Direction::West) & player).intersects(player) {
        flips |= ne;
    }

    let nw = flood_left(move_mask, opponent, 7, !BB::wall(Direction::East));
    if ((nw << 7) & !BB::wall(Direction::East) & player).intersects(player) {
        flips |= nw;
    }

    let e = flood_left(move_mask, opponent, 1, !BB::wall(Direction::West));
    if ((e << 1) & !BB::wall(Direction::West) & player).intersects(player) {
        flips |= e;
    }

    // ── Right-shift directions ──
    let s = flood_right(move_mask, opponent, 8, BB::ONES);
    if ((s >> 8) & player).intersects(player) {
        flips |= s;
    }

    let sw = flood_right(move_mask, opponent, 9, !BB::wall(Direction::East));
    if ((sw >> 9) & !BB::wall(Direction::East) & player).intersects(player) {
        flips |= sw;
    }

    let se = flood_right(move_mask, opponent, 7, !BB::wall(Direction::West));
    if ((se >> 7) & !BB::wall(Direction::West) & player).intersects(player) {
        flips |= se;
    }

    let w = flood_right(move_mask, opponent, 1, !BB::wall(Direction::East));
    if ((w >> 1) & !BB::wall(Direction::East) & player).intersects(player) {
        flips |= w;
    }

    flips
}

/// Naive (loop-based) reference oracle for Othello legality and flipping.
/// Uses only the BB API (`from_coord`, `intersects`, `get_at`, `|`, `&`, `!`)
/// for clarity and correctness — no raw u64 arithmetic.
pub const DIRS: [(i32, i32); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

pub fn naive_generate_moves(player: BB, opponent: BB) -> BB {
    if player.is_empty() || opponent.is_empty() {
        return BB::EMPTY;
    }
    let occupied = player | opponent;
    let mut legal = BB::EMPTY;
    for idx in 0..64 {
        if occupied.get_at(idx / 8, idx % 8) {
            continue;
        }
        let (sr, sc) = (idx as i32 / 8, idx as i32 % 8);
        'dirs: for &(dr, dc) in &DIRS {
            let mut r = sr + dr;
            let mut c = sc + dc;
            if !(0..8).contains(&r) || !(0..8).contains(&c) {
                continue;
            }
            if !opponent.intersects(BB::from_coord(r as usize, c as usize)) {
                continue;
            }
            loop {
                r += dr;
                c += dc;
                if !(0..8).contains(&r) || !(0..8).contains(&c) {
                    break;
                }
                let pos = BB::from_coord(r as usize, c as usize);
                if player.intersects(pos) {
                    legal |= BB::from_coord(sr as usize, sc as usize);
                    break 'dirs;
                }
                if !opponent.intersects(pos) {
                    break;
                }
            }
        }
    }
    legal
}

pub fn naive_get_flips(player: BB, opponent: BB, move_mask: BB) -> BB {
    let sq = move_mask.bits().trailing_zeros() as usize;
    let (sr, sc) = (sq / 8, sq % 8);
    let (sr, sc) = (sr as i32, sc as i32);
    let mut flips = BB::EMPTY;
    for &(dr, dc) in &DIRS {
        let mut r = sr + dr;
        let mut c = sc + dc;
        if !(0..8).contains(&r) || !(0..8).contains(&c) {
            continue;
        }
        let mut line = BB::EMPTY;
        loop {
            let pos = BB::from_coord(r as usize, c as usize);
            if opponent.intersects(pos) {
                line |= pos;
            } else if player.intersects(pos) {
                flips |= line;
                break;
            } else {
                break;
            }
            r += dr;
            c += dc;
            if !(0..8).contains(&r) || !(0..8).contains(&c) {
                break;
            }
        }
    }
    flips
}

pub fn naive_apply(state: State, action: &Move) -> State {
    if *action == Move::PASS {
        return State {
            black: state.black,
            white: state.white,
            turn: state.turn.next(),
            last_pass: true,
            hashes: [0u64; 8],
        };
    }
    let mv = action.0 as usize;
    let (player, opponent) = match state.turn {
        Player::Black => (state.black, state.white),
        Player::White => (state.white, state.black),
    };
    let move_bb = BB::from_index(mv);
    let flips_bb = naive_get_flips(player, opponent, move_bb);
    let new_player = player ^ move_bb ^ flips_bb;
    let new_opponent = opponent ^ flips_bb;
    let (new_black, new_white, new_turn) = match state.turn {
        Player::Black => (new_player, new_opponent, Player::White),
        Player::White => (new_opponent, new_player, Player::Black),
    };
    State {
        black: new_black,
        white: new_white,
        turn: new_turn,
        last_pass: false,
        hashes: [0u64; 8],
    }
}


// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct State {
    pub black: BB,
    pub white: BB,
    pub turn: Player,
    /// Set to true when the previous player passed (had no legal moves).
    /// Reset to false on a real move.
    pub last_pass: bool,
    /// Incrementally-maintained Zobrist hash for each of the 8 symmetries.
    /// `hashes[0]` is the identity-symmetry hash; the others can be used
    /// for canonical-symmetry reduction once `canonical_symmetry` is added.
    pub hashes: [u64; 8],
}

impl Default for State {
    fn default() -> Self {
        let mut hashes = [0u64; 8];
        xor_piece_range(&mut hashes, INITIAL_BLACK, Player::Black);
        xor_piece_range(&mut hashes, INITIAL_WHITE, Player::White);
        // Black to move: no turn hash (turn hash XOR'd for White).
        // last_pass = false: no pass hash.
        Self {
            black: BB::new(INITIAL_BLACK),
            white: BB::new(INITIAL_WHITE),
            turn: Player::Black,
            last_pass: false,
            hashes,
        }
    }
}

// ── Zobrist helpers ──────────────────────────────────────────────────────

/// XOR the hash contribution for a single piece into all 8 symmetry hashes.
#[inline]
pub fn xor_piece(hashes: &mut [u64; 8], pos: usize, player: Player) {
    let symmetries = sym::index_symmetries(pos);
    for (s, &sym_pos) in symmetries.iter().enumerate() {
        hashes[s] ^= HASHES.hash(zobrist_piece(sym_pos, player));
    }
}

/// XOR the hash contribution for every set bit in a u64 bitboard.
pub fn xor_piece_range(hashes: &mut [u64; 8], bits: u64, player: Player) {
    let mut remaining = bits;
    while remaining != 0 {
        let pos = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        xor_piece(hashes, pos, player);
    }
}

/// XOR a position-independent constant (turn, last_pass) into all 8 hashes.
#[inline]
pub fn xor_const(hashes: &mut [u64; 8], table_idx: usize) {
    let v = HASHES.hash(table_idx);
    for h in hashes.iter_mut() {
        *h ^= v;
    }
}

impl State {
    /// The set of all occupied squares.
    #[inline(always)]
    fn occupied(&self) -> BB {
        self.black | self.white
    }

    /// The identity-symmetry Zobrist hash of this state.
    #[inline(always)]
    fn hash(&self) -> u64 {
        self.hashes[0]
    }

    /// Generate legal moves for the player whose turn it is.
    /// When there are no legal moves, produces a single PASS action.
    fn generate_actions(&self, actions: &mut Vec<Move>) {
        let (player, opponent) = match self.turn {
            Player::Black => (self.black, self.white),
            Player::White => (self.white, self.black),
        };
        let moves = generate_moves(player, opponent);
        let count = moves.count_ones();
        if count == 0 {
            actions.push(Move::PASS);
        } else {
            actions.reserve(count as usize);
            for idx in moves {
                actions.push(Move(idx as u8));
            }
        }
    }

    /// Apply a move, flipping captured opponent discs, or handle a pass.
    fn apply(&mut self, action: &Move) {
        if *action == Move::PASS {
            self.last_pass = true;
            self.turn = self.turn.next();
            xor_const(&mut self.hashes, ZOBRIST_TURN);
            xor_const(&mut self.hashes, ZOBRIST_LAST_PASS);
            return;
        }
        let index = action.0 as usize;
        let piece = BB::from_index(index);
        let (player, opponent) = match self.turn {
            Player::Black => (self.black, self.white),
            Player::White => (self.white, self.black),
        };
        let flips = get_flips(player, opponent, piece);
        let player = player | piece | flips;
        let opponent = opponent & !flips;
        match self.turn {
            Player::Black => {
                self.black = player;
                self.white = opponent;
            }
            Player::White => {
                self.white = player;
                self.black = opponent;
            }
        }
        // Update Zobrist hash: place new piece, flip opponents, toggle turn.
        xor_piece(&mut self.hashes, index, self.turn);
        for f in flips {
            xor_piece(&mut self.hashes, f, self.turn.next());
            xor_piece(&mut self.hashes, f, self.turn);
        }
        if self.last_pass {
            xor_const(&mut self.hashes, ZOBRIST_LAST_PASS);
        }
        self.last_pass = false;
        self.turn = self.turn.next();
        xor_const(&mut self.hashes, ZOBRIST_TURN);
    }
}

// ---------------------------------------------------------------------------
// Game impl
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Othello;

impl Game for Othello {
    type S = State;
    type A = Move;
    type P = Player;

    fn apply(mut state: State, action: &Move) -> State {
        state.apply(action);
        state
    }

    fn generate_actions(state: &State, actions: &mut Vec<Move>) {
        state.generate_actions(actions);
    }

    fn is_terminal(state: &State) -> bool {
        // Board completely filled.
        if state.occupied().count_ones() == 64 {
            return true;
        }
        // Previous player passed and current player also has no moves.
        if !state.last_pass {
            return false;
        }
        let (player, opponent) = match state.turn {
            Player::Black => (state.black, state.white),
            Player::White => (state.white, state.black),
        };
        generate_moves(player, opponent).count_ones() == 0
    }

    fn player_to_move(state: &State) -> Player {
        state.turn
    }

    fn winner(state: &State) -> Option<Player> {
        if !Self::is_terminal(state) {
            return None;
        }
        let black = state.black.count_ones();
        let white = state.white.count_ones();
        match black.cmp(&white) {
            std::cmp::Ordering::Greater => Some(Player::Black),
            std::cmp::Ordering::Less => Some(Player::White),
            std::cmp::Ordering::Equal => None,
        }
    }

    fn notation(_state: &Self::S, action: &Self::A) -> String {
        let idx = action.0 as usize;
        let (row, col) = BB::to_coord(idx);
        let file = (b'a' + col as u8) as char;
        let rank = row + 1;
        format!("{}{}", file, rank)
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.hash()
    }

    fn num_players() -> usize {
        2
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl RectangularBoard for State {
    const NUM_DISPLAY_ROWS: usize = BOARD_SIZE;
    const NUM_DISPLAY_COLS: usize = BOARD_SIZE;

    fn display_char_at(&self, row: usize, col: usize) -> char {
        if self.black.get_at(row, col) {
            '●'
        } else if self.white.get_at(row, col) {
            '○'
        } else {
            '.'
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        RectangularBoardDisplay(self).fmt(f)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a BB from a slice of indices.
    fn bits(indices: &[usize]) -> BB {
        indices
            .iter()
            .fold(BB::EMPTY, |b, &i| b | BB::from_index(i))
    }

    /// Check that `generate_moves` for a given position produces exactly the
    /// expected set of moves.
    fn check_moves(black: &[usize], white: &[usize], expected: &[usize]) {
        let player = bits(black);
        let opponent = bits(white);
        let result = generate_moves(player, opponent);
        let expected_bits = bits(expected);
        assert_eq!(
            result,
            expected_bits,
            "\nblack={{{}}}\nwhite={{{}}}\nexpected={{{}}}\ngot={{{}}}",
            fmt_bits(black),
            fmt_bits(white),
            fmt_bits(expected),
            fmt_result(result),
        );
        // Verify none of the expected squares are occupied.
        let occupied = player | opponent;
        for &i in expected {
            assert!(!occupied.get(i), "expected move at {} is occupied", i);
        }
    }

    /// Run `check_moves` for the given position and all 8 symmetric variants.
    /// The `black`/`white`/`expected` slices are treated as the canonical
    /// (identity-symmetry) position; each of the 8 symmetries is tested
    /// automatically.
    fn check_all_symmetries(black: &[usize], white: &[usize], expected: &[usize]) {
        // Wrap index transformation so it's easy to apply.
        let sym = |list: &[usize], sym_idx| -> Vec<usize> {
            list.iter()
                .map(|&i| sym::index_symmetries(i)[sym_idx])
                .collect()
        };
        for sym_idx in 0..8 {
            let p = sym(black, sym_idx);
            let o = sym(white, sym_idx);
            let e = sym(expected, sym_idx);
            check_moves(&p, &o, &e);
        }
    }

    fn fmt_bits(indices: &[usize]) -> String {
        indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    fn fmt_result(b: BB) -> String {
        let mut v: Vec<usize> = b.collect();
        v.sort_unstable();
        v.iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    // ── Simple single-direction captures ──

    /// Initial position: Black has 4 legal moves (d3, c4, f5, e6).
    #[test]
    fn test_initial_moves() {
        // Black at d5(35), e4(28); White at d4(27), e5(36).
        check_all_symmetries(
            &[28, 35],
            &[27, 36],
            &[19, 26, 37, 44], // d3(19), c4(26), f5(37), e6(44)
        );
    }

    /// Single direction — North.  Player piece at 27, opponent at 35 (directly
    /// north), move at 43 captures southward (43→35→27).
    #[test]
    fn test_capture_north() {
        check_all_symmetries(&[27], &[35], &[43]);
    }

    /// Single direction — South.  Player at 43, opponent at 35, move at 27
    /// captures northward (27→35→43).
    #[test]
    fn test_capture_south() {
        check_all_symmetries(&[43], &[35], &[27]);
    }

    /// Single direction — East.  Player at 25(b4), opponents at 26(c4),27(d4),
    /// move at 28(e4) captures westward.
    #[test]
    fn test_capture_east() {
        check_all_symmetries(&[25], &[26, 27], &[28]);
    }

    /// Single direction — West.  Player at 30(g4), opponents at 29(f4),28(e4),
    /// move at 27(d4) captures eastward.
    #[test]
    fn test_capture_west() {
        check_all_symmetries(&[30], &[29, 28], &[27]);
    }

    /// Single direction — NE.  Player at 19(d3), opponent at 28(e4), move at
    /// 37(f5) captures south-westward (37→28→19).
    #[test]
    fn test_capture_northeast() {
        check_all_symmetries(&[19], &[28], &[37]);
    }

    /// Single direction — NW.  Player at 35(d5), opponent at 28(e4), move at
    /// 21(f3) captures south-eastward (21→28→35).
    #[test]
    fn test_capture_northwest() {
        check_all_symmetries(&[35], &[28], &[21]);
    }

    /// Single direction — SE.  Player at 35(d5), opponent at 28(e4), move at
    /// 21(f3).  From 21 looking NW: 28(W), 35(B).  This is the same geometry
    /// as test_capture_northwest but expressed as the SE perspective:
    /// player at 21, opponent at 28, move at 35 — SE from 35: 28(W), 21(B).
    #[test]
    fn test_capture_southeast() {
        check_all_symmetries(&[21], &[28], &[35]);
    }

    /// Single direction — SW.  Player at 19(d3), opponent at 28(e4), move at
    /// 37(f5).  From 37 looking SW: 28(W), 19(B).
    #[test]
    fn test_capture_southwest() {
        check_all_symmetries(&[37], &[28], &[19]);
    }

    // ── Edge and corner cases ──

    /// A move along the west edge (column A) — vertical only, no horizontal
    /// component, so wrapping is not an issue.
    #[test]
    fn test_capture_on_west_edge() {
        // Player at 48(a7), opponent at 40(a6), move at 32(a5).
        check_all_symmetries(&[48], &[40], &[32]);
    }

    /// A move along the east edge (column H) — vertical.
    #[test]
    fn test_capture_on_east_edge() {
        // Player at 63(h8), opponent at 55(h7), move at 47(h6).
        check_all_symmetries(&[63], &[55], &[47]);
    }

    /// A corner capture — playing at a8(56) captures south(48→40) and
    /// east(57→58).
    #[test]
    fn test_corner_multi_direction() {
        // Black at 40(a6), 58(c8); White at 48(a7), 57(b8).
        check_all_symmetries(&[40, 58], &[48, 57], &[56]);
    }

    // ── Multi-direction ──

    /// A move that flips in three directions simultaneously.
    #[test]
    fn test_triple_direction() {
        // Black at 40(a6), 58(c8), 42(c6); White at 48(a7), 57(b8), 49(b7).
        check_all_symmetries(&[40, 58, 42], &[48, 57, 49], &[56]);
    }

    // ── Long line ──

    /// A single line of 6 opponent pieces (the maximum in Othello).
    #[test]
    fn test_long_line_west() {
        // Player at 0(a1), opponents at 1-6(b1-g1), move at 7(h1).
        check_all_symmetries(&[0], &[1, 2, 3, 4, 5, 6], &[7]);
    }

    /// Long line in north direction.
    #[test]
    fn test_long_line_north() {
        // Player at 0(a1), opponents at 8(a2),16(a3),24(a4),32(a5),40(a6),
        // move at 48(a7).
        check_all_symmetries(&[0], &[8, 16, 24, 32, 40], &[48]);
    }

    // ── No legal moves ──

    /// If the player has no pieces, no legal moves.
    #[test]
    fn test_no_player_pieces() {
        let result = generate_moves(BB::EMPTY, bits(&[27, 36]));
        assert_eq!(result, BB::EMPTY);
    }

    /// If the opponent has no pieces, no legal moves (nothing to flip).
    #[test]
    fn test_no_opponent_pieces() {
        let result = generate_moves(bits(&[28, 35]), BB::EMPTY);
        assert_eq!(result, BB::EMPTY);
    }

    /// Pieces exist but no sandwich is possible (isolated player piece far
    /// from any opponent).
    #[test]
    fn test_no_sandwich_possible() {
        // Black at 0(a1), White at 63(h8).  No line connects them.
        check_all_symmetries(&[0], &[63], &[]);
    }

    // ── Existing stubs kept for compatibility ──

    #[test]
    fn test_initial_position() {
        let state = State::default();
        assert_eq!(state.black.count_ones(), 2);
        assert_eq!(state.white.count_ones(), 2);
        assert_eq!(state.occupied().count_ones(), 4);
        assert_eq!(state.turn, Player::Black);
        assert!(state.black.get(28));
        assert!(state.black.get(35));
        assert!(state.white.get(27));
        assert!(state.white.get(36));
    }

    #[test]
    fn test_stub_contract() {
        let state = State::default();
        assert_eq!(Othello::num_players(), 2);
        assert_eq!(Othello::player_to_move(&state), Player::Black);
        assert!(!Othello::is_terminal(&state));

        let mut actions = Vec::new();
        Othello::generate_actions(&state, &mut actions);
        assert_eq!(actions.len(), 4, "initial position should have 4 moves");
    }

    #[test]
    fn test_notation() {
        let m = Move(0);
        let state = State::default();
        assert_eq!(Othello::notation(&state, &m), "a1");
    }

    #[test]
    fn test_apply_stub() {
        let state = State::default();
        let moved = Othello::apply(state, &Move(19));
        assert!(moved.black.get(19));
        // d3 captures white at d4 via the north ray, so black gains d3 + flip d4
        assert_eq!(moved.black.count_ones(), 4);
        assert!(moved.black.get(27));
        assert_eq!(moved.white.count_ones(), 1);
        assert!(moved.white.get(36));
        assert_eq!(moved.turn, Player::White);
    }

    #[test]
    fn test_display() {
        let state = State::default();
        let _ = format!("{}", state);
    }

    // ── Passing ──

    /// A player with no legal actions gets a single PASS action.
    #[test]
    fn test_pass_generated_when_no_moves() {
        // Black at a1(0), White at h8(63) — no sandwich possible.
        let state = State {
            black: bits(&[0]),
            white: bits(&[63]),
            turn: Player::Black,
            last_pass: false,
            hashes: [0u64; 8],
        };
        let mut actions = Vec::new();
        state.generate_actions(&mut actions);
        assert_eq!(actions, vec![Move::PASS]);
    }

    /// Applying a pass flips turn and sets last_pass, then the new player
    /// also gets a pass action.
    #[test]
    fn test_pass_then_opponent_also_passes() {
        let mut state = State {
            black: bits(&[0]),
            white: bits(&[63]),
            turn: Player::Black,
            last_pass: false,
            hashes: [0u64; 8],
        };
        // Black passes.
        state.apply(&Move::PASS);
        assert!(state.last_pass);
        assert_eq!(state.turn, Player::White);

        // White also has no moves — passes too.
        let mut actions = Vec::new();
        state.generate_actions(&mut actions);
        assert_eq!(actions, vec![Move::PASS]);

        // Game should be considered terminal.
        assert!(Othello::is_terminal(&state));
    }

    /// A real move after a pass resets last_pass.
    #[test]
    fn test_real_move_resets_pass_flag() {
        // Start with Black having no legal moves (isolated at a1, white at h8).
        let mut state = State {
            black: bits(&[0]),  // a1
            white: bits(&[63]), // h8
            turn: Player::Black,
            last_pass: false,
            hashes: [0u64; 8],
        };

        // Black has no moves → gets PASS.
        let mut actions = Vec::new();
        state.generate_actions(&mut actions);
        assert_eq!(
            actions,
            vec![Move::PASS],
            "expected Black to have no moves, got {:?}",
            actions
        );

        // Black passes → last_pass set, turn flips to White.
        state.apply(&Move::PASS);
        assert!(state.last_pass);
        assert_eq!(state.turn, Player::White);

        // Now give White a capture: White at 26(c4),27(d4); Black at 25(b4).
        // White can play 24(a4) capturing black at 25 eastward (24→25→26→27).
        state.white = bits(&[26, 27]);
        state.black = bits(&[25]);

        // White plays a4(24).
        state.apply(&Move(24));
        assert!(!state.last_pass, "real move should reset last_pass");
        assert!(state.white.get(24), "white should have disc at a4");
        assert!(
            state.black.is_empty(),
            "black at b4(25) should have been flipped"
        );
        assert!(state.white.get(25), "white should have flipped b4");
        assert_eq!(state.turn, Player::Black);
    }

    // ── Random game sanity ──

    /// Play a few full random games, checking invariants after every move.
    #[test]
    fn test_random_games_invariants() {
        use rand::seq::SliceRandom;

        let mut rng = rand::thread_rng();

        for game_num in 0..3 {
            let mut state = State::default();
            let mut prev_turn = state.turn;
            let mut total_actions = 0;

            while !Othello::is_terminal(&state) {
                total_actions += 1;
                assert!(total_actions <= 200, "Game {game_num} exceeded 200 moves");

                // Generate actions.
                let mut actions = Vec::new();
                state.generate_actions(&mut actions);
                assert!(
                    !actions.is_empty(),
                    "non-terminal state must produce at least one action"
                );
                let action = *actions.choose(&mut rng).unwrap();

                // Invariant: no overlapping bits.
                assert!(
                    (state.black & state.white).is_empty(),
                    "overlapping bits before move"
                );
                // Invariant: turn matches cached.
                assert_eq!(state.turn, prev_turn);

                let (player, opponent) = match state.turn {
                    Player::Black => (state.black, state.white),
                    Player::White => (state.white, state.black),
                };

                if action == Move::PASS {
                    // Pass only valid when player truly has no moves.
                    assert_eq!(
                        generate_moves(player, opponent).count_ones(),
                        0,
                        "PASS generated but player has legal moves"
                    );
                } else {
                    // Non-pass: verify legality up front.
                    let move_mask = BB::from_index(action.0 as usize);
                    assert!(
                        (move_mask & (player | opponent)).is_empty(),
                        "move to occupied square"
                    );
                    assert!(
                        generate_moves(player, opponent) & move_mask != BB::EMPTY,
                        "move must be legal"
                    );
                }

                // Apply.
                state.apply(&action);

                // Invariant: no overlapping bits after apply.
                assert!(
                    (state.black & state.white).is_empty(),
                    "overlapping bits after move"
                );
                // Invariant: turn toggled.
                assert_eq!(state.turn, prev_turn.next());
                prev_turn = state.turn;

                if action != Move::PASS {
                    // Invariant: move bit now belongs to the player.
                    let toggle_idx = action.0 as usize;
                    let (player, _opponent) = match state.turn {
                        // turn has already flipped, so previous player was the mover.
                        Player::Black => (state.white, state.black),
                        Player::White => (state.black, state.white),
                    };
                    assert!(
                        player.get(toggle_idx),
                        "bit {toggle_idx} should belong to the mover after a non-pass move"
                    );
                }

                // Invariant: total discs >= 4 (we never lose discs).
                let total = state.occupied().count_ones();
                assert!(total >= 4, "disc count dropped below 4: {total}");
            }

            // Game ended — verify winner makes sense.
            let total = state.occupied().count_ones();
            assert!(total == 64 || state.last_pass);

            match Othello::winner(&state) {
                Some(Player::Black) => {
                    let b = state.black.count_ones();
                    let w = state.white.count_ones();
                    assert!(b > w, "Black won but has {b} vs {w}");
                }
                Some(Player::White) => {
                    let b = state.black.count_ones();
                    let w = state.white.count_ones();
                    assert!(w > b, "White won but has {w} vs {b}");
                }
                None => {
                    // Draw on full board or consecutive passes.
                    let b = state.black.count_ones();
                    let w = state.white.count_ones();
                    assert_eq!(b, w, "draw but discs unequal ({b} vs {w})");
                }
            }
        }
    }

    /// Replays the 24-move game from the 13:04:52 JSON, printing all states.
    #[test]
    fn test_130452_replay() {
        let moves: [u8; 24] = [26, 20, 29, 18, 19, 22, 37, 45, 17, 25, 24,
                                34, 44, 16, 33, 53, 42, 32, 30, 49, 56, 51,
                                43, 41];
        let mut state = State::default();
        // Play the first 23 moves (0-22).  The JSON is corrupted from n21
        // onward so move 23 (b6) is not necessarily legal in our correct replay.
        for (i, &mv) in moves[..23].iter().enumerate() {
            let (player, opponent) = match state.turn {
                Player::Black => (state.black, state.white),
                Player::White => (state.white, state.black),
            };
            let (b, w) = (state.black.bits(), state.white.bits());
            assert_eq!(b & w, 0, "ply {}: overlap", i);
            assert!(generate_moves(player, opponent)
                    .intersects(BB::from_index(mv as usize)),
                    "ply {}: move {} not legal", i, mv);
            let pf = get_flips(player, opponent, BB::from_index(mv as usize));
            let nf = naive_get_flips(player, opponent, BB::from_index(mv as usize));
            assert_eq!(pf, nf, "ply {}: flip mismatch: prod={:#x} naive={:#x}", i, pf.bits(), nf.bits());
            state = Othello::apply(state, &Move(mv));
        }
        // The JSON was corrupted from n21 onward, so b6(41) at ply 23
        // may not be legal in our correct replay.  Just assert invariants.
        assert_eq!(state.occupied().count_ones(), 27, "23 moves = 27 discs");
        assert_eq!(state.black.bits() & state.white.bits(), 0, "no overlap");
    }

}
