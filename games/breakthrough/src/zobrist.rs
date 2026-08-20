//! Zobrist hashing for a Breakthrough position: one random key per
//! `(cell, color)` pair, XOR'd in for every occupied square, plus a single
//! key toggled when White is to move. `State` is backed by a `u64` bitboard
//! (`bitboard::Board<u64, ..>`), so no board size this game supports can
//! exceed 64 cells -- the table below is sized for the worst case rather
//! than parameterized per `N, M`, so one `static` covers every board size.

use mcts::zobrist::LazyZobristTable;

use crate::Player;

const MAX_CELLS: usize = 64;

/// `2 * MAX_CELLS` piece-placement keys (one per `(cell, color)`) plus one
/// player-to-move key.
const HASHES_LEN: usize = 2 * MAX_CELLS + 1;
static HASHES: LazyZobristTable<HASHES_LEN> = LazyZobristTable::new(0xB2EA);

#[inline]
fn cell_zobrist(index: usize, player: Player) -> u64 {
    debug_assert!(index < MAX_CELLS);
    let color = match player {
        Player::Black => 0,
        Player::White => 1,
    };
    HASHES.hash(2 * index + color)
}

/// Contributes 0 when Black is to move (same "one side is the identity"
/// convention as `druid::zobrist::player_zobrist`), so only the last table
/// slot is needed to toggle between the two players.
#[inline]
fn player_zobrist(player: Player) -> u64 {
    match player {
        Player::Black => 0,
        Player::White => HASHES.hash(2 * MAX_CELLS),
    }
}

/// Hashes `black`/`white`'s occupied cells and `turn` from scratch --
/// `O(popcount)`, meant for `Game::zobrist_hash`, not for incremental
/// per-move updates (unlike Druid's `full_hash`, Breakthrough's `apply`
/// doesn't currently maintain an incremental hash on `State` itself).
pub(crate) fn full_hash(
    black: impl Iterator<Item = usize>,
    white: impl Iterator<Item = usize>,
    turn: Player,
) -> u64 {
    let mut hash = player_zobrist(turn);
    for i in black {
        hash ^= cell_zobrist(i, Player::Black);
    }
    for i in white {
        hash ^= cell_zobrist(i, Player::White);
    }
    hash
}
