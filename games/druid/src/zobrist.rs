//! Zobrist hashing for a Druid position. Board cells, the pending sub-action
//! (move-split only; `Pending::None` contributes 0), player-to-move, and both
//! hands' remaining counts each contribute XOR'd random tables. `Game::apply`
//! updates this incrementally (xor-out old, xor-in new) from a from-scratch
//! `full_hash` that the property tests keep it pinned against.

use mcts::zobrist::LazyZobristTable;
use mcts::game::PlayerIndex;

use crate::state::State;
use crate::types::{Orientation, Pending, PieceKind, Player, Size};

// A naive Zobrist hash, will require a table of size:
//
//     size(N, M) = 2 * ceil(log2(N*M)) * (N*M + N*(M-1) + (N-1)*M)
//
// For a default 10x10 sized board that is 3920 entries. This, is not too high,
// but it is also not very efficient. In Druid, we only need to consider the
// top-down view. Occluded pieces do not need to contribute to the hash. The
// revised hash is better:
//
//     size(N, M) = 2 * N * M * bits_per_height(N, M)
//
// where `bits_per_height` is `ceil(log2(max_cell_height + 1))` -- see
// `zobrist_height_bits`/`max_cell_height`. A cell's height is bounded by a
// player's *hand* (`Hand::new` deals `N*M*2` sarsens), not by the board
// area, so for the standard 10x10 board that's 200 sarsens -> 8 bits ->
// size(10,10) = 2 * 100 * 8 = 1600. There is 8-way symmetry, but this is
// only useful in the early game.
//
// This bounds the largest board size we can support -- see `Size::is_supported`.
pub(crate) const HASHES_LEN: usize = 1600;
pub(crate) static HASHES: LazyZobristTable<HASHES_LEN> = LazyZobristTable::new(0xD401D);

const PENDING_HASHES_LEN: usize = 5;
static PENDING_HASHES: LazyZobristTable<PENDING_HASHES_LEN> = LazyZobristTable::new(0xD402D);

// `player`/hand counts used to be recoverable from the board alone (parity
// of total pieces placed determines the mover, remaining hand counts follow
// from move history) -- so the hash below omitted them. That assumption
// breaks for Druid: a lintel placement raises all 3 touched cells to
// `height(cells[0]) + 1` regardless of the other two cells' prior heights,
// which decouples "how many turns were played" from "what the board looks
// like". Two different, both-legally-reachable (board, pending) values can
// then differ in `player` and/or hand counts while hashing identically,
// silently aliasing unrelated MCTS transposition-table nodes. Hashing
// `player` and both hands' remaining counts closes that gap.
const PLAYER_HASHES_LEN: usize = 1;
static PLAYER_HASHES: LazyZobristTable<PLAYER_HASHES_LEN> = LazyZobristTable::new(0xD403D);

/// XOR contribution of player-to-move. `Black` contributes 0 (same
/// convention as `Pending::None` in `pending_zobrist`), so only one table
/// entry is needed to toggle between the two players.
pub(crate) fn player_zobrist(p: Player) -> u64 {
    match p {
        Player::Black => 0,
        Player::White => PLAYER_HASHES.hash(0),
    }
}

// Sized for 2 players * 2 piece kinds * the largest per-track bit width
// (`zobrist_height_bits`'s max under `HASHES_LEN`, currently 8) = 32.
// `Size::is_supported` checks this bound the same way it checks `HASHES_LEN`.
pub(crate) const HAND_HASHES_LEN: usize = 32;
static HAND_HASHES: LazyZobristTable<HAND_HASHES_LEN> = LazyZobristTable::new(0xD404D);

pub(crate) fn kind_index(k: PieceKind) -> usize {
    match k {
        PieceKind::Sarsen => 0,
        PieceKind::Lintel => 1,
    }
}

/// XOR contribution of one player's remaining count of `kind` pieces --
/// same bit-subset trick as `cell_zobrist`, applied to a bounded counter (a
/// hand count) instead of a board cell's height. Reuses the cell-height bit
/// width (`bits`, from `zobrist_height_bits`): a hand's sarsen count is
/// bounded by exactly that value (`Hand::new`'s `n * 2` is
/// `max_cell_height`'s definition) and its lintel count (`n`) never needs
/// more bits than that, so one shared width covers both tracks.
pub(crate) fn hand_zobrist(player: Player, kind: PieceKind, count: u8, bits: usize) -> u64 {
    let c = count as usize;
    if c == 0 {
        return 0;
    }
    let track = player.to_index() * 2 + kind_index(kind);
    let base = track * bits;
    (0..bits).fold(0, |hash, b| {
        if c & (1 << b) != 0 {
            hash ^ HAND_HASHES.hash(base + b)
        } else {
            hash
        }
    })
}

pub(crate) fn pending_index(p: Pending) -> usize {
    match p {
        Pending::None => 0,
        Pending::Piece(PieceKind::Sarsen) => 1,
        Pending::Piece(PieceKind::Lintel) => 2,
        Pending::Oriented(Orientation::Horizontal) => 3,
        Pending::Oriented(Orientation::Vertical) => 4,
    }
}

pub(crate) fn pending_zobrist(p: Pending) -> u64 {
    match p {
        Pending::None => 0,
        _ => PENDING_HASHES.hash(pending_index(p)),
    }
}

/// Highest height a single cell can reach. A player can only raise a cell's
/// height with pieces from their own hand (repeated sarsens on one cell, or
/// lintels bridging out from it), and `Hand::new` hands out `n * 2` sarsens
/// per player -- so that's the ceiling for one cell, not the board area.
pub(crate) fn max_cell_height(size: Size) -> usize {
    crate::types::Hand::new(size).sarsens as usize
}

// Number of bits used to encode a cell's height: each bit gets its own
// random table entry, XORed in when set, so a height in [0, 2^bits) maps to
// a distinct XOR combination (the entries are independent random u64s, so
// this is injective with overwhelming probability -- the standard trick for
// hashing bounded counters into a Zobrist scheme). `ceil(log2(n))` matches
// the sizing comment above, where `n` is the number of distinct heights a
// cell can take on (`max_cell_height(size) + 1`, since height ranges from 0
// up to and including the max).
pub(crate) fn zobrist_height_bits(size: Size) -> usize {
    let n = max_cell_height(size) + 1;
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

/// XOR contribution of a single cell to the board hash, for a given
/// (height, piece) at position `i`. Shared by the incremental update in
/// `game::apply_turn` and the from-scratch recompute used to validate it in
/// tests -- both need to agree on exactly which bits a cell contributes.
pub(crate) fn cell_zobrist(i: usize, height: u16, piece: Option<Player>, bits: usize) -> u64 {
    let h = height as usize;
    if h == 0 {
        return 0;
    }
    let c = piece.map(|p| p.to_index()).unwrap_or(0);
    let base = (i * 2 + c) * bits;
    (0..bits).fold(0, |hash, b| {
        if h & (1 << b) != 0 {
            hash ^ HASHES.hash(base + b)
        } else {
            hash
        }
    })
}

/// Full from-scratch board hash. `game::apply_turn` no longer uses this on
/// the hot path (see the incremental XOR-delta update there) -- used by the
/// property test that checks the incremental update stays in sync with it,
/// and by `game::HashedState::from_state` to hash a board that didn't arrive
/// via `apply` (e.g. one deserialized from a client-supplied JSON state).
pub(crate) fn recompute_hash(state: &State, bits: usize) -> u64 {
    state.board.iter().enumerate().fold(0, |hash, (i, square)| {
        hash ^ cell_zobrist(i, square.height, square.piece, bits)
    })
}

/// Full from-scratch hash: board cells, pending sub-move, player-to-move,
/// and both players' remaining hand counts -- every component the
/// incremental update in `game::apply_turn` maintains. Used by
/// `Game::HashedState::new`/`from_state` (states that don't arrive via
/// `apply`, so have nothing to update incrementally) and by the property
/// test that checks the incremental update stays in sync with it.
pub(crate) fn full_hash(state: &State, bits: usize) -> u64 {
    recompute_hash(state, bits)
        ^ pending_zobrist(state.pending)
        ^ player_zobrist(state.player)
        ^ hand_zobrist(Player::Black, PieceKind::Sarsen, state.hand_black.sarsens, bits)
        ^ hand_zobrist(Player::Black, PieceKind::Lintel, state.hand_black.lintels, bits)
        ^ hand_zobrist(Player::White, PieceKind::Sarsen, state.hand_white.sarsens, bits)
        ^ hand_zobrist(Player::White, PieceKind::Lintel, state.hand_white.lintels, bits)
}