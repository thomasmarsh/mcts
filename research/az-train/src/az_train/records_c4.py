# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""Reader for ``game-connect4 dump`` v2-connect4 self-play records.

The tic-tac-toe reader (``az_train.records``) packs the whole board into a
``u32``; the standard 6x7 Connect Four board has 42 cells and does not fit,
so the Rust ``dump`` subcommand (``games/connect4/src/dump.rs``) writes the
two occupancy bitboards verbatim instead. A fixed 23-byte head followed by
a variable-length policy tail, packed with no padding, little-endian:

=========  =============  ============
field      type           bytes
=========  =============  ============
black      u64 LE         8
white      u64 LE         8
side       u8             1
ply        u8             1
value      f32 LE         4
n_policy   u8             1
policy     n * (u8, f32)  n_policy * 5
=========  =============  ============

``black`` / ``white`` are raw ``BitBoard<6, 7>`` words: bit ``row * 7 +
col`` set where that player holds the cell, row 0 at the bottom. ``side``
is 0 for Black to move, 1 for White. ``ply`` is the disc count. ``value``
is the final result from the side-to-move player's perspective (``+1`` win,
``-1`` loss, ``0`` draw). The policy tail is the improved-policy target --
``(column_index, probability)`` pairs from a Gumbel Sequential-Halving
visit distribution -- and is empty for ``--label outcome`` dumps.

Records are variable-width, so this walks them sequentially rather than
using ``np.fromfile`` with a fixed dtype.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECORD_HEAD_BYTES = 23
POLICY_ENTRY_BYTES = 5
ROWS = 6
COLS = 7
BOARD_CELLS = ROWS * COLS

_HEAD = struct.Struct("<QQBBfB")
_POLICY_ENTRY = struct.Struct("<Bf")


@dataclass
class Positions:
    """Columnar decode of a dump file. One row per non-terminal position."""

    black: np.ndarray  # (N,) uint64, raw BitBoard<6,7> word
    white: np.ndarray  # (N,) uint64
    side: np.ndarray  # (N,) uint8, 0 = Black to move, 1 = White
    ply: np.ndarray  # (N,) uint8
    value: np.ndarray  # (N,) float32, side-to-move perspective, [-1, 1]
    # (N,) ragged: each entry a list of (column_index, probability) pairs.
    policy: list[list[tuple[int, float]]]

    def __len__(self) -> int:
        return len(self.value)


def game_slices(pos: Positions) -> list[slice]:
    """Recover contiguous games from their recorded plies.

    A dump is a concatenation of complete games.  Checking this here keeps a
    training/validation split from accidentally treating neighbouring boards
    from one game as independent examples.
    """
    if not len(pos):
        return []
    if int(pos.ply[0]) != 0:
        raise ValueError(f"first record has ply {int(pos.ply[0])}, expected 0")
    starts = np.flatnonzero(pos.ply == 0)
    out: list[slice] = []
    for i, start in enumerate(starts):
        end = int(starts[i + 1]) if i + 1 < len(starts) else len(pos)
        expected = np.arange(end - int(start), dtype=np.uint8)
        actual = pos.ply[int(start):end]
        if not np.array_equal(actual, expected):
            bad = int(np.flatnonzero(actual != expected)[0])
            raise ValueError(
                f"game starting at record {int(start)} has ply {int(actual[bad])} "
                f"at offset {bad}, expected {int(expected[bad])}"
            )
        out.append(slice(int(start), end))
    return out


def select_rows(pos: Positions, rows: np.ndarray) -> Positions:
    """Return records at ``rows``, preserving their byte-stream fields."""
    return Positions(
        black=pos.black[rows], white=pos.white[rows], side=pos.side[rows],
        ply=pos.ply[rows], value=pos.value[rows],
        policy=[pos.policy[int(i)] for i in rows],
    )


def split_by_game(
    pos: Positions, validation_fraction: float = 0.2, seed: int = 0
) -> tuple[Positions, Positions, int, int]:
    """Deterministically split complete games into train and validation sets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    games = game_slices(pos)
    if len(games) < 2:
        raise ValueError("need at least two complete games for a train/validation split")
    n_validation = min(len(games) - 1, max(1, round(len(games) * validation_fraction)))
    order = np.random.default_rng(seed).permutation(len(games))
    validation_games = set(int(i) for i in order[:n_validation])
    train_rows = np.concatenate([
        np.arange(s.start, s.stop) for i, s in enumerate(games) if i not in validation_games
    ])
    validation_rows = np.concatenate([
        np.arange(s.start, s.stop) for i, s in enumerate(games) if i in validation_games
    ])
    return (
        select_rows(pos, train_rows),
        select_rows(pos, validation_rows),
        len(games) - n_validation,
        n_validation,
    )


def decode_records(raw: bytes) -> Positions:
    blacks: list[int] = []
    whites: list[int] = []
    sides: list[int] = []
    plies: list[int] = []
    values: list[float] = []
    policies: list[list[tuple[int, float]]] = []

    off = 0
    n = len(raw)
    while off < n:
        if off + RECORD_HEAD_BYTES > n:
            raise ValueError(f"truncated record head at byte {off}")
        black, white, side, ply, value, n_policy = _HEAD.unpack_from(raw, off)
        off += RECORD_HEAD_BYTES
        tail = n_policy * POLICY_ENTRY_BYTES
        if off + tail > n:
            raise ValueError(f"truncated policy tail at byte {off}")
        entries: list[tuple[int, float]] = []
        for i in range(n_policy):
            col, prob = _POLICY_ENTRY.unpack_from(raw, off + i * POLICY_ENTRY_BYTES)
            entries.append((int(col), float(prob)))
        off += tail

        blacks.append(black)
        whites.append(white)
        sides.append(side)
        plies.append(ply)
        values.append(value)
        policies.append(entries)

    return Positions(
        black=np.asarray(blacks, dtype=np.uint64),
        white=np.asarray(whites, dtype=np.uint64),
        side=np.asarray(sides, dtype=np.uint8),
        ply=np.asarray(plies, dtype=np.uint8),
        value=np.asarray(values, dtype=np.float32),
        policy=policies,
    )


def load_positions(path: str | Path) -> Positions:
    """Load every record from a dump file."""
    return decode_records(Path(path).read_bytes())


def encode_records(pos: Positions) -> bytes:
    """Re-encode a :class:`Positions` back to the on-disk byte stream. The
    inverse of :func:`decode_records` -- used to prove the Python codec
    matches Rust's byte-for-byte."""
    out = bytearray()
    for i in range(len(pos)):
        entries = pos.policy[i]
        out += _HEAD.pack(
            int(pos.black[i]),
            int(pos.white[i]),
            int(pos.side[i]),
            int(pos.ply[i]),
            float(pos.value[i]),
            len(entries),
        )
        for col, prob in entries:
            out += _POLICY_ENTRY.pack(col, prob)
    return bytes(out)


def me_opp_planes(pos: Positions) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 42)`` float32 occupancy planes relative to the side to move,
    row-major with bit ``row * 7 + col`` (row 0 at the bottom): ``me[n, c]``
    is 1 where cell ``c`` holds the mover's piece, ``opp`` the opponent's.
    This is exactly the input :mod:`az_train.ntuple_c4` consumes."""
    cell = np.arange(BOARD_CELLS, dtype=np.uint64)
    black_bits = ((pos.black[:, None] >> cell[None, :]) & np.uint64(1)).astype(np.float32)
    white_bits = ((pos.white[:, None] >> cell[None, :]) & np.uint64(1)).astype(np.float32)
    black_to_move = (pos.side == 0)[:, None]
    me = np.where(black_to_move, black_bits, white_bits)
    opp = np.where(black_to_move, white_bits, black_bits)
    return me, opp
