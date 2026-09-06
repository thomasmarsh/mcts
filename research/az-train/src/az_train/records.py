# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""Reader for ``game-<kind> dump`` v2 self-play records.

The Rust ``dump`` subcommand (``games/ttt/src/dump.rs``) writes a fixed
11-byte head followed by a variable-length policy tail, packed with no
padding, little-endian:

=========  =============  ============
field      type           bytes
=========  =============  ============
board      u32 LE         4
side       u8             1
ply        u8             1
value      f32 LE         4
n_policy   u8             1
policy     n * (u8, f32)  n_policy * 5
=========  =============  ============

``board`` is ``game_ttt::Position``'s packed-u32 encoding (2 bits per cell,
digit 1 = X, 2 = O). ``side`` is 0 for X to move, 1 for O. ``ply`` is the
piece count. ``value`` is the final result from the side-to-move player's
perspective (``+1`` win, ``-1`` loss, ``0`` draw). The policy tail is the
improved-policy target -- ``(cell_index, probability)`` pairs from the
Gumbel Sequential-Halving visit distribution -- and is empty for
``--label outcome`` dumps.

Records are variable-width, so this walks them sequentially rather than
using ``np.fromfile`` with a fixed dtype.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECORD_HEAD_BYTES = 11
POLICY_ENTRY_BYTES = 5
BOARD_CELLS = 9

_HEAD = struct.Struct("<IBBfB")
_POLICY_ENTRY = struct.Struct("<Bf")


@dataclass
class Positions:
    """Columnar decode of a dump file. One row per non-terminal position."""

    board: np.ndarray  # (N,) uint32, packed Position encoding
    side: np.ndarray  # (N,) uint8, 0 = X to move, 1 = O
    ply: np.ndarray  # (N,) uint8
    value: np.ndarray  # (N,) float32, side-to-move perspective, [-1, 1]
    # (N,) ragged: each entry a list of (cell_index, probability) pairs.
    policy: list[list[tuple[int, float]]]

    def __len__(self) -> int:
        return len(self.value)


def decode_records(raw: bytes) -> Positions:
    boards: list[int] = []
    sides: list[int] = []
    plies: list[int] = []
    values: list[float] = []
    policies: list[list[tuple[int, float]]] = []

    off = 0
    n = len(raw)
    while off < n:
        if off + RECORD_HEAD_BYTES > n:
            raise ValueError(f"truncated record head at byte {off}")
        board, side, ply, value, n_policy = _HEAD.unpack_from(raw, off)
        off += RECORD_HEAD_BYTES
        tail = n_policy * POLICY_ENTRY_BYTES
        if off + tail > n:
            raise ValueError(f"truncated policy tail at byte {off}")
        entries: list[tuple[int, float]] = []
        for i in range(n_policy):
            action, prob = _POLICY_ENTRY.unpack_from(raw, off + i * POLICY_ENTRY_BYTES)
            entries.append((int(action), float(prob)))
        off += tail

        boards.append(board)
        sides.append(side)
        plies.append(ply)
        values.append(value)
        policies.append(entries)

    return Positions(
        board=np.asarray(boards, dtype=np.uint32),
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
            int(pos.board[i]),
            int(pos.side[i]),
            int(pos.ply[i]),
            float(pos.value[i]),
            len(entries),
        )
        for action, prob in entries:
            out += _POLICY_ENTRY.pack(action, prob)
    return bytes(out)


def me_opp_planes(pos: Positions) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 9)`` float32 occupancy planes relative to the side to move:
    ``me[n, c]`` is 1 where cell ``c`` holds the mover's piece, ``opp`` the
    opponent's."""
    cell = np.arange(BOARD_CELLS, dtype=np.uint32)
    occ = (pos.board[:, None] >> (cell[None, :] * 2)) & np.uint32(0b11)  # 0/1/2
    x_plane = (occ == 1).astype(np.float32)
    o_plane = (occ == 2).astype(np.float32)
    side0 = (pos.side == 0)[:, None]
    me = np.where(side0, x_plane, o_plane)
    opp = np.where(side0, o_plane, x_plane)
    return me, opp
