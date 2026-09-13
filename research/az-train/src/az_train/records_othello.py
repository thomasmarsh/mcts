# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false
"""Reader for ``game-othello dump --label gumbel`` v2 self-play records.

The Rust ``dump`` subcommand (``games/othello/src/dump.rs``) writes a fixed
23-byte head followed by a variable-length policy tail, packed with no
padding, little-endian:

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

``black`` / ``white`` are the raw 64-bit bitboards (bit ``i`` set where that
player holds square ``i``, square 0 = a1, square 63 = h8). ``side`` is 0 for
Black to move, 1 for White. ``ply`` is the disc count minus 4 -- unlike
Connect Four's ply (which increases by exactly one every record, since every
move places a disc), an Othello pass leaves the disc count unchanged, so
consecutive records in one game have *non-decreasing*, not strictly
increasing, ply. ``value`` is the final result from the side-to-move
player's perspective (``+1`` win, ``-1`` loss, ``0`` draw). The policy tail
is the completed-Q improved-policy target -- ``(square, probability)`` pairs,
``square`` a raw ``Move`` byte (``0..=63`` a board square, ``64`` ==
``Move::PASS``) -- and is empty for dumps with no search-derived policy.

Records are variable-width, so this walks them sequentially rather than
using ``np.fromfile`` with a fixed dtype. Mirrors
``az_train.records_c4``'s v2-connect4 reader field-for-field except
``column`` -> ``square`` and the ply-monotonicity relaxation above.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECORD_HEAD_BYTES = 23
POLICY_ENTRY_BYTES = 5
SQUARES = 64
PASS = 64

_HEAD = struct.Struct("<QQBBfB")
_POLICY_ENTRY = struct.Struct("<Bf")


@dataclass
class Positions:
    """Columnar decode of a dump file. One row per non-terminal position."""

    black: np.ndarray  # (N,) uint64
    white: np.ndarray  # (N,) uint64
    side: np.ndarray  # (N,) uint8, 0 = Black to move, 1 = White
    ply: np.ndarray  # (N,) uint8
    value: np.ndarray  # (N,) float32, side-to-move perspective, [-1, 1]
    # (N,) ragged: each entry a list of (square, probability) pairs.
    policy: list[list[tuple[int, float]]]

    def __len__(self) -> int:
        return len(self.value)


def game_slices(pos: Positions) -> list[slice]:
    """Recover contiguous games from their recorded plies.

    A dump is a concatenation of complete games. A game's first record has
    ply 0; unlike Connect Four, a pass does not advance Othello's disc-count
    ply, so within one game ply is only required to be non-decreasing, not
    ``+1`` per record.
    """
    if not len(pos):
        return []
    if int(pos.ply[0]) != 0:
        raise ValueError(f"first record has ply {int(pos.ply[0])}, expected 0")
    starts = np.flatnonzero(pos.ply == 0)
    out: list[slice] = []
    for i, start in enumerate(starts):
        end = int(starts[i + 1]) if i + 1 < len(starts) else len(pos)
        run = pos.ply[int(start) : end]
        if np.any(np.diff(run.astype(np.int64)) < 0):
            bad = int(np.flatnonzero(np.diff(run.astype(np.int64)) < 0)[0]) + 1
            raise ValueError(
                f"game starting at record {int(start)} has ply {int(run[bad])} "
                f"at offset {bad}, which is less than the previous record's ply "
                f"{int(run[bad - 1])}"
            )
        out.append(slice(int(start), end))
    return out


def select_rows(pos: Positions, rows: np.ndarray) -> Positions:
    """Return records at ``rows``, preserving their byte-stream fields."""
    return Positions(
        black=pos.black[rows],
        white=pos.white[rows],
        side=pos.side[rows],
        ply=pos.ply[rows],
        value=pos.value[rows],
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
    train_rows = np.concatenate(
        [np.arange(s.start, s.stop) for i, s in enumerate(games) if i not in validation_games]
    )
    validation_rows = np.concatenate(
        [np.arange(s.start, s.stop) for i, s in enumerate(games) if i in validation_games]
    )
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
        if side > 1 or not np.isfinite(value):
            raise ValueError(f"invalid othello v2 replay fields at byte {off}")
        off += RECORD_HEAD_BYTES
        tail = n_policy * POLICY_ENTRY_BYTES
        if off + tail > n:
            raise ValueError(f"truncated policy tail at byte {off}")
        entries: list[tuple[int, float]] = []
        for i in range(n_policy):
            square, prob = _POLICY_ENTRY.unpack_from(raw, off + i * POLICY_ENTRY_BYTES)
            entries.append((int(square), float(prob)))
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
        for square, prob in entries:
            out += _POLICY_ENTRY.pack(square, prob)
    return bytes(out)


def me_opp_bits(pos: Positions) -> tuple[np.ndarray, np.ndarray]:
    """``(N,)`` uint64 bitboards relative to the side to move: ``me`` is the
    mover's discs, ``opp`` the opponent's."""
    black_to_move = pos.side == 0
    me = np.where(black_to_move, pos.black, pos.white).astype(np.uint64)
    opp = np.where(black_to_move, pos.white, pos.black).astype(np.uint64)
    return me, opp


def concat(parts: list[Positions]) -> Positions:
    if len(parts) == 1:
        return parts[0]
    return Positions(
        black=np.concatenate([p.black for p in parts]),
        white=np.concatenate([p.white for p in parts]),
        side=np.concatenate([p.side for p in parts]),
        ply=np.concatenate([p.ply for p in parts]),
        value=np.concatenate([p.value for p in parts]),
        policy=[e for p in parts for e in p.policy],
    )
