"""Reader for the self-play shards ``games/gonnect/src/cnn/shard.rs`` writes, and the decoding of
their fields into network inputs. ``decode_planes`` must stay identical to
``games/gonnect/src/cnn/encode.rs::Fields::planes``; the Rust-written fixture in
``games/gonnect/cnn/fixtures`` is what keeps the two in step.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"GNCSHRD1"
HEADER_BYTES = 8 + 3 * 4
IN_PLANES = 7
FLAG_BLACK_TO_MOVE = 1
FLAG_SWAP_LEGAL = 2
FLAG_NO_MOVE_LEGAL = 4


def num_actions(size: int) -> int:
    return size * size + 2


def record_dtype(size: int) -> np.dtype:
    return np.dtype(
        [
            ("black", "<u8"),
            ("white", "<u8"),
            ("ko", "<u8"),
            ("legal", "<u8"),
            ("value", "<f4"),
            ("game", "<u4"),
            ("ply", "u1"),
            ("flags", "u1"),
            ("pad", "<u2"),
            ("policy", "<f4", (num_actions(size),)),
        ]
    )


def read_shard(path: str | Path) -> tuple[int, np.ndarray]:
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not a GNCSHRD1 shard")
    size, actions, per = struct.unpack_from("<3I", raw, 8)
    dtype = record_dtype(size)
    if actions != num_actions(size) or per != dtype.itemsize:
        raise ValueError(f"{path}: header disagrees with the record layout")
    body = len(raw) - HEADER_BYTES
    if body % per:
        raise ValueError(f"{path}: truncated record")
    return size, np.frombuffer(raw, dtype=dtype, offset=HEADER_BYTES, count=body // per)


def _bits(mask: np.ndarray, cells: int) -> np.ndarray:
    return ((mask[:, None] >> np.arange(cells, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


def decode_planes(records: np.ndarray, size: int) -> np.ndarray:
    """``(N, 7, size, size)`` uint8 planes, channels first (the torch layout)."""
    n, cells = len(records), size * size
    black_to_move = (records["flags"] & FLAG_BLACK_TO_MOVE).astype(bool)[:, None]
    black, white = _bits(records["black"], cells), _bits(records["white"], cells)
    planes = np.zeros((n, IN_PLANES, cells), dtype=np.uint8)
    planes[:, 0] = np.where(black_to_move, black, white)
    planes[:, 1] = np.where(black_to_move, white, black)
    planes[:, 2] = _bits(records["ko"], cells)
    planes[:, 3] = ((records["flags"] & FLAG_SWAP_LEGAL) != 0)[:, None]
    planes[:, 4] = black_to_move
    planes[:, 5] = _bits(records["legal"], cells)
    planes[:, 6] = 1
    return planes.reshape(n, IN_PLANES, size, size)


def decode_legal(records: np.ndarray, size: int) -> np.ndarray:
    """``(N, actions)`` bool mask of legal action ids: cells, then swap, then no-move."""
    cells = size * size
    legal = np.zeros((len(records), cells + 2), dtype=bool)
    legal[:, :cells] = _bits(records["legal"], cells).astype(bool)
    legal[:, cells] = (records["flags"] & FLAG_SWAP_LEGAL) != 0
    legal[:, cells + 1] = (records["flags"] & FLAG_NO_MOVE_LEGAL) != 0
    return legal


@dataclass
class Positions:
    """Decoded training data. ``game`` is unique across a run's shards (generation-offset)."""

    planes: np.ndarray  # (N, 7, size, size) uint8
    policy: np.ndarray  # (N, actions) float32
    legal: np.ndarray  # (N, actions) bool
    value: np.ndarray  # (N,) float32
    game: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return len(self.value)

    def slice(self, start: int, stop: int) -> Positions:
        return Positions(
            self.planes[start:stop],
            self.policy[start:stop],
            self.legal[start:stop],
            self.value[start:stop],
            self.game[start:stop],
        )

    def take(self, mask: np.ndarray) -> Positions:
        return Positions(
            self.planes[mask],
            self.policy[mask],
            self.legal[mask],
            self.value[mask],
            self.game[mask],
        )


def load_positions(path: str | Path, game_offset: int = 0) -> tuple[int, Positions]:
    size, records = read_shard(path)
    return size, Positions(
        planes=decode_planes(records, size),
        policy=np.ascontiguousarray(records["policy"]),
        legal=decode_legal(records, size),
        value=np.ascontiguousarray(records["value"]),
        game=records["game"].astype(np.int64) + game_offset,
    )
