"""Reader for the self-play shards ``games/druid/src/cnn/shard.rs`` writes, the decoding of their
fields into network inputs, and the board symmetries used for augmentation.

``decode_planes`` must stay identical to ``games/druid/src/cnn/encode.rs::planes``; the
Rust-written fixtures in ``games/druid/cnn/fixtures`` are what keeps the two in step. The shard
stores raw position fields only (no legal-cell mask): Druid's legality is a pure function of the
board and the mover (a sarsen goes on any empty cell or a stack the mover owns on top; a lintel
needs two of its three cells to carry the mover, its ends at equal height and its middle no
higher), so ``legal_cells`` re-derives it here, and the tests check it against the fixtures.

Symmetries: Black joins top and bottom, White left and right, so only reflections keep the
players' roles (quarter turns and transposes swap them). Those are 4 maps: identity, rows
flipped, columns flipped, both. A lintel's cell id is its first (left or top) cell, so the
reflection along its own axis sends anchor ``x`` to ``n - 3 - x``; see ``symmetry_sources``.
"""

from __future__ import annotations

import json
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"DRDSHRD1"
HEADER_BYTES = 8 + 3 * 4
IN_PLANES = 14  # the base encoding; ``CONNECT_PLANES`` appends the six connectivity planes
CONNECT_PLANES = 20
HEIGHT_SCALE = np.float32(8.0)
SYMMETRIES = 4

# Pending phase codes, as shard.rs writes them.
NONE, SARSEN_CHOSEN, LINTEL_CHOSEN, HORIZONTAL, VERTICAL = range(5)


def num_actions(size: int) -> int:
    """Cells, then choose-sarsen, choose-lintel, orient-horizontal, orient-vertical."""
    return size * size + 4


def record_dtype(size: int) -> np.dtype:
    """The current (v2) record layout, matching ``shard.rs``'s ``Record`` -- ``q`` is the mover's
    root search value at self-play time, the same sign convention as ``value``."""
    cells = size * size
    return np.dtype(
        [
            ("heights", "<u2", (cells,)),
            ("owners", "u1", (cells,)),  # 0 empty, 1 Black, 2 White (the top piece)
            ("hands", "u1", (4,)),  # Black sarsens, Black lintels, White sarsens, White lintels
            ("pending", "u1"),
            ("player", "u1"),  # 0 Black to move, 1 White
            ("ply", "<u2"),
            ("value", "<f4"),
            ("game", "<u4"),
            ("q", "<f4"),
            ("policy", "<f4", (num_actions(size),)),
        ]
    )


def record_dtype_v1(size: int) -> np.dtype:
    """The legacy layout, from before the root search value was recorded: identical to
    ``record_dtype`` but with no ``q`` field."""
    cells = size * size
    return np.dtype(
        [
            ("heights", "<u2", (cells,)),
            ("owners", "u1", (cells,)),
            ("hands", "u1", (4,)),
            ("pending", "u1"),
            ("player", "u1"),
            ("ply", "<u2"),
            ("value", "<f4"),
            ("game", "<u4"),
            ("policy", "<f4", (num_actions(size),)),
        ]
    )


def read_shard(path: str | Path) -> tuple[int, np.ndarray]:
    """Reads a v1 or v2 shard (the header's own ``record_bytes`` selects which); a v1 record has
    no stored ``q``, so it comes back with ``q`` set equal to ``value``."""
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not a DRDSHRD1 shard")
    size, actions, per = struct.unpack_from("<3I", raw, 8)
    if actions != num_actions(size):
        raise ValueError(f"{path}: header disagrees with the record layout")
    dtype = record_dtype(size)
    dtype_v1 = record_dtype_v1(size)
    body = len(raw) - HEADER_BYTES
    if per == dtype.itemsize:
        if body % per:
            raise ValueError(f"{path}: truncated record")
        return size, np.frombuffer(raw, dtype=dtype, offset=HEADER_BYTES, count=body // per)
    if per == dtype_v1.itemsize:
        if body % per:
            raise ValueError(f"{path}: truncated record")
        v1 = np.frombuffer(raw, dtype=dtype_v1, offset=HEADER_BYTES, count=body // per)
        records = np.zeros(len(v1), dtype=dtype)
        for name in dtype_v1.names:
            records[name] = v1[name]
        records["q"] = records["value"]
        return size, records
    raise ValueError(f"{path}: header disagrees with the record layout")


def write_shard(path: str | Path, size: int, records: np.ndarray) -> None:
    """Writes ``records`` (``record_dtype(size)``, any subset or reordering of a shard's rows) in
    the current (v2) ``DRDSHRD1`` format ``read_shard`` reads."""
    dtype = record_dtype(size)
    if records.dtype != dtype:
        records = records.astype(dtype)
    header = MAGIC + struct.pack("<3I", size, num_actions(size), dtype.itemsize)
    Path(path).write_bytes(header + records.tobytes())


def _mover(records: np.ndarray) -> np.ndarray:
    """Owner code of the side to move (1 Black, 2 White), ``(N,)``."""
    return (1 + records["player"].astype(np.int64)).astype(np.uint8)


def _lintel_legal(
    heights: np.ndarray, owners: np.ndarray, mover: np.ndarray, size: int, horizontal: bool
) -> np.ndarray:
    """``(N, cells)`` bool: anchors where the mover may place a lintel (hand ignored)."""
    n = len(heights)
    h = heights.reshape(n, size, size).astype(np.int64)
    o = owners.reshape(n, size, size)
    m = mover[:, None, None]
    span = size - 2
    if horizontal:
        sa, sb, sc = ((slice(None), slice(None), slice(k, span + k)) for k in (0, 1, 2))
    else:
        sa, sb, sc = ((slice(None), slice(k, span + k), slice(None)) for k in (0, 1, 2))
    ha, hb, hc = h[sa], h[sb], h[sc]
    oa, ob, oc = o[sa], o[sb], o[sc]
    shape_ok = (ha == hc) & (hb <= ha) & (oa != 0) & (oc != 0)
    count = (oa == m).astype(np.int8) + (oc == m) + ((ob == m) & (hb == ha))
    ok = shape_ok & (count == 2)
    out = np.zeros((n, size, size), dtype=bool)
    out[sa] = ok
    return out.reshape(n, size * size)


def sarsen_legal(owners: np.ndarray, mover: np.ndarray) -> np.ndarray:
    """``(N, cells)`` bool: cells a sarsen may go on (empty, or topped by the mover's piece)."""
    return (owners == 0) | (owners == mover[:, None])


def legal_cells(records: np.ndarray, size: int) -> np.ndarray:
    """``(N, cells)`` bool: cells legal for the pending cell decision, all False while a piece
    kind or an orientation is being chosen (the encoder's plane 3)."""
    heights, owners, mover = records["heights"], records["owners"], _mover(records)
    pending = records["pending"][:, None]
    out = np.zeros(heights.shape, dtype=bool)
    out |= (pending == SARSEN_CHOSEN) & sarsen_legal(owners, mover)
    out |= (pending == HORIZONTAL) & _lintel_legal(heights, owners, mover, size, True)
    out |= (pending == VERTICAL) & _lintel_legal(heights, owners, mover, size, False)
    return out


def decode_legal(records: np.ndarray, size: int) -> np.ndarray:
    """``(N, actions)`` bool mask of legal action ids, matching ``generate_actions``."""
    cells = size * size
    heights, owners, mover = records["heights"], records["owners"], _mover(records)
    pending = records["pending"]
    hands = records["hands"].astype(np.int64)
    own = np.where(records["player"][:, None] == 0, hands[:, 0:2], hands[:, 2:4])
    h_any = _lintel_legal(heights, owners, mover, size, True).any(axis=1)
    v_any = _lintel_legal(heights, owners, mover, size, False).any(axis=1)
    s_any = sarsen_legal(owners, mover).any(axis=1)
    legal = np.zeros((len(records), cells + 4), dtype=bool)
    legal[:, :cells] = legal_cells(records, size)
    start = pending == NONE
    legal[:, cells] = start & (own[:, 0] > 0) & s_any
    legal[:, cells + 1] = start & (own[:, 1] > 0) & (h_any | v_any)
    chose_lintel = pending == LINTEL_CHOSEN
    legal[:, cells + 2] = chose_lintel & h_any
    legal[:, cells + 3] = chose_lintel & v_any
    return legal


def _edge_distances(cost: np.ndarray, axis: int, far: bool) -> np.ndarray:
    """``(N, size, size)`` least path cost from one edge (row or column 0, or the last one when
    ``far``; rows for ``axis`` 1, columns for 2) to each cell, ends included, 4-connected."""
    size = cost.shape[1]
    dist = np.full(cost.shape, np.inf, dtype=np.float32)
    edge = [slice(None)] * 3
    edge[axis] = size - 1 if far else 0
    dist[tuple(edge)] = cost[tuple(edge)]
    while True:
        best = dist.copy()
        for ax in (1, 2):
            for shift in (1, -1):
                moved = np.full(dist.shape, np.inf, dtype=np.float32)
                src, dst = [slice(None)] * 3, [slice(None)] * 3
                src[ax], dst[ax] = (slice(0, -1), slice(1, None)) if shift == 1 else (
                    slice(1, None), slice(0, -1))  # fmt: skip
                moved[tuple(dst)] = dist[tuple(src)]
                best = np.minimum(best, moved + cost)
        if np.array_equal(best, dist):
            return dist
        dist = best


def _side_connectivity(
    top: np.ndarray, side: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One side's ``(through, near, total)``, mirroring ``encode.rs::side_connectivity``: ``top``
    is the ``(N, size, size)`` top-piece colour (0 empty, 1 Black, 2 White) and ``side`` the
    ``(N,)`` colour code (1 Black, whose edges are the rows; 2 White, the columns)."""
    own = top == side[:, None, None]
    cost = np.where(top == 0, 1.0, np.where(own, 0.0, 2.0)).astype(np.float32)
    black = (side == 1)[:, None, None]
    first = np.where(black, _edge_distances(cost, 1, False), _edge_distances(cost, 2, False))
    second = np.where(black, _edge_distances(cost, 1, True), _edge_distances(cost, 2, True))
    through = first + second - cost
    return through, np.minimum(first, second), through.reshape(len(cost), -1).min(axis=1)


def decode_planes(records: np.ndarray, size: int, in_planes: int = IN_PLANES) -> np.ndarray:
    """``(N, in_planes, size, size)`` float32 planes, channels first (the torch layout); the
    encoder's table is in ``games/druid/src/cnn/encode.rs``. ``in_planes`` picks the base
    (``IN_PLANES``) or the connectivity (``CONNECT_PLANES``) encoding."""
    if in_planes not in (IN_PLANES, CONNECT_PLANES):
        raise ValueError(f"no encoding with {in_planes} planes")
    n, cells = len(records), size * size
    mover = _mover(records)[:, None]
    owners = records["owners"]
    black = records["player"] == 0
    hands = records["hands"].astype(np.float32)
    own = np.where(black[:, None], hands[:, 0:2], hands[:, 2:4])
    opp = np.where(black[:, None], hands[:, 2:4], hands[:, 0:2])
    sarsen_start, lintel_start = np.float32(2 * cells), np.float32(cells)
    pending = records["pending"]
    planes = np.zeros((n, in_planes, cells), dtype=np.float32)
    planes[:, 0] = owners == mover
    planes[:, 1] = (owners != 0) & (owners != mover)
    planes[:, 2] = records["heights"].astype(np.float32) / HEIGHT_SCALE
    planes[:, 3] = legal_cells(records, size)
    planes[:, 4] = black[:, None]
    planes[:, 5] = (own[:, 0] / sarsen_start)[:, None]
    planes[:, 6] = (own[:, 1] / lintel_start)[:, None]
    planes[:, 7] = (opp[:, 0] / sarsen_start)[:, None]
    planes[:, 8] = (opp[:, 1] / lintel_start)[:, None]
    for plane, code in zip(
        range(9, 13), (SARSEN_CHOSEN, LINTEL_CHOSEN, HORIZONTAL, VERTICAL), strict=True
    ):
        planes[:, plane] = (pending == code)[:, None]
    planes[:, 13] = 1.0
    if in_planes == CONNECT_PLANES:
        mover_code = np.where(black, 1, 2)
        scale = np.float32(2 * size)
        top = owners.reshape(n, size, size)
        for at, total_at, side in ((14, 18, mover_code), (16, 19, 3 - mover_code)):
            through, near, total = _side_connectivity(top, side, size)
            planes[:, at] = through.reshape(n, cells) / scale
            planes[:, at + 1] = near.reshape(n, cells) / scale
            planes[:, total_at] = (total / scale)[:, None]
    return planes.reshape(n, in_planes, size, size)


def symmetry_sources(size: int) -> np.ndarray:
    """``(SYMMETRIES, 3, cells)`` gather indices: ``x_flat[..., src[s, k]]`` is reflection ``s``
    applied to a row-major grid, where ``k`` picks how the cells are read: 0 as board cells (also
    sarsen placements), 1 as horizontal-lintel anchors, 2 as vertical-lintel anchors. Symmetry 0
    is the identity, 1 flips rows, 2 flips columns, 3 both. Anchors that cannot exist (the last
    two along the lintel's axis) are mapped to each other so every map is a permutation; they are
    never legal, so nothing is read from them."""
    n = size
    rows, cols = np.divmod(np.arange(n * n), n)
    out = np.zeros((SYMMETRIES, 3, n * n), dtype=np.int64)
    for s in range(SYMMETRIES):
        flip_rows, flip_cols = bool(s & 1), bool(s & 2)
        for k, (span_x, span_y) in enumerate(((1, 1), (3, 1), (1, 3))):
            r2 = (n - span_y - rows) % n if flip_rows else rows
            c2 = (n - span_x - cols) % n if flip_cols else cols
            dest = r2 * n + c2  # old cell -> new cell; an involution, so also new -> old
            assert sorted(dest) == list(range(n * n))
            out[s, k] = dest
    return out


def phase_kind(pending: np.ndarray) -> np.ndarray:
    """Which of ``symmetry_sources``' three readings the pending phase's cell decision uses."""
    return np.where(pending == HORIZONTAL, 1, np.where(pending == VERTICAL, 2, 0))


@dataclass
class Positions:
    """Decoded training data. ``game`` is unique across a run's shards (generation-offset)."""

    planes: np.ndarray  # (N, 14, size, size) float32
    policy: np.ndarray  # (N, actions) float32
    legal: np.ndarray  # (N, actions) bool
    value: np.ndarray  # (N,) float32
    game: np.ndarray  # (N,) int64
    kind: np.ndarray  # (N,) int64, phase_kind of each position's pending phase
    q: np.ndarray  # (N,) float32, the mover's root search value (v1 shards: equal to value)

    def __len__(self) -> int:
        return len(self.value)

    def slice(self, start: int, stop: int) -> Positions:
        return self.take(slice(start, stop))

    def take(self, index: np.ndarray | slice) -> Positions:
        return Positions(
            self.planes[index],
            self.policy[index],
            self.legal[index],
            self.value[index],
            self.game[index],
            self.kind[index],
            self.q[index],
        )


def load_positions(
    path: str | Path, game_offset: int = 0, in_planes: int = IN_PLANES
) -> tuple[int, Positions]:
    size, records = read_shard(path)
    return size, Positions(
        planes=decode_planes(records, size, in_planes),
        policy=np.ascontiguousarray(records["policy"]),
        legal=decode_legal(records, size),
        value=np.ascontiguousarray(records["value"]),
        game=records["game"].astype(np.int64) + game_offset,
        kind=phase_kind(records["pending"]).astype(np.int64),
        q=np.ascontiguousarray(records["q"]),
    )


def read_terminal_sidecar(path: str | Path) -> list[dict]:
    """One dict per game played, from ``shard.rs``'s ``write_terminal_sidecar``
    (``<shard>.terminal.jsonl``): ``game``, ``winner`` (0 Black, 1 White, ``None`` for a draw or a
    capped game), ``capped``, and the final board's ``heights``/``owners`` (``size * size`` lists,
    in the same encoding as a record's fields, decoded here to ``int64`` arrays)."""
    games = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            row["heights"] = np.asarray(row["heights"], dtype=np.int64)
            row["owners"] = np.asarray(row["owners"], dtype=np.int64)
            games.append(row)
    return games


# owner code (`Fields.owners`/`Terminal.owners`) for each side, and the `winner` code the plan uses.
_BFS_SIDES = ((0, 1), (1, 2))  # (winner code, owner code): 0 Black, 1 White


def bfs_winner(owners: np.ndarray, size: int) -> tuple[int | None, list[int]]:
    """The winner of a final board's top-owner grid (``owners``, ``size * size``, in
    ``Fields.owners`` encoding: 0 empty, 1 Black, 2 White) by 4-connected BFS: Black connects row 0
    to row ``size - 1``, White column 0 to column ``size - 1`` (``connectivity.rs::Connectivity``).
    A lintel's placement repaints all three of its touched cells to the mover's color before this
    ever runs, so a plain per-cell scan of the final owners already sees the win; no lintel special
    case is needed here. Returns ``(winner, chain)``, the winning side (0 Black, 1 White, or
    ``None``) and its connecting chain's cell indices (row-major, start edge to far edge), or
    ``(None, [])`` if neither side has connected."""
    grid = owners.reshape(size, size)
    for winner, code in _BFS_SIDES:
        color = grid == code
        axis = winner  # Black (0) starts from a row, White (1) from a column.
        start = (
            [(0, c) for c in range(size) if color[0, c]]
            if axis == 0
            else [(r, 0) for r in range(size) if color[r, 0]]
        )
        if not start:
            continue
        parent: dict[tuple[int, int], tuple[int, int] | None] = dict.fromkeys(start)
        queue = deque(start)
        goal = None
        while queue:
            r, c = queue.popleft()
            if (axis == 0 and r == size - 1) or (axis == 1 and c == size - 1):
                goal = (r, c)
                break
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < size and 0 <= nc < size and color[nr, nc] and (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        if goal is not None:
            chain = []
            cur: tuple[int, int] | None = goal
            while cur is not None:
                chain.append(cur[0] * size + cur[1])
                cur = parent[cur]
            return winner, list(reversed(chain))
    return None, []


@dataclass
class AuxTargets:
    """One game's auxiliary training targets (``druid_cnn.fit``'s ownership/winning-chain heads),
    in the fixed (Black's-view) orientation; a batch multiplies ``owners_black`` by each record's
    own mover sign to get the mover's-view target ``fit`` trains against."""

    owners_black: np.ndarray  # (cells,) int8: +1 Black owns the cell, -1 White, 0 empty
    chain_mask: np.ndarray  # (cells,) float32: 1 on the winning chain's cells, else 0
    has_chain: bool  # a side actually connected (bfs_winner found a winner); else chain_mask is 0


def build_aux_targets(games: list[dict], size: int) -> dict[int, AuxTargets]:
    """``AuxTargets`` per game (keyed by its own ``game`` id, unoffset) from a shard's decoded
    terminal sidecar (``read_terminal_sidecar``). The winning chain comes from ``bfs_winner``, not
    the sidecar's own ``winner`` (they agree; this is the one source both ``druid_cnn`` and its
    tests use). A game with no winner (a draw or a capped game) gets an all-zero, fully-masked
    chain target."""
    out: dict[int, AuxTargets] = {}
    cells = size * size
    for row in games:
        winner, chain = bfs_winner(row["owners"], size)
        owners_black = np.zeros(cells, dtype=np.int8)
        owners_black[row["owners"] == 1] = 1
        owners_black[row["owners"] == 2] = -1
        chain_mask = np.zeros(cells, dtype=np.float32)
        if winner is not None:
            chain_mask[chain] = 1.0
        out[int(row["game"])] = AuxTargets(owners_black, chain_mask, winner is not None)
    return out
