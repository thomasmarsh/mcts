"""Reader for ``game-othello dump`` position records.

The Rust ``dump`` subcommand writes fixed-width little-endian records, one
per self-play position:

===========  ========  ======
field        type      bytes
===========  ========  ======
black        u64 LE    8
white        u64 LE    8
side         u8        1
ply          u8        1
target       f32 LE    4
===========  ========  ======

22 bytes per record, packed with no padding. ``side`` is 0 for black to
move, 1 for white to move. ``ply`` is the number of discs on the board
minus 4 (0 at the opening). ``target`` is the final game result from the
side-to-move player's perspective: ``+1.0`` win, ``-1.0`` loss, ``0.0``
draw.

Only ``--label outcome`` dumps (uniform-random self-play, outcome labels)
are produced today; search-labelled harvests are not implemented yet.
"""

from __future__ import annotations

import numpy as np

RECORD_DTYPE: np.dtype = np.dtype(
    [
        ("black", "<u8"),
        ("white", "<u8"),
        ("side", "u1"),
        ("ply", "u1"),
        ("target", "<f4"),
    ],
    align=False,
)

# The single cheapest guard against a silent Rust/Python format skew.
assert RECORD_DTYPE.itemsize == 22, RECORD_DTYPE.itemsize


def load_positions(path: str) -> np.ndarray:
    """Load every record from a dump file into a structured array."""
    return np.fromfile(path, dtype=RECORD_DTYPE)
