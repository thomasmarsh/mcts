"""Numpy counterpart of the Rust D4-equivariant n-tuple policy sidecar
(``games/othello/src/policy.rs``).

Same active tuple features as the value head (``othello_eval.ntuple``), but
with one weight *row* of 64 columns (one per board square) per feature
instead of a scalar weight. A row's columns are indexed in the canonical
(D4 identity) frame; evaluating a real position averages the row over all 8
orientations, mapping each orientation's canonical-frame output back to real
board squares via the inverse of that orientation's permutation -- see the
Rust module doc for the derivation.
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeArgument=false

from __future__ import annotations

import numpy as np

from othello_eval.ntuple import D4, ModelGeometry, featurize

SQUARES = 64


def _inv_table() -> np.ndarray:
    """``(8, 64)``: ``INV[k, D4[k, i]] == i`` for every orientation `k`."""
    inv = np.zeros((8, SQUARES), dtype=np.int64)
    for k in range(8):
        inv[k, D4[k]] = np.arange(SQUARES)
    return inv


INV = _inv_table()


def policy_logits(weights: np.ndarray, positions: np.ndarray, geom: ModelGeometry) -> np.ndarray:
    """``(N, 64)`` D4-symmetrized per-square logits.

    ``weights`` is the flat ``(geom.n_weights * 64,)`` sidecar; reshaped here
    to ``(n_weights, 64)`` rows indexed in the canonical frame.
    """
    feat_idx = featurize(positions, geom)  # (N, n_tuples * 8), tuple-major then orientation
    w = weights.reshape(-1, SQUARES)
    out = np.zeros((positions.shape[0], SQUARES), dtype=np.float64)
    for sym in range(8):
        # Column `t * 8 + sym` is tuple `t`'s active feature under `sym`.
        idx_sym = feat_idx[:, sym::8]  # (N, n_tuples)
        raw = w[idx_sym].sum(axis=1)  # (N, 64), canonical frame
        out += raw[:, INV[sym]]
    return out / 8.0
