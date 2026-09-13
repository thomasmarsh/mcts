"""Fast, deterministic checks for the numpy D4-equivariant policy sidecar
(`othello_eval.policy`) -- no self-play, no real training run here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from othello_eval.ntuple import D4, load_model_toml
from othello_eval.policy import INV, SQUARES, policy_logits
from othello_eval.records import RECORD_DTYPE

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_TOML = REPO_ROOT / "games/othello/ntuple/tests/tiny.toml"


def _positions(rows: list[tuple[int, int, int]]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=RECORD_DTYPE)
    for i, (black, white, side) in enumerate(rows):
        arr[i] = (black, white, side, 0, 0.0)
    return arr


def test_inv_is_the_inverse_permutation_of_d4() -> None:
    for k in range(8):
        assert (INV[k][D4[k]] == np.arange(SQUARES)).all()
        assert (D4[k][INV[k]] == np.arange(SQUARES)).all()


def test_zero_weights_are_uniform_over_every_square() -> None:
    geom = load_model_toml(TINY_TOML)
    weights = np.zeros(geom.n_weights * SQUARES)
    pos = _positions([(1 << 0, 1 << 9, 0)])
    out = policy_logits(weights, pos, geom)
    assert (out == 0.0).all()


def _rotate_bits(bits: int, sym: int) -> int:
    out = 0
    b = bits
    while b:
        i = (b & -b).bit_length() - 1
        b &= b - 1
        out |= 1 << int(D4[sym, i])
    return out


def test_policy_logits_is_d4_equivariant() -> None:
    geom = load_model_toml(TINY_TOML)
    n = geom.n_weights * SQUARES
    weights = np.sin(np.arange(n, dtype=np.float64) * 0.0007)

    black, white = (1 << 0) | (1 << 9) | (1 << 20), (1 << 27) | (1 << 36) | (1 << 45)
    base = policy_logits(weights, _positions([(black, white, 0)]), geom)[0]

    for k in range(8):
        rb, rw = _rotate_bits(black, k), _rotate_bits(white, k)
        got = policy_logits(weights, _positions([(rb, rw, 0)]), geom)[0]
        for sq in range(SQUARES):
            assert abs(got[int(D4[k, sq])] - base[sq]) < 1e-9, (k, sq)


def test_policy_logits_matches_the_rust_reference_fixture() -> None:
    """Same weights formula, geometry and state as
    `games/othello/src/policy.rs`'s `logits_match_python_reference_fixture`
    -- pins that the Rust hot path and this numpy trainer agree (within
    float tolerance) on the D4-symmetrized averaging."""
    geom = load_model_toml(TINY_TOML)
    n = geom.n_weights * SQUARES
    weights = ((np.arange(n, dtype=np.float64) - n / 2) * 0.00002).astype(np.float32).astype(
        np.float64
    )
    pos = _positions([((1 << 0) | (1 << 2), (1 << 1) | (1 << 7), 0)])
    got = policy_logits(weights, pos, geom)[0]
    expected = [
        -0.008340000036696438,
        -0.008339999985764734,
        -0.008339999963936862,
        -0.00834000005852431,
        -0.008340000051248353,
        -0.008339999956660904,
        -0.00833999997121282,
        -0.00834000005852431,
    ]
    for actual, want in zip(got[:8].tolist(), expected, strict=True):
        assert abs(actual - want) < 1e-6, (actual, want)
