# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeStubs=false
"""Forward-pass parity checks for ``az_train.convnet_othello_torch``: no
training loop here, just proving the PyTorch reimplementation of
``OTCNN001`` computes the same function the numpy/Rust sides already agree
on.
"""

from __future__ import annotations

import numpy as np
from othello_eval.convnet import N_WEIGHTS, initial_weights, predict

from az_train.convnet_othello_torch import OTCNN001Torch


def test_flat_round_trips_through_load_and_to_flat() -> None:
    weights = initial_weights(seed=1)
    model = OTCNN001Torch()
    model.load_from_flat(weights)
    assert np.array_equal(model.to_flat(), weights)


def test_predict_matches_numpy_on_a_random_batch() -> None:
    weights = initial_weights(seed=2)
    rng = np.random.default_rng(3)
    n = 5
    me = (rng.random((n, 64)) > 0.7).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.7).astype(np.float32) * (1.0 - me)

    model = OTCNN001Torch()
    model.load_from_flat(weights)
    torch_value, torch_policy = model.predict(me, opp)
    numpy_value, numpy_policy = predict(weights, me, opp)

    assert np.allclose(torch_value, numpy_value, atol=1e-5)
    assert np.allclose(torch_policy, numpy_policy, atol=1e-5)


def test_value_matches_the_cross_language_reference_fixture() -> None:
    """Same weights formula and state as ``othello_eval.convnet``'s
    ``test_value_matches_the_rust_reference_fixture`` and
    ``games/othello/src/convnet.rs``'s ``value_matches_python_reference_
    fixture``. Loading the exact pinned weight vector into
    ``OTCNN001Torch`` and matching this value transitively proves torch ==
    numpy == Rust without a third hand-written fixture."""
    weights = np.array(
        [(i - N_WEIGHTS / 2) * 1e-6 for i in range(N_WEIGHTS)], dtype=np.float32
    )
    black = (1 << 0) | (1 << 2) | (1 << 8)
    white = (1 << 1) | (1 << 7)
    me = np.array([[(black >> j) & 1 for j in range(64)]], dtype=np.float32)
    opp = np.array([[(white >> j) & 1 for j in range(64)]], dtype=np.float32)

    model = OTCNN001Torch()
    model.load_from_flat(weights)
    value, _policy = model.predict(me, opp)
    assert abs(float(value[0]) - 0.0042488095) < 1e-6


def test_policy_matches_the_cross_language_reference_fixture() -> None:
    """Same weights/state as ``othello_eval.convnet``'s
    ``test_policy_matches_the_rust_reference_fixture``."""
    weights = np.array(
        [(i - N_WEIGHTS / 2) * 1e-6 for i in range(N_WEIGHTS)], dtype=np.float32
    )
    black = (1 << 0) | (1 << 2) | (1 << 8)
    white = (1 << 1) | (1 << 7)
    me = np.array([[(black >> j) & 1 for j in range(64)]], dtype=np.float32)
    opp = np.array([[(white >> j) & 1 for j in range(64)]], dtype=np.float32)

    model = OTCNN001Torch()
    model.load_from_flat(weights)
    _value, policy = model.predict(me, opp)
    expected = [
        0.009362561628222466,
        0.00936256255954504,
        0.00936256255954504,
        0.00936256255954504,
        0.00936256255954504,
        0.00936256255954504,
        0.00936256255954504,
        0.009362561628222466,
    ]
    for actual, want in zip(policy[0, :8], expected, strict=True):
        assert abs(float(actual) - want) < 1e-6, (actual, want)
