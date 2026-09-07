# pyright: reportUnknownMemberType=false
import numpy as np

from az_train.policy_c4 import POLICY_WEIGHTS, logits, metrics, targets


def test_zero_sidecar_is_uniform_on_the_legal_columns() -> None:
    raw = logits(np.zeros(POLICY_WEIGHTS, np.float32), np.zeros((1, 42)), np.zeros((1, 42)))
    target, legal = targets([[(0, 0.5), (3, 0.5)]])
    result = metrics(raw, target, legal)
    assert abs(result["cross_entropy"] - np.log(2.0)) < 1e-7


def test_symmetry_average_is_exactly_mirror_equivariant() -> None:
    weights = np.sin(np.arange(POLICY_WEIGHTS, dtype=np.float32))
    me = np.zeros((1, 42), np.float32)
    opp = np.zeros((1, 42), np.float32)
    me[0, [0, 9]] = 1
    opp[0, [1, 20]] = 1
    mirror_me = me.reshape(1, 6, 7)[:, :, ::-1].reshape(1, 42)
    mirror_opp = opp.reshape(1, 6, 7)[:, :, ::-1].reshape(1, 42)
    assert np.allclose(logits(weights, me, opp), logits(weights, mirror_me, mirror_opp)[:, ::-1])


def test_logits_match_rust_reference_fixture() -> None:
    weights = ((np.arange(POLICY_WEIGHTS, dtype=np.float64) - POLICY_WEIGHTS / 2) * 0.00002).astype(
        np.float32
    )
    me = np.zeros((1, 42), np.float32)
    opp = np.zeros((1, 42), np.float32)
    me[0, [0, 2]] = 1
    opp[0, [1, 7]] = 1
    expected = [
        -0.75586009,
        -0.75586015,
        -0.75585949,
        -0.75585961,
        -0.75585949,
        -0.75586015,
        -0.75586045,
    ]
    assert np.allclose(logits(weights, me, opp)[0], expected, atol=1e-6)
