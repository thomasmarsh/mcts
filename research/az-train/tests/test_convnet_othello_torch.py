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
import pytest
import torch
from othello_eval.convnet import (
    BLOCKS,
    CHANNELS,
    N_WEIGHTS,
    VALUE_HIDDEN,
    _literal_loss_gradient_k,  # pyright: ignore[reportPrivateUsage]
    initial_weights,
    initial_weights_k,
    n_weights_for,
    predict,
    predict_k,
)

from az_train.convnet_othello_torch import (
    OTCNN001Torch,
    TrainingStalledError,
    _cosine_lr,  # pyright: ignore[reportPrivateUsage]
    fit_torch,
)


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


def test_one_adam_step_matches_numpy_with_weight_decay_equal_to_2l2() -> None:
    """Proves ``fit_torch``'s L2 equivalence claim (module docstring) end to
    end, rather than leaving it as an asserted derivation: one Adam step
    from the same initial weights, on the same batch, with L2 folded into
    the loss the same way ``fit_torch`` does, must match one step of
    ``othello_eval.convnet``'s own hand-derived-gradient Adam step at the
    same ``l2`` -- including for the policy head, which this loss never
    touches directly but which ``fit_k`` still regularizes unconditionally
    (the reason ``fit_torch`` folds L2 into the loss instead of using
    ``torch.optim.Adam``'s ``weight_decay``, which would silently skip any
    parameter with no gradient from the loss)."""
    l2 = 1e-2  # deliberately large so a wrong factor-of-2 would be obvious
    learning_rate = 2e-3
    seed = 7
    rng = np.random.default_rng(seed)
    n = 32
    me = (rng.random((n, 64)) > 0.7).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.7).astype(np.float32) * (1.0 - me)
    value = rng.uniform(-1.0, 1.0, size=n)

    init = initial_weights_k(seed, BLOCKS, False, CHANNELS, VALUE_HIDDEN)

    # numpy: one Adam step, matching fit_k's update exactly (step=1, so the
    # bias-correction denominators are 1 - beta1 and 1 - beta2).
    _, gradient = _literal_loss_gradient_k(init.copy(), me, opp, value, l2, BLOCKS, False)
    beta1, beta2 = 0.9, 0.999
    moment = (1.0 - beta1) * gradient
    velocity = (1.0 - beta2) * gradient * gradient
    denom = np.sqrt(velocity / (1.0 - beta2)) + 1e-8
    numpy_after = init - learning_rate * (moment / (1.0 - beta1)) / denom

    # torch: one Adam step over autograd, same init, same batch, L2 folded
    # into the loss exactly as fit_torch does.
    model = OTCNN001Torch()
    model.load_from_flat(init)
    weight_params = [p for name, p in model.named_parameters() if not name.endswith(".bias")]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8
    )
    me_t = torch.as_tensor(me, dtype=torch.float32)
    opp_t = torch.as_tensor(opp, dtype=torch.float32)
    value_t = torch.as_tensor(value, dtype=torch.float32)
    optimizer.zero_grad()
    prediction, _ = model.forward_literal(me_t, opp_t)
    reg = sum((p * p).sum() for p in weight_params)
    loss = torch.mean((prediction - value_t) ** 2) + l2 * reg
    loss.backward()
    optimizer.step()
    torch_after = model.to_flat()

    assert np.allclose(torch_after, numpy_after, atol=1e-4, rtol=1e-3), (
        np.max(np.abs(torch_after - numpy_after))
    )


def test_weight_decay_alone_would_have_missed_the_untouched_policy_head() -> None:
    """Regression test for the bug ``fit_torch``'s docstring describes:
    using ``torch.optim.Adam(weight_decay=2*l2)`` for a value-only loss
    leaves the policy head's ``.grad`` as ``None`` (never touched by the
    loss), so the optimizer skips it entirely -- unlike ``fit_k``, which
    regularizes it unconditionally. This pins that gap so it can't
    regress back in silently."""
    l2 = 1e-2
    init = initial_weights_k(0, BLOCKS, False, CHANNELS, VALUE_HIDDEN)
    model = OTCNN001Torch()
    model.load_from_flat(init)
    before = model.policy_dense.weight.detach().clone()

    weight_params = [p for name, p in model.named_parameters() if not name.endswith(".bias")]
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=2.0 * l2)
    me = torch.zeros((1, 64))
    opp = torch.zeros((1, 64))
    value = torch.zeros((1,))
    optimizer.zero_grad()
    prediction, _ = model.forward_literal(me, opp)
    loss = torch.mean((prediction - value) ** 2)  # policy output unused -> policy params ungraphed
    loss.backward()
    assert all(p.grad is None for p in weight_params if p is model.policy_dense.weight)
    optimizer.step()
    assert torch.equal(model.policy_dense.weight, before), (
        "weight_decay alone should leave an untouched parameter unchanged"
    )


def test_k4_tied_geometry_flat_round_trips_and_matches_numpy_predict_k() -> None:
    """Generalization check for ``blocks``/``tied``/``channels``/
    ``value_hidden``, mirroring ``test_predict_k_matches_predict_for_the_
    baseline_geometry``-style parity: a non-default geometry (4 blocks,
    weight-tied, 16 channels) must round-trip through ``load_from_flat``/
    ``to_flat`` and match ``othello_eval.convnet.predict_k`` on the same
    weights, not just the ``OTCNN001``-default geometry."""
    blocks, tied, channels, value_hidden = 4, True, CHANNELS, VALUE_HIDDEN
    weights = initial_weights_k(
        seed=5, blocks=blocks, tied=tied, channels=channels, value_hidden=value_hidden
    )
    model = OTCNN001Torch(blocks, tied, channels, value_hidden)
    model.load_from_flat(weights)
    assert np.array_equal(model.to_flat(), weights)
    assert model.to_flat().shape == (n_weights_for(blocks, tied, channels, value_hidden),)

    rng = np.random.default_rng(6)
    n = 5
    me = (rng.random((n, 64)) > 0.7).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.7).astype(np.float32) * (1.0 - me)
    torch_value, torch_policy = model.predict(me, opp)
    numpy_value, numpy_policy = predict_k(weights, me, opp, blocks, tied, channels, value_hidden)
    assert np.allclose(torch_value, numpy_value, atol=1e-5)
    assert np.allclose(torch_policy, numpy_policy, atol=1e-5)


def test_cosine_lr_is_constant_through_first_half_then_decays_to_zero() -> None:
    epochs = 20
    for epoch in range(1, epochs // 2 + 1):
        assert _cosine_lr(epoch, epochs, 2e-3, decay=True) == 2e-3
    assert _cosine_lr(epochs, epochs, 2e-3, decay=True) == pytest.approx(0.0, abs=1e-12)
    mid = _cosine_lr(15, epochs, 2e-3, decay=True)
    assert 0.0 < mid < 2e-3
    # disabled: stays constant everywhere, including the back half.
    assert _cosine_lr(epochs, epochs, 2e-3, decay=False) == 2e-3


def test_fit_torch_reports_best_checkpoint_alongside_final_epoch() -> None:
    """``fit_torch``'s new best-validation-checkpoint tracking: the reported
    best-checkpoint weights must actually reproduce the best-checkpoint
    validation metrics when reloaded independently, and the best epoch must
    be the epoch in ``validation_epoch_trace`` with the lowest ``value_mse``
    -- not just "some earlier snapshot"."""
    rng = np.random.default_rng(21)
    n = 256
    me = (rng.random((n, 64)) > 0.6).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.6).astype(np.float32) * (1.0 - me)
    value = np.tanh((me.sum(axis=1) - opp.sum(axis=1)) / 8.0).astype(np.float64)
    split = n * 3 // 4
    train_slice, val_slice = slice(0, split), slice(split, n)

    _model, metadata = fit_torch(
        me[train_slice], opp[train_slice], value[train_slice],
        (me[val_slice], opp[val_slice], value[val_slice]),
        l2=1e-4, seed=0, batch_size=64, epochs=20, learning_rate=5e-3,
        report_every=0,
    )
    trace = metadata["validation_epoch_trace"]
    best_trace_mse = min(m["value_mse"] for m in trace)
    best_checkpoint = metadata["best_checkpoint_validation_metrics"]
    assert best_checkpoint["value_mse"] == pytest.approx(best_trace_mse)
    assert 1 <= metadata["best_checkpoint_epoch"] <= 20

    reloaded = OTCNN001Torch(BLOCKS, False, CHANNELS, VALUE_HIDDEN)
    reloaded.load_from_flat(metadata["best_checkpoint_weights"])
    val_prediction, _ = reloaded.predict(me[val_slice], opp[val_slice])
    reloaded_mse = float(np.mean((val_prediction - value[val_slice]) ** 2))
    assert reloaded_mse == pytest.approx(best_checkpoint["value_mse"], abs=1e-5)


def test_fit_torch_reduces_held_out_mse_on_a_tiny_synthetic_fit() -> None:
    """Smoke test for the training loop's plumbing (batching, validation,
    metadata shape) -- not a strength claim, just "does it learn at all" on
    a tiny synthetic problem where the target is a simple function of the
    input the network can actually fit in a handful of epochs."""
    rng = np.random.default_rng(11)
    n = 256
    me = (rng.random((n, 64)) > 0.6).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.6).astype(np.float32) * (1.0 - me)
    value = np.tanh((me.sum(axis=1) - opp.sum(axis=1)) / 8.0).astype(np.float64)
    split = n * 3 // 4
    train_slice, val_slice = slice(0, split), slice(split, n)

    _model, metadata = fit_torch(
        me[train_slice], opp[train_slice], value[train_slice],
        (me[val_slice], opp[val_slice], value[val_slice]),
        l2=1e-4, seed=0, batch_size=64, epochs=40, learning_rate=5e-3,
        report_every=0,
    )
    first_epoch_mse = metadata["validation_epoch_trace"][0]["value_mse"]
    final_mse = metadata["final_validation_metrics"]["value_mse"]
    assert final_mse < first_epoch_mse
    assert metadata["optimizer_steps"] == 40 * ((split + 63) // 64)


def _tiny_synthetic_split() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    rng = np.random.default_rng(11)
    n = 256
    me = (rng.random((n, 64)) > 0.6).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.6).astype(np.float32) * (1.0 - me)
    value = np.tanh((me.sum(axis=1) - opp.sum(axis=1)) / 8.0).astype(np.float64)
    split = n * 3 // 4
    train_slice, val_slice = slice(0, split), slice(split, n)
    return (
        me[train_slice], opp[train_slice], value[train_slice],
        (me[val_slice], opp[val_slice], value[val_slice]),
    )


def test_stall_check_raises_when_pearson_never_escapes_threshold() -> None:
    """A dead-init run sits at ~0 pearson for the entire fit. An unreachable
    threshold (2.0, pearson is bounded in [-1, 1]) simulates that -- the fit
    must stop at the checked epoch instead of running to completion."""
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    with pytest.raises(TrainingStalledError) as exc_info:
        fit_torch(
            me_tr, opp_tr, value_tr, validation,
            l2=1e-4, seed=0, batch_size=64, epochs=40, learning_rate=5e-3,
            report_every=0, stall_check=(2, 2.0),
        )
    assert exc_info.value.epoch == 2


def test_stall_check_does_not_raise_once_check_epoch_is_never_reached() -> None:
    """A ``stall_check`` epoch beyond the fit's total ``epochs`` is never
    reached, so it must never raise regardless of how training actually
    goes -- the check is otherwise a no-op, matching every pre-existing
    caller's behavior with ``stall_check=None``."""
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    _model, metadata = fit_torch(
        me_tr, opp_tr, value_tr, validation,
        l2=1e-4, seed=0, batch_size=64, epochs=40, learning_rate=5e-3,
        report_every=0, stall_check=(1000, 0.05),
    )
    assert len(metadata["validation_epoch_trace"]) == 40
