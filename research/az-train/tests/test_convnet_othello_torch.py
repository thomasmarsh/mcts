# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeStubs=false
"""Forward-pass parity checks for ``az_train.convnet_othello_torch``: no
training loop here, just proving the PyTorch reimplementation of
``OTCNN001`` computes the same function the numpy/Rust sides already agree
on.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from othello_eval.convnet import (
    BLOCKS,
    CHANNELS,
    COLUMNS,
    VALUE_HIDDEN,
    _literal_loss_gradient_k,  # pyright: ignore[reportPrivateUsage]
    initial_weights,
    initial_weights_k,
    kaiming_weights_k,
    n_weights_for,
    predict,
    predict_k,
)

from az_train.convnet_othello_torch import (
    BLOCKS as PRODUCTION_BLOCKS,
)
from az_train.convnet_othello_torch import (
    CHANNELS as PRODUCTION_CHANNELS,
)
from az_train.convnet_othello_torch import (
    INIT_FUNCTIONS,
    AllSeedsStalledError,
    OTCNN001Torch,
    StallCheck,
    TrainingStalledError,
    _cosine_lr,  # pyright: ignore[reportPrivateUsage]
    _lr_schedule,  # pyright: ignore[reportPrivateUsage]
)
from az_train.convnet_othello_torch import fit_torch as _fit_torch_production
from az_train.convnet_othello_torch import fit_torch_with_retry as _fit_torch_with_retry_production

# The trainer's defaults are the production geometry (a 1.78M-weight net);
# these tests exercise the trainer's mechanics and its parity with the numpy
# reference module, which are geometry-independent, so they run at that
# module's small reference geometry to stay fast. Only the cross-language
# fixtures and the default-geometry checks below use the production one.
fit_torch = partial(_fit_torch_production, blocks=BLOCKS, channels=CHANNELS)
fit_torch_with_retry = partial(_fit_torch_with_retry_production, blocks=BLOCKS, channels=CHANNELS)


def _reference_model() -> OTCNN001Torch:
    return OTCNN001Torch(BLOCKS, False, CHANNELS, VALUE_HIDDEN)


def test_flat_round_trips_through_load_and_to_flat() -> None:
    weights = initial_weights(seed=1)
    model = _reference_model()
    model.load_from_flat(weights)
    assert np.array_equal(model.to_flat(), weights)


def test_predict_matches_numpy_on_a_random_batch() -> None:
    weights = initial_weights(seed=2)
    rng = np.random.default_rng(3)
    n = 5
    me = (rng.random((n, 64)) > 0.7).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.7).astype(np.float32) * (1.0 - me)

    model = _reference_model()
    model.load_from_flat(weights)
    torch_value, torch_policy = model.predict(me, opp)
    numpy_value, numpy_policy = predict(weights, me, opp)

    assert np.allclose(torch_value, numpy_value, atol=1e-5)
    assert np.allclose(torch_policy, numpy_policy, atol=1e-5)


def test_predict_chunking_does_not_change_the_result() -> None:
    """``predict``'s internal chunking (bounds peak memory for a large N by
    running the forward pass in pieces instead of all rows at once) must be
    purely a memory-shape change -- each row's D4-averaged prediction is
    independent of every other row's, so a tiny chunk size must produce the
    same output as one big chunk (within float32 tolerance -- different
    batch sizes hit different BLAS/conv op orderings, so this is not
    bit-exact)."""
    weights = initial_weights(seed=4)
    rng = np.random.default_rng(5)
    n = 37
    me = (rng.random((n, 64)) > 0.7).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.7).astype(np.float32) * (1.0 - me)

    model = _reference_model()
    model.load_from_flat(weights)
    whole_value, whole_policy = model.predict(me, opp, chunk_size=10_000)
    chunked_value, chunked_policy = model.predict(me, opp, chunk_size=5)

    assert np.allclose(whole_value, chunked_value, atol=1e-5)
    assert np.allclose(whole_policy, chunked_policy, atol=1e-5)


def _splitmix_weights(n: int, amplitude: float) -> np.ndarray:
    """The identical splitmix64 generator ``games/othello/src/convnet.rs``'s
    ``splitmix_weights`` and ``othello_eval``'s ``test_convnet.py`` use."""
    i = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = i * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return ((u * 2.0 - 1.0) * amplitude).astype(np.float32)


def _production_fixture_prediction() -> tuple[np.ndarray, np.ndarray]:
    model = OTCNN001Torch()
    n = n_weights_for(model.n_blocks, False, model.channels)
    model.load_from_flat(_splitmix_weights(n, 0.07))
    black = (1 << 0) | (1 << 2) | (1 << 8)
    white = (1 << 1) | (1 << 7)
    me = np.array([[(black >> j) & 1 for j in range(64)]], dtype=np.float32)
    opp = np.array([[(white >> j) & 1 for j in range(64)]], dtype=np.float32)
    return model.predict(me, opp)


def test_default_geometry_is_the_rust_production_geometry() -> None:
    """``games/othello/src/convnet.rs``'s ``CHANNELS``/``BLOCKS`` and this
    trainer's defaults must move together: ``CnnValueNet::load`` rejects a
    checkpoint of any other geometry, so a mismatch here means every
    checkpoint the trainer writes is unloadable. 1_779_971 is that file's
    ``CNN_WEIGHTS``."""
    model = OTCNN001Torch()
    assert (model.n_blocks, model.channels) == (6, 128)
    assert (PRODUCTION_BLOCKS, PRODUCTION_CHANNELS) == (6, 128)
    assert n_weights_for(model.n_blocks, False, model.channels, model.value_hidden) == 1_779_971


def test_value_matches_the_cross_language_reference_fixture() -> None:
    """Same weights and state as ``othello_eval.convnet``'s
    ``test_value_matches_the_rust_reference_fixture`` and
    ``games/othello/src/convnet.rs``'s ``value_matches_python_reference_
    fixture``, run at the production geometry through this module's torch
    model: matching this value transitively proves torch == numpy == Rust
    without a third hand-written fixture."""
    value, _policy = _production_fixture_prediction()
    assert abs(float(value[0]) - 0.036056604236364365) < 1e-5


def test_policy_matches_the_cross_language_reference_fixture() -> None:
    """Same weights/state as ``othello_eval.convnet``'s
    ``test_policy_matches_the_rust_reference_fixture``."""
    _value, policy = _production_fixture_prediction()
    expected = [
        0.00868706963956356,
        0.001992151839658618,
        -0.02375856600701809,
        0.001324896002188325,
        0.004524925723671913,
        -0.027064848691225052,
        0.004268915392458439,
        0.011740943416953087,
    ]
    for actual, want in zip(policy[0, :8], expected, strict=True):
        assert abs(float(actual) - want) < 1e-5, (actual, want)


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
    model = _reference_model()
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
    model = _reference_model()
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


def test_stall_check_raises_at_its_step_when_pearson_never_escapes_threshold() -> None:
    """An unreachable threshold (2.0, pearson is bounded in [-1, 1]) must
    stop the fit at exactly the checked optimizer step, mid-epoch, instead of
    running to completion."""
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    steps_per_epoch = -(-len(me_tr) // 64)
    check_step = steps_per_epoch // 2
    assert 0 < check_step < steps_per_epoch
    with pytest.raises(TrainingStalledError) as exc_info:
        fit_torch(
            me_tr, opp_tr, value_tr, validation,
            l2=1e-4, seed=0, batch_size=64, epochs=40, learning_rate=5e-3,
            report_every=0, stall_check=StallCheck(check_step, 2.0),
        )
    assert exc_info.value.step == check_step


def test_stall_check_catches_a_reproduced_dead_seed_within_its_step_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production failure signature: every weight and bias zero gives a
    constant network whose value pearson is exactly 0.0 and which never
    recovers (all-zero conv weights kill every ReLU path, so no gradient
    reaches the trunk). The check must fire at its configured step, not after
    the fit's epochs, while an identical fit from a live init passes it."""
    def zero_init(_seed: int, blocks: int, tied: bool, channels: int, hidden: int) -> np.ndarray:
        return np.zeros(n_weights_for(blocks, tied, channels, hidden), dtype=np.float32)

    monkeypatch.setitem(INIT_FUNCTIONS, "dead", zero_init)
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    check = StallCheck(step=8, min_abs_pearson=0.05)
    with pytest.raises(TrainingStalledError) as exc_info:
        fit_torch(
            me_tr, opp_tr, value_tr, validation, l2=1e-4, seed=0, batch_size=64, epochs=40,
            learning_rate=5e-3, report_every=0, init="dead", stall_check=check,
            trace_steps=(1, 4),
        )
    assert exc_info.value.step == 8
    assert exc_info.value.pearson == 0.0

    _model, metadata = fit_torch(
        me_tr, opp_tr, value_tr, validation, l2=1e-4, seed=0, batch_size=64, epochs=3,
        learning_rate=5e-3, report_every=0, stall_check=StallCheck(8, 0.0), trace_steps=(1, 4),
    )
    assert [t["step"] for t in metadata["step_trace"]] == [1, 4]


def test_stall_check_step_beyond_the_fit_is_never_reached() -> None:
    """A ``stall_check`` step beyond the fit's total steps never fires."""
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    _model, metadata = fit_torch(
        me_tr, opp_tr, value_tr, validation,
        l2=1e-4, seed=0, batch_size=64, epochs=40, learning_rate=5e-3,
        report_every=0, stall_check=StallCheck(10**9, 0.05),
    )
    assert len(metadata["validation_epoch_trace"]) == 40


def test_fit_torch_init_kaiming_starts_from_kaiming_weights_k() -> None:
    """``init="kaiming"`` must actually change what ``fit_torch`` loads, not
    just be accepted and ignored. ``learning_rate=0.0`` makes every Adam
    step's update exactly zero regardless of the gradient, so the
    post-"training" weights are still the initial ones -- an end-to-end
    check that doesn't require reaching into ``fit_torch``'s internals."""
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    expected = kaiming_weights_k(0, BLOCKS, False, CHANNELS, VALUE_HIDDEN)
    model, metadata = fit_torch(
        me_tr, opp_tr, value_tr, validation,
        l2=0.0, seed=0, batch_size=len(me_tr), epochs=1, learning_rate=0.0,
        report_every=0, init="kaiming", lr_decay=False,
    )
    assert np.allclose(model.to_flat(), expected, atol=1e-6)
    assert metadata["optimizer_init"] == "kaiming"


def test_fit_torch_grad_clip_norm_invokes_clip_grad_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    real_clip = torch.nn.utils.clip_grad_norm_
    calls: list[float] = []

    def spy(parameters: object, max_norm: float) -> object:
        calls.append(max_norm)
        return real_clip(parameters, max_norm)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    fit_torch(
        me_tr, opp_tr, value_tr, validation,
        l2=1e-4, seed=0, batch_size=64, epochs=1, learning_rate=1e-3,
        report_every=0, grad_clip_norm=0.5,
    )
    assert calls and all(c == 0.5 for c in calls)


def test_fit_torch_without_grad_clip_norm_never_calls_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def spy(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    fit_torch(
        me_tr, opp_tr, value_tr, validation,
        l2=1e-4, seed=0, batch_size=64, epochs=1, learning_rate=1e-3, report_every=0,
    )
    assert not called


def test_lr_schedule_warmup_ramps_linearly_then_hands_off_to_cosine() -> None:
    epochs, warmup, base = 20, 4, 2e-3
    kw: dict[str, object] = dict(decay=True, warmup_epochs=warmup)
    assert _lr_schedule(1, epochs, base, **kw) == pytest.approx(base * 1 / 4)
    assert _lr_schedule(2, epochs, base, **kw) == pytest.approx(base * 2 / 4)
    assert _lr_schedule(warmup, epochs, base, **kw) == pytest.approx(base)
    after = _lr_schedule(warmup + 1, epochs, base, **kw)
    assert after == _cosine_lr(warmup + 1, epochs, base, decay=True)


def test_lr_schedule_is_a_noop_when_warmup_epochs_is_zero() -> None:
    epochs, base = 20, 2e-3
    for epoch in range(1, epochs + 1):
        assert _lr_schedule(epoch, epochs, base, decay=True, warmup_epochs=0) == _cosine_lr(
            epoch, epochs, base, decay=True
        )


def test_fit_torch_with_retry_succeeds_on_first_seed_when_nothing_stalls() -> None:
    me_tr, opp_tr, value_tr, validation = _tiny_synthetic_split()
    _model, metadata = fit_torch_with_retry(
        me_tr, opp_tr, value_tr, validation,
        l2=1e-4, seed=0, batch_size=64, epochs=40, learning_rate=5e-3,
        report_every=0, stall_check=StallCheck(10**9, 0.05),
    )
    assert metadata["seed_used"] == 0
    assert metadata["seed_attempts"] == []


def test_fit_torch_with_retry_retries_past_a_stalled_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    import az_train.convnet_othello_torch as cot

    calls: list[int] = []

    def fake_fit_torch(
        _me: object, _opp: object, _value: object, _validation: object, _l2: float = 1e-4,
        *, seed: int = 0, stall_check: object = None, **_kwargs: object,
    ) -> tuple[str, dict[str, object]]:
        calls.append(seed)
        if seed == 0:
            raise TrainingStalledError(64, 0.0, 0.05)
        return "model", {"seed": seed}

    monkeypatch.setattr(cot, "fit_torch", fake_fit_torch)
    model, metadata = fit_torch_with_retry(
        np.zeros((1,)), np.zeros((1,)), np.zeros((1,)),
        (np.zeros((1,)), np.zeros((1,)), np.zeros((1,))),
        seed=0, max_retries=3,
    )
    assert calls == [0, 1]
    assert model == "model"
    assert metadata["seed_used"] == 1
    (attempt,) = metadata["seed_attempts"]
    assert (attempt["seed"], attempt["step"], attempt["pearson"]) == (0, 64, 0.0)
    assert metadata["retry_wall_seconds"] == attempt["wall_seconds"] >= 0.0


def test_fit_torch_with_retry_raises_after_every_attempt_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import az_train.convnet_othello_torch as cot

    def always_stall(
        *_args: object, seed: int = 0, **_kwargs: object,
    ) -> tuple[str, dict[str, object]]:
        raise TrainingStalledError(64, 0.0, 0.05)

    monkeypatch.setattr(cot, "fit_torch", always_stall)
    with pytest.raises(AllSeedsStalledError) as exc_info:
        fit_torch_with_retry(
            np.zeros((1,)), np.zeros((1,)), np.zeros((1,)),
            (np.zeros((1,)), np.zeros((1,)), np.zeros((1,))),
            seed=5, max_retries=2,
        )
    assert [a["seed"] for a in exc_info.value.attempts] == [5, 6, 7]


def _rng_batch(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    me = (rng.random((n, 64)) > 0.7).astype(np.float32)
    opp = (rng.random((n, 64)) > 0.7).astype(np.float32) * (1.0 - me)
    value = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    return me, opp, value


def _two_square_targets(me: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same learnable-signal fixture as ``othello_eval.convnet``'s test
    suite's own ``_two_square_targets``: squares 0/1 are the only legal
    columns, target mass sits on square 0 iff ``me``'s square 0 is
    occupied, else square 1."""
    n = me.shape[0]
    target = np.zeros((n, COLUMNS), dtype=np.float64)
    legal = np.zeros((n, COLUMNS), dtype=bool)
    legal[:, 0] = True
    legal[:, 1] = True
    choose_zero = me[:, 0] > 0.5
    target[choose_zero, 0] = 1.0
    target[~choose_zero, 1] = 1.0
    return target, legal


def test_fit_torch_with_policy_reduces_policy_cross_entropy_below_uniform() -> None:
    """The torch-autograd counterpart of ``othello_eval.convnet``'s
    ``test_fit_with_policy_reduces_policy_cross_entropy_below_uniform`` --
    proves :func:`_policy_loss_torch` actually trains the policy head, not
    just that it's wired up and shaped correctly."""
    me, opp, value = _rng_batch(seed=20, n=300)
    policy, legal = _two_square_targets(me)
    val_me, val_opp, val_value = _rng_batch(seed=22, n=60)
    val_policy, val_legal = _two_square_targets(val_me)
    _model, metadata = fit_torch(
        me, opp, value, (val_me, val_opp, val_value),
        seed=0, epochs=30, batch_size=32, learning_rate=2e-3,
        policy=policy, legal=legal,
        validation_policy=val_policy, validation_legal=val_legal,
    )
    final_metrics = metadata["final_validation_metrics"]
    uniform_ce = float(np.mean(np.log(val_legal.sum(axis=1))))
    assert final_metrics["masked_policy_cross_entropy"] < uniform_ce


def test_fit_torch_without_policy_omits_policy_metrics() -> None:
    """Value-only calls (the pre-existing default) report no
    ``masked_policy_cross_entropy`` key -- confirms the addition is
    opt-in and doesn't change value-only callers' metadata shape."""
    me, opp, value = _rng_batch(seed=1, n=20)
    val_me, val_opp, val_value = _rng_batch(seed=2, n=8)
    _model, metadata = fit_torch(
        me, opp, value, (val_me, val_opp, val_value), seed=0, epochs=2, batch_size=8,
    )
    assert "masked_policy_cross_entropy" not in metadata["final_validation_metrics"]


def test_fit_torch_epoch_log_path_writes_one_line_per_validation_epoch(tmp_path: object) -> None:
    from pathlib import Path

    me, opp, value = _rng_batch(seed=3, n=20)
    val_me, val_opp, val_value = _rng_batch(seed=4, n=8)
    log_path = Path(str(tmp_path)) / "epochs.jsonl"
    fit_torch(
        me, opp, value, (val_me, val_opp, val_value), seed=0, epochs=4, batch_size=8,
        validate_every=1, epoch_log_path=str(log_path),
    )
    lines = log_path.read_text().splitlines()
    assert len(lines) == 4
    import json

    first = json.loads(lines[0])
    assert first["epoch"] == 1
    assert "value_mse" in first and "lr" in first and "elapsed_seconds" in first
