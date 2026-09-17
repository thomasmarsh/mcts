# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOptionalMemberAccess=false
# torch's stubs type `nn.ModuleList` iteration as `Module` (losing the
# concrete `_ResidualBlock` element type) and `Conv2d.bias`/`Linear.bias` as
# `Tensor | None` even though this module always constructs them with a
# bias -- both are torch stub gaps, not real bugs here.
"""PyTorch reimplementation of ``othello_eval.convnet``'s ``OTCNN001``
architecture, plus a ``torch.autograd``-based training loop
(:func:`fit_torch`) that replaces the numpy trainer's hand-derived backward
pass.

``OTCNN001Torch`` and :func:`fit_torch` are generalized over
``blocks``/``tied``/``channels``/``value_hidden`` exactly the way
``othello_eval.convnet``'s own ``_unpack_k``/``predict_k``/``fit_k`` family
generalizes the fixed-``OTCNN001`` functions -- the default arguments
(``BLOCKS``, ``tied=False``, ``CHANNELS``, ``VALUE_HIDDEN``) reproduce
``OTCNN001`` exactly, so a caller that never passes these keeps the original
fixed geometry. :func:`fit_torch` also tracks a best-validation-checkpoint
snapshot and applies a cosine learning-rate decay over the back half of
training by default, not as opt-in flags a caller must remember to pass.

``load_from_flat``/``to_flat`` convert to/from the exact flat ``f32``
layout ``othello_eval.convnet._unpack`` documents (stem, ``BLOCKS``
residual blocks, value head, policy head, in that order), so
``write_weights``'s byte format -- and therefore
``games/othello/src/convnet.rs::CnnValueNet::load``'s reader -- needs no
change for a checkpoint fitted under this module to be consumed by the
existing Rust inference hot path. Dense-layer weight matrices are stored
transposed relative to ``nn.Linear`` (numpy's ``(in, out)`` vs. torch's
``(out, in)``); the conversion methods below handle that explicitly.

``fit_torch`` matches ``othello_eval.convnet.fit_k``'s learning-rate/L2/
batch-size semantics. Two points that are *not* free substitutions, both
confirmed (not assumed) by
``tests/test_convnet_othello_torch.py::test_one_adam_step_matches_numpy_
with_weight_decay_equal_to_2l2`` -- one Adam step from the same init, same
batch, must match ``fit_k``'s own hand-derived-gradient step exactly:

1. ``fit_k``'s L2 term is folded into the *loss* as ``l2 * sum(w**2)`` over
   every weight tensor (the even-indexed tensors of ``_unpack_k``'s output
   -- weights, not biases -- regularized unconditionally, whether or not
   that tensor's head is even being trained; see point 2), so its gradient
   contribution is ``2 * l2 * w``. ``torch.optim.Adam``'s ``weight_decay``
   (unlike ``AdamW``'s decoupled version) adds exactly ``grad = grad +
   weight_decay * param`` ahead of the moment/velocity update -- the same
   place -- so plain ``Adam`` is the right optimizer *if* ``weight_decay``
   is set to ``2*l2``, not ``l2``.
2. **But ``weight_decay`` alone is not sufficient here.** ``fit_k``
   regularizes *every* weight tensor unconditionally, policy head included
   even when ``policy=None`` (:func:`fit_torch` is value-only -- it fits the
   value head alone, with no policy target). ``torch.optim.Adam`` only
   applies ``weight_decay``
   to parameters that received a gradient from the loss -- a parameter
   whose ``.grad`` is ``None`` (the policy head's, here, since it never
   feeds into a value-only loss) is skipped entirely, silently leaving it
   un-decayed. So ``fit_torch`` does *not* rely on optimizer
   ``weight_decay`` at all: it adds ``l2 * sum(w**2)`` over every weight
   tensor directly into the loss, exactly mirroring ``fit_k``'s own
   ``value_loss_weight * value_loss + policy_loss + l2 * reg`` formula, and
   uses a plain ``Adam`` with ``weight_decay=0``. This makes every weight
   tensor part of the autograd graph regardless of whether its head is
   live, and produces the identical ``2*l2*w`` gradient contribution
   ``fit_k`` computes by hand -- without depending on which parameters a
   given loss happens to touch.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from othello_eval.convnet import (
    BLOCKS,
    BOARD,
    CHANNELS,
    D4,
    INV,
    POLICY_OUTPUTS,
    SQUARES,
    VALUE_HIDDEN,
    _pearson,  # pyright: ignore[reportPrivateUsage]
    initial_weights_k,
    kaiming_bias05_weights_k,
    kaiming_weights_k,
    n_weights_for,
    orthogonal_bias05_weights_k,
    orthogonal_weights_k,
)
from torch import nn

#: Selectable initializers for :func:`fit_torch`'s ``init`` parameter.
#: ``"fixed_normal"`` is ``initial_weights_k`` -- the existing
#: fixed-0.03-std/0.05-bias initializer, kept as the default so no existing
#: caller's behavior changes. ``"kaiming"``/``"orthogonal"`` are fan-in-aware
#: alternatives that scale (or exactly orthogonalize) each weight tensor by
#: its own fan-in instead of using one fixed std for every tensor regardless
#: of layer size, each its own function in ``othello_eval.convnet`` --
#: ``initial_weights_k`` itself is untouched. ``"kaiming_bias05"``/
#: ``"orthogonal_bias05"`` isolate each scheme's weight-scaling change from
#: its zero-bias pairing by keeping ``initial_weights_k``'s own 0.05 bias
#: constant instead (`local/work/plan/llm-jepa/init-stability.md`'s Session
#: 1/2 follow-up).
INIT_FUNCTIONS: dict[str, Any] = {
    "fixed_normal": initial_weights_k,
    "kaiming": kaiming_weights_k,
    "orthogonal": orthogonal_weights_k,
    "kaiming_bias05": kaiming_bias05_weights_k,
    "orthogonal_bias05": orthogonal_bias05_weights_k,
}


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = torch.relu(self.conv1(x))
        return torch.relu(self.conv2(h) + residual)


class OTCNN001Torch(nn.Module):
    """Same architecture as ``othello_eval.convnet``'s ``_unpack_k`` family:
    a 3x3 stem (2 input planes -> ``channels``), ``blocks`` residual blocks
    (independently-parameterized, or one block's weights reused ``blocks``
    times when ``tied=True``), then a value head (1x1 conv -> dense
    ``64->value_hidden`` -> dense ``value_hidden->1`` -> tanh) and a policy
    head (1x1 conv -> dense ``64->POLICY_OUTPUTS``) sharing the trunk. The
    defaults (``BLOCKS``, ``tied=False``, ``CHANNELS``, ``VALUE_HIDDEN``)
    reproduce ``OTCNN001`` exactly. ``predict`` reproduces
    ``othello_eval.convnet.predict_k``'s D4-orientation-averaging wrapper and
    PASS-as-mean-of-64-logits convention (the mean is left to the caller,
    exactly as ``predict_k`` does -- see its docstring)."""

    def __init__(
        self,
        blocks: int = BLOCKS,
        tied: bool = False,
        channels: int = CHANNELS,
        value_hidden: int = VALUE_HIDDEN,
    ) -> None:
        super().__init__()
        self.n_blocks = blocks
        self.tied = tied
        self.channels = channels
        self.value_hidden = value_hidden
        self.stem = nn.Conv2d(2, channels, 3, padding=1)
        n_block_modules = 1 if tied else blocks
        self.blocks = nn.ModuleList(_ResidualBlock(channels) for _ in range(n_block_modules))
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_dense1 = nn.Linear(BOARD * BOARD, value_hidden)
        self.value_dense2 = nn.Linear(value_hidden, 1)
        self.policy_conv = nn.Conv2d(channels, 1, 1)
        self.policy_dense = nn.Linear(BOARD * BOARD, POLICY_OUTPUTS)

    def forward_literal(
        self, me: torch.Tensor, opp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Value and 64-column policy logits, single (literal) orientation --
        mirrors ``othello_eval.convnet._predict_literal_k`` exactly. ``me``/
        ``opp`` are ``(N, 64)`` float tensors."""
        n = me.shape[0]
        x = torch.stack((me, opp), dim=1).reshape(n, 2, BOARD, BOARD)
        x = torch.relu(self.stem(x))
        for i in range(self.n_blocks):
            block = self.blocks[0] if self.tied else self.blocks[i]
            x = block(x)
        value_features = torch.relu(self.value_conv(x)).reshape(n, BOARD * BOARD)
        value_hidden = torch.relu(self.value_dense1(value_features))
        value = torch.tanh(self.value_dense2(value_hidden)).reshape(n)
        policy_features = torch.relu(self.policy_conv(x)).reshape(n, BOARD * BOARD)
        policy = self.policy_dense(policy_features)
        return value, policy

    @torch.no_grad()
    def predict(
        self, me: np.ndarray, opp: np.ndarray, *, chunk_size: int = 16_384
    ) -> tuple[np.ndarray, np.ndarray]:
        """D4-averaged value and 64-column policy logits from ``(N, 64)``
        numpy occupancy arrays -- matches ``othello_eval.convnet.predict``'s
        symmetry loop and ``INV`` back-permutation exactly.

        Chunks the forward pass over ``chunk_size`` rows at a time rather
        than running all ``N`` rows through the conv stack in one shot --
        at full-corpus scale (N ~800k) a single unbatched pass overflows
        MPS's memory limit (``fit_torch``'s own final train-metrics call
        is exactly this shape: the whole training set, not a minibatch,
        run through ``predict`` once). Chunking bounds peak memory
        regardless of ``N`` without changing the returned values --
        each row's D4-averaged prediction is independent of every other
        row's."""
        device = next(self.parameters()).device
        n = me.shape[0]
        value_total = np.zeros(n, dtype=np.float64)
        policy_total = np.zeros((n, SQUARES), dtype=np.float64)
        for sym in range(8):
            cols = D4[sym]
            inv = INV[sym]
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                me_t = torch.as_tensor(me[start:end, cols], dtype=torch.float32, device=device)
                opp_t = torch.as_tensor(opp[start:end, cols], dtype=torch.float32, device=device)
                v, logits = self.forward_literal(me_t, opp_t)
                value_total[start:end] += v.cpu().numpy().astype(np.float64)
                policy_total[start:end] += logits.cpu().numpy()[:, inv].astype(np.float64)
        return (value_total / 8.0).astype(np.float32), (policy_total / 8.0).astype(np.float32)

    def load_from_flat(self, weights: np.ndarray) -> None:
        """Load the flat ``f32`` layout ``othello_eval.convnet._unpack_k``
        documents, for this model's own ``blocks``/``tied``/``channels``/
        ``value_hidden`` geometry (``n_weights_for(...)`` wide; the
        ``blocks=BLOCKS, tied=False`` default reproduces ``OTCNN001``'s exact
        ``N_WEIGHTS``-wide layout ``_unpack`` documents)."""
        channels, value_hidden = self.channels, self.value_hidden
        expected = n_weights_for(self.n_blocks, self.tied, channels, value_hidden)
        w = np.asarray(weights, dtype=np.float32)
        if w.shape != (expected,):
            raise ValueError(f"expected {expected} weights, got {w.shape}")
        at = 0

        def take(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal at
            n = int(np.prod(shape))
            out = w[at : at + n].reshape(shape)
            at += n
            return out

        with torch.no_grad():
            self.stem.weight.copy_(torch.from_numpy(take((channels, 2, 3, 3))))
            self.stem.bias.copy_(torch.from_numpy(take((channels,))))
            for block in self.blocks:
                block.conv1.weight.copy_(torch.from_numpy(take((channels, channels, 3, 3))))
                block.conv1.bias.copy_(torch.from_numpy(take((channels,))))
                block.conv2.weight.copy_(torch.from_numpy(take((channels, channels, 3, 3))))
                block.conv2.bias.copy_(torch.from_numpy(take((channels,))))
            self.value_conv.weight.copy_(torch.from_numpy(take((1, channels, 1, 1))))
            self.value_conv.bias.copy_(torch.from_numpy(take((1,))))
            self.value_dense1.weight.copy_(
                torch.from_numpy(take((BOARD * BOARD, value_hidden)).T.copy())
            )
            self.value_dense1.bias.copy_(torch.from_numpy(take((value_hidden,))))
            self.value_dense2.weight.copy_(
                torch.from_numpy(take((value_hidden,)).reshape(1, value_hidden))
            )
            self.value_dense2.bias.copy_(torch.from_numpy(take((1,))))
            self.policy_conv.weight.copy_(torch.from_numpy(take((1, channels, 1, 1))))
            self.policy_conv.bias.copy_(torch.from_numpy(take((1,))))
            self.policy_dense.weight.copy_(
                torch.from_numpy(take((BOARD * BOARD, POLICY_OUTPUTS)).T.copy())
            )
            self.policy_dense.bias.copy_(torch.from_numpy(take((POLICY_OUTPUTS,))))
        assert at == expected

    def to_flat(self) -> np.ndarray:
        """Inverse of :meth:`load_from_flat`: the current parameters as the
        flat ``f32`` layout matching this model's own geometry, ready for
        ``othello_eval.convnet.write_weights`` when ``blocks=BLOCKS,
        tied=False, channels=CHANNELS, value_hidden=VALUE_HIDDEN`` (the
        ``OTCNN001``-compatible geometry)."""
        expected = n_weights_for(self.n_blocks, self.tied, self.channels, self.value_hidden)
        parts: list[np.ndarray] = []
        with torch.no_grad():
            parts.append(self.stem.weight.detach().cpu().numpy().ravel())
            parts.append(self.stem.bias.detach().cpu().numpy().ravel())
            for block in self.blocks:
                parts.append(block.conv1.weight.detach().cpu().numpy().ravel())
                parts.append(block.conv1.bias.detach().cpu().numpy().ravel())
                parts.append(block.conv2.weight.detach().cpu().numpy().ravel())
                parts.append(block.conv2.bias.detach().cpu().numpy().ravel())
            parts.append(self.value_conv.weight.detach().cpu().numpy().ravel())
            parts.append(self.value_conv.bias.detach().cpu().numpy().ravel())
            parts.append(self.value_dense1.weight.detach().cpu().numpy().T.ravel())
            parts.append(self.value_dense1.bias.detach().cpu().numpy().ravel())
            parts.append(self.value_dense2.weight.detach().cpu().numpy().reshape(-1))
            parts.append(self.value_dense2.bias.detach().cpu().numpy().ravel())
            parts.append(self.policy_conv.weight.detach().cpu().numpy().ravel())
            parts.append(self.policy_conv.bias.detach().cpu().numpy().ravel())
            parts.append(self.policy_dense.weight.detach().cpu().numpy().T.ravel())
            parts.append(self.policy_dense.bias.detach().cpu().numpy().ravel())
        flat = np.concatenate(parts).astype(np.float32)
        assert flat.shape == (expected,)
        return flat


def _value_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Value-only analogue of ``othello_eval.convnet.validation_metrics_k``
    (``fit_torch`` trains the value head alone, with no policy target)."""
    nonzero = target != 0.0
    metrics = {
        "value_mse": float(np.mean((prediction - target) ** 2)),
        "value_pearson": _pearson(prediction, target),
        "value_sign_agreement": float(
            np.mean(np.sign(prediction[nonzero]) == np.sign(target[nonzero]))
        )
        if np.any(nonzero)
        else 0.0,
    }
    return {name: m if np.isfinite(m) else 0.0 for name, m in metrics.items()}


def _cosine_lr(epoch: int, epochs: int, base_lr: float, *, decay: bool) -> float:
    """Constant at ``base_lr`` through the first half of training, then a
    cosine decay from ``base_lr`` down to 0 across the back half. A no-op
    (always ``base_lr``) when ``decay`` is false, so this stays a pure
    additive default rather than a silent behavior change for any caller
    that opts out."""
    half = epochs // 2
    if not decay or epoch <= half or epochs <= half:
        return base_lr
    progress = (epoch - half) / (epochs - half)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _lr_schedule(
    epoch: int, epochs: int, base_lr: float, *, decay: bool, warmup_epochs: int = 0,
) -> float:
    """:func:`_cosine_lr`, preceded by a linear ramp from ``0`` up to
    ``base_lr`` over the first ``warmup_epochs`` epochs -- a brief early-
    training LR ramp meant to bound the damage a bad batch/bad early
    trajectory can do before the network has settled. A no-op (delegates
    straight to ``_cosine_lr``) when ``warmup_epochs`` is 0, its default --
    existing callers see no behavior change. Ramp and cosine decay don't
    overlap for any short warmup (a handful of epochs, well inside the
    first half of a 120-epoch fit where ``_cosine_lr`` is
    already constant)."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    return _cosine_lr(epoch, epochs, base_lr, decay=decay)


class TrainingStalledError(RuntimeError):
    """Raised by :func:`fit_torch` when ``stall_check`` is set and validation
    pearson hasn't escaped noise by the checked epoch. Some seeded inits
    leave a network dead for its entire run -- pearson pinned at ~0.0 from
    the first validation onward, regardless of how many further epochs run
    -- a property of that particular (seed, architecture) combination that
    both this trainer and the numpy hand-rolled one reproduce identically
    from the same seeded init, not a bug specific to either. Lets a caller
    doing multi-seed sweeps bail out well before the remaining epochs (which
    only re-confirm the same stall) instead of always paying the full fit
    cost."""

    def __init__(self, epoch: int, pearson: float, threshold: float) -> None:
        super().__init__(
            f"training stalled: epoch {epoch} val pearson {pearson:.4f} "
            f"still below {threshold:.4f}"
        )
        self.epoch = epoch
        self.pearson = pearson


def fit_torch(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray], l2: float = 1e-4,
    *, seed: int = 0, batch_size: int = 256, epochs: int = 24, learning_rate: float = 2e-3,
    validate_every: int = 1, report_every: int = 0, device: str = "cpu",
    blocks: int = BLOCKS, tied: bool = False, channels: int = CHANNELS,
    value_hidden: int = VALUE_HIDDEN, lr_decay: bool = True,
    stall_check: tuple[int, float] | None = None,
    init: str = "fixed_normal", grad_clip_norm: float | None = None,
    warmup_epochs: int = 0,
) -> tuple[OTCNN001Torch, dict[str, Any]]:
    """``othello_eval.convnet.fit_k``'s Adam loop, over ``torch.autograd``
    instead of a hand-derived gradient. Value-only (no policy target) --
    an optimizer/engine swap, not a policy-head port. See the module
    docstring for the L2/``weight_decay`` equivalence this relies on.
    Initial weights come from ``initial_weights_k`` (the numpy trainer's own
    seeded initializer), not torch's default
    init, so a torch fit and a numpy fit at the same seed start from
    identical weights -- the only remaining difference is the optimizer
    engine itself. ``blocks``/``tied``/``channels``/``value_hidden`` select
    the trunk geometry (defaults reproduce ``OTCNN001``); ``lr_decay``
    (default on) applies :func:`_cosine_lr`.

    Tracks a best-validation-checkpoint snapshot (by ``value_mse``)
    alongside the final-epoch weights the returned ``model`` carries --
    ``metadata["best_checkpoint_epoch"]``/``best_checkpoint_train_metrics``/
    ``best_checkpoint_validation_metrics``/``best_checkpoint_weights`` (a
    flat ``np.ndarray`` in this geometry's ``n_weights_for`` layout, ready
    for ``load_from_flat`` or ``write_weights``) report it explicitly
    alongside the pre-existing final-epoch ``train_metrics``/
    ``final_validation_metrics`` -- report both, never just one.

    ``stall_check``, if given, is an ``(epoch, min_abs_pearson)`` pair:
    once training reaches that epoch, if ``abs(value_pearson)`` from the
    most recent validation is still below ``min_abs_pearson``, raises
    :class:`TrainingStalledError` immediately rather than running the
    remaining epochs to confirm what's already apparent. Off by default
    (``None``) -- existing callers see no behavior change.

    ``init`` selects the initializer from :data:`INIT_FUNCTIONS` (default
    ``"fixed_normal"`` == ``initial_weights_k``, so existing callers are
    unaffected). ``grad_clip_norm``, if given, clips the global gradient
    norm (``torch.nn.utils.clip_grad_norm_``) before each optimizer step.
    ``warmup_epochs`` (default 0, a no-op) ramps the learning rate linearly
    from 0 over that many initial epochs before ``_cosine_lr`` takes over --
    see :func:`_lr_schedule`. All three are independent training-stability
    levers, each its own opt-in parameter so they can be gated individually
    or in combination.
    """
    if not len(me):
        raise ValueError("CNN fitting requires non-empty rows")
    torch_device = torch.device(device)
    model = OTCNN001Torch(blocks, tied, channels, value_hidden).to(torch_device)
    model.load_from_flat(INIT_FUNCTIONS[init](seed, blocks, tied, channels, value_hidden))

    weight_params = [p for name, p in model.named_parameters() if not name.endswith(".bias")]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    me_t = torch.as_tensor(me, dtype=torch.float32, device=torch_device)
    opp_t = torch.as_tensor(opp, dtype=torch.float32, device=torch_device)
    value_t = torch.as_tensor(value, dtype=torch.float32, device=torch_device)

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    vm, vo, vv = validation
    validation_epoch_trace: list[dict[str, float]] = []
    best_val_mse = float("inf")
    best_epoch = 0
    best_flat: np.ndarray | None = None
    step = 0
    n = len(me)
    for epoch in range(1, epochs + 1):
        epoch_lr = _lr_schedule(
            epoch, epochs, learning_rate, decay=lr_decay, warmup_epochs=warmup_epochs
        )
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            idx = torch.as_tensor(batch, dtype=torch.long, device=torch_device)
            optimizer.zero_grad()
            prediction, _ = model.forward_literal(me_t[idx], opp_t[idx])
            reg = sum((p * p).sum() for p in weight_params)
            loss = torch.mean((prediction - value_t[idx]) ** 2) + l2 * reg
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            step += 1
        if epoch % validate_every == 0 or epoch == epochs:
            val_prediction, _ = model.predict(vm, vo)
            metrics = _value_metrics(val_prediction, vv)
            validation_epoch_trace.append(metrics)
            if stall_check is not None:
                stall_epoch, min_abs_pearson = stall_check
                if epoch >= stall_epoch and abs(metrics["value_pearson"]) < min_abs_pearson:
                    raise TrainingStalledError(epoch, metrics["value_pearson"], min_abs_pearson)
            if metrics["value_mse"] < best_val_mse:
                best_val_mse = metrics["value_mse"]
                best_epoch = epoch
                best_flat = model.to_flat()
            if report_every and (epoch % report_every == 0 or epoch == epochs):
                m = validation_epoch_trace[-1]
                elapsed = time.perf_counter() - started
                print(
                    f"  epoch {epoch:4d}  lr {epoch_lr:.2e}  val mse {m['value_mse']:.4f}  "
                    f"pearson {m['value_pearson']:.4f}  sign-acc {m['value_sign_agreement']:.4f}"
                    f"  ({elapsed:.1f}s)",
                    flush=True,
                )
    train_prediction, _ = model.predict(me, opp)
    if best_flat is None:
        best_flat = model.to_flat()
        best_epoch = epochs
    best_model = OTCNN001Torch(blocks, tied, channels, value_hidden).to(torch_device)
    best_model.load_from_flat(best_flat)
    best_train_prediction, _ = best_model.predict(me, opp)
    best_val_prediction, _ = best_model.predict(vm, vo)
    metadata: dict[str, Any] = {
        "optimizer": "torch_adam_value_mse",
        "optimizer_seed": seed, "optimizer_batch_size": batch_size, "optimizer_epochs": epochs,
        "optimizer_learning_rate": learning_rate, "optimizer_l2": l2, "optimizer_steps": step,
        "optimizer_lr_decay": lr_decay, "optimizer_init": init,
        "optimizer_grad_clip_norm": grad_clip_norm, "optimizer_warmup_epochs": warmup_epochs,
        "device": device,
        "blocks": blocks, "tied": tied, "channels": channels, "value_hidden": value_hidden,
        "n_weights": int(n_weights_for(blocks, tied, channels, value_hidden)),
        "fit_wall_seconds": time.perf_counter() - started,
        "train_metrics": _value_metrics(train_prediction, value),
        "validation_epoch_trace": validation_epoch_trace,
        "final_validation_metrics": validation_epoch_trace[-1],
        "best_checkpoint_epoch": best_epoch,
        "best_checkpoint_train_metrics": _value_metrics(best_train_prediction, value),
        "best_checkpoint_validation_metrics": _value_metrics(best_val_prediction, vv),
        "best_checkpoint_weights": best_flat,
    }
    return model, metadata


class AllSeedsStalledError(RuntimeError):
    """Raised by :func:`fit_torch_with_retry` when every seed it tried,
    ``seed`` through ``seed + max_retries``, stalled -- vanishingly unlikely
    at the dead-rates this plan family measures (even a raw ~1-in-4 rate
    makes 6 consecutive stalls a ~1-in-4096 event), so in practice this
    signals something worse than ordinary seed-to-seed bad luck (a
    misconfigured hypothesis, a bad data split) worth surfacing loudly
    rather than silently exhausting retries."""

    def __init__(self, attempts: list[dict[str, float]]) -> None:
        super().__init__(f"all {len(attempts)} seed attempts stalled: {attempts}")
        self.attempts = attempts


def fit_torch_with_retry(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray], l2: float = 1e-4,
    *, seed: int = 0, max_retries: int = 5, stall_check: tuple[int, float] = (20, 0.05),
    **kwargs: Any,
) -> tuple[OTCNN001Torch, dict[str, Any]]:
    """Seed-hunting wrapper for multi-seed sweeps: fits at ``seed``, and if
    ``stall_check`` raises :class:`TrainingStalledError`, retries at
    ``seed + 1``, ``seed + 2``, ... up to ``max_retries`` additional
    attempts, so a dead seed costs only the (cheap, ``stall_check``-bounded)
    stalled attempt instead of silently consuming a sweep slot with no
    result. Every attempt (stalled or not) is logged in the returned
    metadata's ``"seed_attempts"``; the winning attempt's actual seed is
    ``metadata["seed_used"]`` (which may differ from the requested ``seed``
    -- callers that need the exact seed fitted should read this, not assume
    the one they passed). Raises :class:`AllSeedsStalledError` if every
    attempt through ``seed + max_retries`` stalls. ``**kwargs`` forwards to
    :func:`fit_torch` unchanged (``init``, ``grad_clip_norm``,
    ``warmup_epochs``, ``epochs``, etc.)."""
    attempts: list[dict[str, float]] = []
    for offset in range(max_retries + 1):
        trial_seed = seed + offset
        try:
            model, metadata = fit_torch(
                me, opp, value, validation, l2, seed=trial_seed, stall_check=stall_check, **kwargs,
            )
        except TrainingStalledError as e:
            attempts.append({"seed": trial_seed, "epoch": e.epoch, "pearson": e.pearson})
            continue
        metadata["seed_used"] = trial_seed
        metadata["seed_attempts"] = attempts
        return model, metadata
    raise AllSeedsStalledError(attempts)
