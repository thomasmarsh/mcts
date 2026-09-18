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
(``BLOCKS``, ``tied=False``, ``CHANNELS``, ``VALUE_HIDDEN``) are the
production geometry ``games/othello/src/convnet.rs`` loads, so a caller that
never passes these gets a checkpoint the Rust side accepts (the numpy
reference module's own small geometry has to be requested explicitly).
:func:`fit_torch` also tracks a best-validation-checkpoint
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

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from othello_eval.convnet import (
    BOARD,
    D4,
    INV,
    MAGIC,
    POLICY_OUTPUTS,
    SQUARES,
    VALUE_HIDDEN,
    VERSION,
    _masked_policy_cross_entropy,  # pyright: ignore[reportPrivateUsage]
    _pearson,  # pyright: ignore[reportPrivateUsage]
    initial_weights_k,
    kaiming_bias05_weights_k,
    kaiming_weights_k,
    n_weights_for,
    orthogonal_bias05_weights_k,
    orthogonal_weights_k,
    write_weights,
)
from torch import nn

from az_train import policy_othello
from az_train.records_othello import Positions, concat, load_positions, me_opp_bits, split_by_game

#: Production geometry: what ``OTCNN001Torch``/:func:`fit_torch` build by
#: default, and what ``games/othello/src/convnet.rs``'s compiled-in
#: ``CHANNELS``/``BLOCKS`` load -- ``CnnValueNet::load`` rejects any other
#: geometry's checkpoint by header, so these two constants and that file's
#: must change together. ``othello_eval.convnet``'s own ``CHANNELS``/
#: ``BLOCKS`` are the numpy reference module's small fixed geometry (cheap
#: enough for its test suite) and are deliberately *not* this default.
BLOCKS = 6
CHANNELS = 128

#: Rows per forward call in :meth:`OTCNN001Torch.predict`. At C128/B6 one row's
#: activations are ~33KB per tensor per layer, so a 16k-row chunk needed >1GB of
#: transient MPS memory (invisible to RSS) and the end-of-fit metrics pass over the
#: whole training set tripped the memory watchdog on an 8GB machine. 1024 rows
#: keeps that pass under ~100MB at no measurable throughput cost.
PREDICT_CHUNK_ROWS = 1024

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
    are the production ``OTCNN001`` geometry. ``predict`` reproduces
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
        self, me: np.ndarray, opp: np.ndarray, *, chunk_size: int = PREDICT_CHUNK_ROWS
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


def _metrics(
    prediction: np.ndarray, target: np.ndarray,
    policy_logits64: np.ndarray | None = None,
    policy_target: np.ndarray | None = None, policy_legal: np.ndarray | None = None,
) -> dict[str, float]:
    """Value metrics (mirroring ``othello_eval.convnet.validation_metrics_k``),
    plus ``masked_policy_cross_entropy`` when a policy target/legal mask is
    given -- same ``policy_logits64``/``policy_target``/``policy_legal``
    shapes and ``_with_pass``/``_masked_policy_cross_entropy`` formula
    ``othello_eval.convnet.validation_metrics`` uses, so a torch fit's
    reported policy metric is directly comparable to a numpy ``fit_k`` fit's."""
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
    if policy_logits64 is not None and policy_target is not None and policy_legal is not None:
        metrics["masked_policy_cross_entropy"] = _masked_policy_cross_entropy(
            policy_logits64, policy_target, policy_legal
        )
    return {name: m if np.isfinite(m) else 0.0 for name, m in metrics.items()}


def _policy_loss_torch(
    logits64: torch.Tensor, target: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    """Masked legal-column (65 = 64 squares + PASS) cross entropy, literal
    orientation only -- the torch-autograd counterpart of
    ``othello_eval.convnet._literal_loss_gradient``'s policy-loss term
    (same ``_with_pass``-then-masked-softmax formula, so a torch and numpy
    fit at the same weights compute the identical loss value). ``target``/
    ``legal`` are ``(N, 65)`` (PASS is column 64)."""
    pass_col = logits64.mean(dim=1, keepdim=True)
    logits65 = torch.cat([logits64, pass_col], dim=1)
    masked = torch.where(legal, logits65, torch.full_like(logits65, float("-inf")))
    shifted = masked - masked.max(dim=1, keepdim=True).values
    probability = torch.exp(shifted) * legal
    probability = probability / probability.sum(dim=1, keepdim=True)
    return -(target * torch.log(probability.clamp_min(1e-30))).sum(dim=1).mean()


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


class StallCheck(NamedTuple):
    """Dead-network check for :func:`fit_torch`: once ``step`` optimizer steps
    have run, the value pearson on a fixed validation probe must have
    reached ``min_abs_pearson`` (absolute value), else the fit is abandoned.
    Counted in optimizer steps rather than epochs because a dead network is
    dead within its first few updates, so the right check point does not
    scale with the shard's size."""

    step: int
    min_abs_pearson: float


class TrainingStalledError(RuntimeError):
    """Raised by :func:`fit_torch` when its :class:`StallCheck` fails. Some
    seeded inits leave a network dead for its entire run -- constant value
    output, pearson pinned at exactly 0.0 from the first probe onward,
    regardless of how many further epochs run -- a property of that
    particular (seed, architecture, learning rate) combination. Lets a caller
    bail out after a handful of steps instead of paying the full fit cost."""

    def __init__(self, step: int, pearson: float, threshold: float) -> None:
        super().__init__(
            f"training stalled: step {step} val pearson {pearson:.4f} "
            f"still below {threshold:.4f}"
        )
        self.step = step
        self.pearson = pearson


def fit_torch(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray], l2: float = 1e-4,
    *, seed: int = 0, batch_size: int = 256, epochs: int = 24, learning_rate: float = 2e-3,
    validate_every: int = 1, report_every: int = 0, device: str = "cpu",
    blocks: int = BLOCKS, tied: bool = False, channels: int = CHANNELS,
    value_hidden: int = VALUE_HIDDEN, lr_decay: bool = True,
    stall_check: StallCheck | None = None,
    init: str = "fixed_normal", grad_clip_norm: float | None = None,
    warmup_epochs: int = 0,
    policy: np.ndarray | None = None, legal: np.ndarray | None = None,
    validation_policy: np.ndarray | None = None, validation_legal: np.ndarray | None = None,
    epoch_log_path: str | None = None,
    trace_steps: Sequence[int] = (), probe_size: int = 1024,
) -> tuple[OTCNN001Torch, dict[str, Any]]:
    """``othello_eval.convnet.fit_k``'s Adam loop, over ``torch.autograd``
    instead of a hand-derived gradient. See the module docstring for the
    L2/``weight_decay`` equivalence this relies on.

    ``policy``/``legal`` (each ``(N, 65)``, PASS as column 64, matching
    ``az_train.policy_othello.targets``'s output) add a masked
    cross-entropy policy-head loss on top of the value MSE, mirroring
    ``fit_k``'s own ``value_loss + policy_loss + l2 * reg`` formula exactly
    (:func:`_policy_loss_torch`). Left at their default ``None``, the fit
    stays value-only -- the original behavior, unchanged. ``validation_
    policy``/``validation_legal`` supply the same target/mask shape for
    validation-set ``masked_policy_cross_entropy`` reporting; both pairs
    must be given together or not at all.

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

    ``stall_check``, if given, is a :class:`StallCheck`: right after that
    optimizer step, the value pearson on a fixed ``probe_size``-position
    random subset of the validation set (one un-averaged forward pass, so a
    probe costs milliseconds) must have reached ``min_abs_pearson`` in
    absolute value, else :class:`TrainingStalledError` is raised at once.
    Off by default (``None``). ``trace_steps`` lists extra optimizer steps at
    which to run the same probe and record ``{"step", "value_pearson",
    "value_mse", "value_std"}`` into ``metadata["step_trace"]`` (the raw
    material for choosing a ``StallCheck``); it never affects training.

    ``init`` selects the initializer from :data:`INIT_FUNCTIONS` (default
    ``"fixed_normal"`` == ``initial_weights_k``, so existing callers are
    unaffected). ``grad_clip_norm``, if given, clips the global gradient
    norm (``torch.nn.utils.clip_grad_norm_``) before each optimizer step.
    ``warmup_epochs`` (default 0, a no-op) ramps the learning rate linearly
    from 0 over that many initial epochs before ``_cosine_lr`` takes over --
    see :func:`_lr_schedule`. All three are independent training-stability
    levers, each its own opt-in parameter so they can be gated individually
    or in combination.

    ``epoch_log_path``, if given, appends one JSON line per validation
    epoch (the same dict :func:`_metrics` returns, plus ``epoch``/``lr``/
    ``elapsed_seconds``) to that file as training proceeds, flushed
    immediately -- so a caller watching a long unattended fit can tail real
    per-epoch progress rather than waiting for the final ``report.json``.
    Off by default (``None``) -- existing callers see no behavior change.
    """
    if not len(me):
        raise ValueError("CNN fitting requires non-empty rows")
    if (policy is None) != (legal is None):
        raise ValueError("policy and legal must be given together or not at all")
    if (validation_policy is None) != (validation_legal is None):
        raise ValueError(
            "validation_policy and validation_legal must be given together or not at all"
        )
    torch_device = torch.device(device)
    model = OTCNN001Torch(blocks, tied, channels, value_hidden).to(torch_device)
    model.load_from_flat(INIT_FUNCTIONS[init](seed, blocks, tied, channels, value_hidden))

    weight_params = [p for name, p in model.named_parameters() if not name.endswith(".bias")]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    me_t = torch.as_tensor(me, dtype=torch.float32, device=torch_device)
    opp_t = torch.as_tensor(opp, dtype=torch.float32, device=torch_device)
    value_t = torch.as_tensor(value, dtype=torch.float32, device=torch_device)
    policy_t = None
    legal_t = None
    if policy is not None and legal is not None:
        policy_t = torch.as_tensor(policy, dtype=torch.float32, device=torch_device)
        legal_t = torch.as_tensor(legal, dtype=torch.bool, device=torch_device)

    log_file = open(epoch_log_path, "a") if epoch_log_path is not None else None  # noqa: SIM115
    rng = np.random.default_rng(seed)
    probe_steps = set(trace_steps)
    if stall_check is not None:
        probe_steps.add(stall_check.step)
    probe_rows = np.random.default_rng(0).permutation(len(validation[0]))[:probe_size]
    probe_me, probe_opp = (
        torch.as_tensor(validation[i][probe_rows], dtype=torch.float32, device=torch_device)
        for i in (0, 1)
    )
    probe_value = np.asarray(validation[2][probe_rows], dtype=np.float64)
    step_trace: list[dict[str, float]] = []
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
            prediction, policy_logits64 = model.forward_literal(me_t[idx], opp_t[idx])
            reg = sum((p * p).sum() for p in weight_params)
            loss = torch.mean((prediction - value_t[idx]) ** 2) + l2 * reg
            if policy_t is not None and legal_t is not None:
                loss = loss + _policy_loss_torch(policy_logits64, policy_t[idx], legal_t[idx])
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            step += 1
            if step in probe_steps:
                with torch.no_grad():
                    probe_output = model.forward_literal(probe_me, probe_opp)[0]
                    probe_prediction = probe_output.cpu().numpy().astype(np.float64)
                probe_pearson = _pearson(probe_prediction, probe_value)
                if step in trace_steps:
                    step_trace.append({
                        "step": step, "value_pearson": probe_pearson,
                        "value_mse": float(np.mean((probe_prediction - probe_value) ** 2)),
                        "value_std": float(np.std(probe_prediction)),
                    })
                if (
                    stall_check is not None and step == stall_check.step
                    and abs(probe_pearson) < stall_check.min_abs_pearson
                ):
                    raise TrainingStalledError(step, probe_pearson, stall_check.min_abs_pearson)
        if epoch % validate_every == 0 or epoch == epochs:
            val_prediction, val_policy64 = model.predict(vm, vo)
            metrics = _metrics(
                val_prediction, vv, val_policy64, validation_policy, validation_legal
            )
            validation_epoch_trace.append(metrics)
            if metrics["value_mse"] < best_val_mse:
                best_val_mse = metrics["value_mse"]
                best_epoch = epoch
                best_flat = model.to_flat()
            elapsed = time.perf_counter() - started
            if log_file is not None:
                line = {"epoch": epoch, "lr": epoch_lr, "elapsed_seconds": elapsed, **metrics}
                log_file.write(json.dumps(line) + "\n")
                log_file.flush()
            if report_every and (epoch % report_every == 0 or epoch == epochs):
                m = validation_epoch_trace[-1]
                policy_part = (
                    f"  policy ce {m['masked_policy_cross_entropy']:.4f}"
                    if "masked_policy_cross_entropy" in m
                    else ""
                )
                print(
                    f"  epoch {epoch:4d}  lr {epoch_lr:.2e}  val mse {m['value_mse']:.4f}  "
                    f"pearson {m['value_pearson']:.4f}  sign-acc {m['value_sign_agreement']:.4f}"
                    f"{policy_part}  ({elapsed:.1f}s)",
                    flush=True,
                )
    if log_file is not None:
        log_file.close()
    train_prediction, train_policy64 = model.predict(me, opp)
    if best_flat is None:
        best_flat = model.to_flat()
        best_epoch = epochs
    best_model = OTCNN001Torch(blocks, tied, channels, value_hidden).to(torch_device)
    best_model.load_from_flat(best_flat)
    best_train_prediction, best_train_policy64 = best_model.predict(me, opp)
    best_val_prediction, best_val_policy64 = best_model.predict(vm, vo)
    metadata: dict[str, Any] = {
        "optimizer": "torch_adam_value_mse" if policy is None else "torch_adam_value_policy",
        "optimizer_seed": seed, "optimizer_batch_size": batch_size, "optimizer_epochs": epochs,
        "optimizer_learning_rate": learning_rate, "optimizer_l2": l2, "optimizer_steps": step,
        "optimizer_lr_decay": lr_decay, "optimizer_init": init,
        "optimizer_grad_clip_norm": grad_clip_norm, "optimizer_warmup_epochs": warmup_epochs,
        "device": device,
        "blocks": blocks, "tied": tied, "channels": channels, "value_hidden": value_hidden,
        "n_weights": int(n_weights_for(blocks, tied, channels, value_hidden)),
        "fit_wall_seconds": time.perf_counter() - started,
        "train_metrics": _metrics(train_prediction, value, train_policy64, policy, legal),
        "validation_epoch_trace": validation_epoch_trace,
        "step_trace": step_trace,
        "final_validation_metrics": validation_epoch_trace[-1],
        "best_checkpoint_epoch": best_epoch,
        "best_checkpoint_train_metrics": _metrics(
            best_train_prediction, value, best_train_policy64, policy, legal
        ),
        "best_checkpoint_validation_metrics": _metrics(
            best_val_prediction, vv, best_val_policy64, validation_policy, validation_legal
        ),
        "best_checkpoint_weights": best_flat,
    }
    return model, metadata


#: Derived from 32 seeds' probe traces at C128/B6, batch 32, lr 1e-3 (`az-train-
#: othello-dead-seeds`): a dead net can look alive for its first ~16 steps
#: (pearson up to 0.7) but is pinned at exactly 0.0 by step 24 and never
#: recovers, while every live seed is at >= 0.6 from step 12 on. Step 64 keeps
#: a margin past the last observed death step at a cost of ~64 optimizer
#: steps per dead attempt; 0.05 sits far below any live trace and far above
#: the dead net's exact 0.0.
DEFAULT_STALL_CHECK = StallCheck(step=64, min_abs_pearson=0.05)


#: Measured dead rate at C128/B6, batch 32, lr 1e-3 is ~62% (79 of 128 seeds),
#: so 6 attempts all dying is a ~5% event per fit; a dead attempt costs only
#: ~3.5s, so 21 attempts (0.62**21 ~ 4e-5) are effectively free insurance.
DEFAULT_MAX_RETRIES = 20


class AllSeedsStalledError(RuntimeError):
    """Raised by :func:`fit_torch_with_retry` when every seed it tried,
    ``seed`` through ``seed + max_retries``, stalled. At :data:`DEFAULT_MAX_RETRIES`
    and the measured dead rate that is a ~1-in-25000 event, so it signals
    something worse than seed-to-seed bad luck (a changed learning rate or
    geometry, a bad data split) worth surfacing loudly."""

    def __init__(self, attempts: list[dict[str, float]]) -> None:
        super().__init__(f"all {len(attempts)} seed attempts stalled: {attempts}")
        self.attempts = attempts


def fit_torch_with_retry(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray], l2: float = 1e-4,
    *, seed: int = 0, max_retries: int = DEFAULT_MAX_RETRIES,
    stall_check: StallCheck = DEFAULT_STALL_CHECK,
    **kwargs: Any,
) -> tuple[OTCNN001Torch, dict[str, Any]]:
    """Seed-hunting wrapper: fits at ``seed``, and if ``stall_check`` raises
    :class:`TrainingStalledError`, retries at ``seed + 1``, ``seed + 2``, ...
    up to ``max_retries`` additional attempts, so a dead seed costs only the
    few steps ``stall_check`` takes to notice it instead of a full fit.
    Every stalled attempt is logged in the returned metadata's
    ``"seed_attempts"`` (seed, step, pearson, wall seconds), their total wall
    time is ``metadata["retry_wall_seconds"]``; the winning attempt's actual
    seed is ``metadata["seed_used"]`` (which may differ from the requested
    ``seed`` -- callers that need the exact seed fitted should read this).
    Raises :class:`AllSeedsStalledError` if every attempt through ``seed +
    max_retries`` stalls. ``**kwargs`` forwards to :func:`fit_torch` unchanged
    (``init``, ``grad_clip_norm``, ``warmup_epochs``, ``epochs``, etc.)."""
    attempts: list[dict[str, float]] = []
    for offset in range(max_retries + 1):
        trial_seed = seed + offset
        attempt_started = time.perf_counter()
        try:
            model, metadata = fit_torch(
                me, opp, value, validation, l2, seed=trial_seed, stall_check=stall_check, **kwargs,
            )
        except TrainingStalledError as e:
            attempts.append({
                "seed": trial_seed, "step": e.step, "pearson": e.pearson,
                "wall_seconds": time.perf_counter() - attempt_started,
            })
            continue
        metadata["seed_used"] = trial_seed
        metadata["seed_attempts"] = attempts
        metadata["retry_wall_seconds"] = sum(a["wall_seconds"] for a in attempts)
        return model, metadata
    raise AllSeedsStalledError(attempts)


def me_opp_planes(pos: Positions) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 64)`` float32 0/1 occupancy planes from a decoded ``Positions``
    -- the ``RecordV2``-sourced analogue of ``othello_eval.convnet.
    me_opp_planes`` (which reads a structured v1-record array instead),
    same as ``az_train.convnet_othello.me_opp_planes``."""
    me_bits, opp_bits = me_opp_bits(pos)
    squares = np.arange(SQUARES, dtype=np.uint64)
    me = ((me_bits[:, None] >> squares[None, :]) & np.uint64(1)).astype(np.float32)
    opp = ((opp_bits[:, None] >> squares[None, :]) & np.uint64(1)).astype(np.float32)
    return me, opp


class FitData(NamedTuple):
    """Everything :func:`fit_torch` needs from a set of ``RecordV2`` shards:
    the game-level train/validation split, occupancy planes, and the
    completed-Q policy targets with legal masks."""

    all_positions: Positions
    train: Positions
    validation: Positions
    train_games: int
    validation_games: int
    train_me: np.ndarray
    train_opp: np.ndarray
    va_me: np.ndarray
    va_opp: np.ndarray
    train_policy: np.ndarray
    train_legal: np.ndarray
    va_policy: np.ndarray
    va_legal: np.ndarray


def prepare_fit_data(paths: list[str], validation_fraction: float, split_seed: int) -> FitData:
    """Load and concatenate ``paths``, split by game, and build the arrays
    :func:`fit_torch` consumes. Requires a completed-Q policy target on every
    position (every record must come from ``dump --label gumbel``)."""
    pos = concat([load_positions(p) for p in paths])
    print(f"loaded {len(pos)} positions from {len(paths)} file(s)", flush=True)
    train, validation, train_games, validation_games = split_by_game(
        pos, validation_fraction, split_seed
    )
    has_train_targets = all(entries for entries in train.policy)
    has_validation_targets = all(entries for entries in validation.policy)
    if not (has_train_targets and has_validation_targets):
        raise ValueError(
            "az-train-othello-cnn requires a completed-Q policy target on every "
            "position -- every record must come from `dump --label gumbel`"
        )
    train_me, train_opp = me_opp_planes(train)
    va_me, va_opp = me_opp_planes(validation)
    train_policy, train_legal = policy_othello.targets(train.policy)
    va_policy, va_legal = policy_othello.targets(validation.policy)
    return FitData(
        pos, train, validation, train_games, validation_games, train_me, train_opp, va_me, va_opp,
        train_policy, train_legal, va_policy, va_legal,
    )


def train_cli(argv: list[str] | None = None) -> None:
    """``az-train-othello-cnn`` (torch): fit one Gumbel self-play
    generation's OTCNN001 value+policy head under :func:`fit_torch_with_
    retry` -- the self-play-loop training entrypoint, in place of
    ``az_train.convnet_othello.train_cli``'s numpy/value-only fit (kept,
    importable under ``az-train-othello-cnn-numpy``, but no longer this
    package's console-script default). Same ``RecordV2``-reading/
    checkpoint-writing shape as that numpy CLI: reads
    completed-Q policy targets from every position via ``policy_othello.
    targets``, writes the exact ``OTCNN001`` byte layout ``othello_eval.
    convnet.write_weights`` and ``games/othello/src/convnet.rs::
    CnnValueNet::load`` already agree on, at the production geometry (this
    module's ``BLOCKS``/``CHANNELS``, which that Rust file must match --
    no explicit geometry flag is needed here).

        az-train-othello-cnn --positions gen0.bin,gen1.bin \\
            --out local/output/az/othello-cnn/run0/gen2.bin \\
            --epoch-log local/output/az/othello-cnn/run0/gen2.epochs.jsonl

    ``--device`` defaults to ``mps`` when available (``torch.backends.mps.
    is_available()``), else ``cpu``.
    """
    ap = argparse.ArgumentParser(prog="az-train-othello-cnn")
    ap.add_argument("--positions", required=True, help="comma-separated RecordV2 dump .bin files")
    ap.add_argument("--out", required=True, help="output OTCNN001 checkpoint file")
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--validate-every", type=int, default=1)
    ap.add_argument("--report-every", type=int, default=1)
    ap.add_argument(
        "--device", default=None, help="torch device (default: mps if available, else cpu)"
    )
    ap.add_argument(
        "--epoch-log", default=None, help="path to append one JSON line per validation epoch"
    )
    ap.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help="seed retries past a stalled init",
    )
    ap.add_argument(
        "--stall-check-step", type=int, default=DEFAULT_STALL_CHECK.step,
        help="optimizer step at which a fit whose validation-probe |pearson| is still below "
        "--stall-check-min-pearson is abandoned as a dead seed and retried",
    )
    ap.add_argument(
        "--stall-check-min-pearson", type=float, default=DEFAULT_STALL_CHECK.min_abs_pearson,
    )
    args = ap.parse_args(argv)

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    paths = [p.strip() for p in args.positions.split(",") if p.strip()]
    data = prepare_fit_data(paths, args.validation_fraction, args.split_seed)
    pos, train, validation = data.all_positions, data.train, data.validation
    train_games, validation_games = data.train_games, data.validation_games
    train_me, train_opp, va_me, va_opp = data.train_me, data.train_opp, data.va_me, data.va_opp
    train_policy, train_legal = data.train_policy, data.train_legal
    va_policy, va_legal = data.va_policy, data.va_legal

    print(
        f"=== OTCNN001 (blocks={BLOCKS}, channels={CHANNELS}) fit: {len(train)} train / "
        f"{len(validation)} validation positions, device={device} ===",
        flush=True,
    )
    model, metadata = fit_torch_with_retry(
        train_me, train_opp, train.value.astype(np.float64),
        (va_me, va_opp, validation.value.astype(np.float64)),
        l2=args.l2,
        seed=args.split_seed,
        max_retries=args.max_retries,
        stall_check=StallCheck(args.stall_check_step, args.stall_check_min_pearson),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validate_every=args.validate_every,
        report_every=args.report_every,
        device=device,
        policy=train_policy,
        legal=train_legal,
        validation_policy=va_policy,
        validation_legal=va_legal,
        epoch_log_path=args.epoch_log,
    )
    final = metadata["final_validation_metrics"]
    print(
        f"  final: value mse {final['value_mse']:.4f} pearson {final['value_pearson']:.4f} "
        f"sign-acc {final['value_sign_agreement']:.4f} policy ce "
        f"{final.get('masked_policy_cross_entropy', float('nan')):.4f}",
        flush=True,
    )
    best = metadata["best_checkpoint_validation_metrics"]
    print(
        f"  best (epoch {metadata['best_checkpoint_epoch']}): "
        f"value mse {best['value_mse']:.4f} pearson {best['value_pearson']:.4f} "
        f"sign-acc {best['value_sign_agreement']:.4f} "
        f"policy ce {best.get('masked_policy_cross_entropy', float('nan')):.4f}",
        flush=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_weights(
        str(out), np.asarray(metadata["best_checkpoint_weights"], dtype=np.float32),
        blocks=model.n_blocks, channels=model.channels, value_hidden=model.value_hidden,
    )

    meta = {
        "model": MAGIC.decode(), "version": VERSION, "n_weights": metadata["n_weights"],
        "train": {
            "positions": int(len(pos)), "train_games": train_games,
            "validation_games": validation_games, "sources": paths,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "l2": args.l2, "device": device,
            "seed_used": metadata["seed_used"], "seed_attempts": metadata["seed_attempts"],
        },
        "metrics": {k: v for k, v in metadata.items() if k != "best_checkpoint_weights"},
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {out} ({metadata['n_weights']} weights, "
        f"best epoch {metadata['best_checkpoint_epoch']}) + {out.name}.meta.json",
        flush=True,
    )


if __name__ == "__main__":
    train_cli()
