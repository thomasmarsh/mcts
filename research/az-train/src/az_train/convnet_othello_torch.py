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
pass. No dropout, no LR scheduling, no architecture change here -- this
module is deliberately "same recipe, new engine" only.

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
    N_WEIGHTS,
    POLICY_OUTPUTS,
    SQUARES,
    VALUE_HIDDEN,
    _pearson,  # pyright: ignore[reportPrivateUsage]
    initial_weights_k,
)
from torch import nn


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
    """Same architecture as ``othello_eval.convnet``'s ``OTCNN001``: a 3x3
    stem (2 input planes -> ``CHANNELS``), ``BLOCKS`` residual blocks, then
    a value head (1x1 conv -> dense ``64->VALUE_HIDDEN`` -> dense
    ``VALUE_HIDDEN->1`` -> tanh) and a policy head (1x1 conv -> dense
    ``64->POLICY_OUTPUTS``) sharing the trunk. ``predict`` reproduces
    ``othello_eval.convnet.predict``'s D4-orientation-averaging wrapper and
    PASS-as-mean-of-64-logits convention (the mean is left to the caller,
    exactly as ``predict`` does -- see its docstring)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, CHANNELS, 3, padding=1)
        self.blocks = nn.ModuleList(_ResidualBlock(CHANNELS) for _ in range(BLOCKS))
        self.value_conv = nn.Conv2d(CHANNELS, 1, 1)
        self.value_dense1 = nn.Linear(BOARD * BOARD, VALUE_HIDDEN)
        self.value_dense2 = nn.Linear(VALUE_HIDDEN, 1)
        self.policy_conv = nn.Conv2d(CHANNELS, 1, 1)
        self.policy_dense = nn.Linear(BOARD * BOARD, POLICY_OUTPUTS)

    def forward_literal(
        self, me: torch.Tensor, opp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Value and 64-column policy logits, single (literal) orientation --
        mirrors ``othello_eval.convnet._predict_literal`` exactly. ``me``/
        ``opp`` are ``(N, 64)`` float tensors."""
        n = me.shape[0]
        x = torch.stack((me, opp), dim=1).reshape(n, 2, BOARD, BOARD)
        x = torch.relu(self.stem(x))
        for block in self.blocks:
            x = block(x)
        value_features = torch.relu(self.value_conv(x)).reshape(n, BOARD * BOARD)
        value_hidden = torch.relu(self.value_dense1(value_features))
        value = torch.tanh(self.value_dense2(value_hidden)).reshape(n)
        policy_features = torch.relu(self.policy_conv(x)).reshape(n, BOARD * BOARD)
        policy = self.policy_dense(policy_features)
        return value, policy

    @torch.no_grad()
    def predict(self, me: np.ndarray, opp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """D4-averaged value and 64-column policy logits from ``(N, 64)``
        numpy occupancy arrays -- matches ``othello_eval.convnet.predict``'s
        symmetry loop and ``INV`` back-permutation exactly."""
        device = next(self.parameters()).device
        value_total = np.zeros(me.shape[0], dtype=np.float64)
        policy_total = np.zeros((me.shape[0], SQUARES), dtype=np.float64)
        for sym in range(8):
            cols = D4[sym]
            me_t = torch.as_tensor(me[:, cols], dtype=torch.float32, device=device)
            opp_t = torch.as_tensor(opp[:, cols], dtype=torch.float32, device=device)
            v, logits = self.forward_literal(me_t, opp_t)
            value_total += v.cpu().numpy().astype(np.float64)
            policy_total += logits.cpu().numpy()[:, INV[sym]].astype(np.float64)
        return (value_total / 8.0).astype(np.float32), (policy_total / 8.0).astype(np.float32)

    def load_from_flat(self, weights: np.ndarray) -> None:
        """Load the exact ``OTCNN001`` flat ``f32`` layout
        ``othello_eval.convnet._unpack`` documents."""
        w = np.asarray(weights, dtype=np.float32)
        if w.shape != (N_WEIGHTS,):
            raise ValueError(f"expected {N_WEIGHTS} OTCNN001 weights, got {w.shape}")
        at = 0

        def take(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal at
            n = int(np.prod(shape))
            out = w[at : at + n].reshape(shape)
            at += n
            return out

        with torch.no_grad():
            self.stem.weight.copy_(torch.from_numpy(take((CHANNELS, 2, 3, 3))))
            self.stem.bias.copy_(torch.from_numpy(take((CHANNELS,))))
            for block in self.blocks:
                block.conv1.weight.copy_(torch.from_numpy(take((CHANNELS, CHANNELS, 3, 3))))
                block.conv1.bias.copy_(torch.from_numpy(take((CHANNELS,))))
                block.conv2.weight.copy_(torch.from_numpy(take((CHANNELS, CHANNELS, 3, 3))))
                block.conv2.bias.copy_(torch.from_numpy(take((CHANNELS,))))
            self.value_conv.weight.copy_(torch.from_numpy(take((1, CHANNELS, 1, 1))))
            self.value_conv.bias.copy_(torch.from_numpy(take((1,))))
            self.value_dense1.weight.copy_(
                torch.from_numpy(take((BOARD * BOARD, VALUE_HIDDEN)).T.copy())
            )
            self.value_dense1.bias.copy_(torch.from_numpy(take((VALUE_HIDDEN,))))
            self.value_dense2.weight.copy_(
                torch.from_numpy(take((VALUE_HIDDEN,)).reshape(1, VALUE_HIDDEN))
            )
            self.value_dense2.bias.copy_(torch.from_numpy(take((1,))))
            self.policy_conv.weight.copy_(torch.from_numpy(take((1, CHANNELS, 1, 1))))
            self.policy_conv.bias.copy_(torch.from_numpy(take((1,))))
            self.policy_dense.weight.copy_(
                torch.from_numpy(take((BOARD * BOARD, POLICY_OUTPUTS)).T.copy())
            )
            self.policy_dense.bias.copy_(torch.from_numpy(take((POLICY_OUTPUTS,))))
        assert at == N_WEIGHTS

    def to_flat(self) -> np.ndarray:
        """Inverse of :meth:`load_from_flat`: the current parameters as the
        exact ``OTCNN001`` flat ``f32`` layout, ready for
        ``othello_eval.convnet.write_weights``."""
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
        assert flat.shape == (N_WEIGHTS,)
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


def fit_torch(
    me: np.ndarray, opp: np.ndarray, value: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray], l2: float = 1e-4,
    *, seed: int = 0, batch_size: int = 256, epochs: int = 24, learning_rate: float = 2e-3,
    validate_every: int = 1, report_every: int = 0, device: str = "cpu",
) -> tuple[OTCNN001Torch, dict[str, Any]]:
    """``othello_eval.convnet.fit_k``'s Adam loop, over ``torch.autograd``
    instead of a hand-derived gradient. Value-only (no policy target) --
    an optimizer/engine swap, not a policy-head port. See the module
    docstring for the L2/``weight_decay`` equivalence this relies on.
    Initial weights come from ``initial_weights_k`` (the numpy trainer's own
    seeded initializer), not torch's default
    init, so a torch fit and a numpy fit at the same seed start from
    identical weights -- the only remaining difference is the optimizer
    engine itself.
    """
    if not len(me):
        raise ValueError("CNN fitting requires non-empty rows")
    torch_device = torch.device(device)
    model = OTCNN001Torch().to(torch_device)
    model.load_from_flat(initial_weights_k(seed, BLOCKS, False, CHANNELS, VALUE_HIDDEN))

    weight_params = [p for name, p in model.named_parameters() if not name.endswith(".bias")]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    me_t = torch.as_tensor(me, dtype=torch.float32, device=torch_device)
    opp_t = torch.as_tensor(opp, dtype=torch.float32, device=torch_device)
    value_t = torch.as_tensor(value, dtype=torch.float32, device=torch_device)

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    vm, vo, vv = validation
    validation_epoch_trace: list[dict[str, float]] = []
    step = 0
    n = len(me)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            idx = torch.as_tensor(batch, dtype=torch.long, device=torch_device)
            optimizer.zero_grad()
            prediction, _ = model.forward_literal(me_t[idx], opp_t[idx])
            reg = sum((p * p).sum() for p in weight_params)
            loss = torch.mean((prediction - value_t[idx]) ** 2) + l2 * reg
            loss.backward()
            optimizer.step()
            step += 1
        if epoch % validate_every == 0 or epoch == epochs:
            val_prediction, _ = model.predict(vm, vo)
            validation_epoch_trace.append(_value_metrics(val_prediction, vv))
            if report_every and (epoch % report_every == 0 or epoch == epochs):
                m = validation_epoch_trace[-1]
                elapsed = time.perf_counter() - started
                print(
                    f"  epoch {epoch:4d}  val mse {m['value_mse']:.4f}  "
                    f"pearson {m['value_pearson']:.4f}  sign-acc {m['value_sign_agreement']:.4f}"
                    f"  ({elapsed:.1f}s)",
                    flush=True,
                )
    train_prediction, _ = model.predict(me, opp)
    metadata: dict[str, Any] = {
        "optimizer": "torch_adam_value_mse",
        "optimizer_seed": seed, "optimizer_batch_size": batch_size, "optimizer_epochs": epochs,
        "optimizer_learning_rate": learning_rate, "optimizer_l2": l2, "optimizer_steps": step,
        "device": device,
        "fit_wall_seconds": time.perf_counter() - started,
        "train_metrics": _value_metrics(train_prediction, value),
        "validation_epoch_trace": validation_epoch_trace,
        "final_validation_metrics": validation_epoch_trace[-1],
    }
    return model, metadata
