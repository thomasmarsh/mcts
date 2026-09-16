# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOptionalMemberAccess=false
# torch's stubs type `nn.ModuleList` iteration as `Module` (losing the
# concrete `_ResidualBlock` element type) and `Conv2d.bias`/`Linear.bias` as
# `Tensor | None` even though this module always constructs them with a
# bias -- both are torch stub gaps, not real bugs here.
"""PyTorch reimplementation of ``othello_eval.convnet``'s ``OTCNN001``
architecture. This module reproduces the forward pass only -- no training
loop, no autograd-based fit, no dropout/LR scheduling/etc.

``load_from_flat``/``to_flat`` convert to/from the exact flat ``f32``
layout ``othello_eval.convnet._unpack`` documents (stem, ``BLOCKS``
residual blocks, value head, policy head, in that order), so
``write_weights``'s byte format -- and therefore
``games/othello/src/convnet.rs::CnnValueNet::load``'s reader -- needs no
change for a checkpoint fitted under this module to be consumed by the
existing Rust inference hot path. Dense-layer weight matrices are stored
transposed relative to ``nn.Linear`` (numpy's ``(in, out)`` vs. torch's
``(out, in)``); the conversion methods below handle that explicitly.
"""

from __future__ import annotations

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
