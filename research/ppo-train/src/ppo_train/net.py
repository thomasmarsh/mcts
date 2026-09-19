# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Networks for PPO: the OTCNN001 topology with BatchNorm (torch-only, no fold or export yet).

`OTCNN001BNTorch` keeps `OTCNN001Torch`'s heads and post-activation block shape
`relu(BN2(conv2(relu(BN1(conv1 x)))) + x)`, so every BatchNorm sits directly after a convolution and
can later be folded into it exactly. The reference recipe reports a norm-free residual tower barely
learns under Adam, so `arch = "none"` (plain `OTCNN001Torch`) is only the ablation arm.
"""

from __future__ import annotations

import numpy as np
import torch
from az_train.convnet_othello_torch import OTCNN001Torch
from othello_eval.convnet import BOARD
from torch import nn

from ppo_train.config import NetCfg

NEG_INF = -1e9


class _ResidualBlockBN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(h)) + x)


class OTCNN001BNTorch(OTCNN001Torch):
    def __init__(self, blocks: int, channels: int, value_hidden: int) -> None:
        super().__init__(blocks=blocks, tied=False, channels=channels, value_hidden=value_hidden)
        self.stem = nn.Conv2d(2, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(_ResidualBlockBN(channels) for _ in range(blocks))

    def forward_literal(
        self, me: torch.Tensor, opp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = me.shape[0]
        x = torch.stack((me, opp), dim=1).reshape(n, 2, BOARD, BOARD)
        x = torch.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            x = block(x)
        value_features = torch.relu(self.value_conv(x)).reshape(n, BOARD * BOARD)
        value = torch.tanh(self.value_dense2(torch.relu(self.value_dense1(value_features))))
        policy_features = torch.relu(self.policy_conv(x)).reshape(n, BOARD * BOARD)
        return value.reshape(n), self.policy_dense(policy_features)

    def load_from_flat(self, weights: np.ndarray) -> None:
        raise NotImplementedError("BN variant needs the fold to plain OTCNN001 first")

    def to_flat(self) -> np.ndarray:
        raise NotImplementedError("BN variant needs the fold to plain OTCNN001 first")


def build_net(cfg: NetCfg) -> OTCNN001Torch:
    if cfg.arch == "bn":
        return OTCNN001BNTorch(cfg.blocks, cfg.channels, cfg.value_hidden)
    return OTCNN001Torch(blocks=cfg.blocks, channels=cfg.channels, value_hidden=cfg.value_hidden)


def policy_value(
    net: OTCNN001Torch, obs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legal-masked 65-way logits (PASS = mean of the 64 square logits, as in OTCNN001) and value.

    `obs` is the `(n, 2, 8, 8)` mover-relative float observation, `mask` the `(n, 65)` legal mask.
    """
    n = obs.shape[0]
    value, squares = net.forward_literal(obs[:, 0].reshape(n, -1), obs[:, 1].reshape(n, -1))
    logits = torch.cat((squares, squares.mean(dim=-1, keepdim=True)), dim=-1)
    return torch.where(mask, logits, torch.full_like(logits, NEG_INF)), value
