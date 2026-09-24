# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
"""Residual CNN over a square board grid, the torch side of ``crates/grid-cnn``.

The network is described by :class:`Geometry` (board size, input planes, channels, residual
blocks, policy/value head widths). Training uses batch-norm; :func:`export_flat` folds it into the
convolutions so the exported weight vector (and the Rust MLX forward that reads it) is only
convolutions, ReLUs, residual additions and dense layers. The flat layout and file format are
documented in ``crates/grid-cnn/src/lib.rs``; this module writes exactly that.

``python -m az_train.gridcnn fixture <dir>`` regenerates the cross-language fixture the Rust
tests read: a small net with random weights and non-trivial batch-norm statistics, and its torch
outputs on random planes.
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

MAGIC = b"GRIDCNN1"
VERSION = 1  # dense heads; the agnostic head's files are VERSION_AGNOSTIC (one extra header field)
VERSION_AGNOSTIC = 2
BN_EPS = 1e-5
HEAD_DENSE = "dense"
HEAD_AGNOSTIC = "agnostic"
HEADS = (HEAD_DENSE, HEAD_AGNOSTIC)


@dataclass(frozen=True)
class Geometry:
    size: int
    in_planes: int
    channels: int
    blocks: int
    policy_planes: int
    policy_out: int
    value_planes: int
    value_hidden: int
    # "dense": dense layers over the flattened head feature maps (tied to ``size``, the layout
    # every existing checkpoint uses). "agnostic": per-cell 1x1-conv policy logits plus a dense
    # layer on the pooled trunk for the ``policy_out - size^2`` non-cell logits, and a value head
    # on the trunk's global mean and max (``value_planes`` is unused): no weight depends on
    # ``size``, so a trained net loads at any board size. Not part of the header's 8 geometry
    # fields; the file version says which.
    head: str = HEAD_DENSE

    @property
    def cells(self) -> int:
        return self.size * self.size

    def n_weights(self) -> int:
        c, cells = self.channels, self.cells

        def conv3(i: int, o: int) -> int:
            return o * i * 9 + o

        def conv1(i: int, o: int) -> int:
            return o * i + o

        def dense(i: int, o: int) -> int:
            return i * o + o

        if self.head == HEAD_AGNOSTIC:
            return (
                conv3(self.in_planes, c)
                + self.blocks * 2 * conv3(c, c)
                + conv1(c, self.policy_planes)
                + conv1(self.policy_planes, 1)
                + dense(c, self.policy_out - cells)
                + dense(2 * c, self.value_hidden)
                + dense(self.value_hidden, 1)
            )
        return (
            conv3(self.in_planes, c)
            + self.blocks * 2 * conv3(c, c)
            + conv1(c, self.policy_planes)
            + dense(self.policy_planes * cells, self.policy_out)
            + conv1(c, self.value_planes)
            + dense(self.value_planes * cells, self.value_hidden)
            + dense(self.value_hidden, 1)
        )


class _ConvBn(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm2d(out_ch, eps=BN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class _Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.c1 = _ConvBn(channels, channels, 3)
        self.c2 = _ConvBn(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))) + x)


class GridCNN(nn.Module):
    def __init__(self, g: Geometry) -> None:
        super().__init__()
        self.geometry = g
        self.stem = _ConvBn(g.in_planes, g.channels, 3)
        self.blocks = nn.ModuleList(_Block(g.channels) for _ in range(g.blocks))
        if g.head not in HEADS:
            raise ValueError(f"unknown head {g.head!r}, expected one of {HEADS}")
        self.policy_conv = _ConvBn(g.channels, g.policy_planes, 1)
        if g.head == HEAD_AGNOSTIC:
            if g.policy_out < g.cells:
                raise ValueError("the agnostic head needs policy_out >= size^2")
            self.policy_cell = nn.Conv2d(g.policy_planes, 1, 1)
            self.policy_extra = nn.Linear(g.channels, g.policy_out - g.cells)
            self.value_dense1 = nn.Linear(2 * g.channels, g.value_hidden)
        else:
            self.policy_dense = nn.Linear(g.policy_planes * g.cells, g.policy_out)
            self.value_conv = _ConvBn(g.channels, g.value_planes, 1)
            self.value_dense1 = nn.Linear(g.value_planes * g.cells, g.value_hidden)
        self.value_dense2 = nn.Linear(g.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` is ``(N, in_planes, size, size)``; returns ``(value (N,), logits (N, A))``."""
        h = torch.relu(self.stem(x))
        for block in self.blocks:
            h = block(h)
        n = x.shape[0]
        if self.geometry.head == HEAD_AGNOSTIC:
            pooled = h.mean(dim=(2, 3))
            cell_logits = self.policy_cell(torch.relu(self.policy_conv(h))).reshape(n, -1)
            logits = torch.cat([cell_logits, self.policy_extra(pooled)], dim=1)
            both = torch.cat([pooled, h.amax(dim=(2, 3))], dim=1)
            value = torch.tanh(self.value_dense2(torch.relu(self.value_dense1(both))))
            return value.reshape(n), logits
        logits = self.policy_dense(torch.relu(self.policy_conv(h)).reshape(n, -1))
        v = torch.relu(self.value_conv(h)).reshape(n, -1)
        value = torch.tanh(self.value_dense2(torch.relu(self.value_dense1(v)))).reshape(n)
        return value, logits

    def weight_tensors(self) -> list[torch.Tensor]:
        """Convolution and dense weights (not biases or batch-norm), the tensors L2 applies to."""
        return [m.weight for m in self.modules() if isinstance(m, nn.Conv2d | nn.Linear)]


def _fold(cb: _ConvBn) -> tuple[np.ndarray, np.ndarray]:
    """The conv's weight and bias with the following eval-mode batch-norm folded in."""
    bn = cb.bn
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    w = cb.conv.weight.detach() * scale.reshape(-1, 1, 1, 1)
    b = (cb.conv.bias.detach() - bn.running_mean) * scale + bn.bias.detach()
    return w.cpu().numpy(), b.cpu().numpy()


@torch.no_grad()
def export_flat(model: GridCNN) -> np.ndarray:
    """The flat weight vector in the layout ``crates/grid-cnn`` reads (batch-norm folded using
    its running statistics, i.e. the eval-mode function)."""
    parts: list[np.ndarray] = []

    def conv(cb: _ConvBn, one_by_one: bool = False) -> None:
        w, b = _fold(cb)
        parts.append(w.reshape(w.shape[0], -1).ravel() if one_by_one else w.ravel())
        parts.append(b.ravel())

    def dense(lin: nn.Linear) -> None:
        # torch stores (out, in); the file stores (in, out) row-major.
        parts.append(lin.weight.detach().cpu().numpy().T.ravel())
        parts.append(lin.bias.detach().cpu().numpy().ravel())

    conv(model.stem)
    for block in model.blocks:
        conv(block.c1)
        conv(block.c2)
    conv(model.policy_conv, one_by_one=True)
    if model.geometry.head == HEAD_AGNOSTIC:
        parts.append(model.policy_cell.weight.detach().cpu().numpy().reshape(1, -1).ravel())
        parts.append(model.policy_cell.bias.detach().cpu().numpy().ravel())
        dense(model.policy_extra)
    else:
        dense(model.policy_dense)
        conv(model.value_conv, one_by_one=True)
    dense(model.value_dense1)
    dense(model.value_dense2)
    flat = np.concatenate(parts).astype(np.float32)
    assert flat.shape == (model.geometry.n_weights(),), (flat.shape, model.geometry.n_weights())
    return flat


def write_weights(path: str | Path, g: Geometry, flat: np.ndarray) -> None:
    flat = np.asarray(flat, dtype="<f4")
    if flat.shape != (g.n_weights(),):
        raise ValueError(f"expected {g.n_weights()} weights, got {flat.shape}")
    fields = astuple(g)[:8]
    if g.head == HEAD_AGNOSTIC:
        header = MAGIC + struct.pack("<I", VERSION_AGNOSTIC) + struct.pack("<8I", *fields)
        header += struct.pack("<I", 1)  # head kind: agnostic
    else:
        header = MAGIC + struct.pack("<I", VERSION) + struct.pack("<8I", *fields)
    Path(path).write_bytes(header + struct.pack("<Q", flat.size) + flat.tobytes())


def read_weights(path: str | Path) -> tuple[Geometry, np.ndarray]:
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not a GRIDCNN1 weights file")
    (version,) = struct.unpack_from("<I", raw, 8)
    if version not in (VERSION, VERSION_AGNOSTIC):
        raise ValueError(f"{path}: unsupported version {version}")
    head = HEAD_AGNOSTIC if version == VERSION_AGNOSTIC else HEAD_DENSE
    g = Geometry(*struct.unpack_from("<8I", raw, 12), head=head)
    count_at = 44 + (4 if head == HEAD_AGNOSTIC else 0)
    (count,) = struct.unpack_from("<Q", raw, count_at)
    if count != g.n_weights():
        raise ValueError(f"{path}: header says {count} weights, geometry needs {g.n_weights()}")
    flat = np.frombuffer(raw, dtype="<f4", offset=count_at + 8)
    if flat.size != count:
        raise ValueError(f"{path}: expected {count} weights, found {flat.size}")
    return g, flat.astype(np.float32)


def warm_start(model: GridCNN, state: dict[str, torch.Tensor], mode: str) -> list[str]:
    """Load a net trained at another board size into ``model``; returns the copied parameter and
    buffer names. ``trunk`` copies the stem and residual blocks (convolutional, so any board size)
    and leaves the heads at their fresh initialisation; ``full`` loads every entry and needs the
    same head kind (an agnostic net, whose every weight is size independent)."""
    if mode == "full":
        model.load_state_dict(state)
        return list(state)
    if mode != "trunk":
        raise ValueError(f"unknown warm-start mode {mode!r}, expected trunk or full")
    trunk = {k: v for k, v in state.items() if k.startswith(("stem.", "blocks."))}
    if not trunk:
        raise ValueError("the checkpoint has no trunk entries")
    missing = model.load_state_dict(trunk, strict=False).unexpected_keys
    if missing:
        raise ValueError(f"trunk entries the model does not have: {missing}")
    return list(trunk)


def d4_apply(x: torch.Tensor, sym: int) -> torch.Tensor:
    """Symmetry ``sym`` (0..7) applied to the last two (row, col) dims: flip columns first when
    ``sym >= 4``, then ``sym % 4`` quarter turns."""
    if sym >= 4:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, sym % 4, dims=(-2, -1))


def zero_model(g: Geometry) -> GridCNN:
    """A net whose folded export is all zeros: uniform policy, value 0."""
    model = GridCNN(g)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_var.fill_(1.0)
    return model


def _fill_random(model: GridCNN, seed: int) -> None:
    """Deterministic (numpy-generated, so torch-version independent) weights with non-trivial
    batch-norm statistics, so the fold is really exercised."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            scale = 1.5 if p.dim() > 1 else 0.3
            if name.endswith("bn.weight"):
                arr = 1.0 + 0.2 * rng.standard_normal(p.shape)
            else:
                arr = (
                    scale
                    * rng.standard_normal(p.shape)
                    / max(1.0, np.sqrt(p[0].numel() if p.dim() > 1 else 1))
                )
            p.copy_(torch.from_numpy(arr.astype(np.float32)))
        for name, buf in sorted(model.named_buffers()):
            if name.endswith("running_mean"):
                buf.copy_(
                    torch.from_numpy((0.2 * rng.standard_normal(buf.shape)).astype(np.float32))
                )
            elif name.endswith("running_var"):
                buf.copy_(torch.from_numpy((0.5 + rng.random(buf.shape)).astype(np.float32)))


FIXTURE_GEOMETRY = Geometry(
    size=7,
    in_planes=7,
    channels=8,
    blocks=2,
    policy_planes=4,
    policy_out=51,
    value_planes=2,
    value_hidden=16,
)
FIXTURE_GEOMETRY_AGNOSTIC = Geometry(
    size=7,
    in_planes=7,
    channels=8,
    blocks=2,
    policy_planes=4,
    policy_out=51,
    value_planes=2,
    value_hidden=16,
    head=HEAD_AGNOSTIC,
)
FIXTURE_POSITIONS = 5


def fixture(
    seed: int = 7, g: Geometry = FIXTURE_GEOMETRY
) -> tuple[Geometry, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(geometry, flat weights, planes (n, size, size, in_planes), values, logits)`` from the
    torch model in eval mode. Planes are NHWC here because that is what the Rust forward takes."""
    model = GridCNN(g)
    _fill_random(model, seed)
    model.eval()
    rng = np.random.default_rng(seed + 1)
    planes = (rng.random((FIXTURE_POSITIONS, g.size, g.size, g.in_planes)) < 0.4).astype(np.float32)
    planes[..., -1] = 1.0
    with torch.no_grad():
        x = torch.from_numpy(planes).permute(0, 3, 1, 2).contiguous()
        value, logits = model(x)
    return g, export_flat(model), planes, value.numpy(), logits.numpy()


def write_fixture(out_dir: str | Path, g: Geometry = FIXTURE_GEOMETRY) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g, flat, planes, value, logits = fixture(g=g)
    write_weights(out / "weights.bin", g, flat)
    (out / "io.json").write_text(
        json.dumps(
            {
                "n": int(planes.shape[0]),
                "planes": planes.ravel().tolist(),
                "values": value.tolist(),
                "logits": logits.ravel().tolist(),
            }
        )
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "fixture":
        write_fixture(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "fixture-agnostic":
        write_fixture(sys.argv[2], FIXTURE_GEOMETRY_AGNOSTIC)
    else:
        raise SystemExit("usage: python -m az_train.gridcnn fixture|fixture-agnostic <dir>")
