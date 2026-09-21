import numpy as np
import torch

from az_train import gonnect_cnn as gc
from az_train import gridcnn
from az_train.gonnect_records import Positions

SIZE = 5
CELLS = SIZE * SIZE
ACTIONS = CELLS + 2
CPU = torch.device("cpu")


def small_geometry() -> gridcnn.Geometry:
    return gridcnn.Geometry(
        size=SIZE, in_planes=7, channels=8, blocks=1, policy_planes=2, policy_out=ACTIONS,
        value_planes=1, value_hidden=8,
    )  # fmt: skip


def synthetic(n: int, seed: int = 0) -> Positions:
    """Positions whose outcome is a readable function of the planes (own minus opponent stone
    count on the left half) and whose policy is the one-hot of a fixed cell."""
    rng = np.random.default_rng(seed)
    planes = np.zeros((n, 7, SIZE, SIZE), dtype=np.uint8)
    planes[:, 0] = rng.random((n, SIZE, SIZE)) < 0.3
    planes[:, 1] = (rng.random((n, SIZE, SIZE)) < 0.3) & (planes[:, 0] == 0)
    planes[:, 6] = 1
    left = planes[:, 0, :, :2].sum(axis=(1, 2)).astype(np.int64) - planes[:, 1, :, :2].sum(
        axis=(1, 2)
    )
    value = np.where(left >= 0, 1.0, -1.0).astype(np.float32)
    legal = np.ones((n, ACTIONS), dtype=bool)
    policy = np.zeros((n, ACTIONS), dtype=np.float32)
    policy[:, 12] = 1.0
    return Positions(planes, policy, legal, value, np.arange(n, dtype=np.int64) // 5)


def test_augmentation_moves_planes_policy_and_legal_mask_together():
    n = 64
    x = torch.zeros(n, 7, SIZE, SIZE)
    pi = torch.zeros(n, ACTIONS)
    legal = torch.zeros(n, ACTIONS, dtype=torch.uint8)
    rng = np.random.default_rng(1)
    cell = torch.from_numpy(rng.integers(0, CELLS, n))
    x.reshape(n, 7, CELLS)[torch.arange(n), 0, cell] = 1
    pi[torch.arange(n), cell] = 1
    legal[torch.arange(n), cell] = 1
    pi[:, CELLS] = 0.5
    legal[:, CELLS + 1] = 1
    ax, api, alegal = gc._augment(x, pi, legal, gc.d4_sources(SIZE, CPU), SIZE)
    moved = ax.reshape(n, 7, CELLS)[:, 0].argmax(dim=1)
    assert (moved != cell).any(), "some samples must actually move"
    assert torch.equal(api[:, :CELLS].argmax(dim=1), moved)
    assert torch.equal(alegal[:, :CELLS].argmax(dim=1), moved)
    assert torch.equal(api[:, CELLS:], pi[:, CELLS:])
    assert torch.equal(alegal[:, CELLS:], legal[:, CELLS:])


def test_a_short_fit_lowers_the_loss_and_logs_every_step():
    torch.manual_seed(0)
    positions = synthetic(400)
    train, val = gc.split_validation(positions, 10)
    assert len(val) > 0 and len(train) + len(val) == 400
    model = gridcnn.GridCNN(small_geometry())
    opt = gc.make_optimizer(model, 3e-3)
    tcfg = {
        "batch_size": 32, "steps_per_generation": 120, "l2": 1e-5, "validate_every": 60,
        "stall_check_step": 60, "stall_check_min_value_std": 0.0,
    }  # fmt: skip
    rows: list[dict] = []
    step, summary = gc.fit(
        model, opt, gc.Data(train, CPU), val, tcfg,
        gen=0, global_step=0, device=CPU, step_log=rows.append, stall_check=False,
    )  # fmt: skip
    assert step == 120 and len(rows) == 120
    assert summary["loss_last"] < summary["loss_first"]
    assert summary["validation"][-1]["value_pearson"] > 0.3
    assert summary["validation"][-1]["policy_top1"] > 0.9


def test_a_dead_value_head_raises_stalled_at_the_check():
    positions = synthetic(200)
    train, val = gc.split_validation(positions, 4)
    model = gc.new_model(small_geometry(), 0, CPU)
    opt = gc.make_optimizer(model, 0.0)  # learning rate 0: the value head cannot come alive
    tcfg = {
        "batch_size": 8, "steps_per_generation": 10, "l2": 0.0, "validate_every": 100,
        "stall_check_step": 5, "stall_check_min_value_std": 10.0,
    }  # fmt: skip
    try:
        gc.fit(
            model, opt, gc.Data(train, CPU), val, tcfg,
            gen=0, global_step=0, device=CPU, step_log=lambda r: None, stall_check=True,
        )  # fmt: skip
    except gc.Stalled:
        return
    raise AssertionError("expected Stalled")


def test_checkpoint_round_trip_reproduces_the_exported_weights(tmp_path):
    g = small_geometry()
    model = gc.new_model(g, 3, CPU)
    opt = gc.make_optimizer(model, 1e-3)
    positions = synthetic(100)
    x = torch.from_numpy(positions.planes).float()
    model.train()
    model(x)  # move the batch-norm statistics off their defaults
    gc.atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict()}, tmp_path / "c.pt")
    gc.export_weights(model, tmp_path / "w.bin")

    reloaded = gridcnn.GridCNN(g)
    reloaded.load_state_dict(torch.load(tmp_path / "c.pt", weights_only=False)["model"])
    gc.export_weights(reloaded, tmp_path / "w2.bin")
    assert (tmp_path / "w.bin").read_bytes() == (tmp_path / "w2.bin").read_bytes()
    _, flat = gridcnn.read_weights(tmp_path / "w.bin")
    assert flat.shape == (g.n_weights(),)
    model.eval()
    reloaded.eval()
    assert torch.equal(model(x)[1], reloaded(x)[1])


def test_wilson_interval_brackets_the_score():
    lo, hi = gc.wilson(20, 40)
    assert lo < 0.5 < hi and 0.34 < lo < 0.36 and 0.64 < hi < 0.66
