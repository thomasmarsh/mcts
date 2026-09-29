from pathlib import Path

import numpy as np
import torch

from az_train import druid_cnn as dc
from az_train import druid_records as dr

FIXTURES = Path(__file__).resolve().parents[3] / "games/druid/cnn/fixtures"


def test_augment_agrees_with_decoding_the_reflected_position():
    for size in (5, 7):
        _, records = dr.read_shard(FIXTURES / f"encode-{size}.shard.bin")
        pos = dr.load_positions(FIXTURES / f"encode-{size}.shard.bin")[1]
        src = torch.from_numpy(dr.symmetry_sources(size))
        x, pi, legal, kind = (
            torch.from_numpy(a)
            for a in (pos.planes, pos.policy, pos.legal.astype(np.uint8), pos.kind)
        )
        for s in range(dr.SYMMETRIES):
            flipped = records.copy()
            for name in ("heights", "owners"):
                g = records[name].reshape(len(records), size, size)
                g = g[:, ::-1, :] if s & 1 else g
                g = g[:, :, ::-1] if s & 2 else g
                flipped[name] = g.reshape(len(records), -1)
            ax, api, alegal = dc.augment(x, pi, legal, kind, src, size, torch.full((len(x),), s))
            np.testing.assert_array_equal(ax.numpy(), dr.decode_planes(flipped, size))
            np.testing.assert_array_equal(alegal.numpy() > 0, dr.decode_legal(flipped, size))
            np.testing.assert_allclose(api.numpy().sum(1), pos.policy.sum(1), rtol=1e-6)
            np.testing.assert_array_equal(api.numpy() > 0, alegal.numpy() > 0)


def _row(draw: float, pearson: float | None, over: float | None) -> dict:
    return {
        "diagnostics": {
            "selfplay": {"draw_rate": draw},
            "value": {"held": {"pearson": pearson, "mse_vs_constant": over}},
        }
    }


def test_druid_warnings_flag_draws_and_an_anti_correlated_value_head():
    healthy = [_row(0.02, 0.3, 0.8)] * 10
    assert dc.druid_warnings(healthy) == []
    sick = [_row(0.2, -0.2, 1.3)] * 10
    text = " | ".join(dc.druid_warnings(sick))
    assert "draw rate" in text and "Pearson" in text and "constant predictor" in text
    assert dc.druid_warnings([]) == []
    # Only the last 10 generations count, and a missing Pearson (constant head) is skipped.
    assert dc.druid_warnings([_row(0.9, -1.0, 2.0)] * 5 + healthy) == []
    assert dc.druid_warnings([_row(0.0, None, None)] * 3) == []


def test_geometry_of_reads_the_head_kind():
    net = {"size": 7, "channels": 8, "blocks": 1, "policy_planes": 2, "value_planes": 1}
    net["value_hidden"] = 4
    assert dc.geometry_of({"net": net}).head == "dense"
    agnostic = dc.geometry_of({"net": {**net, "head": "agnostic"}})
    assert agnostic.head == "agnostic" and agnostic.policy_out == 7 * 7 + 4


def test_record_window_batches_equal_the_decoded_window(tmp_path):
    fixtures = Path(__file__).resolve().parents[3] / "games/druid/cnn/fixtures"
    size, records = dr.read_shard(fixtures / "encode-7.shard.bin")
    positions = dr.load_positions(fixtures / "encode-7.shard.bin")[1]
    cpu = torch.device("cpu")
    idx = torch.tensor([0, 3, 5, 3, len(records) - 1])
    got = dc.RecordWindow(records, size, cpu).batch(idx)
    want = dc.Data(positions, cpu).batch(idx)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == w.dtype
        torch.testing.assert_close(g, w, rtol=0, atol=0)


def _tiny_model():
    net = {"size": 5, "channels": 8, "blocks": 1, "policy_planes": 2, "value_planes": 2}
    cfg = {"net": {**net, "value_hidden": 8, "head": "agnostic"}}
    return dc.gridcnn.GridCNN(dc.geometry_of(cfg))


def test_adamw_decays_only_conv_and_linear_weights():
    model = _tiny_model()
    opt = dc.build_optimizer(model, 1e-3, {"optimizer": "adamw", "weight_decay": 0.05})
    assert isinstance(opt, torch.optim.AdamW)
    decayed, plain = opt.param_groups
    assert decayed["weight_decay"] == 0.05 and plain["weight_decay"] == 0.0
    assert {id(p) for p in decayed["params"]} == {id(w) for w in model.weight_tensors()}
    assert len(decayed["params"]) + len(plain["params"]) == len(list(model.parameters()))
    assert plain["params"]  # biases and batch-norm exist and are exempt


def test_adam_is_default_with_l2_and_adamw_has_none_in_the_loss():
    model = _tiny_model()
    assert type(dc.build_optimizer(model, 1e-3, {})) is torch.optim.Adam
    w, dev = model.weight_tensors(), torch.device("cpu")
    assert dc.coupled_l2(w, {"l2": 1e-4}, dev) > 0
    assert dc.coupled_l2(w, {"optimizer": "adamw", "weight_decay": 0.05, "l2": 1e-4}, dev) == 0
    # Resume restores lr and weight_decay from the checkpoint over freshly built groups.
    cfg = {"optimizer": "adamw", "weight_decay": 0.05}
    saved = dc.build_optimizer(model, 1e-3, cfg).state_dict()
    fresh = dc.build_optimizer(model, 5e-4, {**cfg, "weight_decay": 0.5})
    fresh.load_state_dict(saved)
    assert fresh.param_groups[0]["lr"] == 1e-3
    assert fresh.param_groups[0]["weight_decay"] == 0.05


def test_warm_start_widens_the_stem_with_zeros_and_keeps_the_function():
    net = {"size": 5, "channels": 8, "blocks": 1, "policy_planes": 2, "value_planes": 2}
    base = {"net": {**net, "value_hidden": 8, "head": "agnostic"}}
    wide = {"net": {**base["net"], "connectivity": True}}
    assert dc.geometry_of(base).in_planes == 14 and dc.geometry_of(wide).in_planes == 20
    old = dc.gridcnn.GridCNN(dc.geometry_of(base))
    new = dc.gridcnn.GridCNN(dc.geometry_of(wide))
    for m in (old, new):
        m.eval()
    copied = dc.gridcnn.warm_start(new, old.state_dict(), "full")
    assert "stem.conv.weight" in copied
    assert not new.stem.conv.weight[:, 14:].any()
    x = torch.rand(3, 20, 5, 5)
    with torch.no_grad():
        (v0, l0), (v1, l1) = old(x[:, :14]), new(x)
    torch.testing.assert_close(v1, v0)
    torch.testing.assert_close(l1, l0)


def test_widen_stem_optimizer_state_pads_the_stem_and_keeps_everything_else_exact():
    net = {"size": 5, "channels": 8, "blocks": 1, "policy_planes": 2, "value_planes": 2}
    base = {"net": {**net, "value_hidden": 8, "head": "agnostic"}}
    wide = {"net": {**base["net"], "connectivity": True}}
    tcfg = {"optimizer": "adamw", "weight_decay": 0.05}

    old_model = dc.gridcnn.GridCNN(dc.geometry_of(base))
    old_opt = dc.build_optimizer(old_model, 1e-3, tcfg)
    x = torch.rand(4, 14, 5, 5)
    for _ in range(3):
        value, logits = old_model(x)
        old_opt.zero_grad()
        (value.sum() + logits.sum()).backward()
        old_opt.step()

    new_model = dc.gridcnn.GridCNN(dc.geometry_of(wide))
    new_model.load_state_dict(dc.gridcnn.widen_stem(new_model, old_model.state_dict()))
    widened_opt_state = dc.widen_stem_optimizer_state(
        old_model, new_model, old_opt.state_dict(), 1e-3, tcfg
    )

    # The widened model still computes the old model's function on the old planes.
    wide_x = torch.rand(3, 20, 5, 5)
    old_model.eval()
    new_model.eval()
    with torch.no_grad():
        v0, l0 = old_model(wide_x[:, :14])
        v1, l1 = new_model(wide_x)
    torch.testing.assert_close(v1, v0)
    torch.testing.assert_close(l1, l0)

    new_opt = dc.build_optimizer(new_model, 1e-3, tcfg)
    new_opt.load_state_dict(widened_opt_state)

    stem_state = new_opt.state[new_model.stem.conv.weight]
    old_stem_state = old_opt.state[old_model.stem.conv.weight]
    assert stem_state["step"] == old_stem_state["step"]
    for field in ("exp_avg", "exp_avg_sq"):
        assert stem_state[field].shape[1] == 20
        torch.testing.assert_close(stem_state[field][:, :14], old_stem_state[field])
        assert not stem_state[field][:, 14:].any()

    # Every other parameter's optimizer state, including biases and batch-norm, carries over
    # bit for bit -- only the stem weight's entry was touched.
    old_by_name = dict(old_model.named_parameters())
    new_by_name = dict(new_model.named_parameters())
    for name, p in old_by_name.items():
        if name == "stem.conv.weight":
            continue
        for field in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                new_opt.state[new_by_name[name]][field], old_opt.state[p][field], rtol=0, atol=0
            )

    new_model.train()
    value, logits = new_model(wide_x)
    new_opt.zero_grad()
    (value.sum() + logits.sum()).backward()
    new_opt.step()  # the widened optimizer runs a step over the widened model without error


def test_blended_value_target_is_the_outcome_at_zero_weight_and_hand_computed_at_half():
    outcome = torch.tensor([1.0, -1.0, 0.0])
    q = torch.tensor([0.2, 0.4, -0.6])
    torch.testing.assert_close(dc.blended_value_target(outcome, q, 0.0), outcome)
    want_half = torch.tensor([0.6, -0.3, -0.3])  # 0.5 * outcome + 0.5 * q, worked by hand
    torch.testing.assert_close(dc.blended_value_target(outcome, q, 0.5), want_half)
