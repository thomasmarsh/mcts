import dataclasses
from pathlib import Path

import numpy as np
import pytest
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


# ---------------------------------------------------------------------------- auxiliary heads


def _aux_net_cfg(aux: bool) -> dict:
    net = {"size": 5, "channels": 8, "blocks": 1, "policy_planes": 2, "value_planes": 2}
    cfg = {"net": {**net, "value_hidden": 8, "head": "agnostic"}}
    if aux:
        cfg["train"] = {"aux_ownership_weight": 1.0, "aux_chain_weight": 0.5}
    return cfg


def test_geometry_of_turns_on_aux_heads_only_above_zero_weight():
    assert dc.geometry_of(_aux_net_cfg(False)).aux_heads is False
    zero = {**_aux_net_cfg(False), "train": {"aux_chain_weight": 0.0}}
    assert dc.geometry_of(zero).aux_heads is False
    assert dc.geometry_of(_aux_net_cfg(True)).aux_heads is True


def test_default_aux_weights_add_no_auxiliary_parameters():
    g = dc.geometry_of(_aux_net_cfg(False))
    model = dc.gridcnn.GridCNN(g)
    assert not any(k.startswith("aux_") for k in model.state_dict())


def test_aux_heads_output_shapes_and_forward_ignores_them():
    g = dc.geometry_of(_aux_net_cfg(True))
    model = dc.gridcnn.GridCNN(g)
    x = torch.rand(3, 14, 5, 5)
    value, logits, ownership, chain_logits = model.forward_with_aux(x)
    assert ownership.shape == (3, 25) and chain_logits.shape == (3, 25)
    assert torch.all(ownership <= 1) and torch.all(ownership >= -1)
    v2, l2 = model(x)
    torch.testing.assert_close(v2, value)
    torch.testing.assert_close(l2, logits)

    plain = dc.gridcnn.GridCNN(dc.geometry_of(_aux_net_cfg(False)))
    with pytest.raises(ValueError):
        plain.forward_with_aux(x)


def test_export_flat_is_identical_whether_or_not_aux_heads_are_present():
    g = dc.geometry_of(_aux_net_cfg(False))
    model = dc.gridcnn.GridCNN(g)
    torch.manual_seed(3)
    for p in model.parameters():
        p.data.normal_()
    exported = dc.gridcnn.export_flat(model)

    aux_model = dc.gridcnn.GridCNN(dataclasses.replace(g, aux_heads=True))
    aux_model.load_state_dict({**aux_model.state_dict(), **model.state_dict()})
    exported_aux = dc.gridcnn.export_flat(aux_model)
    np.testing.assert_array_equal(exported, exported_aux)


def test_symmetry_gather_of_aux_targets_matches_the_planes_transform():
    size = 5
    src = torch.from_numpy(dr.symmetry_sources(size))
    owners = np.arange(size * size, dtype=np.int64) % 3
    owners_black = np.where(owners == 1, 1, np.where(owners == 2, -1, 0)).astype(np.int8)
    t = torch.from_numpy(owners_black).unsqueeze(0)
    for s in range(dr.SYMMETRIES):
        board = src[s, 0]
        got = t.gather(1, board.unsqueeze(0))[0].numpy()
        grid = owners_black.reshape(size, size)
        grid = grid[::-1, :] if s & 1 else grid
        grid = grid[:, ::-1] if s & 2 else grid
        np.testing.assert_array_equal(got, grid.reshape(-1))


def test_add_aux_heads_optimizer_state_keeps_every_existing_parameter_exact():
    tcfg = {"optimizer": "adamw", "weight_decay": 0.05}
    old_model = dc.gridcnn.GridCNN(dc.geometry_of(_aux_net_cfg(False)))
    old_opt = dc.build_optimizer(old_model, 1e-3, tcfg)
    x = torch.rand(4, 14, 5, 5)
    for _ in range(3):
        value, logits = old_model(x)
        old_opt.zero_grad()
        (value.sum() + logits.sum()).backward()
        old_opt.step()

    new_geometry = dc.geometry_of(_aux_net_cfg(True))
    new_model = dc.gridcnn.GridCNN(new_geometry)
    new_model.load_state_dict({**new_model.state_dict(), **old_model.state_dict()})
    widened_opt_state = dc.add_aux_heads_optimizer_state(
        old_model, new_model, old_opt.state_dict(), 1e-3, tcfg
    )

    new_opt = dc.build_optimizer(new_model, 1e-3, tcfg)
    new_opt.load_state_dict(widened_opt_state)

    old_by_name = dict(old_model.named_parameters())
    new_by_name = dict(new_model.named_parameters())
    for name, p in old_by_name.items():
        old_state, new_state = old_opt.state[p], new_opt.state[new_by_name[name]]
        assert new_state["step"] == old_state["step"]
        for field in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(new_state[field], old_state[field], rtol=0, atol=0)

    # The two new heads start with no optimizer state at all (Adam lazily inits it on first step).
    aux_names = ("aux_ownership.weight", "aux_ownership.bias", "aux_chain.weight", "aux_chain.bias")
    for name in aux_names:
        assert new_opt.state[new_by_name[name]] == {}

    new_model.train()
    value, logits, ownership, chain_logits = new_model.forward_with_aux(x)
    new_opt.zero_grad()
    (value.sum() + logits.sum() + ownership.sum() + chain_logits.sum()).backward()
    new_opt.step()  # runs without error even though the two new params have no prior state
