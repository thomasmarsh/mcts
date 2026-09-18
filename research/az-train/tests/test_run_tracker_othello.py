from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from az_train.run_tracker_othello import (
    best_epoch,
    build_run_data,
    last_fit_attempt,
    parse_launch_log,
    render_page,
)

GATE_TEXT = (
    "head to head vs baseline: candidate 6-1-3 (W-D-L) over 10, score share 0.650, "
    "Wilson LB 0.354  -> FAIL\n"
    "rollout anchor: candidate 9-0-1 (W-D-L) over 10, score share 0.900, Wilson LB 0.596  -> PASS\n"
    "edax yardstick (level 3) games=  10  W-D-L 0-0-10  win_rate=0.000  ci=[0.000, 0.278]\n"
    "GATE: FAIL\n"
)

LAUNCH_LOG = """wrote gen0.cnn.bin (1779971 weights, blocks=6, channels=128)
=== generation 0: self-play (200 games, 32 sims, engine=batched) @ Fri Sep 18 10:00:00 EDT 2026 ===
batched gumbel self-play: games=200
=== generation 0 -> 1: train @ Fri Sep 18 10:20:00 EDT 2026 ===
=== generation 1: gates (10 games) @ Fri Sep 18 10:27:00 EDT 2026 ===
=== generation 1: self-play (200 games, 32 sims, engine=batched) @ Fri Sep 18 10:40:00 EDT 2026 ===
=== generation 1 -> 2: train @ Fri Sep 18 11:00:00 EDT 2026 ===
"""


def epoch_rows(n: int, *, dead_after: int | None = None) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for e in range(1, n + 1):
        mse = 0.9 - 0.06 * min(e, 5) + 0.01 * max(0, e - 5)
        pearson = 0.6 if dead_after is None or e <= dead_after else 0.0
        rows.append(
            {
                "epoch": e,
                "lr": 0.001,
                "elapsed_seconds": 10.0 * e,
                "value_mse": mse,
                "value_pearson": pearson,
                "value_sign_agreement": 0.7,
                "masked_policy_cross_entropy": 1.8 - 0.01 * e,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_meta(run: Path, n: int, *, attempts: int, retry: float, fit: float) -> None:
    meta = {
        "train": {
            "positions": 12000,
            "train_games": 180,
            "validation_games": 20,
            "sources": ["a.bin"],
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 0.001,
            "seed_used": attempts,
            "seed_attempts": [
                {"seed": s, "step": 64, "pearson": 0.0, "wall_seconds": retry / max(attempts, 1)}
                for s in range(attempts)
            ],
        },
        "metrics": {
            "retry_wall_seconds": retry,
            "fit_wall_seconds": fit,
            "best_checkpoint_epoch": 5,
            "best_checkpoint_validation_metrics": {"value_pearson": 0.6, "value_mse": 0.6},
            "final_validation_metrics": {"value_pearson": 0.55, "value_mse": 0.65},
        },
    }
    (run / f"gen{n}.cnn.bin.meta.json").write_text(json.dumps(meta))


def make_run(tmp_path: Path) -> Path:
    """Generation 1 finished, generation 2 fitting with 8 epochs logged."""
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "launch.log").write_text(LAUNCH_LOG)
    write_jsonl(run / "gen1.epochs.jsonl", epoch_rows(30))
    write_meta(run, 1, attempts=3, retry=30.0, fit=270.0)
    (run / "gen1.gate-vs-gen0.txt").write_text(GATE_TEXT)
    write_jsonl(
        run / "log.jsonl",
        [{"generation": 1, "wall_seconds": 2400.0}],
    )
    write_jsonl(run / "gen2.epochs.jsonl", epoch_rows(8))
    (run / "watch.log").write_text(
        "10:00:01 available_mb=1500 group_rss_mb=40\n"
        "10:00:02 available_mb=1100 group_rss_mb=41\n"
        "10:00:03 available_mb=1400 group_rss_mb=41\n"
    )
    return run


NOW = datetime(2026, 9, 18, 11, 10, 0)


def test_parse_launch_log_maps_banners_to_produced_generations() -> None:
    b = parse_launch_log(LAUNCH_LOG)
    assert b["selfplay"][1] == datetime(2026, 9, 18, 10, 0, 0)
    assert b["selfplay"][2] == datetime(2026, 9, 18, 10, 40, 0)
    assert b["train"][1] == datetime(2026, 9, 18, 10, 20, 0)
    assert b["gates"][1] == datetime(2026, 9, 18, 10, 27, 0)
    assert b["config"] == {"games": 200, "sims": 32, "engine": "batched", "gate_games": 10}
    assert b["done"] is None
    assert not b["all_seeds_stalled"]


def test_best_epoch_is_the_lowest_validation_mse_first_occurrence() -> None:
    rows = epoch_rows(30)
    best = best_epoch(rows)
    assert best is not None
    assert best["epoch"] == 5
    assert best_epoch([]) is None


def test_last_fit_attempt_takes_the_final_epoch_restart() -> None:
    rows = epoch_rows(3) + epoch_rows(2)
    last, segments = last_fit_attempt(rows)
    assert segments == 2
    assert [r["epoch"] for r in last] == [1, 2]


def test_mid_run_snapshot_reports_what_exists_and_what_is_missing(tmp_path: Path) -> None:
    data = build_run_data(make_run(tmp_path), NOW, planned_gens=2)
    assert data["status"] == "running"
    assert data["current"] == {"generation": 2, "phase": "fit"}
    assert data["memory"]["min_available_mb"] == 1100
    assert data["memory"]["min_available_at"] == "10:00:02"

    g1, g2 = data["generations"]
    assert g1["state"] == "done"
    assert g1["best_epoch"] == 5
    assert g1["live_network"] is True
    assert g1["late_death"] is False
    assert g1["fit"]["stalled_attempts"] == 3
    assert g1["fit"]["retry_fraction_of_fit"] == 30.0 / 300.0
    assert g1["fit"]["retry_fraction_of_generation"] == 30.0 / 2400.0
    assert g1["wall_vs_estimate"] == 2400.0 / 5400.0
    assert g1["phases"]["selfplay"]["seconds"] == 20 * 60
    assert g1["phases"]["fit"]["seconds"] == 7 * 60
    assert g1["phases"]["gates"]["seconds"] == 13 * 60
    assert g1["gate_vs_gen0"]["checks"]["h2h"]["verdict"] == "FAIL"
    assert g1["gate_vs_gen0"]["checks"]["rollout_anchor"]["wins"] == 9
    assert g1["gate_vs_gen0"]["checks"]["edax_yardstick"]["level"] == 3
    assert g1["gate_vs_prev"] is None
    assert g1["missing"] == []

    assert g2["state"] == "fit"
    assert g2["phases"]["fit"] == {
        "state": "running",
        "seconds": 10 * 60,
        "started": "2026-09-18T11:00:00",
    }
    assert g2["fit"] is None
    assert g2["live_network"] is None
    assert len(g2["epochs"]) == 8
    assert g2["best_epoch"] == 5
    assert any("Seed attempts" in m for m in g2["missing"])


def test_generation_that_has_not_started_is_listed_as_pending(tmp_path: Path) -> None:
    data = build_run_data(make_run(tmp_path), NOW, planned_gens=3)
    g3 = data["generations"][2]
    assert g3["state"] == "pending"
    assert g3["missing"] == ["Not started."]
    assert g3["epochs"] == []


def test_late_death_and_all_seeds_stalled_are_detected(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    write_jsonl(run / "gen2.epochs.jsonl", epoch_rows(10, dead_after=4))
    with (run / "launch.log").open("a") as f:
        f.write("az_train.convnet_othello_torch.AllSeedsStalledError: all 21 seeds stalled\n")
    data = build_run_data(run, NOW)
    assert data["generations"][1]["late_death"] is True
    assert data["generations"][1]["dead_epochs"] == [5, 6, 7, 8, 9, 10]
    assert data["all_seeds_stalled"] is True


def test_watchdog_kill_and_finished_run_set_the_status(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    with (run / "watch.log").open("a") as f:
        f.write("11:00:04 WATCHDOG: available 800MB < floor 900MB -- killing process group 5\n")
    data = build_run_data(run, NOW)
    assert data["status"] == "killed by the memory watchdog"
    assert data["memory"]["killed"] is True
    assert data["current"] is None

    clean = make_run(tmp_path / "second")
    with (clean / "launch.log").open("a") as f:
        f.write("=== done @ Fri Sep 18 11:05:00 EDT 2026 ===\n")
    assert build_run_data(clean, NOW)["status"] == "finished"


def test_a_reused_shard_keeps_its_original_selfplay_time_and_supersedes_the_killed_fit(
    tmp_path: Path,
) -> None:
    run = make_run(tmp_path)
    with (run / "launch.log").open("a") as f:
        f.write(
            "=== generation 1 -> 2: train @ Fri Sep 18 11:02:00 EDT 2026 ===\n"
            "=== generation 1: self-play reused (complete shard from an earlier run, 200 games, "
            "32 sims) @ Fri Sep 18 11:05:00 EDT 2026 ===\n"
        )
    g2 = build_run_data(run, NOW)["generations"][1]
    assert g2["phases"]["selfplay"] == {
        "state": "reused",
        "seconds": 0.0,
        "original_seconds": 22 * 60,
        "started": "2026-09-18T11:05:00",
    }
    assert g2["phases"]["fit"]["state"] == "running"
    assert g2["state"] == "fit"


def test_wall_clock_adds_back_the_reused_selfplay(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "launch.log").write_text(
        "=== generation 0: self-play (200 games, 32 sims, engine=batched) "
        "@ Fri Sep 18 10:00:00 EDT 2026 ===\n"
        "=== generation 0 -> 1: train @ Fri Sep 18 10:18:00 EDT 2026 ===\n"
        "=== generation 0: self-play reused (complete shard from an earlier run, 200 games, "
        "32 sims) @ Fri Sep 18 11:00:00 EDT 2026 ===\n"
        "=== generation 0 -> 1: train @ Fri Sep 18 11:00:05 EDT 2026 ===\n"
        "=== generation 1: gates (10 games) @ Fri Sep 18 11:07:00 EDT 2026 ===\n"
    )
    write_jsonl(run / "log.jsonl", [{"generation": 1, "wall_seconds": 1200.0}])
    g1 = build_run_data(run, NOW)["generations"][0]
    assert g1["phases"]["selfplay"]["original_seconds"] == 18 * 60
    assert g1["phases"]["fit"]["seconds"] == 7 * 60 - 5
    assert g1["wall_seconds"] == 1200.0
    assert g1["wall_seconds_incl_reused_selfplay"] == 1200.0 + 18 * 60
    assert g1["wall_vs_estimate"] == (1200.0 + 18 * 60) / 5400.0


def test_earlier_watchdog_kills_are_listed_but_do_not_mark_the_run_killed(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    (run / "watch.killed-1.log").write_text(
        "10:00:04 WATCHDOG: available 800MB < floor 900MB -- killing process group 5\n"
    )
    data = build_run_data(run, NOW)
    assert data["status"] == "running"
    assert len(data["earlier_kills"]) == 1


def test_a_half_written_epoch_line_is_ignored(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    with (run / "gen2.epochs.jsonl").open("a") as f:
        f.write('{"epoch": 9, "lr"')
    assert len(build_run_data(run, NOW)["generations"][1]["epochs"]) == 8


def test_rendered_page_embeds_parseable_data_and_uses_no_em_dash(tmp_path: Path) -> None:
    data = build_run_data(make_run(tmp_path), NOW, planned_gens=2)
    page = render_page(data)
    assert "/*__RUN_DATA__*/" not in page
    marker = "const DATA = "
    start = page.index(marker) + len(marker)
    end = page.index(";\n", start)
    assert json.loads(page[start:end].replace("<\\/", "</"))["status"] == "running"
    assert "—" not in page
