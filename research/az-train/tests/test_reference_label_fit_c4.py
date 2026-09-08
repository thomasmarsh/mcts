# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
# pyright: reportIndexIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false
# ruff: noqa: E501
import struct
from pathlib import Path

import numpy as np

from az_train.mirror_diagnostic_c4 import read_reference_corpus
from az_train.records_c4 import Positions, encode_records
from az_train.reference_label_fit_c4 import (
    _proven_split,
    _replay_rows,
    legal_columns_from_boards,
    ply_band,
    run_arms,
    uniform_legal_policy,
    value_report,
)

_HEADER = struct.Struct("<8sIIB6x")
_RECORD = struct.Struct("<QQBBIBBBBf")

COLS = 7


def _bit(row: int, col: int) -> int:
    return 1 << (row * COLS + col)


def _write_corpus(path: Path, records: list[tuple]) -> None:
    payload = _HEADER.pack(b"C4REFD01", 1, len(records), 0xFF)
    for rec in records:
        payload += _RECORD.pack(*rec)
    path.write_bytes(payload)


def _full_column(col: int) -> int:
    return sum(_bit(row, col) for row in range(6))


def test_legal_columns_detects_full_columns() -> None:
    black = np.array([_full_column(0) | _bit(0, 3)], dtype=np.uint64)
    white = np.array([_full_column(6)], dtype=np.uint64)
    legal = legal_columns_from_boards(black, white)
    assert legal.tolist() == [[False, True, True, True, True, True, False]]


def test_uniform_legal_policy_is_normalized_over_legal_columns() -> None:
    legal = np.array([[True, False, True, True, False, False, False]])
    policy = uniform_legal_policy(legal)
    np.testing.assert_allclose(policy.sum(axis=1), 1.0)
    assert policy[0, 1] == 0.0 and policy[0, 0] == policy[0, 2] == policy[0, 3]


def test_uniform_legal_policy_rejects_a_full_board_row() -> None:
    try:
        uniform_legal_policy(np.zeros((1, 7), dtype=bool))
    except ValueError:
        return
    raise AssertionError("expected a ValueError when a position has no legal column")


def test_ply_band_edges() -> None:
    assert ply_band(np.array([0, 14, 15, 22, 23, 41])).tolist() == [
        "opening", "opening", "middle", "middle", "late", "late",
    ]


def test_value_report_neutral_on_empty_one_class_and_constant() -> None:
    report = value_report(
        prediction=np.array([0.4, 0.4]),
        target=np.array([1.0, 1.0]),
        exact=np.array([True, False]),
        ply=np.array([10, 25]),
        side=np.array([0, 1]),
    )
    overall = report["overall"]
    assert overall["value_pearson"] == 0.0  # constant target
    assert overall["balanced_sign_accuracy"] == 1.0  # only the +1 class present
    bounded = report["by_proof"]["bounded"]
    assert bounded["count"] == 1 and bounded["value_pearson"] == 0.0
    opening = report["by_ply_band"]["opening"]
    assert opening["count"] == 1
    assert report["by_ply_band"]["late"]["count"] == 1
    assert report["zero_predictor"]["sign_agreement"] == 0.0


def _canonical_corpus_records() -> list[tuple]:
    # black, white, side, ply, group, split, label, proof_depth, max_depth, source_outcome
    # label: 0 exact_win, 1 exact_loss, 3 bounded_win, 4 bounded_loss, 5 unresolved
    return [
        (_bit(0, 0), _bit(0, 1), 0, 2, 10, 0, 0, 8, 10, 1.0),
        (_bit(0, 0) | _bit(0, 2), _bit(0, 1), 1, 3, 10, 0, 4, 6, 10, -1.0),
        (_bit(0, 3), _bit(0, 4) | _bit(0, 5), 1, 3, 11, 0, 1, 8, 10, -1.0),
        (_bit(0, 2), _bit(0, 3), 0, 2, 20, 1, 3, 8, 10, 1.0),
        (_bit(0, 1) | _bit(0, 5), _bit(0, 2), 1, 3, 20, 1, 1, 8, 10, 1.0),
        (_bit(0, 4), _bit(0, 2), 0, 2, 21, 1, 5, 0, 10, 0.0),  # unresolved -> dropped
    ]


def test_reference_reader_exposes_group_and_source_outcome(tmp_path: Path) -> None:
    path = tmp_path / "corpus.c4ref"
    _write_corpus(path, _canonical_corpus_records())
    corpus = read_reference_corpus(path)
    assert corpus.group.tolist() == [10, 10, 11, 20, 20, 21]
    assert corpus.source_outcome[0] == 1.0 and corpus.source_outcome[1] == -1.0


def test_proven_split_drops_unresolved_and_keeps_groups_within_one_split(tmp_path: Path) -> None:
    path = tmp_path / "corpus.c4ref"
    _write_corpus(path, _canonical_corpus_records())
    corpus = read_reference_corpus(path)
    train = _proven_split(corpus, 0)
    val = _proven_split(corpus, 1)
    assert train["value"].size == 3 and val["value"].size == 2  # unresolved row dropped
    assert set(np.unique(train["group"]).tolist()).isdisjoint(np.unique(val["group"]).tolist())
    assert np.all(np.isfinite(train["value"])) and np.all(np.isfinite(val["value"]))


def _tiny_replay(path: Path) -> None:
    rng = np.random.default_rng(0)
    black, white, side, ply, value, policy = [], [], [], [], [], []
    for game in range(3):
        for step in range(3):
            black.append(int(rng.integers(0, 1 << 20)))
            white.append(int(rng.integers(0, 1 << 20)))
            side.append(step % 2)
            ply.append(step)
            value.append(1.0 if game % 2 == 0 else -1.0)
            policy.append([(1, 0.5), (4, 0.5)])
    pos = Positions(
        black=np.asarray(black, dtype=np.uint64),
        white=np.asarray(white, dtype=np.uint64),
        side=np.asarray(side, dtype=np.uint8),
        ply=np.asarray(ply, dtype=np.uint8),
        value=np.asarray(value, dtype=np.float32),
        policy=policy,
    )
    path.write_bytes(encode_records(pos))


def test_run_arms_smoke_produces_three_scored_arms(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.c4ref"
    _write_corpus(corpus_path, _canonical_corpus_records())
    replay_path = tmp_path / "replay.bin"
    _tiny_replay(replay_path)

    corpus = read_reference_corpus(corpus_path)
    replay = _replay_rows([replay_path])
    result = run_arms(corpus, replay, tmp_path / "out", seed=1, epochs=1, l2=1e-4)

    assert set(result["arms"]) == {
        "equivariant_trained_on_proven",
        "literal_trained_on_proven",
        "literal_trained_on_selfplay_outcome",
    }
    for arm in result["arms"].values():
        overall = arm["validation_report"]["overall"]
        assert np.isfinite(overall["value_pearson"]) and np.isfinite(overall["value_mse"])
        assert (tmp_path / "out" / arm["weights_final"]).exists()
    for name in ("literal_trained_on_proven", "literal_trained_on_selfplay_outcome"):
        arm = result["arms"][name]
        assert (tmp_path / "out" / arm["weights_early_stopped"]).exists()
        assert np.isfinite(arm["validation_report_early_stopped"]["overall"]["value_pearson"])
    assert np.isfinite(result["source_outcome_vs_proven_label_validation"]["value_pearson"])
