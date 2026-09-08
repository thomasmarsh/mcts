# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
# ruff: noqa: E501
import struct
from pathlib import Path
from typing import cast

import numpy as np

from az_train.convnet_c4 import N_WEIGHTS
from az_train.mirror_diagnostic_c4 import (
    _mirror_value_target,
    evaluate_value_orientations,
    read_reference_corpus,
)

_HEADER = struct.Struct("<8sIIB6x")
_RECORD = struct.Struct("<QQBBIBBBBf")


def _write_corpus(path: Path, records: list[tuple]) -> None:
    payload = _HEADER.pack(b"C4REFD01", 1, len(records), 0xFF)
    for rec in records:
        payload += _RECORD.pack(*rec)
    path.write_bytes(payload)


def _corpus_records() -> list[tuple]:
    # black, white, side, ply, group, split, label, proof_depth, max_depth, source_outcome
    return [
        (0b1, 0b10, 0, 2, 0, 0, 0, 8, 10, 1.0),   # train, exact win
        (0b1, 0b110, 1, 3, 1, 0, 4, 6, 10, -1.0),  # train, bounded loss
        (0b101, 0b010, 1, 3, 2, 1, 3, 8, 10, 1.0),  # validation, bounded win
        (0b1, 0b10, 0, 2, 3, 1, 5, 0, 10, 0.0),    # validation, unresolved -> dropped
    ]


def test_reference_reader_parses_split_and_proven_labels(tmp_path: Path) -> None:
    path = tmp_path / "corpus.c4ref"
    _write_corpus(path, _corpus_records())
    corpus = read_reference_corpus(path)
    assert corpus.split.tolist() == [0, 0, 1, 1]
    assert corpus.label_sign[0] == 1.0 and corpus.label_sign[1] == -1.0
    assert np.isnan(corpus.label_sign[3])
    assert corpus.exact.tolist() == [True, False, False, False]


def test_reference_reader_rejects_foreign_magic(tmp_path: Path) -> None:
    path = tmp_path / "bad.c4ref"
    path.write_bytes(b"NOTREFXX" + b"\x00" * 40)
    try:
        read_reference_corpus(path)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a non-C4REFD01 artifact")


def test_mirror_value_target_is_identity() -> None:
    target = np.array([-1.0, 0.0, 1.0, 0.5])
    assert np.array_equal(_mirror_value_target(target), target)


def test_variant_c_equals_mirror_averaged_baseline_by_construction() -> None:
    rng = np.random.default_rng(7)
    weights = (rng.standard_normal(N_WEIGHTS) * 0.05).astype(np.float32)
    me = np.zeros((4, 42), dtype=np.float32)
    opp = np.zeros_like(me)
    me[0, [0, 7]], opp[0, [1, 8]] = 1.0, 1.0
    me[1, [2, 3]], opp[1, [9, 10]] = 1.0, 1.0
    me[2, [4]], opp[2, [5, 6]] = 1.0, 1.0
    me[3, [0, 1, 2]], opp[3, [7, 8, 14]] = 1.0, 1.0
    target = np.array([1.0, -1.0, 1.0, -1.0])
    result = evaluate_value_orientations(weights, me, opp, target)
    assert result["mirror_averaged"] == result["mirror_averaged_target_mirrored"]
    literal = cast("dict[str, float]", result["literal"])
    assert set(literal) == {"count", "value_mse", "value_pearson", "sign_agreement", "balanced_sign_accuracy"}
