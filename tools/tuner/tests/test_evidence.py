from __future__ import annotations

from pathlib import Path

import pytest

from tuner_cli.event_payloads import RunInterruptedPayload
from tuner_cli.evidence import EvidenceWriter, atomic_json, read_events, tail_events


def test_canonical_manifest_and_contiguous_evidence(tmp_path: Path) -> None:
    writer = EvidenceWriter(tmp_path / "evidence.jsonl")
    writer.append(RunInterruptedPayload("test", None))
    writer.append(RunInterruptedPayload("test", None))
    assert [event.sequence for event in read_events(tmp_path / "evidence.jsonl")] == [1, 2]
    atomic_json(tmp_path / "report.json", {"b": 2, "a": 1})
    assert (tmp_path / "report.json").read_text() == '{"a":1,"b":2}\n'


def _write_log(path: Path, count: int) -> None:
    writer = EvidenceWriter(path)
    for index in range(count):
        writer.append(RunInterruptedPayload(f"stage-{index}", None))


def test_tail_events_from_zero_returns_everything(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    _write_log(path, 3)
    events, max_seq = tail_events(path, since_seq=0)
    assert [event.sequence for event in events] == [1, 2, 3]
    assert max_seq == 3


def test_tail_events_from_mid_and_end(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    _write_log(path, 3)
    events, max_seq = tail_events(path, since_seq=1)
    assert [event.sequence for event in events] == [2, 3]
    assert max_seq == 3
    events, max_seq = tail_events(path, since_seq=3)
    assert events == []
    assert max_seq == 3


def test_tail_events_withholds_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    _write_log(path, 2)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":5,"sequence":3,"type":"run_inter')
    events, max_seq = tail_events(path, since_seq=0)
    assert [event.sequence for event in events] == [1, 2]
    assert max_seq == 2
    # The writer finishes the line -> it is now delivered.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('rupted","payload":{"stage":"s","pair_id":null}}\n')
    events, max_seq = tail_events(path, since_seq=2)
    assert [event.sequence for event in events] == [3]
    assert max_seq == 3


def test_tail_events_missing_log_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing evidence log"):
        tail_events(tmp_path / "nope.jsonl", since_seq=0)
