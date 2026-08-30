from __future__ import annotations

from pathlib import Path

from tuner_cli.event_payloads import RunInterruptedPayload
from tuner_cli.evidence import EvidenceWriter, atomic_json, read_events


def test_canonical_manifest_and_contiguous_evidence(tmp_path: Path) -> None:
    writer = EvidenceWriter(tmp_path / "evidence.jsonl")
    writer.append(RunInterruptedPayload("test", None))
    writer.append(RunInterruptedPayload("test", None))
    assert [event.sequence for event in read_events(tmp_path / "evidence.jsonl")] == [1, 2]
    atomic_json(tmp_path / "report.json", {"b": 2, "a": 1})
    assert (tmp_path / "report.json").read_text() == '{"a":1,"b":2}\n'
