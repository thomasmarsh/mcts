from __future__ import annotations

from tuner_cli.evidence import EvidenceWriter, atomic_json, read_events, write_manifest


def test_canonical_manifest_and_contiguous_evidence(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = write_manifest(tmp_path / "manifest.json", {"schema_version": 1, "z": 2, "a": 1})
    assert manifest["fingerprint"]
    writer = EvidenceWriter(tmp_path / "evidence.jsonl")
    writer.append("first", {"z": 2, "a": 1})
    writer.append("second", {})
    assert [event["sequence"] for event in read_events(tmp_path / "evidence.jsonl")] == [1, 2]
    atomic_json(tmp_path / "report.json", {"b": 2, "a": 1})
    assert (tmp_path / "report.json").read_text() == '{"a":1,"b":2}\n'
