from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_run import FakeModel, FakeTarget, _budgeted_options

from tuner_cli.__main__ import main
from tuner_cli.run import run_foreground
from tuner_cli.telemetry import Span, TelemetryWriter, chrome_trace, read_spans


def test_span_records_a_positive_duration_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path)
    writer.mark("session", resumed=0)
    with writer.span("wait", {"phase": "tuning", "pairs": 6}):
        pass
    spans = read_spans(path)
    assert [span.name for span in spans] == ["session", "wait"]
    assert spans[0].dur_us == 0
    assert spans[1].dur_us >= 0
    assert spans[1].args == {"phase": "tuning", "pairs": 6}
    assert spans[1].pid == spans[0].pid


def test_read_spans_drops_a_torn_final_line_and_rejects_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text(Span("fold", 10, 1, 1).line() + "\n" + '{"name": "wait"', encoding="utf-8")
    assert [span.name for span in read_spans(path)] == ["fold"]
    with pytest.raises(ValueError, match="missing telemetry sidecar"):
        read_spans(tmp_path / "absent.jsonl")


def test_chrome_trace_lays_each_name_on_its_own_lane_sorted_by_start() -> None:
    spans = [
        Span("wait", 300, 50, 1),
        Span("propose", 100, 20, 1),
        Span("fold", 200, 5, 1),
    ]
    document = json.loads(chrome_trace(spans))
    assert document["displayTimeUnit"] == "ms"
    complete = [event for event in document["traceEvents"] if event["ph"] == "X"]
    assert [event["name"] for event in complete] == ["propose", "fold", "wait"]
    assert [event["ts"] for event in complete] == [100, 200, 300]
    # propose and fold and wait each get a distinct lane.
    assert len({event["tid"] for event in complete}) == 3
    thread_names = {
        event["tid"]: event["args"]["name"]
        for event in document["traceEvents"]
        if event["name"] == "thread_name"
    }
    assert {thread_names[event["tid"]] for event in complete} == {"propose", "fold", "wait"}


def test_resume_sessions_append_and_show_as_separate_process_rows(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text(Span("fold", 10, 1, 111).line() + "\n", encoding="utf-8")
    with TelemetryWriter(path).span("wait"):
        pass
    spans = read_spans(path)
    assert len(spans) == 2
    document = json.loads(chrome_trace(spans))
    process_rows = [e for e in document["traceEvents"] if e["name"] == "process_name"]
    assert len(process_rows) == 2


def test_foreground_run_writes_a_loadable_trace_profile(tmp_path: Path) -> None:
    options = _budgeted_options(tmp_path, 14, run_name="traced")
    run_foreground(options, FakeTarget(), model_proposer=FakeModel())
    sidecar = options.run_dir / "telemetry.jsonl"
    names = {span.name for span in read_spans(sidecar)}
    assert {"session", "cold_fold", "fold", "propose", "wait"} <= names
    wait_spans = [span for span in read_spans(sidecar) if span.name == "wait"]
    assert all(span.args.get("phase") in {"tuning", "validation"} for span in wait_spans)

    out = tmp_path / "trace.json"
    assert main(["trace", str(options.run_dir), "-o", str(out)]) == 0
    document = json.loads(out.read_text())
    assert document["traceEvents"] and document["displayTimeUnit"] == "ms"


def test_trace_subcommand_reports_a_missing_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["trace", str(tmp_path)]) == 1
    assert "tuner trace failed" in capsys.readouterr().err
