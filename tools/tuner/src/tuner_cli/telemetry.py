"""Wall-clock telemetry sidecar: non-scientific span records for the run loop.

``telemetry.jsonl`` sits beside ``evidence.jsonl`` and records where wall-clock
time actually goes while a run turns its loop -- the cold fold, the per-turn
tail fold, model proposal, pair dispatch, and the wait on game subprocesses,
each tagged with the phase it served. It carries exactly the timestamps
``evidence.jsonl`` deliberately omits: no replay,
fingerprint, projection, or scientific-event count ever reads this file.

``tuner trace <run-dir>`` renders the sidecar as a Chrome / Perfetto JSON Trace
Profile that loads in ``ui.perfetto.dev`` or ``chrome://tracing``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .codec import integer, json_object, strict_json, string

# One lane per span name, in the order they occur in a loop turn. A name not
# listed here shares the trailing "other" lane.
LANES: tuple[str, ...] = (
    "session",
    "cold_fold",
    "fold",
    "propose",
    "dispatch",
    "wait",
    "diagnostic",
)

SpanArgs = dict[str, str | int]


@dataclass(frozen=True, slots=True)
class Span:
    """One completed interval on the run-loop thread.

    ``start_us`` is Unix-epoch microseconds (so the spans of separate
    ``--resume`` sessions land in the right place on one wall timeline);
    ``dur_us`` comes from a monotonic clock for precision. ``pid`` is the
    process that recorded it -- a resumed run shows as a fresh process lane
    with a visible gap where the machine slept.
    """

    name: str
    start_us: int
    dur_us: int
    pid: int
    args: SpanArgs = field(default_factory=lambda: dict[str, str | int]())

    def line(self) -> str:
        record: dict[str, str | int | SpanArgs] = {
            "name": self.name,
            "start_us": self.start_us,
            "dur_us": self.dur_us,
            "pid": self.pid,
            "args": dict(self.args),
        }
        return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _span_args(value: object) -> SpanArgs:
    fields = json_object(value, "telemetry span args")
    result: SpanArgs = {}
    for key, item in fields.items():
        if isinstance(item, bool):
            result[key] = int(item)
        elif isinstance(item, (str, int)):
            result[key] = item
    return result


def _decode_span(text: str) -> Span:
    fields = json_object(strict_json(text, "telemetry line"), "telemetry span")
    return Span(
        string(fields.get("name"), "telemetry span name"),
        integer(fields.get("start_us"), "telemetry span start"),
        integer(fields.get("dur_us"), "telemetry span duration"),
        integer(fields.get("pid"), "telemetry span pid"),
        _span_args(fields.get("args", {})),
    )


def read_spans(path: Path) -> list[Span]:
    if not path.is_file():
        raise ValueError(f"missing telemetry sidecar: {path}")
    text = path.read_text(encoding="utf-8")
    # A torn trailing line (the loop was killed mid-append) is simply dropped.
    return [_decode_span(line) for line in text.split("\n")[:-1] if line]


class TelemetryWriter:
    """Append-only span recorder for one run-loop process.

    Spans are emitted only from the single loop thread, so no locking is
    needed. Writes are flushed but not fsync'd: losing the last few
    milliseconds of a killed run's telemetry costs nothing.
    """

    __slots__ = ("_path", "_pid")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._pid = os.getpid()

    def _write(self, span: Span) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(span.line() + "\n")

    def mark(self, name: str, **args: str | int) -> None:
        """Record a zero-length marker (a session boundary, say)."""
        self._write(Span(name, int(time.time() * 1_000_000), 0, self._pid, dict(args)))

    @contextmanager
    def span(self, name: str, args: SpanArgs | None = None) -> Generator[None, None, None]:
        start_wall = time.time()
        start_mono = time.monotonic_ns()
        try:
            yield
        finally:
            dur_us = (time.monotonic_ns() - start_mono) // 1_000
            start_us = int(start_wall * 1_000_000)
            self._write(Span(name, start_us, dur_us, self._pid, dict(args or {})))


TraceEvent = dict[str, str | int | SpanArgs]


def _lane(name: str) -> int:
    return LANES.index(name) if name in LANES else len(LANES)


def _lane_name(tid: int) -> str:
    return LANES[tid] if tid < len(LANES) else "other"


def _metadata(spans: list[Span]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for pid in sorted({span.pid for span in spans}):
        name: SpanArgs = {"name": f"tuner pid {pid}"}
        events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": name})
        for tid in sorted({_lane(span.name) for span in spans if span.pid == pid}):
            lane: SpanArgs = {"name": _lane_name(tid)}
            events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": lane})
    return events


def _complete_event(span: Span) -> TraceEvent:
    return {
        "name": span.name,
        "cat": "run_loop",
        "ph": "X",
        "ts": span.start_us,
        "dur": span.dur_us,
        "pid": span.pid,
        "tid": _lane(span.name),
        "args": dict(span.args),
    }


def chrome_trace(spans: list[Span]) -> str:
    """A Chrome Trace Event Format document (``traceEvents`` complete events).

    Each span becomes a ``"ph": "X"`` event on a per-name lane so the profile
    reads as a stacked time-split; each recording process gets its own process
    row so ``--resume`` sessions are visually distinct.
    """
    ordered = sorted(spans, key=lambda item: item.start_us)
    events = [*_metadata(spans), *(_complete_event(span) for span in ordered)]
    return json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}, sort_keys=True)
