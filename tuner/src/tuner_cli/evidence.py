"""Canonical manifest, append-only evidence, and atomic JSON file helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .identity import JsonValue, canonical_json, fingerprint


def atomic_json(path: Path, value: object, *, create_once: bool = False) -> None:
    """Publish a canonical JSON document atomically, never exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical_json(value) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if create_once:
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_manifest(path: Path, manifest: dict[str, JsonValue]) -> dict[str, JsonValue]:
    with_fingerprint = {**manifest, "fingerprint": fingerprint(manifest)}
    atomic_json(path, with_fingerprint, create_once=True)
    return with_fingerprint


class EvidenceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._sequence = 0
        with path.open("x", encoding="utf-8"):
            pass

    def append(self, event_type: str, payload: object) -> dict[str, JsonValue]:
        self._sequence += 1
        event: dict[str, JsonValue] = {
            "schema_version": 1,
            "sequence": self._sequence,
            "type": event_type,
            "payload": json.loads(canonical_json(payload)),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event


def read_events(path: Path) -> Iterator[dict[str, JsonValue]]:
    expected = 1
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if (
                not isinstance(event, dict)
                or event.get("schema_version") != 1
                or event.get("sequence") != expected
            ):
                raise ValueError("evidence sequence is invalid")
            expected += 1
            yield event
