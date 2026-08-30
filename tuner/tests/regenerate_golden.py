"""Regenerate the checked-in version-4 golden fixtures.

Run from the tuner project root::

    uv run python tests/regenerate_golden.py

Review the resulting diff carefully: any change here is a change to an
operator-visible scientific artifact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from golden_support import FIXTURES, GoldenTarget, golden_options, write_binary, write_objective

from tuner_cli.evidence import read_events, scientific_projection
from tuner_cli.run import run_foreground


def _interrupted_prefix(evidence: str) -> str:
    lines = evidence.splitlines()
    for index, line in enumerate(lines):
        if '"type":"pair_started"' in line:
            return "\n".join(lines[: index + 1]) + "\n"
    raise RuntimeError("golden evidence has no pair_started event")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        run_dir = tmp / "run"
        run_foreground(
            golden_options(write_binary(tmp), run_dir, write_objective(tmp)), GoldenTarget()
        )
        manifest = (run_dir / "manifest.json").read_bytes()
        evidence = (run_dir / "evidence.jsonl").read_text(encoding="utf-8")
        report = (run_dir / "report.json").read_bytes()

    (FIXTURES / "manifest.json").write_bytes(manifest)
    (FIXTURES / "evidence.jsonl").write_text(evidence, encoding="utf-8")
    (FIXTURES / "report.json").write_bytes(report)
    (FIXTURES / "evidence.interrupted.jsonl").write_text(
        _interrupted_prefix(evidence), encoding="utf-8"
    )
    projection = scientific_projection(read_events(FIXTURES / "evidence.jsonl"))
    (FIXTURES / "scientific_projection.json").write_text(projection + "\n", encoding="utf-8")
    print(f"wrote golden fixtures to {FIXTURES}")


if __name__ == "__main__":
    main()
