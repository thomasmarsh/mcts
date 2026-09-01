"""Regenerate the checked-in version-4 golden fixtures.

Run from the tuner project root::

    uv run python tests/regenerate_golden.py

Review the resulting diff carefully: any change here is a change to an
operator-visible scientific artifact.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from golden_support import (
    ACTIVE_FIXTURES,
    FIXTURES,
    ActiveHalvingGoldenTarget,
    GoldenTarget,
    active_halving_golden_options,
    golden_options,
    write_binary,
    write_objective,
)

from tuner_cli.evidence import read_events, scientific_projection
from tuner_cli.run import RunOptions, run_foreground
from tuner_cli.target import Target


def _interrupted_prefix(evidence: str) -> str:
    lines = evidence.splitlines()
    for index, line in enumerate(lines):
        if '"type":"pair_started"' in line:
            return "\n".join(lines[: index + 1]) + "\n"
    raise RuntimeError("golden evidence has no pair_started event")


def _generate(
    fixtures: Path,
    options: Callable[[Path, Path, Path], RunOptions],
    target: Callable[[], Target],
) -> None:
    fixtures.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        run_dir = tmp / "run"
        run_foreground(options(write_binary(tmp), run_dir, write_objective(tmp)), target())
        manifest = (run_dir / "manifest.json").read_bytes()
        evidence = (run_dir / "evidence.jsonl").read_text(encoding="utf-8")
        report = (run_dir / "report.json").read_bytes()

    (fixtures / "manifest.json").write_bytes(manifest)
    (fixtures / "evidence.jsonl").write_text(evidence, encoding="utf-8")
    (fixtures / "report.json").write_bytes(report)
    (fixtures / "evidence.interrupted.jsonl").write_text(
        _interrupted_prefix(evidence), encoding="utf-8"
    )
    projection = scientific_projection(read_events(fixtures / "evidence.jsonl"))
    (fixtures / "scientific_projection.json").write_text(projection + "\n", encoding="utf-8")
    print(f"wrote golden fixtures to {fixtures}")


def main() -> None:
    _generate(FIXTURES, golden_options, GoldenTarget)
    _generate(ACTIVE_FIXTURES, active_halving_golden_options, ActiveHalvingGoldenTarget)


if __name__ == "__main__":
    main()
