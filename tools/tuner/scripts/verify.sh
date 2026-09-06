#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

uv lock --project tools/tuner --check
uv sync --project tools/tuner
(cd tools/tuner && uv run pyright src)
uv run --project tools/tuner ruff format --check tools/tuner/src tools/tuner/tests
uv run --project tools/tuner ruff check tools/tuner/src tools/tuner/tests
uv run --project tools/tuner pytest -q tools/tuner/tests
