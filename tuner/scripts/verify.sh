#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

uv lock --project tuner --check
uv sync --project tuner
(cd tuner && uv run pyright src)
uv run --project tuner ruff format --check tuner/src tuner/tests
uv run --project tuner ruff check tuner/src tuner/tests
uv run --project tuner pytest -q tuner/tests
