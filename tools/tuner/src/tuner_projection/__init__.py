"""Rebuildable read-only SQLite projection of version-4 tuner run artifacts.

This package flattens the typed values produced by ``tuner_cli.replay`` and
``tuner_cli.report`` into relational rows. It computes no statistic, interval,
ranking, or audit value of its own; the ``run-dir`` triple stays the sole
scientific authority and the SQLite file is disposable.
"""

from __future__ import annotations

from .build import project_runs
from .schema import PROJECTION_SCHEMA_VERSION

__all__ = ["PROJECTION_SCHEMA_VERSION", "project_runs"]
