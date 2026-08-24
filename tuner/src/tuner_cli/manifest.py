"""Immutable resolved manifests for persistent tuning sessions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from .config import SearchConfig
from .lifecycle import LIFECYCLE_SCHEMA_VERSION, strict_json_dumps

MANIFEST_SCHEMA_VERSION: Final = 1


class SessionForkRequired(ValueError):
    """The requested launch changes immutable session semantics."""


def canonical_json(value: Any) -> str:
    """Serialize semantic data deterministically for fingerprinting."""
    return strict_json_dumps(value, sort_keys=True)


def manifest_fingerprint(semantic_inputs: dict[str, Any]) -> str:
    """Return the SHA-256 fingerprint of resolved semantic inputs."""
    return hashlib.sha256(canonical_json(semantic_inputs).encode("utf-8")).hexdigest()


def search_space_hash(cfg: SearchConfig) -> str:
    """Fingerprint the schema returned by the game binary's ``tune describe``."""
    return manifest_fingerprint(
        {
            "parameters": [asdict(parameter) for parameter in cfg.parameters],
            "conditions": [asdict(condition) for condition in cfg.conditions],
        }
    )


def build_session_manifest(
    cfg: SearchConfig,
    *,
    game_kind: str | None,
    binary: Path,
    git_sha: str,
    study_name: str,
    storage: str,
) -> dict[str, Any]:
    """Build the versioned, immutable snapshot for one logical session.

    Trial count and coordinator worker count are operational launch controls;
    they intentionally do not participate in the fingerprint.
    """
    cfg.validate()
    semantic_inputs = session_semantic_inputs(
        cfg,
        game_kind=game_kind,
        binary=binary,
        git_sha=git_sha,
        study_name=study_name,
        storage=storage,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "fingerprint": manifest_fingerprint(semantic_inputs),
        "semantic_inputs": semantic_inputs,
    }


def session_semantic_inputs(
    cfg: SearchConfig,
    *,
    game_kind: str | None,
    binary: Path,
    git_sha: str,
    study_name: str,
    storage: str,
) -> dict[str, Any]:
    """Project resolved policy inputs that define a logical tuning session."""
    return {
        "game": {
            "kind": game_kind or binary.name.removeprefix("game-"),
            "config": cfg.target.game_config,
        },
        "optimizer": {
            "direction": "maximize",
            "sampler": {
                "kind": cfg.optimizer.sampler.kind,
                "seed": cfg.optimizer.seed,
                "deterministic": cfg.optimizer.deterministic,
                "startup_trials": cfg.optimizer.sampler.startup_trials,
            },
            "pruning": asdict(cfg.optimizer.pruning),
            "resource": asdict(cfg.optimizer.resource),
        },
        "rating": {
            "model": "ThurstoneMostellerPart",
            "score": "mu_minus_k_sigma",
            "sigma_stop": cfg.optimizer.rating.sigma_stop,
            "conservative_k": cfg.optimizer.rating.conservative_k,
        },
        "evaluator": {
            "rounds": cfg.target.rounds,
            "max_iterations": cfg.target.max_iterations,
            "max_time_ms": cfg.target.max_time_ms,
            "baselines": cfg.target.baselines,
            "baseline_configs": cfg.target.baseline_configs,
        },
        "search_space": {
            "hash": search_space_hash(cfg),
            "parameters": [asdict(parameter) for parameter in cfg.parameters],
            "conditions": [asdict(condition) for condition in cfg.conditions],
        },
        "engine": {"binary": str(binary), "git_sha": git_sha},
        "study": {"name": study_name, "storage": storage},
        "schema_versions": {
            "manifest": MANIFEST_SCHEMA_VERSION,
            "lifecycle": LIFECYCLE_SCHEMA_VERSION,
        },
    }


def write_manifest_atomic(path: str | Path, manifest: dict[str, Any]) -> None:
    """Create a manifest once, rejecting later semantic changes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_existing_manifest(destination, manifest)
        return
    _replace_manifest_atomically(destination, canonical_json(manifest) + "\n")


def _validate_existing_manifest(destination: Path, manifest: dict[str, Any]) -> None:
    """Reject a requested manifest whose semantic fingerprint changed."""
    existing = json.loads(destination.read_text(encoding="utf-8"))
    if existing.get("fingerprint") != manifest.get("fingerprint"):
        raise SessionForkRequired(
            f"fork required: session manifest at {destination} has a different fingerprint"
        )


def _replace_manifest_atomically(destination: Path, contents: str) -> None:
    """Durably create a new manifest without exposing a partial file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as temporary:
        temporary.write(contents)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
