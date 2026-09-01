"""Extract a Druid-realism calibration from recorded tuner runs.

The mechanism sweep (`mechanism_sim`) needs three facts about real tuning
evidence, none of which depend on the rules of any game:

1. how a single pair's utility scatters around a stratum's latent propensity,
2. how far apart candidate strengths sit inside one cohort, and how tight the
   eta-2 cut boundary gap is,
3. how correlated a candidate's per-stratum deviations are.

We read these once from already-recorded runs and freeze them in a checked-in
JSON file (config-as-data). Nothing here re-runs a tuner.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import Manifest, read_manifest
from .codec import JsonObject, JsonValue, elements, integer, json_object, raw_number, strict_json
from .domain import Observation, ReplayState
from .evidence import read_events
from .replay import replay

# The pair utility alphabet: two games, each win/draw/loss -> {0, .25, .5, .75, 1}.
UTILITY_ALPHABET: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_BINS = 5
MIN_BIN_SAMPLES = 30
CALIBRATION_SCHEMA = "mechanism-calibration-v1"


@dataclass(frozen=True, slots=True)
class UtilityBin:
    lo: float
    hi: float
    count: int
    mean_propensity: float
    cdf: tuple[tuple[float, float], ...]

    def to_json(self) -> JsonObject:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "count": self.count,
            "mean_propensity": self.mean_propensity,
            "cdf": [[value, cumulative] for value, cumulative in self.cdf],
        }

    @classmethod
    def from_json(cls, raw: JsonObject) -> UtilityBin:
        cdf: list[tuple[float, float]] = []
        for entry in elements(raw["cdf"], "cdf point"):
            pair = elements(entry, "cdf pair")
            cdf.append((raw_number(pair[0], "cdf value"), raw_number(pair[1], "cdf cumulative")))
        return cls(
            raw_number(raw["lo"], "bin lower"),
            raw_number(raw["hi"], "bin upper"),
            integer(raw["count"], "bin count"),
            raw_number(raw["mean_propensity"], "bin mean propensity"),
            tuple(cdf),
        )


@dataclass(frozen=True, slots=True)
class Calibration:
    generated: str
    provenance: JsonObject
    pair_utility_bins: tuple[UtilityBin, ...]
    strength_mean: float
    strength_std: float
    boundary_gap_mean: float
    boundary_gap_std: float
    pairs_per_stratum: tuple[int, ...]
    deviation_correlation: float
    deviation_std: float

    def to_json(self) -> JsonObject:
        bins: list[JsonValue] = [item.to_json() for item in self.pair_utility_bins]
        return {
            "schema": CALIBRATION_SCHEMA,
            "generated": self.generated,
            "provenance": self.provenance,
            "pair_utility_bins": bins,
            "overall_strength": {"mean": self.strength_mean, "std": self.strength_std},
            "boundary_gap": {"mean": self.boundary_gap_mean, "std": self.boundary_gap_std},
            "stratum_structure": {
                "pairs_per_stratum_18": list(self.pairs_per_stratum),
                "deviation_correlation": self.deviation_correlation,
                "deviation_std": self.deviation_std,
            },
        }

    @classmethod
    def from_json(cls, raw: JsonObject) -> Calibration:
        if raw.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError("unexpected calibration schema")
        strength = json_object(raw["overall_strength"], "overall_strength")
        gap = json_object(raw["boundary_gap"], "boundary_gap")
        structure = json_object(raw["stratum_structure"], "stratum_structure")
        provenance = raw.get("provenance", {})
        return cls(
            str(raw.get("generated", "")),
            json_object(provenance, "provenance"),
            tuple(
                UtilityBin.from_json(json_object(item, "bin"))
                for item in elements(raw["pair_utility_bins"], "pair_utility_bins")
            ),
            raw_number(strength["mean"], "strength mean"),
            raw_number(strength["std"], "strength std"),
            raw_number(gap["mean"], "gap mean"),
            raw_number(gap["std"], "gap std"),
            tuple(
                integer(item, "stratum pairs")
                for item in elements(structure["pairs_per_stratum_18"], "pairs_per_stratum_18")
            ),
            raw_number(structure["deviation_correlation"], "deviation correlation"),
            raw_number(structure["deviation_std"], "deviation std"),
        )

    def bin_for(self, propensity: float) -> UtilityBin:
        return min(
            self.pair_utility_bins,
            key=lambda item: (
                0.0
                if item.lo <= propensity < item.hi
                else min(abs(propensity - item.lo), abs(propensity - item.hi))
            ),
        )


@dataclass(frozen=True, slots=True)
class _StratumRecord:
    propensity: float
    utilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _RunFacts:
    records: list[_StratumRecord]
    boundary_gaps: list[float]
    strengths: list[float]
    stratum_counts: dict[str, int]
    propensity_vectors: list[dict[str, float]]
    deviation_stds: list[float]


def _tuning_maximum_observations(manifest: Manifest, state: ReplayState) -> list[Observation]:
    prefix_id = manifest.tuning_prefix.prefix_id
    completed = {
        candidate.candidate_id
        for cohort in state.completed_cohorts
        for candidate in cohort.candidates
    }
    return [
        item
        for item in state.observations
        if item.phase == "tuning"
        and item.context.task_prefix.prefix_id == prefix_id
        and item.candidate_id in completed
    ]


def _stratum_of_case(manifest: Manifest) -> tuple[str, ...]:
    return tuple(case.stratum_id for case in manifest.prefix_cases("tuning"))


def _cohort_index_of(state: ReplayState, candidate_id: str) -> int:
    for cohort in state.completed_cohorts:
        if any(item.candidate_id == candidate_id for item in cohort.candidates):
            return cohort.cohort_index
    raise ValueError("observation candidate is outside every completed cohort")


def _collect_run(run_dir: Path) -> _RunFacts:
    manifest = read_manifest(run_dir / "manifest.json")
    events = read_events(run_dir / "evidence.jsonl")
    state = replay(manifest, events)
    if state.terminal_status != "complete":
        raise ValueError(f"{run_dir.name}: run is not complete")

    strata_by_case = _stratum_of_case(manifest)
    counts: dict[str, int] = defaultdict(int)
    for stratum_id in strata_by_case:
        counts[stratum_id] += 1

    records: list[_StratumRecord] = []
    strengths: list[float] = []
    vectors: list[dict[str, float]] = []
    deviation_stds: list[float] = []
    per_cohort: dict[int, list[float]] = defaultdict(list)
    kept = max(manifest.finalists, (manifest.cohort_size + 1) // 2)

    for obs in _tuning_maximum_observations(manifest, state):
        by_stratum: dict[str, list[float]] = defaultdict(list)
        for stratum_id, utility in zip(strata_by_case, obs.pair_utilities, strict=True):
            by_stratum[stratum_id].append(utility)
        propensities: dict[str, float] = {}
        for stratum_id, values in by_stratum.items():
            propensity = sum(values) / len(values)
            propensities[stratum_id] = propensity
            records.append(_StratumRecord(propensity, tuple(values)))
        vectors.append(propensities)
        if len(propensities) > 1:
            deviation_stds.append(statistics.pstdev(propensities.values()))
        strengths.append(obs.estimate.mean)
        per_cohort[_cohort_index_of(state, obs.candidate_id)].append(obs.estimate.mean)

    gaps: list[float] = []
    for values in per_cohort.values():
        if len(values) > kept:
            ordered = sorted(values, reverse=True)
            gaps.append(ordered[kept - 1] - ordered[kept])

    return _RunFacts(records, gaps, strengths, dict(counts), vectors, deviation_stds)


def _utility_cdf(utilities: Sequence[float]) -> tuple[tuple[float, float], ...]:
    total = len(utilities)
    cumulative = 0
    cdf: list[tuple[float, float]] = []
    for value in UTILITY_ALPHABET:
        cumulative += sum(1 for item in utilities if math.isclose(item, value))
        cdf.append((value, cumulative / total))
    return tuple(cdf)


def _deviation_correlation(vectors: list[dict[str, float]]) -> float:
    """Average pairwise Pearson correlation between stratum columns across
    candidates. Near 1 a candidate strong against one opponent stratum tends to
    be strong against another; near 0 the strata move independently."""
    strata = sorted({key for vector in vectors for key in vector})
    columns = {
        stratum: [vector[stratum] for vector in vectors if stratum in vector] for stratum in strata
    }
    correlations: list[float] = []
    for left in range(len(strata)):
        for right in range(left + 1, len(strata)):
            a = columns[strata[left]]
            b = columns[strata[right]]
            if len(a) != len(b) or len(a) < 3:
                continue
            try:
                correlations.append(statistics.correlation(a, b))
            except statistics.StatisticsError:
                continue
    if not correlations:
        return 0.0
    return max(-1.0, min(1.0, statistics.mean(correlations)))


def _quantiles(values: Sequence[float], points: Sequence[float]) -> list[JsonValue]:
    ordered = sorted(values)
    result: list[JsonValue] = []
    for point in points:
        position = point * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        value = ordered[lower] * (1 - fraction) + ordered[upper] * fraction
        result.append([point, value])
    return result


@dataclass(slots=True)
class _Aggregate:
    records: list[_StratumRecord]
    gaps: list[float]
    strengths: list[float]
    vectors: list[dict[str, float]]
    deviation_stds: list[float]
    stratum_counts: dict[str, int]
    runs: list[JsonValue]


def _gather(run_dirs: Sequence[Path]) -> _Aggregate:
    aggregate = _Aggregate([], [], [], [], [], {}, [])
    for run_dir in run_dirs:
        facts = _collect_run(run_dir)
        aggregate.records.extend(facts.records)
        aggregate.gaps.extend(facts.boundary_gaps)
        aggregate.strengths.extend(facts.strengths)
        aggregate.vectors.extend(facts.propensity_vectors)
        aggregate.deviation_stds.extend(facts.deviation_stds)
        aggregate.stratum_counts = facts.stratum_counts
        aggregate.runs.append(
            {
                "run_id": run_dir.name,
                "stratum_records": len(facts.records),
                "cohort_boundary_gaps": len(facts.boundary_gaps),
                "candidate_strengths": len(facts.strengths),
            }
        )
    return aggregate


def _build_bins(records: Sequence[_StratumRecord], bins: int) -> list[UtilityBin]:
    edges = [index / bins for index in range(bins + 1)]
    binned: list[list[float]] = [[] for _ in range(bins)]
    for record in records:
        index = min(bins - 1, int(record.propensity * bins))
        binned[index].extend(record.utilities)
    result: list[UtilityBin] = []
    for index, utilities in enumerate(binned):
        if len(utilities) < MIN_BIN_SAMPLES:
            continue
        result.append(
            UtilityBin(
                edges[index],
                edges[index + 1],
                len(utilities),
                sum(utilities) / len(utilities),
                _utility_cdf(utilities),
            )
        )
    if not result:
        raise ValueError("no propensity bin reached the minimum sample count")
    return result


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _pstdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def build_calibration(run_dirs: Sequence[Path], bins: int = DEFAULT_BINS) -> Calibration:
    if not run_dirs:
        raise ValueError("calibration needs at least one recorded run directory")
    data = _gather(run_dirs)
    points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    provenance: JsonObject = {
        "runs": data.runs,
        "total_stratum_records": len(data.records),
        "total_candidate_strengths": len(data.strengths),
        "total_cohort_boundary_gaps": len(data.gaps),
        "propensity_bins": bins,
        "min_bin_samples": MIN_BIN_SAMPLES,
        "strength_quantiles": _quantiles(data.strengths, points),
        "boundary_gap_quantiles": _quantiles(data.gaps, points) if data.gaps else [],
    }
    return Calibration(
        datetime.now(timezone.utc).isoformat(),
        provenance,
        tuple(_build_bins(data.records, bins)),
        _mean(data.strengths),
        _pstdev(data.strengths),
        _mean(data.gaps),
        _pstdev(data.gaps),
        tuple(sorted(data.stratum_counts.values(), reverse=True)),
        _deviation_correlation(data.vectors),
        _mean(data.deviation_stds),
    )


def write_calibration(
    run_dirs: Sequence[Path], out_path: Path, bins: int = DEFAULT_BINS
) -> Calibration:
    calibration = build_calibration(run_dirs, bins)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration.to_json(), indent=2, sort_keys=True) + "\n")
    return calibration


def load_calibration(path: Path) -> Calibration:
    return Calibration.from_json(json_object(strict_json(Path(path).read_text()), "calibration"))
