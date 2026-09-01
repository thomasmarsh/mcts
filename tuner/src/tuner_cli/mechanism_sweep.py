"""Grid sweep of the shadow-race mechanism and its preregistered PASS gate.

Runs `mechanism_sim.run_trial` across a `(boundary_gap, spread_scale)` grid, many
trials per cell, and aggregates the eviction metrics with Wilson intervals. The
counting / aggregation / interval math has a fast deterministic test; the full
sweep is a `tuner-mechanism` entry point, not part of the automated suite.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .artifacts import Manifest
from .codec import JsonObject, JsonValue
from .mechanism_calibration import Calibration
from .mechanism_sim import TrialClassification, run_trial

# Preregistered PASS thresholds, fixed before the full sweep is run and not moved
# after seeing results.
TOP_SET_FALSE_EVICTION_UPPER = 0.03  # clause 1: X in [1, 3] %
BOUNDARY_REVERSAL_UPPER_OVERALL = 0.03  # clause 2: Y %
BOUNDARY_REVERSAL_UPPER_WORST_CELL = 0.06  # clause 2: Z % (worst near-tie cell)
BOUNDARY_REVERSAL_RATIO_LIMIT = 2.0  # clause 3: K, per-eviction rate vs paired

DEFAULT_BOUNDARY_GAPS: tuple[float, ...] = (-0.04, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2)
DEFAULT_SPREAD_SCALES: tuple[float, ...] = (0.6, 1.0, 1.5)
DEFAULT_TRIALS = 3000
_Z = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = _Z) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True, slots=True)
class PolicyRates:
    trials: int
    eliminated: int
    mean_eliminated_per_trial: float
    mean_unique_pairs_saved: float
    top_set_false_eviction_rate: float
    top_set_false_eviction_upper: float
    boundary_reversal_rate: float
    boundary_reversal_upper: float
    rule_tie_eviction_rate: float
    per_stratum_dangerous_flip_rate: float

    def to_json(self) -> JsonObject:
        return {
            "trials": self.trials,
            "eliminated": self.eliminated,
            "mean_eliminated_per_trial": self.mean_eliminated_per_trial,
            "mean_unique_pairs_saved": self.mean_unique_pairs_saved,
            "top_set_false_eviction_rate": self.top_set_false_eviction_rate,
            "top_set_false_eviction_upper": self.top_set_false_eviction_upper,
            "boundary_reversal_rate": self.boundary_reversal_rate,
            "boundary_reversal_upper": self.boundary_reversal_upper,
            "rule_tie_eviction_rate": self.rule_tie_eviction_rate,
            "per_stratum_dangerous_flip_rate": self.per_stratum_dangerous_flip_rate,
        }


@dataclass(slots=True)
class PolicyTally:
    trials: int = 0
    eliminated: int = 0
    top_set_false_evictions: int = 0
    boundary_reversals: int = 0
    rule_tie_evictions: int = 0
    per_stratum_dangerous_flips: int = 0
    unique_pairs_saved: int = 0

    def add(self, result: TrialClassification) -> None:
        self.trials += 1
        self.eliminated += result.eliminated
        self.top_set_false_evictions += result.top_set_false_evictions
        self.boundary_reversals += result.boundary_reversals
        self.rule_tie_evictions += result.rule_tie_evictions
        self.per_stratum_dangerous_flips += result.per_stratum_dangerous_flips
        self.unique_pairs_saved += result.unique_pairs_saved

    def rates(self) -> PolicyRates:
        evicted = max(self.eliminated, 1)
        trials = max(self.trials, 1)
        return PolicyRates(
            self.trials,
            self.eliminated,
            self.eliminated / trials,
            self.unique_pairs_saved / trials,
            self.top_set_false_evictions / evicted,
            wilson_interval(self.top_set_false_evictions, self.eliminated)[1],
            self.boundary_reversals / evicted,
            wilson_interval(self.boundary_reversals, self.eliminated)[1],
            self.rule_tie_evictions / evicted,
            self.per_stratum_dangerous_flips / evicted,
        )


@dataclass(frozen=True, slots=True)
class CellResult:
    boundary_gap: float
    spread_scale: float
    halving: PolicyRates
    paired: PolicyRates

    def to_json(self) -> JsonObject:
        return {
            "boundary_gap": self.boundary_gap,
            "spread_scale": self.spread_scale,
            "halving": self.halving.to_json(),
            "paired": self.paired.to_json(),
        }


@dataclass(frozen=True, slots=True)
class GateResult:
    clauses: dict[str, bool]
    worst_cell_boundary_reversal_upper: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SweepResult:
    boundary_gaps: tuple[float, ...]
    spread_scales: tuple[float, ...]
    trials_per_cell: int
    seed: int
    paired_resamples: int
    cells: tuple[CellResult, ...]
    overall_halving: PolicyRates
    overall_paired: PolicyRates
    gate: GateResult

    def to_json(self) -> JsonObject:
        cells: list[JsonValue] = [cell.to_json() for cell in self.cells]
        clauses: JsonObject = dict(self.gate.clauses)
        return {
            "schema": "mechanism-sweep-v1",
            "config": {
                "boundary_gaps": list(self.boundary_gaps),
                "spread_scales": list(self.spread_scales),
                "trials_per_cell": self.trials_per_cell,
                "seed": self.seed,
                "paired_resamples": self.paired_resamples,
            },
            "cells": cells,
            "overall": {
                "halving": self.overall_halving.to_json(),
                "paired": self.overall_paired.to_json(),
            },
            "gate": {
                "clauses": clauses,
                "worst_cell_boundary_reversal_upper": (
                    self.gate.worst_cell_boundary_reversal_upper
                ),
                "passed": self.gate.passed,
            },
        }


def evaluate_gate(cells: tuple[CellResult, ...], overall_halving: PolicyRates) -> GateResult:
    top_set_ok = overall_halving.top_set_false_eviction_upper <= TOP_SET_FALSE_EVICTION_UPPER

    worst_cell_upper = max(cell.halving.boundary_reversal_upper for cell in cells)
    reversal_ok = (
        overall_halving.boundary_reversal_upper <= BOUNDARY_REVERSAL_UPPER_OVERALL
        and worst_cell_upper <= BOUNDARY_REVERSAL_UPPER_WORST_CELL
    )

    ratio_ok = True
    for cell in cells:
        halving_rate = cell.halving.boundary_reversal_rate
        paired_rate = cell.paired.boundary_reversal_rate
        if paired_rate == 0.0:
            if halving_rate > BOUNDARY_REVERSAL_UPPER_WORST_CELL:
                ratio_ok = False
        elif halving_rate > BOUNDARY_REVERSAL_RATIO_LIMIT * paired_rate:
            ratio_ok = False

    saved_ok = all(
        cell.halving.mean_unique_pairs_saved >= cell.paired.mean_unique_pairs_saved
        for cell in cells
    )

    clauses = {
        "1_top_set_false_eviction_bounded": top_set_ok,
        "2_boundary_reversal_bounded": reversal_ok,
        "3_boundary_reversal_ratio_bounded": ratio_ok,
        "4_halving_saves_at_least_as_much": saved_ok,
    }
    return GateResult(clauses, worst_cell_upper, all(clauses.values()))


def run_sweep(
    calibration: Calibration,
    manifest: Manifest,
    *,
    boundary_gaps: tuple[float, ...] = DEFAULT_BOUNDARY_GAPS,
    spread_scales: tuple[float, ...] = DEFAULT_SPREAD_SCALES,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    paired_resamples: int = 512,
) -> SweepResult:
    cells: list[CellResult] = []
    overall_halving = PolicyTally()
    overall_paired = PolicyTally()
    for gap in boundary_gaps:
        for spread in spread_scales:
            halving = PolicyTally()
            paired = PolicyTally()
            for trial in range(trials):
                rng = random.Random(f"{seed}|{gap!r}|{spread!r}|{trial}")
                outcome = run_trial(calibration, manifest, rng, gap, spread, paired_resamples)
                halving.add(outcome["halving"])
                paired.add(outcome["paired"])
                overall_halving.add(outcome["halving"])
                overall_paired.add(outcome["paired"])
            cells.append(CellResult(gap, spread, halving.rates(), paired.rates()))
    gate = evaluate_gate(tuple(cells), overall_halving.rates())
    return SweepResult(
        boundary_gaps,
        spread_scales,
        trials,
        seed,
        paired_resamples,
        tuple(cells),
        overall_halving.rates(),
        overall_paired.rates(),
        gate,
    )


def format_summary(sweep: SweepResult) -> str:
    lines: list[str] = [
        f"mechanism sweep: {len(sweep.cells)} cells x {sweep.trials_per_cell} trials "
        f"(seed {sweep.seed}, paired resamples {sweep.paired_resamples})",
        f"  {'gap':>6} {'spread':>7} | {'h_elim':>7} {'h_topF':>8} "
        f"{'h_reversal (95% upper)':>26} {'h_saved':>8} | {'p_reversal':>11} {'p_saved':>8}",
    ]
    for cell in sweep.cells:
        h = cell.halving
        p = cell.paired
        lines.append(
            f"  {cell.boundary_gap:>6.2f} {cell.spread_scale:>7.2f} | "
            f"{h.mean_eliminated_per_trial:>7.2f} {h.top_set_false_eviction_rate:>8.4f} "
            f"{h.boundary_reversal_rate:>13.4f} ({h.boundary_reversal_upper:>7.4f}) "
            f"{h.mean_unique_pairs_saved:>8.2f} | "
            f"{p.boundary_reversal_rate:>11.4f} {p.mean_unique_pairs_saved:>8.2f}"
        )
    over = sweep.overall_halving
    lines.append("")
    lines.append(
        f"  overall halving: top-set-false {over.top_set_false_eviction_rate:.4f} "
        f"(upper {over.top_set_false_eviction_upper:.4f}), boundary-reversal "
        f"{over.boundary_reversal_rate:.4f} (upper {over.boundary_reversal_upper:.4f}), "
        f"rule-tie {over.rule_tie_eviction_rate:.4f}"
    )
    lines.append("")
    for name, ok in sweep.gate.clauses.items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    lines.append(f"\n  verdict: {'PASS' if sweep.gate.passed else 'FAIL'}")
    return "\n".join(lines)
