// compute-model.ts — pure derivation behind the Run Science "Compute
// ledger" section. Flattens `report.compute` into a per-phase table
// (tuning / validation / diagnostic), a treemap grouping of pair-attempt
// dispositions per phase, and headline totals. No rendering, no `fetch`.

import type { JsonValue } from "../../types.js";
import type { ProjectionComputePhase } from "../tuner-types.js";
import { asArray, asNumber, asObject, asString } from "./json-util.js";
import { formatWall } from "./progress-model.js";
import type { KpiItem } from "../primitives/KpiRow.js";
import type { TreemapGroup } from "../primitives/Treemap.js";

const PHASES = ["tuning", "validation", "diagnostic"] as const;
const PHASE_LABEL: Record<string, string> = {
  tuning: "Tuning",
  validation: "Validation",
  diagnostic: "Diagnostic",
};

export interface PhaseLedger {
  phase: string;
  label: string;
  budget: number;
  pairAttempts: number;
  completedPairs: number;
  failedAttempts: number;
  censoredAttempts: number;
  overrunPairAttempts: number;
  unspentPairAttempts: number;
  physicalGames: number;
  searchIterations: number;
  wallTimeMs: number;
}

export interface ComputeExtension {
  label: string;
  detail: string;
}

export interface ComputeLedgerView {
  present: boolean;
  phases: PhaseLedger[];
  treemap: TreemapGroup[];
  kpis: KpiItem[];
  extensions: ComputeExtension[];
}

const EMPTY: ComputeLedgerView = { present: false, phases: [], treemap: [], kpis: [], extensions: [] };

function n(v: JsonValue | undefined): number {
  return asNumber(v) ?? 0;
}

/** Build the ledger from the projection's `compute_phases` rows (live during a
 * run) with the `report.compute` overlay when it exists. The rows carry
 * attempts / completed / failed / censored / games / iterations / wall; the
 * report additionally carries the per-phase budget, overrun and unspent
 * counts and the extension log. */
export function deriveComputeLedger(
  report: JsonValue | undefined,
  computePhases?: ProjectionComputePhase[],
): ComputeLedgerView {
  const compute = asObject(asObject(report)?.["compute"]);
  const rows = computePhases ?? [];
  if (compute) return fromReport(compute);
  if (rows.length === 0) return EMPTY;

  const byPhase = new Map(rows.map((r) => [r.phase, r]));
  const phases: PhaseLedger[] = PHASES.flatMap((phase) => {
    const r = byPhase.get(phase);
    if (!r) return [];
    return [
      {
        phase,
        label: PHASE_LABEL[phase] ?? phase,
        budget: 0,
        pairAttempts: r.pair_attempts,
        completedPairs: r.completed_pairs,
        failedAttempts: r.failed_attempts,
        censoredAttempts: r.censored_attempts,
        overrunPairAttempts: 0,
        unspentPairAttempts: 0,
        physicalGames: r.physical_games,
        searchIterations: r.search_iterations,
        wallTimeMs: r.wall_time_ms,
      },
    ];
  });
  return assemble(phases, []);
}

function fromReport(compute: Record<string, JsonValue>): ComputeLedgerView {
  const budget = asObject(compute["budget"]) ?? {};
  const phases: PhaseLedger[] = PHASES.flatMap((phase) => {
    const p = asObject(compute[phase]);
    if (!p) return [];
    return [
      {
        phase,
        label: PHASE_LABEL[phase] ?? phase,
        budget: n(budget[`${phase}_pair_attempts`]),
        pairAttempts: n(p["pair_attempts"]),
        completedPairs: n(p["completed_pairs"]),
        failedAttempts: n(p["failed_attempts"]),
        censoredAttempts: n(p["censored_attempts"]),
        overrunPairAttempts: n(p["overrun_pair_attempts"]),
        unspentPairAttempts: n(p["unspent_pair_attempts"]),
        physicalGames: n(p["physical_games"]),
        searchIterations: n(p["search_iterations"]),
        wallTimeMs: n(p["wall_time_ms"]),
      },
    ];
  });

  const extensions: ComputeExtension[] = asArray(compute["extensions"]).flatMap((eRaw) => {
    const e = asObject(eRaw);
    if (!e) return [];
    return [
      {
        label: asString(e["at"]) ?? asString(e["timestamp"]) ?? "extension",
        detail: asString(e["detail"]) ?? JSON.stringify(e),
      },
    ];
  });

  return assemble(phases, extensions, asString(compute["policy_version"]));
}

function assemble(
  phases: PhaseLedger[],
  extensions: ComputeExtension[],
  policyVersion?: string | null,
): ComputeLedgerView {
  const treemap: TreemapGroup[] = phases
    .filter((p) => p.budget > 0 || p.pairAttempts > 0)
    .map((p) => ({
      key: p.phase,
      label: p.label,
      children: [
        { label: "completed", value: p.completedPairs },
        { label: "failed", value: p.failedAttempts },
        { label: "censored", value: p.censoredAttempts },
        { label: "overrun", value: p.overrunPairAttempts },
        { label: "unspent", value: p.unspentPairAttempts },
      ].filter((c) => c.value > 0),
    }))
    .filter((g) => g.children.length > 0);

  const sum = (pick: (p: PhaseLedger) => number): number => phases.reduce((a, p) => a + pick(p), 0);
  const kpis: KpiItem[] = [
    { label: "Pair attempts", value: String(sum((p) => p.pairAttempts)) },
    { label: "Physical games", value: String(sum((p) => p.physicalGames)) },
    { label: "Search iterations", value: String(sum((p) => p.searchIterations)) },
    { label: "Wall time", value: formatWall(sum((p) => p.wallTimeMs)) },
    { label: "Policy", value: policyVersion ?? "—" },
  ];

  return { present: phases.length > 0, phases, treemap, kpis, extensions };
}
