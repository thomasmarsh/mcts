// elimination-model.ts — pure derivation behind the Run Science
// "Shadow / active elimination" section. A run carries at most one of
// `report.shadow_elimination` (recorded, never enforced) or
// `report.active_elimination` (enforced pruning with a randomized audit).
// This flattens whichever is present into calibration bins, headline KPIs,
// and a suspension flag. No rendering, no `fetch`.

import type { JsonValue } from "../../types.js";
import { asArray, asNumber, asObject, asString } from "./json-util.js";
import type { KpiItem } from "../primitives/KpiRow.js";

export interface CalibrationBin {
  /** Predicted-probability band. */
  lower: number;
  upper: number;
  meanPrediction: number;
  observedRate: number;
  count: number;
}

export interface EliminationView {
  present: boolean;
  mode: "shadow" | "active" | "none";
  /** Active elimination actually prunes candidates; shadow only records. */
  enforced: boolean;
  policyKind: string | null;
  policyVersion: string | null;
  suspended: boolean;
  suspensionReason: string | null;
  /** Shadow mode only — active mode carries no calibration. */
  calibrationBins: CalibrationBin[];
  kpis: KpiItem[];
}

const EMPTY: EliminationView = {
  present: false,
  mode: "none",
  enforced: false,
  policyKind: null,
  policyVersion: null,
  suspended: false,
  suspensionReason: null,
  calibrationBins: [],
  kpis: [],
};

function num(v: JsonValue | undefined): string {
  const n = asNumber(v);
  return n == null ? "—" : String(n);
}

function rate(v: JsonValue | undefined): string {
  const n = asNumber(v);
  return n == null ? "—" : `${(n * 100).toFixed(1)}%`;
}

function deriveShadow(se: Record<string, JsonValue>): EliminationView {
  const summary = asObject(se["summary"]) ?? {};
  const scope = asObject(se["scope"]) ?? {};
  const policy = asObject(se["policy"]) ?? {};

  let reversals = 0;
  let eliminationReversals = 0;
  for (const sRaw of asArray(se["strata"])) {
    const s = asObject(sRaw);
    reversals += asNumber(s?.["reversals"]) ?? 0;
    eliminationReversals += asNumber(s?.["elimination_reversals"]) ?? 0;
  }

  const calibrationBins: CalibrationBin[] = asArray(se["calibration_bins"]).flatMap((bRaw) => {
    const b = asObject(bRaw);
    if (!b) return [];
    return [
      {
        lower: asNumber(b["lower"]) ?? 0,
        upper: asNumber(b["upper"]) ?? 1,
        meanPrediction: asNumber(b["mean_prediction"]) ?? 0,
        observedRate: asNumber(b["observed_success_rate"]) ?? 0,
        count: asNumber(b["count"]) ?? 0,
      },
    ];
  });

  const kpis: KpiItem[] = [
    { label: "Counterfactual eliminations", value: num(summary["counterfactual_eliminations"]) },
    {
      label: "Top-set false eliminations",
      value: num(summary["top_set_false_eliminations"]),
      hint: `rate ${rate(summary["top_set_false_elimination_rate"])}`,
    },
    {
      label: "Trash precision",
      value: summary["trash_precision"] == null ? "n/a" : rate(summary["trash_precision"]),
      hint: `${num(summary["true_trash_eliminations"])} true-trash eliminations`,
    },
    { label: "Brier score", value: (asNumber(summary["brier_score"]) ?? 0).toFixed(3) },
    {
      label: "Stratum reversals",
      value: String(reversals),
      hint: `${eliminationReversals} at an elimination boundary`,
    },
    { label: "Completed cohorts", value: num(scope["completed_cohorts"]) },
    {
      label: "Recorded looks",
      value: num(scope["recorded_looks"]),
      hint: `${num(scope["active_path_looks"])} on the active path`,
    },
    {
      label: "Held-out validation",
      value: scope["held_out_validation_used"] === true ? "used" : "not used",
    },
  ];

  return {
    present: true,
    mode: "shadow",
    enforced: false,
    policyKind: asString(policy["policy_kind"]),
    policyVersion: asString(policy["policy_version"]),
    suspended: false,
    suspensionReason: null,
    calibrationBins,
    kpis,
  };
}

function deriveActive(ae: Record<string, JsonValue>): EliminationView {
  const summary = asObject(ae["summary"]) ?? {};
  const policy = asObject(ae["policy"]) ?? {};
  const interval = asObject(ae["active_interval"]) ?? {};
  const suspended = ae["suspended"] === true;
  const suspension = asObject(ae["suspension"]);

  const kpis: KpiItem[] = [
    { label: "Pruned", value: num(summary["pruned"]) },
    { label: "Nominal eliminations", value: num(summary["nominal_eliminations"]) },
    { label: "Elimination decisions", value: num(summary["elimination_decisions"]) },
    {
      label: "Audited continuations",
      value: num(summary["audited_continuations"]),
      hint: `${num(summary["audit_continued"])} continued past the cut`,
    },
    {
      label: "Audited boundary reversals",
      value: num(summary["audited_boundary_reversals"]),
      hint: `estimated ${num(summary["estimated_boundary_reversals"])} · rate ${rate(
        summary["estimated_reversal_rate"],
      )}`,
    },
    {
      label: "Observed audit reversal rate",
      value: summary["observed_audit_reversal_rate"] == null
        ? "n/a"
        : rate(summary["observed_audit_reversal_rate"]),
    },
    { label: "Planned pair savings", value: num(summary["planned_unique_pair_savings"]) },
    {
      label: "Active cohorts",
      value: `${num(interval["first_cohort_index"])}–${
        interval["last_cohort_index"] == null ? "end" : num(interval["last_cohort_index"])
      }`,
      hint: `audit probability ${rate(policy["audit_probability"])}`,
    },
  ];

  return {
    present: true,
    mode: "active",
    enforced: true,
    policyKind: asString(policy["policy_kind"]),
    policyVersion: asString(policy["policy_version"]),
    suspended,
    suspensionReason: suspended
      ? asString(suspension?.["reason"]) ?? "audited-boundary-reversal safety rule tripped"
      : null,
    calibrationBins: [],
    kpis,
  };
}

export function deriveElimination(report: JsonValue | undefined): EliminationView {
  const root = asObject(report);
  if (!root) return EMPTY;
  const active = asObject(root["active_elimination"]);
  if (active) return deriveActive(active);
  const shadow = asObject(root["shadow_elimination"]);
  if (shadow) return deriveShadow(shadow);
  return EMPTY;
}
