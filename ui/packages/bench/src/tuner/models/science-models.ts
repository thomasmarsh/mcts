// science-models.ts — pure derivations for the Run Science charts that read
// straight off `report.json`: the convergence step line and the per-cohort
// observation forests. All chart maths lives here; the primitives stay
// layout-only.

import type { JsonValue } from "../../types.js";
import { asArray, asNumber, asObject, asString } from "./json-util.js";
import { shortCandidateId } from "./verdict-model.js";

// --- Convergence -------------------------------------------------------

export interface ConvergenceStep {
  cohortIndex: number;
  /** 1-based x position (one step per completed cohort). */
  x: number;
  label: string;
  /** The leading candidate's largest margin over its elimination boundary
   * recorded in this cohort — the "best so far" signal. */
  bestMargin: number;
  leaderCandidateId: string | null;
  leaderShortId: string | null;
}

export interface Convergence {
  steps: ConvergenceStep[];
  domain: [number, number];
  present: boolean;
}

export function deriveConvergence(report: JsonValue | undefined): Convergence {
  const cohorts = asArray(asObject(asObject(report)?.["shadow_elimination"])?.["cohorts"]);
  const steps: ConvergenceStep[] = [];

  cohorts.forEach((cRaw, ordinal) => {
    const c = asObject(cRaw);
    const cohortIndex = asNumber(c?.["cohort_index"]) ?? ordinal;
    let bestMargin = -Infinity;
    let leaderCandidateId: string | null = null;
    for (const pRaw of asArray(c?.["paths"] ?? c?.["candidate_paths"])) {
      const p = asObject(pRaw);
      const cid = asString(p?.["candidate_id"]);
      for (const lRaw of asArray(p?.["looks"])) {
        const m = asNumber(asObject(lRaw)?.["maximum_mean_difference"]);
        if (m != null && m > bestMargin) {
          bestMargin = m;
          leaderCandidateId = cid;
        }
      }
    }
    if (bestMargin === -Infinity) bestMargin = 0;
    steps.push({
      cohortIndex,
      x: ordinal + 1,
      label: `Cohort ${cohortIndex}`,
      bestMargin,
      leaderCandidateId,
      leaderShortId: leaderCandidateId ? shortCandidateId(leaderCandidateId) : null,
    });
  });

  const maxY = steps.reduce((a, s) => Math.max(a, s.bestMargin), 0);
  return {
    steps,
    domain: [0, maxY > 0 ? maxY * 1.1 : 1],
    present: steps.length > 0,
  };
}

// --- Observations -----------------------------------------------------

export interface ObservationRow {
  candidateId: string;
  shortId: string;
  /** Mean of this candidate's per-opponent means. */
  mean: number;
  /** Conservative envelope: the widest per-opponent interval. */
  lower: number;
  upper: number;
  opponents: number;
}

export interface Observations {
  cohortIndex: number | null;
  rows: ObservationRow[];
  domain: [number, number];
  present: boolean;
}

/** Per-candidate performance at the maximum tuning prefix, summarised from
 * `opponent_response_analysis` — the one report section that carries a
 * mean and an interval per (candidate, opponent). The row interval is the
 * envelope across opponents, not a re-estimated CI. */
export function deriveObservations(report: JsonValue | undefined): Observations {
  const ora = asObject(asObject(report)?.["opponent_response_analysis"]);
  const cands = asArray(ora?.["candidates"]);
  const cohortIndex = asNumber(asObject(ora?.["scope"])?.["cohort_index"]);

  const rows: ObservationRow[] = [];
  for (const cRaw of cands) {
    const c = asObject(cRaw);
    const candidateId = asString(c?.["candidate_id"]);
    if (!candidateId) continue;
    const responses = asArray(c?.["opponent_responses"]);
    const means: number[] = [];
    let lower = Infinity;
    let upper = -Infinity;
    for (const rRaw of responses) {
      const r = asObject(rRaw);
      const mean = asNumber(r?.["mean"]);
      const iv = asObject(r?.["interval"]);
      const lo = asNumber(iv?.["lower"]);
      const hi = asNumber(iv?.["upper"]);
      if (mean != null) means.push(mean);
      if (lo != null) lower = Math.min(lower, lo);
      if (hi != null) upper = Math.max(upper, hi);
    }
    if (means.length === 0) continue;
    const mean = means.reduce((a, b) => a + b, 0) / means.length;
    rows.push({
      candidateId,
      shortId: shortCandidateId(candidateId),
      mean,
      lower: Number.isFinite(lower) ? lower : mean,
      upper: Number.isFinite(upper) ? upper : mean,
      opponents: means.length,
    });
  }
  rows.sort((a, b) => b.mean - a.mean);

  let domain: [number, number] = [0, 1];
  if (rows.length > 0) {
    const lo = Math.min(...rows.map((r) => r.lower));
    const hi = Math.max(...rows.map((r) => r.upper));
    const pad = (hi - lo) * 0.05 || 0.05;
    domain = [lo - pad, hi + pad];
  }

  return { cohortIndex, rows, domain, present: rows.length > 0 };
}
