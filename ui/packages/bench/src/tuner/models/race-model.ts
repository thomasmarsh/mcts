// race-model.ts — pure derivation behind the Run Science "Cohort race"
// section. Turns `report.shadow_elimination.cohorts` into one grid per
// cohort: rows are candidates, columns are the common prefixes they were
// looked at, and each cell is the shadow disposition recorded at that
// prefix. Candidate sources come from the projection candidate list.

import type { JsonValue } from "../../types.js";
import type { ProjectionCandidate } from "../tuner-types.js";
import { asArray, asObject, asString, shortId } from "./json-util.js";
import { shortCandidateId } from "./verdict-model.js";

export interface RacePrefix {
  prefixId: string;
  shortId: string;
  /** 1-based column position. */
  index: number;
}

export interface RaceRow {
  candidateId: string;
  shortId: string;
  source: string | null;
  protected: boolean;
  finalTopSet: boolean;
  firstEliminationPrefixId: string | null;
  /** One entry per prefix column; `null` where the candidate had no look. */
  cells: (string | null)[];
}

export interface CohortRace {
  cohortIndex: number;
  prefixes: RacePrefix[];
  rows: RaceRow[];
}

export interface RaceGraph {
  cohorts: CohortRace[];
  /** All dispositions that actually appear, for a legend. */
  dispositions: string[];
  enforced: boolean;
  policyKind: string | null;
  present: boolean;
}

export function deriveCohortRace(
  report: JsonValue | undefined,
  candidates: ProjectionCandidate[] | undefined,
): RaceGraph {
  const shadow = asObject(asObject(report)?.["shadow_elimination"]);
  const cohortsRaw = asArray(shadow?.["cohorts"]);
  const sourceById = new Map<string, string>();
  for (const c of candidates ?? []) sourceById.set(c.candidate_id, c.source);

  const dispositions = new Set<string>();
  const cohorts: CohortRace[] = cohortsRaw.map((cRaw, ordinal) => {
    const c = asObject(cRaw);
    const cohortIndex = numberOr(c?.["cohort_index"], ordinal);
    const paths = asArray(c?.["paths"] ?? c?.["candidate_paths"]);

    const prefixOrder: string[] = [];
    for (const pRaw of paths) {
      for (const lRaw of asArray(asObject(pRaw)?.["looks"])) {
        const pid = asString(asObject(lRaw)?.["prefix_id"]);
        if (pid && !prefixOrder.includes(pid)) prefixOrder.push(pid);
      }
    }
    const prefixes: RacePrefix[] = prefixOrder.map((prefixId, i) => ({
      prefixId,
      shortId: shortId(prefixId),
      index: i + 1,
    }));

    const rows: RaceRow[] = paths.map((pRaw) => {
      const p = asObject(pRaw);
      const candidateId = asString(p?.["candidate_id"]) ?? "";
      const byPrefix = new Map<string, string>();
      for (const lRaw of asArray(p?.["looks"])) {
        const l = asObject(lRaw);
        const pid = asString(l?.["prefix_id"]);
        const disp = asString(l?.["disposition"]);
        if (pid && disp) {
          byPrefix.set(pid, disp);
          dispositions.add(disp);
        }
      }
      return {
        candidateId,
        shortId: shortCandidateId(candidateId),
        source: sourceById.get(candidateId) ?? null,
        protected: p?.["protected"] === true,
        finalTopSet: p?.["final_top_set"] === true,
        firstEliminationPrefixId: asString(p?.["first_elimination_prefix_id"]),
        cells: prefixOrder.map((pid) => byPrefix.get(pid) ?? null),
      };
    });

    return { cohortIndex, prefixes, rows };
  });

  const policy = asObject(shadow?.["policy"]);
  return {
    cohorts,
    dispositions: [...dispositions].sort(),
    enforced: policy?.["enforced"] === true,
    policyKind: asString(policy?.["kind"]),
    present: cohorts.some((c) => c.rows.length > 0),
  };
}

function numberOr(v: JsonValue | undefined, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}
