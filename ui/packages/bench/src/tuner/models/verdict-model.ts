// verdict-model.ts — pure derivation behind `<ShipVerdict>` and the run
// overview's validation table. Answers the operator's one question: "which
// config should I ship, and can I trust that answer?" from three projection
// resources — the ranked validation rows, the candidate configs, and the
// verbatim `report.json` (for the claim, the missing production axes, the
// unresolved ties, and the limitations). No rendering, no `fetch`; every
// number the ship verdict shows is derived here and unit-tested.

import type {
  ProjectionCandidate,
  ProjectionValidation,
  ProjectionValidationRow,
} from "../tuner-types.js";
import type { JsonValue } from "../../types.js";

/** `candidate-<64 hex>` → a short, still-unambiguous stem for chips/labels. */
export function shortCandidateId(id: string): string {
  const bare = id.replace(/^candidate-/, "");
  return bare.length > 12 ? bare.slice(0, 12) : bare;
}

export interface VerdictCandidate {
  candidateId: string;
  shortId: string;
  fingerprint: string | null;
  source: string | null;
  config: JsonValue | null;
  rank: number;
  estimate: number;
  lower: number;
  upper: number;
  wins: number;
  draws: number;
  losses: number;
}

export interface ShipVerdict {
  /** Rank-1 validation row, enriched with its config. Null when the run has
   * no validation rows yet (still running, or failed before validation). */
  finalist: VerdictCandidate | null;
  runnerUp: VerdictCandidate | null;
  /** Every validation row, rank order, enriched. */
  ranked: VerdictCandidate[];
  /** Pairs the sample cannot separate — "cannot ship X over Y". */
  ties: { left: string; right: string; leftShort: string; rightShort: string }[];
  /** `report.validation_claim.claim`, verbatim (e.g. `mechanics_smoke`). */
  claim: string | null;
  /** Plain-language warnings: a non-production claim, each missing production
   * axis, then the report's own `limitations`. Shown inline so nobody ships
   * a smoke-test winner by accident. */
  caveats: string[];
  /** [min lower, max upper] across the ranked rows, widened a little; a safe
   * `[-1, 1]` when there are no rows. Feeds `<IntervalBar>` / `<Forest>`. */
  domain: [number, number];
}

export interface VerdictInput {
  validation: ProjectionValidation | undefined;
  candidates: ProjectionCandidate[] | undefined;
  report: JsonValue | undefined;
}

function asObject(v: JsonValue | undefined): Record<string, JsonValue> | null {
  return v !== null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, JsonValue>)
    : null;
}

function asStringArray(v: JsonValue | undefined): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

function humanize(token: string): string {
  return token.replace(/_/g, " ");
}

function tiesFrom(input: VerdictInput): ShipVerdict["ties"] {
  const raw =
    (input.validation && Array.isArray(input.validation.unresolved_ties)
      ? input.validation.unresolved_ties
      : null) ?? asObject(input.report)?.["unresolved_ties"];
  if (!Array.isArray(raw)) return [];
  const out: ShipVerdict["ties"] = [];
  for (const entry of raw) {
    const o = asObject(entry);
    const left = o?.["left_candidate_id"];
    const right = o?.["right_candidate_id"];
    if (typeof left === "string" && typeof right === "string") {
      out.push({
        left,
        right,
        leftShort: shortCandidateId(left),
        rightShort: shortCandidateId(right),
      });
    }
  }
  return out;
}

function caveatsFrom(report: Record<string, JsonValue> | null, claim: string | null): string[] {
  const out: string[] = [];
  if (claim && claim !== "production_validation") {
    out.push(
      claim === "mechanics_smoke"
        ? "Mechanics smoke — not a production validation; do not ship on this alone"
        : `Validation claim: ${humanize(claim)} — not a full production validation`,
    );
  }
  const vc = asObject(report?.["validation_claim"]);
  for (const axis of asStringArray(vc?.["missing_production_axes"])) {
    out.push(`Missing production axis: ${humanize(axis)}`);
  }
  for (const lim of asStringArray(report?.["limitations"])) {
    out.push(lim);
  }
  return out;
}

function enrich(
  row: ProjectionValidationRow,
  byId: Map<string, ProjectionCandidate>,
): VerdictCandidate {
  const c = byId.get(row.candidate_id);
  return {
    candidateId: row.candidate_id,
    shortId: shortCandidateId(row.candidate_id),
    fingerprint: c?.fingerprint ?? null,
    source: c?.source ?? null,
    config: c?.canonical_config ?? null,
    rank: row.rank,
    estimate: row.estimate,
    lower: row.lower,
    upper: row.upper,
    wins: row.wins,
    draws: row.draws,
    losses: row.losses,
  };
}

export function deriveVerdict(input: VerdictInput): ShipVerdict {
  const report = asObject(input.report);
  const claim =
    typeof asObject(report?.["validation_claim"])?.["claim"] === "string"
      ? (asObject(report?.["validation_claim"])!["claim"] as string)
      : null;

  const byId = new Map<string, ProjectionCandidate>();
  for (const c of input.candidates ?? []) byId.set(c.candidate_id, c);

  const rows = [...(input.validation?.rows ?? [])].sort((a, b) => a.rank - b.rank);
  const ranked = rows.map((r) => enrich(r, byId));

  let domain: [number, number] = [-1, 1];
  if (ranked.length > 0) {
    const lo = Math.min(...ranked.map((r) => r.lower));
    const hi = Math.max(...ranked.map((r) => r.upper));
    const pad = (hi - lo) * 0.05 || 0.05;
    domain = [lo - pad, hi + pad];
  }

  return {
    finalist: ranked[0] ?? null,
    runnerUp: ranked[1] ?? null,
    ranked,
    ties: tiesFrom(input),
    claim,
    caveats: caveatsFrom(report, claim),
    domain,
  };
}
