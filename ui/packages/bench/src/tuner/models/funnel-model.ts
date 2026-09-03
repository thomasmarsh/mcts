// funnel-model.ts — pure derivation behind the Run Science "Proposal
// search" section. Maps `report.proposal_search` to a per-source funnel
// (configured → attempted → accepted, with rejections) plus a handful of
// headline numbers. No rendering, no `fetch`.

import type { JsonValue } from "../../types.js";
import type { ProjectionProposal } from "../tuner-types.js";
import { asArray, asNumber, asObject, asString, asStringArray, shortId } from "./json-util.js";

/** Proposal sources in the order the funnel lists them. */
const SOURCE_ORDER = [
  "schema_default",
  "bootstrap_random",
  "smac_model",
  "irace_model",
  "qmc_search",
  "random_search",
  "random_reserve",
] as const;

const SOURCE_LABEL: Record<string, string> = {
  schema_default: "Schema default",
  bootstrap_random: "Bootstrap random",
  smac_model: "SMAC model",
  irace_model: "irace model",
  qmc_search: "QMC search",
  random_search: "Random search",
  random_reserve: "Random reserve",
};

export interface ProposalStage {
  source: string;
  label: string;
  /** Configured budget for this source, when the schema pins one. */
  configured: number | null;
  attempted: number;
  accepted: number;
  rejected: number;
}

export interface ProposalFunnel {
  stages: ProposalStage[];
  kpis: { label: string; value: string }[];
  /** True when `proposal_search` was present and had at least one attempt. */
  present: boolean;
}

function configuredFor(source: string, configured: Record<string, JsonValue> | null): number | null {
  if (source === "schema_default") return 1;
  if (!configured) return null;
  const key =
    source === "bootstrap_random"
      ? "bootstrap"
      : source === "smac_model"
        ? "model"
        : source === "random_reserve"
          ? "random_reserve"
          : null;
  return key ? asNumber(configured[key]) : null;
}

export function deriveProposalFunnel(
  report: JsonValue | undefined,
  proposals?: ProjectionProposal[],
): ProposalFunnel {
  const ps = asObject(asObject(report)?.["proposal_search"]);
  if (!ps) return fromRows(proposals ?? []);

  const configured = asObject(ps["configured"]);
  const attempts = asObject(ps["actual_source_attempts"]) ?? {};
  const rejections = asObject(ps["rejections_by_source"]) ?? {};

  const acceptedBySource = new Map<string, number>();
  for (const entry of asArray(ps["accepted"])) {
    const src = asString(asObject(entry)?.["source"]);
    if (src) acceptedBySource.set(src, (acceptedBySource.get(src) ?? 0) + 1);
  }

  const stages: ProposalStage[] = SOURCE_ORDER.map((source) => ({
    source,
    label: SOURCE_LABEL[source] ?? source,
    configured: configuredFor(source, configured),
    attempted: asNumber(attempts[source]) ?? 0,
    accepted: acceptedBySource.get(source) ?? 0,
    rejected: asNumber(rejections[source]) ?? 0,
  })).filter((s) => s.attempted > 0 || (s.configured ?? 0) > 0 || s.accepted > 0);

  const excluded = asStringArray(configured?.["excluded_families"]);
  const frontier = asString(ps["final_frontier_id"]);
  const kpis: { label: string; value: string }[] = [
    { label: "Model", value: asString(ps["model_version"]) ?? "—" },
    { label: "Cohorts", value: String(asNumber(configured?.["cohorts"]) ?? "—") },
    { label: "Retained elites", value: String(asNumber(configured?.["retained_elites"]) ?? "—") },
    { label: "Final observations", value: String(asNumber(ps["final_observation_count"]) ?? "—") },
    { label: "Final frontier", value: frontier ? shortId(frontier) : "—" },
    { label: "Excluded families", value: excluded.length ? excluded.join(", ") : "none" },
  ];

  const totalAttempts = stages.reduce((a, s) => a + s.attempted, 0);
  return { stages, kpis, present: totalAttempts > 0 };
}

/** Live funnel from the projection `proposals` rows (no `report.json` yet).
 * Each row is one proposal attempt; `disposition` is `accepted` / `rejected`
 * / `null` (still pending). Configured budgets are unknown without the
 * report, so those stay `null`. */
function fromRows(proposals: ProjectionProposal[]): ProposalFunnel {
  const bySource = new Map<string, { attempted: number; accepted: number; rejected: number }>();
  for (const p of proposals) {
    const bucket = bySource.get(p.source) ?? { attempted: 0, accepted: 0, rejected: 0 };
    bucket.attempted += 1;
    if (p.disposition === "accepted") bucket.accepted += 1;
    else if (p.disposition === "rejected") bucket.rejected += 1;
    bySource.set(p.source, bucket);
  }
  const ordered = [...SOURCE_ORDER, ...[...bySource.keys()].filter((s) => !SOURCE_ORDER.includes(s as never))];
  const stages: ProposalStage[] = ordered
    .filter((s) => bySource.has(s))
    .map((source) => {
      const b = bySource.get(source)!;
      return {
        source,
        label: SOURCE_LABEL[source] ?? source,
        configured: source === "schema_default" ? 1 : null,
        attempted: b.attempted,
        accepted: b.accepted,
        rejected: b.rejected,
      };
    });
  const pending = proposals.filter((p) => p.disposition == null).length;
  const kpis: { label: string; value: string }[] = [
    { label: "Proposals", value: String(proposals.length) },
    { label: "Pending", value: String(pending) },
  ];
  return { stages, kpis, present: stages.length > 0 };
}
