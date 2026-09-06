// diagnostic-model.ts — pure derivation behind the Run Science "Diagnostic
// matchup graph" section. Reshapes `report.diagnostic_matchup_graph` into
// nodes (objective-ranked candidates), directed material edges, material
// cycle components, and the shortlist effect. No rendering, no `fetch`.

import type { JsonValue } from "../../types.js";
import { asArray, asNumber, asObject, asString, asStringArray } from "./json-util.js";
import { shortCandidateId } from "./verdict-model.js";
import type { KpiItem } from "../primitives/KpiRow.js";

export interface DiagnosticNode {
  candidateId: string;
  shortId: string;
  rank: number;
  /** In a material cycle component. */
  inCycle: boolean;
}

export interface DiagnosticEdge {
  from: string;
  to: string;
  estimate: number | null;
  lower: number | null;
  upper: number | null;
  pairCount: number;
  /** True when the winning direction is undetermined (edge shown undirected). */
  undirected: boolean;
}

export interface DiagnosticCycle {
  members: string[];
  witness: string[];
}

export interface DiagnosticGraphView {
  present: boolean;
  /** A diagnostic budget was allocated or edges were measured. */
  hasBudget: boolean;
  nodes: DiagnosticNode[];
  edges: DiagnosticEdge[];
  cycles: DiagnosticCycle[];
  shortlist: {
    objectiveIds: string[];
    finalistIds: string[];
    reserveId: string | null;
    displacedId: string | null;
    reserveDisplaced: boolean;
  };
  kpis: KpiItem[];
}

const EMPTY: DiagnosticGraphView = {
  present: false,
  hasBudget: false,
  nodes: [],
  edges: [],
  cycles: [],
  shortlist: {
    objectiveIds: [],
    finalistIds: [],
    reserveId: null,
    displacedId: null,
    reserveDisplaced: false,
  },
  kpis: [],
};

export function deriveDiagnosticGraph(report: JsonValue | undefined): DiagnosticGraphView {
  const g = asObject(asObject(report)?.["diagnostic_matchup_graph"]);
  if (!g) return EMPTY;

  const scope = asObject(g["scope"]) ?? {};
  const cycles: DiagnosticCycle[] = asArray(g["material_cycle_components"]).flatMap((cRaw) => {
    const c = asObject(cRaw);
    if (!c) return [];
    return [
      {
        members: asStringArray(c["candidate_ids"]).map(shortCandidateId),
        witness: asStringArray(c["witness_cycle_candidate_ids"]).map(shortCandidateId),
      },
    ];
  });
  const cycleMembers = new Set(
    asArray(g["material_cycle_components"]).flatMap((cRaw) =>
      asStringArray(asObject(cRaw)?.["candidate_ids"]),
    ),
  );

  const nodes: DiagnosticNode[] = asArray(g["nodes"])
    .flatMap((nRaw) => {
      const n = asObject(nRaw);
      const candidateId = asString(n?.["candidate_id"]);
      if (!candidateId) return [];
      return [
        {
          candidateId,
          shortId: shortCandidateId(candidateId),
          rank: asNumber(n?.["objective_rank"]) ?? 0,
          inCycle: cycleMembers.has(candidateId),
        },
      ];
    })
    .sort((a, b) => a.rank - b.rank);

  const edges: DiagnosticEdge[] = asArray(g["edges"]).flatMap((eRaw) => {
    const e = asObject(eRaw);
    const left = asString(e?.["left_candidate_id"]);
    const right = asString(e?.["right_candidate_id"]);
    if (!left || !right) return [];
    const direction = asString(e?.["material_direction"]);
    const iv = asObject(e?.["interval"]);
    // `material_direction` names the dominant side; fall back to an
    // undirected edge when the diagnostic could not resolve one.
    const from = direction === "right" ? right : left;
    const to = direction === "right" ? left : right;
    return [
      {
        from,
        to,
        estimate: asNumber(e?.["estimate"]),
        lower: asNumber(iv?.["lower"]),
        upper: asNumber(iv?.["upper"]),
        pairCount: asNumber(e?.["pair_count"]) ?? 0,
        undirected: direction == null || direction === "none",
      },
    ];
  });

  const shortlistRaw = asObject(g["shortlist_effect"]) ?? {};
  const displacedId = asString(shortlistRaw["displaced_candidate_id"]);
  const shortlist = {
    objectiveIds: asStringArray(shortlistRaw["objective_candidate_ids"]),
    finalistIds: asStringArray(shortlistRaw["finalist_ids"]),
    reserveId: asString(shortlistRaw["reserve_candidate_id"]),
    displacedId,
    reserveDisplaced: displacedId != null,
  };

  const budget = asNumber(scope["pair_attempt_budget"]) ?? 0;
  const allocations = asNumber(asObject(g["allocations"])?.["count"]) ?? 0;
  const effort = asObject(scope["search_effort"]);
  const kpis: KpiItem[] = [
    { label: "Candidates", value: String(nodes.length) },
    { label: "Direct edges", value: String(edges.length) },
    { label: "Cycle components", value: String(cycles.length) },
    { label: "Pair-attempt budget", value: String(budget), hint: `${allocations} allocated` },
    {
      label: "Search effort",
      value: effort ? `${asNumber(effort["value"]) ?? "—"} ${asString(effort["kind"]) ?? ""}`.trim() : "—",
    },
  ];

  return {
    present: nodes.length > 0,
    hasBudget: budget > 0 || edges.length > 0 || allocations > 0,
    nodes,
    edges,
    cycles,
    shortlist,
    kpis,
  };
}
