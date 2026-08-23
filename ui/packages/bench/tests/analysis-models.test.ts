import { describe, expect, it } from "vitest";
import {
  UNASSIGNED_BRACKET,
  analysisSampleMetadata,
  bracketFacets,
  decisionGroupRows,
  exactPlotRows,
  highlightSelectedTrial,
  opponentDistances,
  poolRevisionCoverage,
  pruningFunnelRows,
  reasonSymbol,
  resourceDomains,
  rungFunnelRows,
  stateSymbol,
  trialPageSummary,
  trialTrajectories,
} from "../src/tuning/analysis-models.js";
import type { TuningAnalysisOverview, TuningTrialDetailView, TuningTrialPage } from "../src/types.js";

const overview: TuningAnalysisOverview = {
  schema_version: 1,
  policy: null,
  objective: { metric: "score", direction: "maximize", complete_trials_only: true },
  cursor: { session_sequence: 4 },
  coverage: {
    trials: { total: 5, queued: 1, running: 1, terminal: 3, completed: 2, failed: 1, pruned: 0, cancelled: 0 },
    reports: 4,
    pairs: { total: 6, running: 1, complete: 4, failed: 1, unmatched_pool_revisions: 2 },
    points: { total: 8, returned: 5, sampled: true },
  },
  bracket_resources: [
    { bracket_id: "bracket-10", resource: 10, rung_resource: 10, reports: 2, trials: 2 },
    { bracket_id: "bracket-2", resource: 2, rung_resource: 2, reports: 1, trials: 1 },
    { bracket_id: null, resource: 3, rung_resource: null, reports: 1, trials: 1 },
  ],
  decision_groups: [
    { outcome: "prune", reason: "hyperband_prune", pruning_exempt: false, reports: 1 },
    { outcome: "complete", reason: "max_pairs", pruning_exempt: true, reports: 0 },
  ],
  points: [
    { trial_id: "trial-2", trial_number: 2, trial_status: "complete", resource: 10, rating: { mu: 30, sigma: 2 }, score: 24, outcome: "complete", reason: "max_pairs", pruning_exempt: false, bracket_id: "bracket-10", rung_resource: 10 },
    { trial_id: "trial-1", trial_number: 1, trial_status: "pruned", resource: 2, rating: { mu: 20, sigma: 3 }, score: 11, outcome: "continue", reason: "startup_exempt", pruning_exempt: true, bracket_id: "bracket-2", rung_resource: 2 },
    { trial_id: "trial-1", trial_number: 1, trial_status: "pruned", resource: 10, rating: { mu: 22, sigma: 3 }, score: 13, outcome: "prune", reason: "hyperband_prune", pruning_exempt: false, bracket_id: "bracket-10", rung_resource: 10 },
    { trial_id: "trial-tie", trial_number: 4, trial_status: "complete", resource: 10, rating: { mu: 30, sigma: 2 }, score: 24, outcome: "complete", reason: "max_pairs", pruning_exempt: false, bracket_id: "bracket-10", rung_resource: 10 },
    { trial_id: "trial-3", trial_number: 3, trial_status: "failed", resource: 3, rating: { mu: 10, sigma: 4 }, score: -2, outcome: "failed", reason: "worker_failed", pruning_exempt: false, bracket_id: null, rung_resource: null },
  ],
  best: { score: 24, trial_ids: ["trial-2", "trial-tie"] },
  pool_revisions: [
    { pool_snapshot_fingerprint: "rev-3", display_ordinal: 3, observed_at: "2026-08-23T12:03:00Z", pair_count: 2, anchors: [{ anchor_ordinal: 1, anchor_id: "anchor-3", config: { family: "ucb1" }, rating: { mu: 20, sigma: 2 }, provenance: "candidate", insertion_reason: "promotion", source_trial_id: "trial-2" }] },
    { pool_snapshot_fingerprint: "rev-1", display_ordinal: 1, observed_at: "2026-08-23T12:01:00Z", pair_count: 1, anchors: [] },
  ],
};

const page: TuningTrialPage = {
  schema_version: 1,
  total_count: 3,
  limit: 2,
  next_cursor: "next",
  cursor: { session_sequence: 4 },
  trials: [
    { trial_id: "trial-2", trial_number: 2, attempt_id: "attempt", state: "complete", reason: "max_pairs", rating: { mu: 30, sigma: 2 }, score: 24, family: "ucb1", config_summary: "", bracket_id: "bracket-10", resource: 10, pair_count: 2, wins: 3, losses: 1, draws: 2, elapsed_ms: 40, search_iterations_total: 100, search_move_time_ms: 12, has_detail: true },
    { trial_id: "trial-1", trial_number: 1, attempt_id: "attempt", state: "pruned", reason: "hyperband_prune", rating: { mu: 20, sigma: 3 }, score: 11, family: "rave", config_summary: "", bracket_id: "bracket-2", resource: 2, pair_count: 1, wins: 1, losses: 2, draws: 0, elapsed_ms: 20, search_iterations_total: 25, search_move_time_ms: 7, has_detail: true },
  ],
};

describe("tuning analysis models", () => {
  it("keeps unassigned and explicitly empty bracket facets while sorting resources numerically", () => {
    const facets = bracketFacets(overview, ["empty"]);
    expect(facets.map((facet) => [facet.key, facet.resources, facet.empty])).toEqual([
      [UNASSIGNED_BRACKET, [3], false],
      ["bracket-2", [2], false],
      ["bracket-10", [10], false],
      ["empty", [], true],
    ]);
    expect(resourceDomains(overview, ["empty"])).toMatchObject({ shared: [2, 3, 10], local: facets });
  });

  it("exposes sampled point metadata, exact scores, best ties, and selection without mutating snapshots", () => {
    const rows = exactPlotRows(overview, "trial-1");
    expect(analysisSampleMetadata(overview)).toEqual({ total: 8, returned: 5, sampled: true });
    expect(rows.map((row) => [row.trial_id, row.resource, row.score, row.best, row.selected])).toEqual([
      ["trial-1", 2, 11, false, true],
      ["trial-1", 10, 13, false, true],
      ["trial-2", 10, 24, true, false],
      ["trial-3", 3, -2, false, false],
      ["trial-tie", 10, 24, true, false],
    ]);
    expect(overview.points[0]).not.toHaveProperty("selected");
    expect(highlightSelectedTrial(page.trials, "trial-2").map((row) => row.selected)).toEqual([true, false]);
  });

  it("models every decision group and known or unknown state/reason symbols", () => {
    expect(decisionGroupRows(overview).map((row) => [row.outcome, row.reason, row.reports, row.outcomeSymbol.symbol, row.reasonSymbol.symbol])).toEqual([
      ["complete", "max_pairs", 0, "✓", "✓"],
      ["prune", "hyperband_prune", 1, "↯", "↯"],
    ]);
    expect(stateSymbol("mystery")).toEqual({ key: "mystery", symbol: "?", label: "mystery" });
    expect(reasonSymbol(null)).toEqual({ key: "unreported", symbol: "−", label: "Not reported" });
  });

  it("sorts funnel rungs and terminates each trajectory at its final reported point", () => {
    expect(rungFunnelRows(overview, "bracket-10").map((row) => [row.bracket, row.resource, row.selected])).toEqual([
      ["Unassigned", 3, false],
      ["bracket-2", 2, false],
      ["bracket-10", 10, true],
    ]);
    const trajectories = trialTrajectories(overview, "trial-1");
    expect(trajectories.map((path) => [path.trialId, path.reportCount, path.points.map((point) => point.terminal)])).toEqual([
      ["trial-1", 2, [false, true]],
      ["trial-2", 1, [true]],
      ["trial-3", 1, [true]],
      ["trial-tie", 1, [true]],
    ]);
  });

  it("keeps every typed pruning reason in one disjoint exact funnel", () => {
    const allReasons = {
      ...overview,
      coverage: { ...overview.coverage, reports: 28 },
      decision_groups: [
        { outcome: "continue", reason: "below_min_pairs", pruning_exempt: false, reports: 1 },
        { outcome: "continue", reason: "startup_exempt", pruning_exempt: true, reports: 2 },
        { outcome: "continue", reason: "pruning_disabled", pruning_exempt: false, reports: 3 },
        { outcome: "continue", reason: "hyperband_keep", pruning_exempt: false, reports: 4 },
        { outcome: "prune", reason: "hyperband_prune", pruning_exempt: false, reports: 5 },
        { outcome: "complete", reason: "confidence", pruning_exempt: false, reports: 6 },
        { outcome: "complete", reason: "max_pairs", pruning_exempt: false, reports: 7 },
      ],
    };
    const funnel = pruningFunnelRows(allReasons);
    expect(funnel.map((row) => [row.key, row.reason, row.reports])).toEqual([
      ["below_minimum", "below_min_pairs", 1], ["startup_exempt", "startup_exempt", 2],
      ["pruning_disabled", "pruning_disabled", 3], ["continued", "hyperband_keep", 4],
      ["pruned", "hyperband_prune", 5], ["confidence_completed", "confidence", 6],
      ["max_completed", "max_pairs", 7],
    ]);
    expect(funnel.reduce((total, row) => total + row.reports, 0)).toBe(allReasons.coverage.reports);
    expect(funnel.every((row) => row.description.endsWith("."))).toBe(true);
  });

  it("summarizes only returned W/L/D and compute rows while disclosing page coverage", () => {
    expect(trialPageSummary(page, "trial-1")).toMatchObject({
      wld: { wins: 4, losses: 3, draws: 2, total: 9 },
      compute: { elapsedMs: 60, searchIterationsTotal: 125, searchMoveTimeMs: 19 },
      totalCount: 3,
      returnedCount: 2,
      sampled: true,
    });
  });

  it("retains pool revision gaps and computes candidate-opponent distance from recorded pair ratings", () => {
    expect(poolRevisionCoverage(overview)).toMatchObject({
      revisionCount: 2,
      pairCount: 3,
      anchorCount: 1,
      unmatchedPoolRevisions: 2,
      revisions: [{ display_ordinal: 1, gapBefore: [] }, { display_ordinal: 3, gapBefore: [2] }],
    });
    const trial: Pick<TuningTrialDetailView, "pairs"> = {
      pairs: [{ pair_id: "pair-2", pair_index: 2, state: "complete", seed: 1, round: 1, opponent: { anchor_id: "anchor", config: {}, mu: 17, sigma: 1, label: null, provenance: null }, pool_snapshot_fingerprint: "rev", pool_revision: null, rating_before: { mu: 20, sigma: 3 }, rating_after: null, score: 1, failure: null, games: [] }],
    };
    expect(opponentDistances(trial)).toEqual([{
      pairId: "pair-2", pairIndex: 2, opponentId: "anchor", candidateMu: 20, opponentMu: 17,
      deltaMu: 3, absoluteMuDistance: 3, candidateSigma: 3, opponentSigma: 1, deltaSigma: 2,
    }]);
  });
});
