import { describe, expect, it } from "vitest";

import {
  budgetPairsFromManifest,
  deriveContenderRecords,
  deriveVitals,
  formatRecord,
  summarizeVitals,
} from "../../src/tuner/models/telemetry-model.js";
import type {
  ProjectionComputePhase,
  ProjectionManifestSummary,
  ProjectionPairRow,
  ProjectionTelemetry,
} from "../../src/tuner/tuner-types.js";

const telemetry = (over: Partial<ProjectionTelemetry> = {}): ProjectionTelemetry => ({
  run_id: "r",
  sessions: 1,
  first_start_us: 1_000_000_000,
  last_end_us: 3_000_000_000,
  wall_span_us: 2_000_000_000,
  loop_active_us: 200_000_000,
  wait_us: 800_000_000,
  lanes: [],
  ...over,
});

const phase = (over: Partial<ProjectionComputePhase>): ProjectionComputePhase => ({
  phase: "tuning",
  pair_attempts: 0,
  completed_pairs: 0,
  failed_attempts: 0,
  censored_attempts: 0,
  physical_games: 0,
  search_iterations: 0,
  wall_time_ms: 0,
  ...over,
});

describe("deriveVitals", () => {
  it("derives parallelism, overhead, rate and ETA from the rollup", () => {
    const v = deriveVitals({
      telemetry: telemetry(),
      compute: [phase({ completed_pairs: 40, wall_time_ms: 1_600_000 })],
      budgetPairs: 100,
      live: false,
      nowMs: 0,
    });
    // game compute 1_600_000 ms over 800_000 ms of waiting.
    expect(v.effectiveParallelism).toBe(2);
    // loop 200_000 ms of a 1_000_000 ms critical path.
    expect(v.loopOverheadFraction).toBe(0.2);
    // elapsed wall = sidecar extent 2_000_000 ms / 40 pairs.
    expect(v.secPerPair).toBe(50);
    // 60 pairs left * 50 s.
    expect(v.etaMs).toBe(3_000_000);
    expect(v.completedPairs).toBe(40);
  });

  it("uses now - startedAt for a live run's elapsed wall", () => {
    const v = deriveVitals({
      telemetry: telemetry(),
      compute: [phase({ completed_pairs: 10 })],
      live: true,
      startedAt: new Date(1_000).toISOString(),
      nowMs: 101_000,
    });
    expect(v.elapsedWallMs).toBe(100_000);
  });

  it("leaves rate and ETA null before any pair completes or without a budget", () => {
    const noPairs = deriveVitals({
      telemetry: telemetry(),
      compute: [phase({})],
      budgetPairs: 100,
      live: false,
      nowMs: 0,
    });
    expect(noPairs.secPerPair).toBeNull();
    expect(noPairs.etaMs).toBeNull();

    const noBudget = deriveVitals({
      telemetry: telemetry(),
      compute: [phase({ completed_pairs: 5 })],
      live: false,
      nowMs: 0,
    });
    expect(noBudget.etaMs).toBeNull();
  });

  it("leaves parallelism null with no wait span yet", () => {
    const v = deriveVitals({
      telemetry: telemetry({ wait_us: 0, loop_active_us: 0 }),
      compute: [phase({})],
      live: false,
      nowMs: 0,
    });
    expect(v.effectiveParallelism).toBeNull();
    expect(v.loopOverheadFraction).toBeNull();
  });

  it("clamps the ETA at zero once the budget is met", () => {
    const v = deriveVitals({
      telemetry: telemetry(),
      compute: [phase({ completed_pairs: 120, wall_time_ms: 100 })],
      budgetPairs: 100,
      live: false,
      nowMs: 0,
    });
    expect(v.etaMs).toBe(0);
  });
});

describe("budgetPairsFromManifest", () => {
  const manifest = (over: Partial<ProjectionManifestSummary> = {}): ProjectionManifestSummary => ({
    manifest_run_id: "r",
    manifest_fingerprint: "f",
    game_kind: "druid",
    objective_id: "o",
    cohort_size: 4,
    finalists: 2,
    seed: 1,
    task_seed: 2,
    shadow_policy_kind: "none",
    active_elimination: false,
    tuning_pair_budget: 84,
    validation_pair_budget: 4,
    diagnostic_pair_budget: 0,
    ...over,
  });

  it("sums the three budget legs", () => {
    expect(budgetPairsFromManifest(manifest())).toBe(88);
  });

  it("is null for a legacy manifest missing a leg, or no manifest", () => {
    expect(budgetPairsFromManifest(manifest({ diagnostic_pair_budget: null }))).toBeNull();
    expect(budgetPairsFromManifest(null)).toBeNull();
  });
});

describe("summarizeVitals", () => {
  it("formats the KPI tiles, budget fraction and ETA", () => {
    const view = summarizeVitals({
      telemetry: telemetry(),
      compute: [phase({ completed_pairs: 40, wall_time_ms: 1_600_000 })],
      budgetPairs: 100,
      live: false,
      nowMs: 0,
    });
    expect(view.progressFraction).toBe(0.4);
    expect(view.etaLabel).toBe("50m 0s");
    const byLabel = new Map(view.kpis.map((k) => [k.label, k.value]));
    expect(byLabel.get("effective parallelism")).toBe("2.00×");
    expect(byLabel.get("loop overhead")).toBe("20%");
    expect(byLabel.get("s / pair")).toBe("50.0s");
    expect(byLabel.get("pairs done")).toBe("40 / 100");
  });

  it("drops the budget bar and ETA when the budget is unknown, and shows a sessions tile on resume", () => {
    const view = summarizeVitals({
      telemetry: telemetry({ sessions: 2 }),
      compute: [phase({ completed_pairs: 5 })],
      live: false,
      nowMs: 0,
    });
    expect(view.progressFraction).toBeNull();
    expect(view.etaLabel).toBeNull();
    expect(view.kpis.some((k) => k.label === "sessions" && k.value === "2")).toBe(true);
  });
});

describe("deriveContenderRecords", () => {
  const pair = (over: Partial<ProjectionPairRow>): ProjectionPairRow => ({
    pair_id: "p",
    phase: "tuning",
    candidate_id: "c1",
    task_id: "t",
    opponent_id: "o1",
    pair_utility: 0,
    ...over,
  });

  it("buckets a candidate's pairs into W/L/D per opponent", () => {
    const rows = [
      pair({ opponent_id: "o1", pair_utility: 0.4 }),
      pair({ opponent_id: "o1", pair_utility: -0.2 }),
      pair({ opponent_id: "o2", pair_utility: 0 }),
      pair({ opponent_id: "o2", pair_utility: 0.1 }),
      pair({ candidate_id: "other", opponent_id: "o1", pair_utility: 0.9 }),
    ];
    const rec = deriveContenderRecords("c1", rows);
    expect(rec.byOpponent.map((r) => r.opponentId)).toEqual(["o1", "o2"]);
    expect(rec.byOpponent[0]).toEqual({ opponentId: "o1", wins: 1, losses: 1, draws: 0 });
    expect(rec.byOpponent[1]).toEqual({ opponentId: "o2", wins: 1, losses: 0, draws: 1 });
    expect(rec.total).toEqual({ opponentId: "", wins: 2, losses: 1, draws: 1 });
  });

  it("formats a record as W–L–D", () => {
    expect(formatRecord({ opponentId: "o", wins: 3, losses: 1, draws: 2 })).toBe("3–1–2");
  });
});
