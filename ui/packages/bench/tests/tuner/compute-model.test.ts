import { describe, expect, it } from "vitest";
import { deriveComputeLedger } from "../../src/tuner/models/compute-model.js";

describe("deriveComputeLedger", () => {
  it("is absent without a compute section", () => {
    expect(deriveComputeLedger({}).present).toBe(false);
  });

  it("flattens per-phase rows, a treemap, and totals", () => {
    const view = deriveComputeLedger({
      compute: {
        policy_version: "safe-boundary-pair-attempts-v1",
        budget: { tuning_pair_attempts: 84, validation_pair_attempts: 4, diagnostic_pair_attempts: 0 },
        tuning: {
          pair_attempts: 82,
          completed_pairs: 80,
          failed_attempts: 1,
          censored_attempts: 1,
          overrun_pair_attempts: 0,
          unspent_pair_attempts: 2,
          physical_games: 164,
          search_iterations: 328,
          wall_time_ms: 164,
        },
        validation: {
          pair_attempts: 4,
          completed_pairs: 4,
          failed_attempts: 0,
          censored_attempts: 0,
          overrun_pair_attempts: 0,
          unspent_pair_attempts: 0,
          physical_games: 8,
          search_iterations: 16,
          wall_time_ms: 8,
        },
        diagnostic: {
          pair_attempts: 0,
          completed_pairs: 0,
          failed_attempts: 0,
          censored_attempts: 0,
          overrun_pair_attempts: 0,
          unspent_pair_attempts: 0,
          physical_games: 0,
          search_iterations: 0,
          wall_time_ms: 0,
        },
      },
    });
    expect(view.present).toBe(true);
    expect(view.phases.map((p) => p.phase)).toEqual(["tuning", "validation", "diagnostic"]);
    // diagnostic had no budget/attempts → dropped from the treemap
    expect(view.treemap.map((g) => g.key)).toEqual(["tuning", "validation"]);
    const tuning = view.treemap[0]!;
    expect(tuning.children.map((c) => c.label)).toEqual(["completed", "failed", "censored", "unspent"]);
    expect(view.kpis.find((k) => k.label === "Physical games")?.value).toBe("172");
    expect(view.kpis.find((k) => k.label === "Pair attempts")?.value).toBe("86");
  });
});
