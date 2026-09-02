import { describe, expect, it } from "vitest";
import { deriveProgress, formatWall } from "../../src/tuner/models/progress-model.js";
import type { ProjectionComputePhase } from "../../src/tuner/tuner-types.js";

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

describe("deriveProgress", () => {
  it("uses wall clock while live and the ledger once exited", () => {
    const live = deriveProgress({
      status: "live",
      startedAt: "2026-01-01T00:00:00Z",
      nowMs: Date.parse("2026-01-01T00:05:00Z"),
      compute: [],
    });
    expect(live.live).toBe(true);
    expect(live.wallMs).toBe(300_000);
    expect(live.phase).toBe("starting");

    const done = deriveProgress({
      status: "exited",
      startedAt: "2026-01-01T00:00:00Z",
      nowMs: Date.parse("2026-01-01T09:00:00Z"),
      compute: [phase({ wall_time_ms: 1000 }), phase({ phase: "validation", wall_time_ms: 500 })],
    });
    expect(done.wallMs).toBe(1500);
    expect(done.phase).toBe("validation");
  });

  it("sums pair counts across phases and computes the completed fraction", () => {
    const p = deriveProgress({
      status: "live",
      startedAt: null,
      nowMs: 0,
      compute: [
        phase({ pair_attempts: 10, completed_pairs: 6, failed_attempts: 1 }),
        phase({ phase: "validation", pair_attempts: 10, completed_pairs: 4, censored_attempts: 2 }),
      ],
    });
    expect(p.pairs).toEqual({ completed: 10, attempted: 20, failed: 1, censored: 2 });
    expect(p.fraction).toBe(0.5);
  });
});

describe("formatWall", () => {
  it("scales the unit to the magnitude", () => {
    expect(formatWall(4_000)).toBe("4s");
    expect(formatWall(125_000)).toBe("2m 5s");
    expect(formatWall(7_400_000)).toBe("2h 3m");
  });
});
