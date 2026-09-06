import { describe, expect, it } from "vitest";
import { derivePairInspector } from "../../src/tuner/models/pair-model.js";
import type { ProjectionGameRow, ProjectionPairRow } from "../../src/tuner/tuner-types.js";

const pair: ProjectionPairRow = {
  pair_id: "pair-abc123def456ff",
  phase: "tuning",
  candidate_id: "candidate-aaaa1111bbbb",
  task_id: "task-deadbeef0000",
  opponent_id: "schema-default",
  pair_utility: 0.25,
};

const game = (over: Partial<ProjectionGameRow>): ProjectionGameRow => ({
  game_id: "game-1111",
  pair_id: pair.pair_id,
  candidate_side: "first",
  outcome: "draw",
  plies: 10,
  elapsed_ms: 500,
  candidate_iterations_total: 1000,
  opponent_iterations_total: 900,
  ...over,
});

describe("derivePairInspector", () => {
  it("classifies candidate-relative outcomes and totals both games", () => {
    const view = derivePairInspector(pair, [
      game({ game_id: "game-a", candidate_side: "first", outcome: "candidate_win", plies: 8 }),
      game({ game_id: "game-b", candidate_side: "second", outcome: "opponent_win", plies: 12 }),
    ]);
    expect(view.games.map((g) => g.result)).toEqual(["win", "loss"]);
    expect(view.wins).toBe(1);
    expect(view.losses).toBe(1);
    expect(view.draws).toBe(0);
    expect(view.totalPlies).toBe(20);
    expect(view.totalElapsedMs).toBe(1000);
    expect(view.candidateIterations).toBe(2000);
    expect(view.pairUtility).toBe(0.25);
    expect(view.shortPairId).toBe("abc123def456");
  });

  it("treats draw and unknown outcomes distinctly", () => {
    const view = derivePairInspector(pair, [
      game({ outcome: "draw" }),
      game({ outcome: "weird" }),
    ]);
    expect(view.draws).toBe(1);
    expect(view.games[1]!.result).toBe("unknown");
    expect(view.games[1]!.resultLabel).toBe("weird");
  });

  it("has no traces for v4 tuner games", () => {
    const view = derivePairInspector(pair, [game({})]);
    expect(view.games[0]!.hasTrace).toBe(false);
  });
});
