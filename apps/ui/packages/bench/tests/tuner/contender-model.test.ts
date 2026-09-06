import { describe, expect, it } from "vitest";

import {
  deriveContenders,
  findContender,
  formatWDL,
} from "../../src/tuner/models/contender-model.js";
import type {
  ProjectionCandidate,
  ProjectionContenderRow,
} from "../../src/tuner/tuner-types.js";

const row = (over: Partial<ProjectionContenderRow>): ProjectionContenderRow => ({
  candidate_id: "candidate-aaaa000000000000",
  opponent_id: "op-mcts-1k",
  phase: "tuning",
  wins: 0,
  losses: 0,
  draws: 0,
  sum_utility: 0,
  ...over,
});

const cand = (over: Partial<ProjectionCandidate>): ProjectionCandidate => ({
  candidate_id: "candidate-aaaa000000000000",
  fingerprint: "f",
  canonical_config: {},
  cohort_index: 0,
  cohort_slot: 0,
  source: "schema_default",
  parent_candidate_id: null,
  ...over,
});

describe("deriveContenders", () => {
  it("aggregates W/D/L per opponent and overall, and rolls the phase split", () => {
    const rows = [
      row({ opponent_id: "op-a", phase: "tuning", wins: 3, losses: 1, draws: 0, sum_utility: 2 }),
      row({ opponent_id: "op-b", phase: "tuning", wins: 1, losses: 2, draws: 1, sum_utility: -1 }),
      row({
        opponent_id: "op-a",
        phase: "validation",
        wins: 2,
        losses: 0,
        draws: 0,
        sum_utility: 2,
      }),
    ];
    const c = deriveContenders(rows, [cand({})])[0]!;
    expect(c.overall).toEqual({ wins: 6, losses: 3, draws: 1 });
    expect(c.games).toBe(10);
    expect(c.byOpponent.map((o) => [o.opponentId, formatWDL(o)])).toEqual([
      ["op-a", "5 / 0 / 1"],
      ["op-b", "1 / 1 / 2"],
    ]);
    expect(c.byPhase.map((p) => p.phase)).toEqual(["tuning", "validation"]);
    expect(c.phaseReached).toBe("validation");
    expect(c.meanUtility).toBeCloseTo(3 / 10);
  });

  it("sorts by mean utility descending and includes candidates with no pairs", () => {
    const rows = [
      row({ candidate_id: "candidate-low", sum_utility: -4, wins: 0, losses: 4, draws: 0 }),
      row({ candidate_id: "candidate-high", sum_utility: 4, wins: 4, losses: 0, draws: 0 }),
    ];
    const candidates = [
      cand({ candidate_id: "candidate-high", cohort_index: 1 }),
      cand({ candidate_id: "candidate-low", cohort_index: 1 }),
      cand({ candidate_id: "candidate-fresh", cohort_index: 2 }),
    ];
    const out = deriveContenders(rows, candidates);
    expect(out.map((c) => c.candidateId)).toEqual([
      "candidate-high",
      "candidate-low",
      "candidate-fresh",
    ]);
    const fresh = findContender(out, "candidate-fresh")!;
    expect(fresh.games).toBe(0);
    expect(fresh.meanUtility).toBeNull();
    expect(fresh.phaseReached).toBeNull();
    expect(fresh.cohortIndex).toBe(2);
  });

  it("returns nothing with neither rows nor candidates", () => {
    expect(deriveContenders([], undefined)).toEqual([]);
  });
});

describe("formatWDL", () => {
  it("renders W / D / L in that order", () => {
    expect(formatWDL({ wins: 3, losses: 1, draws: 2 })).toBe("3 / 2 / 1");
  });
});
