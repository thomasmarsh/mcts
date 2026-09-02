import { describe, expect, it } from "vitest";
import { deriveConvergence, deriveObservations } from "../../src/tuner/models/science-models.js";

const report = {
  shadow_elimination: {
    cohorts: [
      {
        cohort_index: 0,
        candidate_paths: [
          { candidate_id: "candidate-aaaa1111", looks: [{ maximum_mean_difference: 0.0 }] },
        ],
      },
      {
        cohort_index: 1,
        candidate_paths: [
          { candidate_id: "candidate-aaaa1111", looks: [{ maximum_mean_difference: 0.0 }] },
          { candidate_id: "candidate-cccc3333", looks: [{ maximum_mean_difference: 0.5 }] },
        ],
      },
    ],
  },
  opponent_response_analysis: {
    scope: { cohort_index: 1 },
    candidates: [
      {
        candidate_id: "candidate-cccc3333",
        opponent_responses: [
          { opponent_id: "schema-default", mean: 1.0, interval: { lower: 0.48, upper: 1.0 } },
          { opponent_id: "historical", mean: 1.0, interval: { lower: 0.48, upper: 1.0 } },
        ],
      },
      {
        candidate_id: "candidate-aaaa1111",
        opponent_responses: [
          { opponent_id: "schema-default", mean: 0.5, interval: { lower: 0.0, upper: 1.0 } },
          { opponent_id: "historical", mean: 0.4, interval: { lower: 0.1, upper: 0.9 } },
        ],
      },
    ],
  },
};

describe("deriveConvergence", () => {
  it("emits one step per cohort with the leader's best margin", () => {
    const c = deriveConvergence(report);
    expect(c.present).toBe(true);
    expect(c.steps.map((s) => s.bestMargin)).toEqual([0, 0.5]);
    expect(c.steps[1]!.leaderShortId).toBe("cccc3333");
    expect(c.steps.map((s) => s.x)).toEqual([1, 2]);
    expect(c.domain[1]).toBeCloseTo(0.55);
  });

  it("is absent without cohorts", () => {
    expect(deriveConvergence({}).present).toBe(false);
  });
});

describe("deriveObservations", () => {
  it("summarises each candidate across opponents, sorted by mean", () => {
    const o = deriveObservations(report);
    expect(o.present).toBe(true);
    expect(o.cohortIndex).toBe(1);
    expect(o.rows.map((r) => r.shortId)).toEqual(["cccc3333", "aaaa1111"]);
    const worst = o.rows[1]!;
    expect(worst.mean).toBeCloseTo(0.45);
    expect(worst.lower).toBe(0.0);
    expect(worst.upper).toBe(1.0);
    expect(worst.opponents).toBe(2);
  });

  it("is absent without opponent_response_analysis", () => {
    expect(deriveObservations({}).present).toBe(false);
  });
});
