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

describe("science models from projection rows (live, no report)", () => {
  const cohorts = [
    { cohort_index: 0, candidate_ids: ["candidate-aaaa1111", "candidate-bbbb2222"], retained_candidate_ids: [] },
    { cohort_index: 1, candidate_ids: ["candidate-cccc3333"], retained_candidate_ids: [] },
  ];
  const observations = [
    { observation_id: "o1", candidate_id: "candidate-aaaa1111", phase: "tuning", prefix_id: "p1", mean: 0.4, lower: 0.2, upper: 0.6 },
    { observation_id: "o2", candidate_id: "candidate-bbbb2222", phase: "tuning", prefix_id: "p1", mean: 0.55, lower: 0.3, upper: 0.8 },
    { observation_id: "o3", candidate_id: "candidate-cccc3333", phase: "tuning", prefix_id: "p2", mean: 0.7, lower: 0.5, upper: 0.9 },
  ];

  it("deriveConvergence: one step per cohort, y = best observed mean among its members", () => {
    const c = deriveConvergence(undefined, cohorts, observations);
    expect(c.present).toBe(true);
    expect(c.steps.map((s) => s.bestMargin)).toEqual([0.55, 0.7]);
    expect(c.steps[1]!.leaderShortId).toBe("cccc3333");
  });

  it("deriveObservations: per-candidate forest rows from the observation rows", () => {
    const o = deriveObservations(undefined, observations, [
      { candidate_id: "candidate-aaaa1111", fingerprint: "f", canonical_config: {}, cohort_index: 0, cohort_slot: 0, source: "s", parent_candidate_id: null },
    ]);
    expect(o.present).toBe(true);
    expect(o.rows.map((r) => r.shortId)).toEqual(["cccc3333", "bbbb2222", "aaaa1111"]);
    expect(o.rows[0]).toMatchObject({ mean: 0.7, lower: 0.5, upper: 0.9 });
  });

  it("still uses the report when it is present", () => {
    const c = deriveConvergence(report, cohorts, observations);
    expect(c.steps).toHaveLength(2);
    expect(c.steps[1]!.bestMargin).toBe(0.5); // from maximum_mean_difference, not observations
  });
});
