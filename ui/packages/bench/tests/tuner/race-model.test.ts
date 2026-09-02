import { describe, expect, it } from "vitest";
import { deriveCohortRace } from "../../src/tuner/models/race-model.js";
import type { ProjectionCandidate } from "../../src/tuner/tuner-types.js";

const report = {
  shadow_elimination: {
    policy: { enforced: false, kind: "paired_bootstrap" },
    cohorts: [
      {
        cohort_index: 0,
        candidate_paths: [
          {
            candidate_id: "candidate-aaaa1111",
            final_top_set: true,
            first_elimination_prefix_id: null,
            protected: false,
            looks: [
              { prefix_id: "prefix-p6xxxx", disposition: "continue" },
              { prefix_id: "prefix-p12xxx", disposition: "continue" },
            ],
          },
          {
            candidate_id: "candidate-bbbb2222",
            final_top_set: false,
            first_elimination_prefix_id: "prefix-p12xxx",
            protected: false,
            looks: [{ prefix_id: "prefix-p12xxx", disposition: "eliminate" }],
          },
        ],
      },
    ],
  },
};

const candidates: ProjectionCandidate[] = [
  {
    candidate_id: "candidate-aaaa1111",
    fingerprint: "aaaa1111",
    canonical_config: {},
    cohort_index: 0,
    cohort_slot: 0,
    source: "smac_model",
    parent_candidate_id: null,
  },
];

describe("deriveCohortRace", () => {
  it("builds a prefix-ordered grid with a cell per look", () => {
    const race = deriveCohortRace(report, candidates);
    expect(race.present).toBe(true);
    expect(race.enforced).toBe(false);
    expect(race.policyKind).toBe("paired_bootstrap");
    const cohort = race.cohorts[0]!;
    expect(cohort.prefixes.map((p) => p.index)).toEqual([1, 2]);
    const [rowA, rowB] = cohort.rows;
    expect(rowA!.source).toBe("smac_model");
    expect(rowA!.finalTopSet).toBe(true);
    expect(rowA!.cells).toEqual(["continue", "continue"]);
    // b was only looked at the second prefix.
    expect(rowB!.cells).toEqual([null, "eliminate"]);
    expect(rowB!.firstEliminationPrefixId).toBe("prefix-p12xxx");
  });

  it("collects the distinct dispositions for a legend", () => {
    expect(deriveCohortRace(report, []).dispositions).toEqual(["continue", "eliminate"]);
  });

  it("is absent without a shadow_elimination record", () => {
    expect(deriveCohortRace({}, []).present).toBe(false);
  });
});
