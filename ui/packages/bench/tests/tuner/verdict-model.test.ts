import { describe, expect, it } from "vitest";
import { deriveVerdict, shortCandidateId } from "../../src/tuner/models/verdict-model.js";
import type { ProjectionCandidate, ProjectionValidation } from "../../src/tuner/tuner-types.js";

const cand = (over: Partial<ProjectionCandidate>): ProjectionCandidate => ({
  candidate_id: "candidate-aaaa",
  fingerprint: "aaaa",
  canonical_config: { select: "b" },
  cohort_index: 0,
  cohort_slot: 0,
  source: "smac_model",
  parent_candidate_id: null,
  ...over,
});

const validation = (): ProjectionValidation => ({
  rows: [
    {
      candidate_id: "candidate-bbbb",
      rank: 2,
      estimate: 0.1,
      lower: -0.2,
      upper: 0.4,
      wins: 1,
      draws: 2,
      losses: 1,
    },
    {
      candidate_id: "candidate-aaaa",
      rank: 1,
      estimate: 0.5,
      lower: 0.2,
      upper: 0.8,
      wins: 3,
      draws: 1,
      losses: 0,
    },
  ],
  unresolved_ties: [{ left_candidate_id: "candidate-aaaa", right_candidate_id: "candidate-bbbb" }],
});

describe("shortCandidateId", () => {
  it("strips the prefix and truncates", () => {
    expect(shortCandidateId("candidate-0123456789abcdef")).toBe("0123456789ab");
    expect(shortCandidateId("schema-default")).toBe("schema-defau");
  });
});

describe("deriveVerdict", () => {
  it("ranks the finalist, enriches with config, and derives the domain", () => {
    const v = deriveVerdict({
      validation: validation(),
      candidates: [
        cand({ candidate_id: "candidate-aaaa" }),
        cand({ candidate_id: "candidate-bbbb", source: "bootstrap_random" }),
      ],
      report: undefined,
    });
    expect(v.finalist?.candidateId).toBe("candidate-aaaa");
    expect(v.finalist?.config).toEqual({ select: "b" });
    expect(v.finalist?.source).toBe("smac_model");
    expect(v.runnerUp?.candidateId).toBe("candidate-bbbb");
    expect(v.ranked.map((r) => r.rank)).toEqual([1, 2]);
    expect(v.domain[0]).toBeLessThan(-0.2);
    expect(v.domain[1]).toBeGreaterThan(0.8);
  });

  it("surfaces ties from the validation payload", () => {
    const v = deriveVerdict({ validation: validation(), candidates: [], report: undefined });
    expect(v.ties).toEqual([
      { left: "candidate-aaaa", right: "candidate-bbbb", leftShort: "aaaa", rightShort: "bbbb" },
    ]);
  });

  it("builds caveats from a mechanics-smoke claim, missing axes, and limitations", () => {
    const v = deriveVerdict({
      validation: validation(),
      candidates: [],
      report: {
        validation_claim: { claim: "mechanics_smoke", missing_production_axes: ["search_effort"] },
        limitations: ["default-only starting state"],
      },
    });
    expect(v.claim).toBe("mechanics_smoke");
    expect(v.caveats[0]).toMatch(/Mechanics smoke/);
    expect(v.caveats).toContain("Missing production axis: search effort");
    expect(v.caveats).toContain("default-only starting state");
  });

  it("emits no finalist and a safe domain when there are no validation rows", () => {
    const v = deriveVerdict({
      validation: { rows: [], unresolved_ties: null },
      candidates: [],
      report: undefined,
    });
    expect(v.finalist).toBeNull();
    expect(v.domain).toEqual([-1, 1]);
    expect(v.caveats).toEqual([]);
  });
});
