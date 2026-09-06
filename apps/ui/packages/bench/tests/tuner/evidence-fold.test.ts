import { describe, expect, it } from "vitest";
import { foldEvidence, tickerLines, describeEvent } from "../../src/tuner/models/evidence-fold.js";
import type { EvidenceEnvelope } from "../../src/tuner/tuner-types.js";

let seq = 0;
const ev = (type: EvidenceEnvelope["type"], payload: unknown): EvidenceEnvelope => ({
  sequence: ++seq,
  type,
  payload,
});

describe("foldEvidence", () => {
  it("tracks phase, resetting the per-phase pair counters on a transition", () => {
    seq = 0;
    const progress = foldEvidence([
      ev("proposal_created", { source: "smac_model", candidate_id: "candidate-aaa111", cohort_index: 0 }),
      ev("pair_started", { phase: "tuning", pair_id: "pair-1", candidate_id: "candidate-aaa111" }),
      ev("pair_completed", { phase: "tuning", pair_id: "pair-1", candidate_id: "candidate-aaa111", pair_utility: 0.2 }),
      ev("allocation_decided", { allocation: { kind: "begin_validation", tuning_prefix_id: "p" } }),
      ev("pair_started", { phase: "validation", pair_id: "pair-2", candidate_id: "candidate-aaa111" }),
    ]);
    expect(progress.phase).toBe("validation");
    // The tuning pairs were zeroed when validation began.
    expect(progress.pairs).toEqual({ started: 1, completed: 0, failed: 0 });
    expect(progress.lastEventSeq).toBe(5);
  });

  it("counts pair_* per current phase and tracks best-so-far by max utility", () => {
    seq = 0;
    const progress = foldEvidence([
      ev("pair_completed", { phase: "tuning", candidate_id: "candidate-aaa", pair_utility: 0.1 }),
      ev("pair_completed", { phase: "tuning", candidate_id: "candidate-bbb", pair_utility: 0.35 }),
      ev("pair_failed", { phase: "tuning", pair_id: "p3" }),
      ev("pair_completed", { phase: "tuning", candidate_id: "candidate-ccc", pair_utility: 0.2 }),
    ]);
    expect(progress.pairs).toEqual({ started: 0, completed: 3, failed: 1 });
    expect(progress.bestSoFar).toEqual({ candidateId: "candidate-bbb", pairUtility: 0.35 });
  });

  it("tallies proposals by source and remembers the last cohort seen", () => {
    seq = 0;
    const progress = foldEvidence([
      ev("proposal_created", { source: "smac_model", candidate_id: "c1", cohort_index: 1 }),
      ev("proposal_accepted", { source: "smac_model", candidate_id: "c1", cohort_index: 1 }),
      ev("proposal_created", { source: "bootstrap_random", candidate_id: "c2", cohort_index: 1 }),
      ev("proposal_rejected", { source: "bootstrap_random", candidate_id: "c2", cohort_index: 1 }),
      ev("cohort_completed", { cohort_index: 2, candidate_ids: ["c1", "c2"], retained_candidate_ids: ["c1"] }),
    ]);
    expect(progress.proposals).toEqual({
      smac_model: { created: 1, accepted: 1, rejected: 0 },
      bootstrap_random: { created: 1, accepted: 0, rejected: 1 },
    });
    expect(progress.cohortIndex).toBe(2);
  });

  it("reaches the done phase on run_completed", () => {
    seq = 0;
    expect(foldEvidence([ev("run_completed", { finalist_ids: ["c1"] })]).phase).toBe("done");
  });
});

describe("tickerLines / describeEvent", () => {
  it("formats each event type as one human line", () => {
    seq = 0;
    expect(
      describeEvent(ev("pair_completed", {
        phase: "tuning",
        pair_id: "pair-c7d8e19aaaa",
        candidate_id: "candidate-77bb",
        opponent_id: "baseline",
        pair_utility: 0.031,
      })),
    ).toBe("pair c7d8e19 done · 77bb vs baseline · +0.031");

    expect(
      describeEvent(ev("allocation_decided", { allocation: { kind: "begin_validation" } })),
    ).toBe("validation started");

    expect(
      describeEvent(ev("proposal_accepted", { source: "smac_model", candidate_id: "candidate-a1b2c3d" })),
    ).toBe("proposal smac_model accepted (a1b2c3d)");

    expect(
      describeEvent(ev("cohort_completed", {
        cohort_index: 1,
        candidate_ids: ["a", "b", "c", "d", "e"],
        retained_candidate_ids: ["a", "b", "c"],
      })),
    ).toBe("cohort 1 complete — 3 promoted, 2 eliminated");
  });

  it("caps the ticker at the requested limit, newest last", () => {
    seq = 0;
    const envelopes = Array.from({ length: 10 }, () => ev("run_interrupted", { stage: "s" }));
    const lines = tickerLines(envelopes, 3);
    expect(lines.map((l) => l.seq)).toEqual([8, 9, 10]);
  });
});
