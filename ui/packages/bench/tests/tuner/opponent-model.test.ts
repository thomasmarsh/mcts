import { describe, expect, it } from "vitest";
import { deriveOpponentResponse } from "../../src/tuner/models/opponent-model.js";

const report = {
  opponent_response_analysis: {
    scope: { opponent_ids: ["schema-default", "historical"], interval_method: "hoeffding_pair_bound_v1" },
    candidates: [
      {
        candidate_id: "candidate-aaaa1111",
        opponent_responses: [
          { opponent_id: "schema-default", mean: 0.7, interval: { lower: 0.4, upper: 0.9 }, games: 14 },
          { opponent_id: "historical", mean: 0.3, interval: { lower: 0.1, upper: 0.6 }, games: 14 },
        ],
      },
      {
        candidate_id: "candidate-bbbb2222",
        opponent_responses: [
          { opponent_id: "schema-default", mean: 0.5, interval: { lower: 0.2, upper: 0.8 }, games: 14 },
          { opponent_id: "historical", mean: 0.5, interval: { lower: 0.2, upper: 0.8 }, games: 14 },
        ],
      },
    ],
    pairwise_interactions: [
      {
        left_candidate_id: "candidate-aaaa1111",
        right_candidate_id: "candidate-bbbb2222",
        contrasts: [
          { opponent_id: "schema-default", relation: "left_better", mean_difference: 0.2 },
          { opponent_id: "historical", relation: "tie", mean_difference: 0.0 },
        ],
        ranking_reversals: [{ opponent_id: "historical" }],
      },
    ],
  },
};

describe("deriveOpponentResponse", () => {
  it("is absent without the analysis section", () => {
    expect(deriveOpponentResponse({}).present).toBe(false);
  });

  it("builds a candidate × opponent grid sorted by mean, flagging interactions", () => {
    const view = deriveOpponentResponse(report);
    expect(view.present).toBe(true);
    expect(view.opponentIds).toEqual(["schema-default", "historical"]);
    expect(view.rows.map((r) => r.shortId)).toEqual(["aaaa1111", "bbbb2222"]);
    const top = view.rows[0]!;
    expect(top.mean).toBeCloseTo(0.5);
    expect(top.cells[0]!.mean).toBe(0.7);
    // material contrast at schema-default, ranking reversal at historical
    expect(top.cells[0]!.flagged).toBe(true);
    expect(top.cells[1]!.flagged).toBe(true);
    expect(view.kpis.find((k) => k.label === "Ranking reversals")?.value).toBe("1");
    expect(view.kpis.find((k) => k.label === "Material interactions")?.value).toBe("1");
  });

  it("falls back to opponents seen in the responses when scope omits them", () => {
    const view = deriveOpponentResponse({
      opponent_response_analysis: {
        candidates: [
          {
            candidate_id: "candidate-cccc3333",
            opponent_responses: [{ opponent_id: "mcts-lite", mean: 0.6, interval: { lower: 0, upper: 1 } }],
          },
        ],
      },
    });
    expect(view.opponentIds).toEqual(["mcts-lite"]);
  });
});
