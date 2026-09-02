import { describe, expect, it } from "vitest";
import { deriveDiagnosticGraph } from "../../src/tuner/models/diagnostic-model.js";

describe("deriveDiagnosticGraph", () => {
  it("is absent without a graph", () => {
    expect(deriveDiagnosticGraph({}).present).toBe(false);
  });

  it("reports nodes but no budget when the objective order was accepted directly", () => {
    const view = deriveDiagnosticGraph({
      diagnostic_matchup_graph: {
        scope: { pair_attempt_budget: 0, search_effort: { kind: "iterations", value: 3 } },
        allocations: { count: 0 },
        nodes: [
          { candidate_id: "candidate-aaaa1111", objective_rank: 1 },
          { candidate_id: "candidate-bbbb2222", objective_rank: 0 },
        ],
        edges: [],
        material_cycle_components: [],
        shortlist_effect: {},
      },
    });
    expect(view.present).toBe(true);
    expect(view.hasBudget).toBe(false);
    expect(view.nodes.map((n) => n.shortId)).toEqual(["bbbb2222", "aaaa1111"]);
  });

  it("directs edges by material_direction and marks cycle members", () => {
    const view = deriveDiagnosticGraph({
      diagnostic_matchup_graph: {
        scope: { pair_attempt_budget: 12 },
        allocations: { count: 6 },
        nodes: [
          { candidate_id: "candidate-aaaa1111", objective_rank: 0 },
          { candidate_id: "candidate-bbbb2222", objective_rank: 1 },
          { candidate_id: "candidate-cccc3333", objective_rank: 2 },
        ],
        edges: [
          {
            left_candidate_id: "candidate-aaaa1111",
            right_candidate_id: "candidate-bbbb2222",
            material_direction: "right",
            estimate: -0.3,
            interval: { lower: -0.6, upper: -0.1 },
            pair_count: 4,
          },
          {
            left_candidate_id: "candidate-bbbb2222",
            right_candidate_id: "candidate-cccc3333",
            material_direction: "left",
            estimate: 0.2,
            interval: { lower: 0.05, upper: 0.4 },
            pair_count: 4,
          },
        ],
        material_cycle_components: [
          {
            candidate_ids: ["candidate-aaaa1111", "candidate-bbbb2222", "candidate-cccc3333"],
            witness_cycle_candidate_ids: ["candidate-aaaa1111", "candidate-bbbb2222"],
          },
        ],
        shortlist_effect: {
          objective_candidate_ids: ["candidate-aaaa1111", "candidate-bbbb2222"],
          finalist_ids: ["candidate-aaaa1111", "candidate-cccc3333"],
          reserve_candidate_id: "candidate-cccc3333",
          displaced_candidate_id: "candidate-bbbb2222",
        },
      },
    });
    expect(view.hasBudget).toBe(true);
    expect(view.edges[0]).toMatchObject({ from: "candidate-bbbb2222", to: "candidate-aaaa1111" });
    expect(view.edges[1]).toMatchObject({ from: "candidate-bbbb2222", to: "candidate-cccc3333" });
    expect(view.nodes.every((n) => n.inCycle)).toBe(true);
    expect(view.cycles[0]!.members).toEqual(["aaaa1111", "bbbb2222", "cccc3333"]);
    expect(view.shortlist.reserveDisplaced).toBe(true);
  });
});
