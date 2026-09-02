import { describe, expect, it } from "vitest";
import { deriveProposalFunnel } from "../../src/tuner/models/funnel-model.js";

const proposalSearch = {
  proposal_search: {
    configured: {
      bootstrap: 2,
      model: 2,
      random_reserve: 2,
      cohorts: 2,
      retained_elites: 2,
      excluded_families: [],
    },
    actual_source_attempts: {
      schema_default: 1,
      bootstrap_random: 1,
      smac_model: 3,
      random_reserve: 3,
      irace_model: 0,
    },
    rejections_by_source: { schema_default: 0, bootstrap_random: 0, smac_model: 1, random_reserve: 1 },
    accepted: [
      { source: "schema_default" },
      { source: "bootstrap_random" },
      { source: "smac_model" },
      { source: "smac_model" },
      { source: "random_reserve" },
      { source: "random_reserve" },
    ],
    model_version: "smac-2.4-public-ask-v1",
    final_observation_count: 4,
    final_frontier_id: "frontier-c3f2baf765fe13e09bb6a4c30286fb3b53c75aa825101c6b6f223170f9b3255e",
  },
};

describe("deriveProposalFunnel", () => {
  it("maps each active source to configured / attempted / accepted / rejected", () => {
    const f = deriveProposalFunnel(proposalSearch);
    expect(f.present).toBe(true);
    expect(f.stages.map((s) => s.source)).toEqual([
      "schema_default",
      "bootstrap_random",
      "smac_model",
      "random_reserve",
    ]);
    const smac = f.stages.find((s) => s.source === "smac_model")!;
    expect(smac).toMatchObject({ configured: 2, attempted: 3, accepted: 2, rejected: 1 });
    expect(f.stages.find((s) => s.source === "schema_default")!.configured).toBe(1);
  });

  it("surfaces headline KPIs including the shortened frontier id", () => {
    const f = deriveProposalFunnel(proposalSearch);
    expect(f.kpis).toContainEqual({ label: "Model", value: "smac-2.4-public-ask-v1" });
    expect(f.kpis).toContainEqual({ label: "Cohorts", value: "2" });
    expect(f.kpis).toContainEqual({ label: "Final frontier", value: "c3f2baf765fe" });
    expect(f.kpis).toContainEqual({ label: "Excluded families", value: "none" });
  });

  it("is absent when the report has no proposal_search", () => {
    expect(deriveProposalFunnel({}).present).toBe(false);
    expect(deriveProposalFunnel(undefined).stages).toEqual([]);
  });
});
