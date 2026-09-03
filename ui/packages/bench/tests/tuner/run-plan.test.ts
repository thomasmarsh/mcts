import { describe, expect, it } from "vitest";
import { summarizeRunPlan } from "../../src/tuner/models/run-plan-model.js";
import type { RunPlan } from "../../src/tuner/tuner-types.js";

const resolved: RunPlan = {
  ok: true,
  errors: [],
  game_kind: "atari-go",
  objective_id: "atari-go-reference-v1",
  game_config: "{}",
  game_config_is_override: false,
  panel_fingerprint: "pf",
  opponents: [
    {
      id: "schema-default",
      label: "Default",
      role: "default",
      weight: 1,
      source: "schema_default",
      config: '{"family":"rave"}',
      fingerprint: "abc",
    },
    {
      id: "historical",
      role: "historical_reference",
      weight: 1,
      source: "inline",
      config: '{"family":"mcts"}',
    },
  ],
  space: {
    schema_id: "strategy",
    families: ["mcts", "rave"],
    excluded_families: ["negamax"],
    parameters: [
      {
        name: "family",
        kind: "categorical",
        bounds: null,
        choices: ["mcts", "rave"],
        default: "mcts",
        constant_value: null,
        active_when: null,
      },
    ],
  },
  efforts: {
    tuning: { kind: "iterations", value: 1000 },
    validation: { kind: "iterations", value: 10000 },
    production: { kind: "time_ms", value: 500 },
  },
  budgets: {
    cohort_size: 8,
    finalists: 2,
    bootstrap_candidates: 3,
    random_reserve_candidates: 2,
    tuning_pairs: 4,
    tuning_pair_budget: 64,
    validation_pair_budget: 24,
    diagnostic_pair_budget: 0,
    production_validation_pairs: 20,
    proposer_policy: "smac_mixed",
    derived: {
      initial_cohort_pairs: 32,
      validation_pairs_per_finalist: 12,
      production_pairs: 20,
    },
  },
  epoch: { epoch_id: "epoch-x", fingerprint: "deadbeefcafef00d" },
};

describe("summarizeRunPlan", () => {
  it("expands the schema-default opponent and reshapes the space + budgets", () => {
    const s = summarizeRunPlan(resolved);
    expect(s.resolved).toBe(true);
    const def = s.opponents.find((o) => o.id === "schema-default");
    expect(def?.config).toBe('{"family":"rave"}');
    expect(s.families).toEqual(["mcts", "rave"]);
    expect(s.excludedFamilies).toEqual(["negamax"]);
    expect(s.effortKpis.map((k) => k.value)).toEqual([
      "1000 iters",
      "10000 iters",
      "500 ms",
    ]);
    expect(s.budgetKpis.find((k) => k.label === "initial cohort")?.value).toBe("32 pairs");
    expect(s.epochFingerprint).toBe("deadbeefcafef00d");
  });

  it("stays unresolved and carries errors for a rejected request", () => {
    const s = summarizeRunPlan({ ok: false, errors: ["finalists must be smaller than cohort size"] });
    expect(s.resolved).toBe(false);
    expect(s.errors[0]).toContain("finalists must be smaller");
    expect(s.opponents).toEqual([]);
  });

  it("is empty for an absent plan", () => {
    expect(summarizeRunPlan(undefined).resolved).toBe(false);
  });
});
