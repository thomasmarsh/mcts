// profile-model.test.ts — pure tests for the launch-profile draft model:
// file JSON ⇄ draft round-trips, constraint-row reconstruction, and the
// client-side validator.

import { describe, expect, it } from "vitest";
import {
  draftFromProfileContent,
  draftToProfileContent,
  emptyProfileDraft,
  rowsFromConstraints,
  validateProfileDraft,
  profileSlugKey,
} from "../../src/tuner/models/profile-model.js";
import type { ParamSchema } from "../../src/tuner/models/constraint-editor-model.js";

const schema: ParamSchema = {
  parameters: [
    { name: "algorithm", type: "categorical", choices: ["mcts", "random", "negamax"] },
    { name: "select", type: "categorical", choices: ["ucb1", "rave"] },
    { name: "c", type: "float", bounds: [0.5, 3.0], default: 1.4 },
  ],
  conditions: [
    { if: { algorithm: "mcts" }, then: ["select"] },
    { if: { select: ["ucb1"] }, then: ["c"] },
  ],
};

describe("profileSlugKey", () => {
  it("slugs an id and falls back to 'profile'", () => {
    expect(profileSlugKey("Druid UCB1 Sweep")).toBe("druid-ucb1-sweep");
    expect(profileSlugKey("  ")).toBe("profile");
  });
});

describe("draftToProfileContent", () => {
  it("emits constraints, efforts, and budgets from a filled draft", () => {
    const draft = emptyProfileDraft("druid");
    draft.profileId = "druid-sweep";
    draft.objectiveKey = "druid-strength";
    draft.constraintRows = rowsFromConstraints(schema, [
      { set: { algorithm: { choices: ["mcts"] } } },
      { set: { c: { range: [1.2, 1.8] } } },
    ]).rows;
    draft.efforts.tuning = { value: "500", unit: "iterations" };
    draft.efforts.production = { value: "2000", unit: "iterations" };
    draft.budgets.tuningPairBudget = "40";
    draft.budgets.finalists = "3";

    const content = draftToProfileContent(draft, schema) as Record<string, unknown>;
    expect(content.game_kind).toBe("druid");
    expect(content.objective_key).toBe("druid-strength");
    expect(content.constraints).toEqual([
      { set: { algorithm: { choices: ["mcts"] } } },
      { set: { c: { range: [1.2, 1.8] } } },
    ]);
    expect(content.efforts).toEqual({
      tuning: { kind: "iterations", value: 500 },
      production: { kind: "iterations", value: 2000 },
    });
    expect(content.budgets).toEqual({
      tuning_pair_budget: 40,
      validation_pair_budget: 24,
      production_validation_pairs: 8,
      finalists: 3,
    });
  });

  it("omits constraints and efforts when empty", () => {
    const draft = emptyProfileDraft("druid");
    draft.objectiveKey = "o";
    const content = draftToProfileContent(draft, schema) as Record<string, unknown>;
    expect(content.constraints).toBeUndefined();
    expect(content.efforts).toBeUndefined();
    expect(content.budgets).toBeDefined();
  });
});

describe("draftFromProfileContent", () => {
  it("round-trips a profile file back to an equivalent draft", () => {
    const original = emptyProfileDraft("druid");
    original.profileId = "p";
    original.objectiveKey = "druid-strength";
    original.constraintRows = rowsFromConstraints(schema, [
      { set: { select: { choices: ["ucb1"] } } },
    ]).rows;
    original.efforts.validation = { value: "300", unit: "time_ms" };
    original.budgets.cohortSize = "8";

    const content = draftToProfileContent(original, schema);
    const { draft, warnings } = draftFromProfileContent(content, schema);
    expect(warnings).toEqual([]);
    expect(draft.gameKind).toBe("druid");
    expect(draft.objectiveKey).toBe("druid-strength");
    expect(draft.efforts.validation).toEqual({ value: "300", unit: "time_ms" });
    expect(draft.budgets.cohortSize).toBe("8");
    expect(draftToProfileContent(draft, schema)).toEqual(content);
  });

  it("warns and drops a constraint on an unknown parameter", () => {
    const { warnings } = draftFromProfileContent(
      { game_kind: "druid", objective_key: "o", constraints: [{ set: { bogus: { fix: 1 } } }] },
      schema,
    );
    expect(warnings.some((w) => w.includes("bogus"))).toBe(true);
  });

  it("tolerates a non-object file", () => {
    const { draft, warnings } = draftFromProfileContent(42 as never, schema, "druid");
    expect(draft.gameKind).toBe("druid");
    expect(warnings).toHaveLength(1);
  });
});

describe("rowsFromConstraints", () => {
  it("accepts the bare sugar map form", () => {
    const { rows } = rowsFromConstraints(schema, { c: { fix: 2 } });
    expect(rows.c!.mode).toBe("fix");
    expect(rows.c!.fix).toBe("2");
  });
});

describe("validateProfileDraft", () => {
  it("flags a missing objective, a bad budget, and an effort that exceeds production", () => {
    const draft = emptyProfileDraft("druid");
    draft.budgets.tuningPairBudget = "0";
    draft.efforts.production = { value: "100", unit: "iterations" };
    draft.efforts.tuning = { value: "200", unit: "iterations" };
    const errors = validateProfileDraft(draft, schema);
    expect(errors).toContain("pick an objective");
    expect(errors.some((e) => e.includes("tuning pair budget"))).toBe(true);
    expect(errors.some((e) => e.includes("cannot exceed production"))).toBe(true);
  });

  it("passes a well-formed draft", () => {
    const draft = emptyProfileDraft("druid");
    draft.objectiveKey = "o";
    expect(validateProfileDraft(draft, schema)).toEqual([]);
  });
});
