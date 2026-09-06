import { describe, expect, it } from "vitest";
import type { TunerInfo } from "../../src/types.js";
import {
  axisGroups,
  deriveConstraints,
  emptyRow,
  emptyRows,
  modesFor,
  predicateParents,
  type ConstraintRow,
  type ConstraintRows,
} from "../../src/tuner/models/constraint-editor-model.js";

// A two-algorithm schema: `algorithm` gates `select` + `max_depth`, and
// `select` in turn gates `c`. `q_init` is an ungated schema constant.
const SCHEMA: TunerInfo = {
  id: "strategy",
  baselines: ["strong"],
  eval_rounds: 5,
  game_config: {},
  parameters: [
    { name: "algorithm", type: "categorical", choices: ["mcts", "random"], default: "mcts" },
    { name: "select", type: "categorical", choices: ["ucb1", "ucb1_tuned", "rave"], default: "ucb1" },
    { name: "c", type: "float", bounds: [0, 3], default: 1.41 },
    { name: "max_depth", type: "int", bounds: [1, 10], default: 4 },
    { name: "q_init", type: "constant", value: "Infinity" },
  ],
  conditions: [
    { if: { algorithm: "mcts" }, then: ["select", "max_depth"] },
    { if: { select: ["ucb1", "ucb1_tuned"] }, then: ["c"] },
  ],
};

function rows(overrides: Record<string, Partial<ConstraintRow>>): ConstraintRows {
  const base = emptyRows(SCHEMA);
  for (const [name, patch] of Object.entries(overrides)) {
    base[name] = { ...base[name]!, ...patch };
  }
  return base;
}

describe("axisGroups", () => {
  it("puts ungated parameters in the top group, then one group per gating axis", () => {
    const groups = axisGroups(SCHEMA);
    expect(groups.map((g) => g.axis)).toEqual([null, "algorithm", "select"]);
    expect(groups[0]!.parameters.map((p) => p.name)).toEqual(["algorithm", "q_init"]);
    expect(groups[1]!.parameters.map((p) => p.name)).toEqual(["select", "max_depth"]);
    expect(groups[2]!.parameters.map((p) => p.name)).toEqual(["c"]);
  });
});

describe("modesFor", () => {
  it("offers modes matching the parameter's domain", () => {
    expect(modesFor(SCHEMA.parameters[0]!)).toEqual(["free", "fix", "choices"]);
    expect(modesFor(SCHEMA.parameters[2]!)).toEqual(["free", "fix", "range"]);
    expect(modesFor(SCHEMA.parameters[4]!)).toEqual(["free"]);
  });
});

describe("predicateParents", () => {
  it("returns categorical ancestors up the condition graph", () => {
    expect(predicateParents(SCHEMA, "c").map((p) => p.name)).toEqual(["algorithm", "select"]);
    expect(predicateParents(SCHEMA, "select").map((p) => p.name)).toEqual(["algorithm"]);
    expect(predicateParents(SCHEMA, "algorithm")).toEqual([]);
  });
});

describe("emptyRow", () => {
  it("seeds range bounds and the full retained choice set from the schema", () => {
    expect(emptyRow(SCHEMA.parameters[2]!)).toMatchObject({ mode: "free", low: "0", high: "3" });
    expect(emptyRow(SCHEMA.parameters[1]!).retained).toEqual(["ucb1", "ucb1_tuned", "rave"]);
  });
});

describe("deriveConstraints", () => {
  it("emits nothing for all-free rows", () => {
    expect(deriveConstraints(SCHEMA, emptyRows(SCHEMA))).toEqual({ constraints: [], errors: [] });
  });

  it("fixes a numeric parameter inside its domain", () => {
    const { constraints, errors } = deriveConstraints(SCHEMA, rows({ c: { mode: "fix", fix: "1.5" } }));
    expect(errors).toEqual([]);
    expect(constraints).toEqual([{ set: { c: { fix: 1.5 } } }]);
  });

  it("rejects a fix outside the schema bounds", () => {
    const { errors } = deriveConstraints(SCHEMA, rows({ c: { mode: "fix", fix: "9" } }));
    expect(errors).toEqual(["c: fix 9 is outside [0, 3]"]);
  });

  it("rejects a non-integer fix on an int parameter", () => {
    const { errors } = deriveConstraints(SCHEMA, rows({ max_depth: { mode: "fix", fix: "3.5" } }));
    expect(errors).toEqual(["max_depth: fix needs an integer"]);
  });

  it("narrows a numeric range", () => {
    const { constraints, errors } = deriveConstraints(
      SCHEMA,
      rows({ max_depth: { mode: "range", low: "2", high: "6" } }),
    );
    expect(errors).toEqual([]);
    expect(constraints).toEqual([{ set: { max_depth: { range: [2, 6] } } }]);
  });

  it("rejects a range that escapes the schema bounds or inverts", () => {
    expect(
      deriveConstraints(SCHEMA, rows({ c: { mode: "range", low: "-1", high: "2" } })).errors,
    ).toEqual(["c: range escapes schema bounds [0, 3]"]);
    expect(
      deriveConstraints(SCHEMA, rows({ c: { mode: "range", low: "2", high: "1" } })).errors,
    ).toEqual(["c: range low must be below high"]);
  });

  it("restricts a categorical to a proper subset", () => {
    const { constraints, errors } = deriveConstraints(
      SCHEMA,
      rows({ select: { mode: "choices", retained: ["ucb1", "rave"] } }),
    );
    expect(errors).toEqual([]);
    expect(constraints).toEqual([{ set: { select: { choices: ["ucb1", "rave"] } } }]);
  });

  it("flags a categorical with every box unticked", () => {
    const { errors } = deriveConstraints(SCHEMA, rows({ select: { mode: "choices", retained: [] } }));
    expect(errors).toEqual(["select: must leave at least one choice"]);
  });

  it("flags a choices row that drops nothing", () => {
    const { errors } = deriveConstraints(
      SCHEMA,
      rows({ select: { mode: "choices", retained: ["ucb1", "ucb1_tuned", "rave"] } }),
    );
    expect(errors).toEqual(["select: choices must drop at least one value"]);
  });

  it("attaches a valid when predicate", () => {
    const { constraints, errors } = deriveConstraints(
      SCHEMA,
      rows({ c: { mode: "range", low: "1.2", high: "1.8", when: { select: ["ucb1", "ucb1_tuned"] } } }),
    );
    expect(errors).toEqual([]);
    expect(constraints).toEqual([
      { set: { c: { range: [1.2, 1.8] } }, when: { select: ["ucb1", "ucb1_tuned"] } },
    ]);
  });

  it("rejects a when predicate on a non-categorical parent or an out-of-domain value", () => {
    expect(
      deriveConstraints(SCHEMA, rows({ c: { mode: "fix", fix: "1", when: { max_depth: ["4"] } } }))
        .errors,
    ).toEqual(['c: when parent "max_depth" is not a categorical parameter']);
    expect(
      deriveConstraints(SCHEMA, rows({ c: { mode: "fix", fix: "1", when: { select: ["nope"] } } }))
        .errors,
    ).toEqual(['c: when value "nope" is not a choice of "select"']);
  });

  it("expresses 'exclude an algorithm' as a choices row", () => {
    const { constraints } = deriveConstraints(
      SCHEMA,
      rows({ algorithm: { mode: "choices", retained: ["mcts"] } }),
    );
    expect(constraints).toEqual([{ set: { algorithm: { choices: ["mcts"] } } }]);
  });
});
