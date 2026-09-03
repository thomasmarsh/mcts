import { describe, expect, it } from "vitest";
import type { JsonValue, TunerInfo } from "../../src/types.js";
import {
  activeParamNames,
  draftFromContent,
  draftToContent,
  emptyDraft,
  flattenDotted,
  nestDotted,
  reduceWeights,
  slugKey,
  validateDraft,
  type ObjectiveDraft,
} from "../../src/tuner/models/objective-model.js";

const ATARIGO: TunerInfo = {
  id: "strategy",
  baselines: ["strong"],
  eval_rounds: 5,
  parameters: [{ name: "c", type: "float", bounds: [0, 2], default: 1.4 }],
  conditions: [],
  game_config: { size: 13 },
  game_config_schema: {
    parameters: [{ name: "size", type: "int", bounds: [3, 19], default: 13 }],
    conditions: [],
  },
};

// Copies of the checked-in seed corpus (`tuner/objectives/*.json`), kept here
// so the round-trip is exercised without a filesystem read. If a seed's
// canonical shape changes, update both.
const SEEDS: Record<string, JsonValue> = {
  "druid-reference-v1": {
    schema_version: 1,
    objective_id: "druid-reference-v1",
    game_kind: "druid",
    opponents: [
      {
        id: "schema-default",
        label: "Schema default",
        role: "default",
        weight: 1,
        config: { source: "schema_default" },
      },
      {
        id: "historical-ucb1",
        label: "Historical UCB1",
        role: "historical_reference",
        weight: 2,
        config: {
          source: "inline",
          value: { c: 1.414, family: "ucb1", final_action: "robust_child", q_init: "Infinity" },
        },
      },
      {
        id: "historical-strong",
        label: "Historical strong configuration",
        role: "historical_reference",
        weight: 3,
        config: {
          source: "inline",
          value: {
            c: 1.414,
            epsilon: 0.3,
            family: "ucb1_dm_nst",
            final_action: "robust_child",
            nst_backoff_threshold: 5,
            q_init: "Infinity",
          },
        },
      },
    ],
    start_distribution: { kind: "default_only" },
  },
};

describe("reduceWeights", () => {
  it("divides a panel by its gcd", () => {
    expect(reduceWeights([2, 4, 6])).toEqual([1, 2, 3]);
    expect(reduceWeights([3, 5])).toEqual([3, 5]);
    expect(reduceWeights([7])).toEqual([1]);
    expect(reduceWeights([1, 1])).toEqual([1, 1]);
  });
});

describe("draftFromContent / draftToContent", () => {
  it.each(Object.entries(SEEDS))("round-trips %s to its parsed JSON", (_name, content) => {
    const { draft, warnings } = draftFromContent(content);
    expect(warnings).toEqual([]);
    expect(draftToContent(draft)).toEqual(content);
  });

  it("emptyDraft serialises to the minimum legal shape and reduces weights", () => {
    const draft = emptyDraft("nim");
    draft.objectiveId = "nim-v1";
    draft.opponents[0]!.weight = 2;
    draft.opponents[1]!.label = "Historical";
    draft.opponents[1]!.weight = 4;
    draft.opponents[1]!.configText = '{"c":1.2}';
    draft.opponents[1]!.config = { c: 1.2 };
    const wire = draftToContent(draft);
    expect(wire["schema_version"]).toBe(1);
    expect(wire["start_distribution"]).toEqual({ kind: "default_only" });
    const opponents = wire["opponents"] as Array<{ weight: number; role: string }>;
    expect(opponents.map((o) => o.weight)).toEqual([1, 2]);
    expect(opponents.map((o) => o.role)).toEqual(["default", "historical_reference"]);
  });

  it("moves a misplaced schema-default opponent to index 0", () => {
    const { draft, warnings } = draftFromContent({
      objective_id: "x",
      game_kind: "nim",
      opponents: [
        {
          id: "a",
          label: "A",
          role: "historical_reference",
          weight: 1,
          config: { source: "inline", value: {} },
        },
        { id: "d", label: "D", role: "default", weight: 1, config: { source: "schema_default" } },
      ],
    });
    expect(warnings).toEqual([]);
    expect(draft.opponents[0]!.kind).toBe("schema_default");
  });
});

describe("validateDraft", () => {
  const good = (): ObjectiveDraft => {
    const d = emptyDraft("nim");
    d.objectiveId = "nim-v1";
    d.opponents[1] = {
      id: "hist",
      label: "Historical",
      kind: "inline",
      weight: 1,
      config: { c: 1.4 },
      configText: '{"c":1.4}',
      configMode: "form",
    };
    return d;
  };

  it("passes a well-formed panel", () => {
    expect(validateDraft(good())).toEqual([]);
  });

  it("requires an objective id", () => {
    const d = good();
    d.objectiveId = "";
    expect(validateDraft(d).some((e) => /objective id/i.test(e))).toBe(true);
  });

  it("needs at least one inline opponent", () => {
    const d = good();
    d.opponents = [d.opponents[0]!];
    expect(validateDraft(d).some((e) => /at least one/i.test(e))).toBe(true);
  });

  it("rejects duplicate opponent ids", () => {
    const d = good();
    d.opponents[1]!.id = "schema-default";
    expect(validateDraft(d).some((e) => /duplicate/i.test(e))).toBe(true);
  });

  it("rejects a non-positive weight", () => {
    const d = good();
    d.opponents[1]!.weight = 0;
    expect(validateDraft(d).some((e) => /positive integer/i.test(e))).toBe(true);
  });

  it("rejects unparseable inline JSON", () => {
    const d = good();
    d.opponents[1]!.configText = "{not json";
    expect(validateDraft(d).some((e) => /not valid JSON/i.test(e))).toBe(true);
  });

  it("flags an unknown parameter against the schema", () => {
    const d = good();
    d.opponents[1]!.configText = '{"bogus":1}';
    const schema: TunerInfo = {
      id: "s",
      baselines: [],
      eval_rounds: 1,
      parameters: [{ name: "c", type: "float", bounds: [0, 2], default: 1.4 }],
      conditions: [],
      game_config: {},
    };
    expect(validateDraft(d, schema).some((e) => /unknown parameter "bogus"/.test(e))).toBe(true);
  });

  it("flags a family config missing a parameter the binary would require", () => {
    const d = good();
    d.opponents[1]!.configText = '{"family":"ucb1","q_init":"Parent"}';
    const schema: TunerInfo = {
      id: "s",
      baselines: [],
      eval_rounds: 1,
      parameters: [
        { name: "family", type: "categorical", choices: ["ucb1"], default: "ucb1" },
        { name: "q_init", type: "categorical", choices: ["Parent", "Infinity"], default: "Infinity" },
        { name: "c", type: "float", bounds: [0, 3], default: 1.4 },
        {
          name: "final_action",
          type: "categorical",
          choices: ["robust_child"],
          default: "robust_child",
        },
      ],
      conditions: [
        { if: { family: "ucb1" }, then: ["c"] },
        { if: { family: "ucb1" }, then: ["final_action"] },
      ],
      game_config: {},
    };
    const errors = validateDraft(d, schema);
    expect(errors.some((e) => /missing required parameters .*"c".*"final_action"/.test(e))).toBe(true);
  });
});

describe("activeParamNames", () => {
  const schema: TunerInfo = {
    id: "s",
    baselines: [],
    eval_rounds: 1,
    parameters: [
      { name: "family", type: "categorical", choices: ["ucb1", "ucb1_dm_nst"], default: "ucb1" },
      { name: "c", type: "float", bounds: [0, 3], default: 1.4 },
      { name: "nst_backoff_threshold", type: "int", bounds: [1, 10], default: 5 },
    ],
    conditions: [{ if: { family: "ucb1_dm_nst" }, then: ["nst_backoff_threshold"] }],
    game_config: {},
  };

  it("gates a conditioned parameter on its parent value", () => {
    expect(activeParamNames(schema, {}).has("nst_backoff_threshold")).toBe(false);
    expect(activeParamNames(schema, { family: "ucb1_dm_nst" }).has("nst_backoff_threshold")).toBe(
      true,
    );
    expect(activeParamNames(schema, {}).has("c")).toBe(true);
  });
});

describe("flattenDotted / nestDotted", () => {
  it("round-trips a nested object through dotted keys", () => {
    const nested = { size: { w: 7, h: 9 } };
    expect(flattenDotted(nested)).toEqual({ "size.w": 7, "size.h": 9 });
    expect(nestDotted({ "size.w": 7, "size.h": 9 })).toEqual(nested);
  });

  it("treats a scalar top-level key as its own leaf", () => {
    expect(flattenDotted({ size: 9 })).toEqual({ size: 9 });
    expect(nestDotted({ size: 9 })).toEqual({ size: 9 });
  });
});

describe("game_config round-trip", () => {
  const seed = (gameConfig?: JsonValue): JsonValue => ({
    schema_version: 1,
    objective_id: "atarigo-v1",
    game_kind: "atarigo",
    opponents: [
      {
        id: "schema-default",
        label: "Schema default",
        role: "default",
        weight: 1,
        config: { source: "schema_default" },
      },
      {
        id: "hist",
        label: "Historical",
        role: "historical_reference",
        weight: 1,
        config: { source: "inline", value: { c: 1.2 } },
      },
    ],
    start_distribution: { kind: "default_only" },
    ...(gameConfig !== undefined ? { game_config: gameConfig } : {}),
  });

  it("omits game_config when the objective has none", () => {
    const { draft, warnings } = draftFromContent(seed());
    expect(warnings).toEqual([]);
    expect(draftToContent(draft)).toEqual(seed());
    expect("game_config" in draftToContent(draft)).toBe(false);
  });

  it("round-trips an objective carrying game_config byte-identically", () => {
    const { draft, warnings } = draftFromContent(seed({ size: 9 }));
    expect(warnings).toEqual([]);
    expect(draft.gameConfig).toEqual({ size: 9 });
    expect(draftToContent(draft)).toEqual(seed({ size: 9 }));
  });
});

describe("validateDraft — game_config", () => {
  const atariDraft = (gameConfig: JsonValue): ObjectiveDraft => {
    const d = emptyDraft("atarigo");
    d.objectiveId = "atarigo-v1";
    d.opponents[1] = {
      id: "hist",
      label: "Historical",
      kind: "inline",
      weight: 1,
      config: { c: 1.2 },
      configText: '{"c":1.2}',
      configMode: "form",
    };
    d.gameConfig = flattenDotted(gameConfig);
    d.gameConfigText = JSON.stringify(gameConfig);
    return d;
  };

  it("accepts an in-bounds size", () => {
    expect(validateDraft(atariDraft({ size: 9 }), ATARIGO)).toEqual([]);
  });

  it("rejects an out-of-bounds size", () => {
    expect(
      validateDraft(atariDraft({ size: 25 }), ATARIGO).some((e) => /within \[3, 19\]/.test(e)),
    ).toBe(true);
  });

  it("rejects an unknown game-setup field", () => {
    expect(
      validateDraft(atariDraft({ variant: "x" }), ATARIGO).some((e) => /unknown field "variant"/.test(e)),
    ).toBe(true);
  });

  it("rejects a game_config equal to the default", () => {
    expect(
      validateDraft(atariDraft({ size: 13 }), ATARIGO).some((e) => /matches the default/.test(e)),
    ).toBe(true);
  });

  it("rejects any override for a fixed-board game", () => {
    const d = atariDraft({ size: 9 });
    const fixed: TunerInfo = { ...ATARIGO, game_config: {}, game_config_schema: undefined };
    expect(validateDraft(d, fixed).some((e) => /board is fixed/.test(e))).toBe(true);
  });
});

describe("slugKey", () => {
  it("makes a filesystem-safe key", () => {
    expect(slugKey("Nim Reference v1")).toBe("nim-reference-v1");
    expect(slugKey("  ")).toBe("objective");
  });
});
