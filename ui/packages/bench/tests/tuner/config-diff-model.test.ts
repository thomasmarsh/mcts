import { describe, expect, it } from "vitest";
import {
  configDiffRows,
  flattenConfig,
  schemaDefaults,
} from "../../src/tuner/models/config-diff-model.js";
import type { TunerParameter } from "../../src/types.js";

describe("flattenConfig", () => {
  it("flattens nested objects to dotted leaf paths", () => {
    expect(flattenConfig({ a: 1, b: { c: "x", d: true } })).toEqual({
      a: "1",
      "b.c": "x",
      "b.d": "true",
    });
  });
});

describe("schemaDefaults", () => {
  it("reads default, falling back to value", () => {
    const params: TunerParameter[] = [
      { name: "c", type: "float", default: 1.4 },
      { name: "policy", type: "categorical", default: "ucb1" },
      { name: "k", type: "constant", value: 3 },
      { name: "nodef", type: "float" },
    ];
    expect(schemaDefaults(params)).toEqual({ c: "1.4", policy: "ucb1", k: "3" });
  });
});

describe("configDiffRows", () => {
  it("pairs base against candidate and flags changes", () => {
    const rows = configDiffRows({ c: "1.4", policy: "ucb1" }, { c: 1.7 });
    expect(rows).toEqual([
      { path: "c", base: "1.4", candidate: "1.7", changed: true },
      { path: "policy", base: "ucb1", candidate: null, changed: true },
    ]);
  });

  it("treats a null config as empty", () => {
    expect(configDiffRows({ c: "1.4" }, null)).toEqual([
      { path: "c", base: "1.4", candidate: null, changed: true },
    ]);
  });
});
