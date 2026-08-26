// family-activation.test.ts — unit tests for the parent/child activation
// walk directly against `fixtureTunerInfo`, independent of any rendering,
// per AGENTS.md's "test the algorithm on a small hand-verifiable input"
// precedent (there applied to `examples/*.rs`, here to the equivalent TS
// activation logic).

import { describe, expect, it } from "vitest";
import { activeNames, withDefaultsFilled } from "../src/family-activation.js";
import { fixtureTunerInfo } from "./schema-fixture.js";

const { parameters, conditions } = fixtureTunerInfo;

describe("activeNames", () => {
  it("always includes roots, regardless of their value", () => {
    const active = activeNames(parameters, conditions, { family: "random" });
    expect(active.has("family")).toBe(true);
    expect(active.has("contempt")).toBe(true);
  });

  it("activates a direct child once its parent's value matches", () => {
    const active = activeNames(parameters, conditions, { family: "ucb1" });
    expect(active.has("c")).toBe(true);
    expect(active.has("rave_ucb")).toBe(false);
  });

  it("leaves a child inactive when no parent value matches", () => {
    const active = activeNames(parameters, conditions, { family: "random" });
    expect(active.has("c")).toBe(false);
    expect(active.has("rave_ucb")).toBe(false);
  });

  it("activates a grandchild across two passes (family -> rave_ucb -> c)", () => {
    const active = activeNames(parameters, conditions, { family: "rave_ucb", rave_ucb: "tuned" });
    expect(active.has("rave_ucb")).toBe(true);
    expect(active.has("c")).toBe(true);
  });

  it("does not activate the grandchild if the intermediate value doesn't match", () => {
    // `rave_ucb` itself is active (its parent `family` matches), but its
    // value is outside `c`'s `[ucb1, tuned]` condition -- not a real choice
    // for this fixture, but exercises that activation checks the actual
    // value, not just parent activation.
    const active = activeNames(parameters, conditions, { family: "rave_ucb", rave_ucb: "other" });
    expect(active.has("rave_ucb")).toBe(true);
    expect(active.has("c")).toBe(false);
  });

  it("treats a field named by two conditions as active if either is satisfied", () => {
    const viaFamily = activeNames(parameters, conditions, { family: "ucb1" });
    expect(viaFamily.has("c")).toBe(true);
    const viaRaveUcb = activeNames(parameters, conditions, {
      family: "rave_ucb",
      rave_ucb: "tuned",
    });
    expect(viaRaveUcb.has("c")).toBe(true);
  });
});

describe("withDefaultsFilled", () => {
  it("fills defaults for every active field, given only a family seed", () => {
    const result = withDefaultsFilled(parameters, conditions, { family: "ucb1" });
    expect(result).toEqual({
      family: "ucb1",
      contempt: "off",
      c: 1.4142135623730951,
    });
  });

  it("fills a grandchild's default across the same call (family -> rave_ucb -> c)", () => {
    const result = withDefaultsFilled(parameters, conditions, { family: "rave_ucb" });
    expect(result.rave_ucb).toBe("ucb1");
    expect(result.c).toBe(1.4142135623730951);
  });

  it("never overwrites a value already present", () => {
    const result = withDefaultsFilled(parameters, conditions, { family: "ucb1", c: 2.5 });
    expect(result.c).toBe(2.5);
  });

  it("does not fill defaults for inactive fields", () => {
    const result = withDefaultsFilled(parameters, conditions, { family: "random" });
    expect(result).toEqual({ family: "random", contempt: "off" });
    expect(result.c).toBeUndefined();
    expect(result.rave_ucb).toBeUndefined();
  });

  it("reveals a newly-activated field's default without disturbing existing values", () => {
    const withContemptOn = withDefaultsFilled(parameters, conditions, {
      family: "ucb1",
      c: 2.5,
      contempt: "on",
    });
    expect(withContemptOn.c).toBe(2.5);
    expect(withContemptOn.contempt_factor).toBe(0);
  });
});
