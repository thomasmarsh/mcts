import { describe, expect, it } from "vitest";
import { buildPreset } from "../../src/tuner/models/preset-copy.js";

describe("buildPreset", () => {
  it("serialises a candidate config into a presets.json blob", () => {
    const r = buildPreset({
      candidateId: "candidate-0123456789abcdef00",
      gameKind: "nim",
      config: { family: "b", c: 1.7 },
    });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.preset).toEqual({
      id: "tuned-nim-0123456789ab",
      label: "Tuned nim (0123456789ab)",
      game: "nim",
      params: { family: "b", c: 1.7 },
    });
    expect(JSON.parse(r.text)).toEqual(r.preset);
  });

  it("refuses a candidate with no object config", () => {
    expect(buildPreset({ candidateId: "candidate-x", gameKind: "nim", config: null }).ok).toBe(
      false,
    );
    expect(buildPreset({ candidateId: "candidate-x", gameKind: "nim", config: [1, 2] }).ok).toBe(
      false,
    );
  });
});
