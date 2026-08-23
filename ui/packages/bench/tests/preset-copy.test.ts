import { describe, expect, it, vi } from "vitest";
import {
  buildPresetSpec,
  candidatePresetSource,
  copyPreset,
  opponentPresetSource,
  safePresetId,
} from "../src/tuning/preset-copy.js";
import type { PresetSource } from "../src/tuning/preset-copy.js";
import type { TuningPoolAnchor, TuningPoolRevision, TuningTrialDetailView } from "../src/types.js";

const candidate: Pick<TuningTrialDetailView, "trial_id" | "trial_number" | "config"> = {
  trial_id: "trial / A",
  trial_number: 7,
  config: { family: "ucb1", c: 1.4, mcgs: false },
};
const anchor: TuningPoolAnchor = {
  anchor_ordinal: 2, anchor_id: "pool/A", config: { family: "rave", threshold: 50, mcgs: true },
  rating: { mu: 22, sigma: 1 }, provenance: "candidate", insertion_reason: "promotion", source_trial_id: "trial-9",
};
const revision: Pick<TuningPoolRevision, "display_ordinal"> = { display_ordinal: 3 };

describe("preset copy", () => {
  it("serializes candidate and opponent snapshots in the frozen deterministic shape", () => {
    const candidateBuild = buildPresetSpec(candidatePresetSource(candidate));
    const opponentBuild = buildPresetSpec(opponentPresetSource(anchor, revision, { max_time_ms: 250 }));
    expect(candidateBuild).toMatchObject({ enabled: true, preset: { id: "candidate-trial_x20__x2f__x20__x41_", description: "Candidate snapshot from trial 7 (trial / A).", params: candidate.config, max_iterations: 10_000, threads: 1, use_transpositions: true } });
    expect(opponentBuild).toMatchObject({ enabled: true, preset: { id: "opponent-pool_x2f__x41_", description: "Opponent snapshot pool/A from pool revision 3.", params: anchor.config, max_time_ms: 250, threads: 1, use_transpositions: true } });
    if (!candidateBuild.enabled || !opponentBuild.enabled) throw new Error("fixtures must be valid");
    expect(candidateBuild.text).toBe(`{
    "id": "candidate-trial_x20__x2f__x20__x41_",
    "label": "Tuned candidate",
    "description": "Candidate snapshot from trial 7 (trial / A).",
    "params": {
        "c": 1.4,
        "family": "ucb1",
        "mcgs": false
    },
    "max_iterations": 10000,
    "threads": 1,
    "use_transpositions": true
}`);
    expect(opponentBuild.text).toBe(`{
    "id": "opponent-pool_x2f__x41_",
    "label": "Pool opponent",
    "description": "Opponent snapshot pool/A from pool revision 3.",
    "params": {
        "family": "rave",
        "mcgs": true,
        "threshold": 50
    },
    "max_time_ms": 250,
    "threads": 1,
    "use_transpositions": true
}`);
    expect(candidate.config).toEqual({ family: "ucb1", c: 1.4, mcgs: false });
    expect(anchor.config).toEqual({ family: "rave", threshold: 50, mcgs: true });
  });

  it("uses the recorded mcgs parameter as capability, not its selected boolean value", () => {
    for (const [params, expected] of [
      [{ family: "ucb1", mcgs: false }, true],
      [{ family: "ucb1", mcgs: true }, true],
      [{ family: "ucb1" }, false],
    ] as const) {
      const result = buildPresetSpec({ kind: "candidate", sourceId: "source", sourceDescription: "Recorded.", params });
      expect(result.enabled && result.preset.use_transpositions).toBe(expected);
    }
  });

  it("selects exactly one valid budget and reports legacy or invalid snapshots with typed reasons", () => {
    const cases: Array<{ input: PresetSource; expected: string; enabled: boolean }> = [
      { input: { kind: "candidate", sourceId: "a", sourceDescription: "", params: { family: "ucb1" }, max_iterations: 12 }, expected: "max_iterations", enabled: true },
      { input: { kind: "candidate", sourceId: "a", sourceDescription: "", params: { family: "ucb1" }, max_time_ms: 12 }, expected: "max_time_ms", enabled: true },
      { input: { kind: "candidate", sourceId: "a", sourceDescription: "", params: { family: "ucb1" } }, expected: "max_iterations", enabled: true },
      { input: { kind: "candidate", sourceId: "a", sourceDescription: "", params: { family: "ucb1" }, max_iterations: 0 }, expected: "invalid_budget", enabled: false },
      { input: { kind: "candidate", sourceId: "a", sourceDescription: "", params: { family: "ucb1" }, max_time_ms: 1, max_iterations: 1 }, expected: "multiple_budgets", enabled: false },
      { input: { kind: "candidate", sourceId: "a", sourceDescription: "", params: null }, expected: "legacy_missing_config", enabled: false },
      { input: { kind: "candidate", sourceId: "a", sourceDescription: "", params: {} }, expected: "missing_family", enabled: false },
      { input: { kind: "candidate", sourceId: "a", sourceDescription: "", params: { family: "ucb1", mcgs: "yes" } }, expected: "invalid_mcgs", enabled: false },
    ];
    for (const { input, expected, enabled } of cases) {
      const result = buildPresetSpec(input);
      expect(result.enabled).toBe(enabled);
      if (result.enabled) expect(Object.hasOwn(result.preset, expected)).toBe(true);
      else expect(result.reason.code).toBe(expected);
    }
    expect(safePresetId("candidate", "")).toBeNull();
    expect(safePresetId("opponent", "A_A")).toBe("opponent-_x41__x5f__x41_");
  });

  it("returns accessible clipboard success, rejection, and disabled states", async () => {
    const valid = buildPresetSpec(candidatePresetSource(candidate));
    const writer = { writeText: vi.fn().mockResolvedValue(undefined) };
    await expect(copyPreset(valid, writer)).resolves.toEqual({ status: "success", announcement: "Preset copied to clipboard." });
    expect(writer.writeText).toHaveBeenCalledWith(expect.stringContaining('"id": "candidate-trial'));
    await expect(copyPreset(valid, { writeText: vi.fn().mockRejectedValue(new Error("denied")) })).resolves.toEqual({ status: "failure", announcement: "Could not copy preset to clipboard." });
    await expect(copyPreset(buildPresetSpec({ kind: "candidate", sourceId: "legacy", sourceDescription: "", params: null }), writer)).resolves.toMatchObject({ status: "disabled" });
  });
});
