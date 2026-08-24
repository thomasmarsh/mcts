import { describe, expect, it } from "vitest";
import { buildTunerOverrides, validateTunerLaunchPolicy, type TunerLaunchPolicyInput } from "../src/tuner-launch-policy.js";

const input = (overrides: Partial<TunerLaunchPolicyInput> = {}): TunerLaunchPolicyInput => ({
  nTrials: 12,
  nWorkers: "",
  deterministic: true,
  seed: 4,
  minPairs: 2,
  maxPairs: 6,
  pruningEnabled: false,
  reductionFactor: 3,
  pruningStartupTrials: 5,
  sigmaStop: "",
  tpeStartupTrials: 3,
  maxIterations: "",
  maxTimeMs: "",
  ...overrides,
});

describe("tuner launch policy", () => {
  it("builds resolved-policy overrides with pruning disabled", () => {
    const value = input();
    expect(validateTunerLaunchPolicy(value)).toBeNull();
    expect(buildTunerOverrides(value)).toEqual([
      "optimizer.n_trials=12",
      "optimizer.deterministic=True",
      "optimizer.seed=4",
      "optimizer.resource.min_pairs=2",
      "optimizer.resource.max_pairs=6",
      "optimizer.sampler.startup_trials=3",
      "optimizer.pruning.enabled=False",
    ]);
  });

  it("allows parallel pair slots and includes pruning controls when enabled", () => {
    const value = input({ pruningEnabled: true, nWorkers: "3", sigmaStop: "2" });
    expect(validateTunerLaunchPolicy(value)).toBeNull();
    expect(buildTunerOverrides(value)).toEqual(expect.arrayContaining([
      "optimizer.n_workers=3",
      "optimizer.pruning.enabled=True",
      "optimizer.pruning.reduction_factor=3",
      "optimizer.pruning.startup_trials=5",
      "rating.sigma_stop=2",
    ]));
  });

  it("rejects an invalid explicit evaluation-slot count", () => {
    expect(validateTunerLaunchPolicy(input({ nWorkers: "1.5" }))).toBe(
      "Workers are concurrent evaluation slots and must be a positive whole number.",
    );
  });

  it("rejects inverted pair bounds", () => {
    expect(validateTunerLaunchPolicy(input({ minPairs: 7, maxPairs: 6 }))).toBe(
      "Minimum pairs must be at least 1 and no greater than maximum pairs.",
    );
  });
});
