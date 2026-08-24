export interface TunerLaunchPolicyInput {
  nTrials: number;
  nWorkers: string;
  deterministic: boolean;
  seed: number;
  minPairs: number;
  maxPairs: number;
  pruningEnabled: boolean;
  reductionFactor: number;
  pruningStartupTrials: number;
  sigmaStop: string;
  tpeStartupTrials: number;
  maxIterations: string;
  maxTimeMs: string;
}

export function validateTunerLaunchPolicy(input: TunerLaunchPolicyInput): string | null {
  if (input.nTrials < 1) return "Target trials must be at least 1.";
  if (input.minPairs < 1 || input.minPairs > input.maxPairs) {
    return "Minimum pairs must be at least 1 and no greater than maximum pairs.";
  }
  const workers = input.nWorkers.trim();
  if (workers !== "" && (!/^\d+$/.test(workers) || Number(workers) < 1)) {
    return "Workers are concurrent evaluation slots and must be a positive whole number.";
  }
  return null;
}

/** Build only the launcher inputs that the resolved policy accepts. */
export function buildTunerOverrides(input: TunerLaunchPolicyInput): string[] {
  const overrides = [
    `optimizer.n_trials=${input.nTrials}`,
    `optimizer.deterministic=${input.deterministic ? "True" : "False"}`,
    `optimizer.seed=${input.seed}`,
    `optimizer.resource.min_pairs=${input.minPairs}`,
    `optimizer.resource.max_pairs=${input.maxPairs}`,
    `optimizer.sampler.startup_trials=${input.tpeStartupTrials}`,
    `optimizer.pruning.enabled=${input.pruningEnabled ? "True" : "False"}`,
  ];
  const workers = input.nWorkers.trim();
  if (workers !== "") overrides.push(`optimizer.n_workers=${workers}`);
  if (input.pruningEnabled) {
    overrides.push(`optimizer.pruning.reduction_factor=${input.reductionFactor}`);
    overrides.push(`optimizer.pruning.startup_trials=${input.pruningStartupTrials}`);
  }
  const sigmaStop = input.sigmaStop.trim();
  if (sigmaStop !== "") overrides.push(`rating.sigma_stop=${sigmaStop}`);
  const maxIterations = input.maxIterations.trim();
  if (maxIterations !== "") overrides.push(`target.max_iterations=${maxIterations}`);
  const maxTimeMs = input.maxTimeMs.trim();
  if (maxTimeMs !== "") overrides.push(`target.max_time_ms=${maxTimeMs}`);
  return overrides;
}
