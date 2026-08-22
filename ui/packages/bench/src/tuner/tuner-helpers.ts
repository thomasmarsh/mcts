/** Shared helpers for tuner-run detail components — extracting OpenSkill
 * metrics from a `TrialRow`'s `extra` payload, formatting scores, and
 * computing confidence bands.
 *
 * Every trial from the optuna-driven pipeline stores its OpenSkill estimate
 * in `extra`: `{"mu": ..., "sigma": ..., "opponents": [...], "git_sha": ...}`,
 * with `cost = -(mu - 3*sigma)` on the wire. */

import type { TrialRow, TunerInfo } from "../index.js";

// ---------------------------------------------------------------------------
// Optuna-metric extraction
// ---------------------------------------------------------------------------

/** The OpenSkill `mu` for a trial, or `null` if not yet evaluated. */
export function trialMu(t: TrialRow): number | null {
  const extra = t.extra as Record<string, unknown> | null;
  const mu = extra?.mu;
  return typeof mu === "number" ? mu : null;
}

/** The OpenSkill `sigma` for a trial, or `null` if not yet evaluated. */
export function trialSigma(t: TrialRow): number | null {
  const extra = t.extra as Record<string, unknown> | null;
  const sigma = extra?.sigma;
  return typeof sigma === "number" ? sigma : null;
}

/** The primary score `mu - 3*sigma` (higher is better). Returns `null`
 * when the trial hasn't been evaluated yet (cost is null). */
export function trialScore(t: TrialRow): number | null {
  if (t.cost === null) return null;
  const mu = trialMu(t);
  const sigma = trialSigma(t);
  if (mu !== null && sigma !== null) return mu - 3 * sigma;
  // Fallback for callers that still read old wire data: cost =
  // -(mu - 3*sigma), so -cost recovers the original score.
  return -t.cost;
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

export function fmtScore(score: number | null): string {
  return score === null ? "—" : score.toFixed(3);
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

/** 95% confidence band for an OpenSkill rating: `mu ± 2*sigma`. Returns
 * `null` for a trial not yet evaluated. */
export function skillInterval(
  t: TrialRow,
): { lower: number; upper: number } | null {
  const mu = trialMu(t);
  const sigma = trialSigma(t);
  if (mu === null || sigma === null) return null;
  return { lower: mu - 2 * sigma, upper: mu + 2 * sigma };
}

// ---------------------------------------------------------------------------
// Resolve rounds from the launch config
// ---------------------------------------------------------------------------

/** The `rounds` actually used for this run's trials — an operator can
 * `--override target.rounds=N` away from the tuner's declared default at
 * launch time, and only the run's own launch config (not the tuner metadata)
 * reflects that. Falls back to the tuner's default when the run didn't
 * override it. */
export function resolveRounds(
  launchConfig: unknown,
  tuner: Pick<TunerInfo, "eval_rounds"> | null,
): number {
  const overrides = (launchConfig as { overrides?: unknown } | null)
    ?.overrides;
  if (Array.isArray(overrides)) {
    for (const o of overrides) {
      if (typeof o === "string" && o.startsWith("target.rounds=")) {
        const n = Number(o.slice("target.rounds=".length));
        if (Number.isFinite(n) && n > 0) return n;
      }
    }
  }
  return tuner?.eval_rounds ?? 20;
}

// ---------------------------------------------------------------------------
// Grouping (intensifier pooling)
// ---------------------------------------------------------------------------

/** A group of trials sharing the same `config` and baseline instance,
 * pooled to produce a tighter score estimate. */
export interface TrialGroup {
  trials: TrialRow[];
  meanMu: number;
  meanSigma: number;
  meanScore: number;
  ci: { lower: number; upper: number };
}

/** Grouping key for CI pooling — same `config` *and* same baseline instance
 * (identified from `extra.instance`). */
export function groupKey(t: TrialRow): string {
  const extra = t.extra as { instance?: unknown } | null;
  const instance = typeof extra?.instance === "string" ? extra.instance : "";
  return JSON.stringify(t.config) + "::" + instance;
}

/** Build pooled trial groups from scored trials. Each group's score is the
 * mean `mu - 3*sigma` and the CI is `mean mu ± 2*mean sigma`. */
export function buildGroups(scored: TrialRow[]): Map<string, TrialGroup> {
  const byKey = new Map<string, TrialRow[]>();
  for (const t of scored) {
    const key = groupKey(t);
    const list = byKey.get(key);
    if (list) list.push(t);
    else byKey.set(key, [t]);
  }

  const result = new Map<string, TrialGroup>();
  for (const [key, trials] of byKey) {
    const meanMu =
      trials.reduce((s, t) => s + (trialMu(t) ?? 0), 0) / trials.length;
    const meanSigma =
      trials.reduce((s, t) => s + (trialSigma(t) ?? 0), 0) / trials.length;
    const meanScore = meanMu - 3 * meanSigma;
    result.set(key, {
      trials,
      meanMu,
      meanSigma,
      meanScore,
      ci: { lower: meanMu - 2 * meanSigma, upper: meanMu + 2 * meanSigma },
    });
  }
  return result;
}