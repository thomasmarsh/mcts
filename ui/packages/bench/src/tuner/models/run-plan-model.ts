// run-plan-model.ts — pure derivation behind the launch form's read-only
// "Run plan" panel. The server (`tuner_cli plan`) already did every
// resolution; this only reshapes the `RunPlan` response into the rows and
// stat tiles the panel renders. No rendering, no `fetch`.

import type { RunPlan, RunPlanOpponent, RunPlanParameter } from "../tuner-types.js";
import type { KpiItem } from "../primitives/KpiRow.js";

export interface RunPlanSummary {
  /** True once resolution got far enough to carry the panel / space / epoch —
   * i.e. the request is at least structurally launchable. */
  resolved: boolean;
  errors: string[];
  gameKind: string | null;
  objectiveId: string | null;
  opponents: RunPlanOpponent[];
  /** The effective `family` choice set after exclusions / overrides. */
  families: string[];
  excludedFamilies: string[];
  parameters: RunPlanParameter[];
  gameConfig: string | null;
  gameConfigIsOverride: boolean;
  epochFingerprint: string | null;
  effortKpis: KpiItem[];
  budgetKpis: KpiItem[];
}

const EMPTY: RunPlanSummary = {
  resolved: false,
  errors: [],
  gameKind: null,
  objectiveId: null,
  opponents: [],
  families: [],
  excludedFamilies: [],
  parameters: [],
  gameConfig: null,
  gameConfigIsOverride: false,
  epochFingerprint: null,
  effortKpis: [],
  budgetKpis: [],
};

function effort(value: { kind: string; value: number } | undefined): string {
  if (!value) return "default";
  return value.kind === "time_ms" ? `${value.value} ms` : `${value.value} iters`;
}

export function summarizeRunPlan(plan: RunPlan | undefined): RunPlanSummary {
  if (!plan) return EMPTY;
  const budgets = plan.budgets;
  return {
    resolved: plan.opponents !== undefined,
    errors: plan.errors ?? [],
    gameKind: plan.game_kind ?? null,
    objectiveId: plan.objective_id ?? null,
    opponents: plan.opponents ?? [],
    families: (plan.space?.families ?? []).map((f) => String(f)),
    excludedFamilies: plan.space?.excluded_families ?? [],
    parameters: plan.space?.parameters ?? [],
    gameConfig: plan.game_config ?? null,
    gameConfigIsOverride: plan.game_config_is_override ?? false,
    epochFingerprint: plan.epoch?.fingerprint ?? null,
    effortKpis: plan.efforts
      ? [
          { label: "tuning effort", value: effort(plan.efforts.tuning) },
          { label: "validation effort", value: effort(plan.efforts.validation) },
          { label: "production effort", value: effort(plan.efforts.production) },
        ]
      : [],
    budgetKpis: budgets
      ? [
          {
            label: "initial cohort",
            value: `${budgets.derived.initial_cohort_pairs} pairs`,
            hint: `${budgets.cohort_size} candidates × ${budgets.tuning_pairs} pairs`,
          },
          { label: "tuning budget", value: `${budgets.tuning_pair_budget} pairs` },
          {
            label: "validation width",
            value: `${budgets.derived.validation_pairs_per_finalist} pairs/finalist`,
            hint: `${budgets.validation_pair_budget} budget ÷ ${budgets.finalists} finalists`,
          },
          { label: "production", value: `${budgets.derived.production_pairs} pairs` },
        ]
      : [],
  };
}
