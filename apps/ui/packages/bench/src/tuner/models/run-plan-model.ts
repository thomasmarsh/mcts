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
  /** Residual domain of the root `algorithm` categorical after constraints. */
  algorithms: string[];
  /** Categorical axes (`select`, `simulate`, …) whose domain was narrowed for
   * this run, as `name → residual values`. Axes left at full width are omitted. */
  narrowedVariants: { name: string; values: string[] }[];
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
  algorithms: [],
  narrowedVariants: [],
  parameters: [],
  gameConfig: null,
  gameConfigIsOverride: false,
  epochFingerprint: null,
  effortKpis: [],
  budgetKpis: [],
};

/** A `set` constraint that pins a non-`algorithm` categorical axis to a
 * `choices` subset is exactly the "this run narrows a variant set" signal. */
function narrowedVariants(constraints: unknown[]): { name: string; values: string[] }[] {
  const out: { name: string; values: string[] }[] = [];
  for (const entry of constraints) {
    const set = (entry as { set?: Record<string, unknown> } | null)?.set;
    if (!set || typeof set !== "object") continue;
    for (const [axis, spec] of Object.entries(set)) {
      if (axis === "algorithm") continue;
      const choices = (spec as { choices?: unknown } | null)?.choices;
      if (Array.isArray(choices)) out.push({ name: axis, values: choices.map((c) => String(c)) });
    }
  }
  return out;
}

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
    algorithms: (plan.space?.algorithms ?? []).map((a) => String(a)),
    narrowedVariants: narrowedVariants(plan.space?.constraints ?? []),
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
