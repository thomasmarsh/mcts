// profile-model.ts — the pure core of the launch-profile editor. A launch
// profile is a saved `{game, objective, constraints, efforts, budgets}`
// bundle a tuner run is started from (it is not an objective — it only
// references an `objective_key`). This module converts between the profile
// file JSON the server stores (`server/src/bench/tuner_profiles.rs`) and the
// editor's form-shaped `ProfileDraft`, and runs the same client-side checks
// the launch form applies before the debounced preflight round-trip.
//
// No rendering, no fetch. The constraint rows are a pure function of the
// game's tuning schema, so nothing here hardcodes a parameter or algorithm
// name.

import type { JsonValue } from "../../types.js";
import type { SpaceOverride } from "../tuner-types.js";
import {
  deriveConstraints,
  emptyRows,
  type ConstraintRows,
  type ParamSchema,
} from "./constraint-editor-model.js";

export type ProfilePhase = "tuning" | "validation" | "production";
export type EffortUnit = "iterations" | "time_ms";

export const PROFILE_PHASES: readonly ProfilePhase[] = ["tuning", "validation", "production"];

/** One phase's search effort as the editor holds it: a raw value string and
 * a unit. An empty / non-positive value means "fall back to the CLI default"
 * and is not persisted. */
export interface ProfileEffortDraft {
  value: string;
  unit: EffortUnit;
}

/** The five budget knobs the editor exposes, all held as raw strings. The
 * first three are always written; `cohortSize` / `finalists` only when set. */
export interface ProfileBudgetsDraft {
  tuningPairBudget: string;
  validationPairBudget: string;
  productionValidationPairs: string;
  cohortSize: string;
  finalists: string;
}

export interface ProfileDraft {
  profileId: string;
  gameKind: string;
  objectiveKey: string;
  constraintRows: ConstraintRows;
  efforts: Record<ProfilePhase, ProfileEffortDraft>;
  budgets: ProfileBudgetsDraft;
}

export interface ProfileDraftParse {
  draft: ProfileDraft;
  warnings: string[];
}

/** Mirrors the launch form's own defaults so a fresh profile is launchable. */
const DEFAULT_BUDGETS: ProfileBudgetsDraft = {
  tuningPairBudget: "32",
  validationPairBudget: "24",
  productionValidationPairs: "8",
  cohortSize: "",
  finalists: "",
};

function emptyEfforts(): Record<ProfilePhase, ProfileEffortDraft> {
  return {
    tuning: { value: "", unit: "iterations" },
    validation: { value: "", unit: "iterations" },
    production: { value: "", unit: "iterations" },
  };
}

export function emptyProfileDraft(gameKind = ""): ProfileDraft {
  return {
    profileId: "",
    gameKind,
    objectiveKey: "",
    constraintRows: {},
    efforts: emptyEfforts(),
    budgets: { ...DEFAULT_BUDGETS },
  };
}

/** A filesystem-safe key from a profile id (`^[a-zA-Z0-9._-]+$`). */
export function profileSlugKey(profileId: string): string {
  return (
    profileId
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "profile"
  );
}

/** `"5"` → `5`, `""` / junk / non-positive → undefined. */
function posInt(raw: string): number | undefined {
  const t = raw.trim();
  if (t === "") return undefined;
  const n = Number(t);
  if (!Number.isFinite(n) || n <= 0) return undefined;
  return Math.trunc(n);
}

function asObject(value: JsonValue | undefined): Record<string, JsonValue> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, JsonValue>)
    : null;
}

/** Reverse of the constraint editor: turn a stored `constraints` blob (the
 * unified array form or the bare `{ name: op }` sugar map) back into editor
 * rows against `schema`. Unknown parameters are dropped and reported. */
export function rowsFromConstraints(
  schema: ParamSchema,
  constraints: JsonValue | undefined,
): { rows: ConstraintRows; warnings: string[] } {
  const rows = emptyRows(schema);
  const warnings: string[] = [];
  if (constraints == null) return { rows, warnings };

  const entries: Array<{ when?: Record<string, JsonValue>; set: Record<string, JsonValue> }> = [];
  if (Array.isArray(constraints)) {
    for (const raw of constraints) {
      const obj = asObject(raw);
      if (!obj) {
        warnings.push("dropped a constraint entry that was not an object");
        continue;
      }
      const set = asObject(obj.set) ?? obj;
      const when = asObject(obj.when) ?? undefined;
      entries.push({ set: set as Record<string, JsonValue>, when });
    }
  } else {
    const obj = asObject(constraints);
    if (obj) entries.push({ set: obj });
  }

  for (const entry of entries) {
    const when: Record<string, string[]> = {};
    for (const [parent, values] of Object.entries(entry.when ?? {})) {
      if (Array.isArray(values)) when[parent] = values.map(String);
    }
    for (const [name, opRaw] of Object.entries(entry.set)) {
      const spec = schema.parameters.find((p) => p.name === name);
      if (!spec) {
        warnings.push(`dropped a constraint on unknown parameter "${name}"`);
        continue;
      }
      const op = asObject(opRaw) as SpaceOverride | null;
      const base = rows[name] ?? emptyRows({ parameters: [spec], conditions: [] })[name]!;
      if (op && "fix" in op) {
        rows[name] = { ...base, mode: "fix", fix: String(op.fix), when };
      } else if (op && "range" in op && Array.isArray(op.range)) {
        rows[name] = {
          ...base,
          mode: "range",
          low: String(op.range[0]),
          high: String(op.range[1]),
          when,
        };
      } else if (op && "choices" in op && Array.isArray(op.choices)) {
        rows[name] = { ...base, mode: "choices", retained: op.choices.map(String), when };
      } else {
        warnings.push(`dropped an unrecognised constraint on "${name}"`);
      }
    }
  }
  return { rows, warnings };
}

/** Build the profile file JSON from a draft. `constraints` / `efforts` are
 * omitted entirely when empty; `budgets` always carries the three core
 * pair-budget values plus `cohort_size` / `finalists` when set. */
export function draftToProfileContent(draft: ProfileDraft, schema: ParamSchema): JsonValue {
  const content: Record<string, JsonValue> = {
    profile_id: draft.profileId.trim(),
    game_kind: draft.gameKind,
    objective_key: draft.objectiveKey,
  };

  const constraints = deriveConstraints(schema, draft.constraintRows).constraints;
  if (constraints.length > 0) content.constraints = constraints as unknown as JsonValue;

  const efforts: Record<string, JsonValue> = {};
  for (const phase of PROFILE_PHASES) {
    const value = posInt(draft.efforts[phase].value);
    if (value !== undefined) {
      efforts[phase] = { kind: draft.efforts[phase].unit, value };
    }
  }
  if (Object.keys(efforts).length > 0) content.efforts = efforts;

  const budgets: Record<string, JsonValue> = {
    tuning_pair_budget: posInt(draft.budgets.tuningPairBudget) ?? 32,
    validation_pair_budget: posInt(draft.budgets.validationPairBudget) ?? 24,
    production_validation_pairs: posInt(draft.budgets.productionValidationPairs) ?? 8,
  };
  const cohort = posInt(draft.budgets.cohortSize);
  if (cohort !== undefined) budgets.cohort_size = cohort;
  const finalists = posInt(draft.budgets.finalists);
  if (finalists !== undefined) budgets.finalists = finalists;
  content.budgets = budgets;

  return content;
}

/** Parse a stored profile file into an editor draft. Tolerant: unparseable
 * pieces are dropped and reported in `warnings` rather than throwing. */
export function draftFromProfileContent(
  content: JsonValue,
  schema: ParamSchema,
  fallbackGame = "",
): ProfileDraftParse {
  const warnings: string[] = [];
  const root = asObject(content);
  if (!root) {
    return { draft: emptyProfileDraft(fallbackGame), warnings: ["profile file is not a JSON object"] };
  }

  const str = (name: string): string => {
    const v = root[name];
    return typeof v === "string" ? v : "";
  };

  const draft = emptyProfileDraft(str("game_kind") || fallbackGame);
  draft.profileId = str("profile_id");
  draft.objectiveKey = str("objective_key");

  const { rows, warnings: cw } = rowsFromConstraints(schema, root.constraints);
  draft.constraintRows = rows;
  warnings.push(...cw);

  const efforts = asObject(root.efforts);
  if (efforts) {
    for (const phase of PROFILE_PHASES) {
      const e = asObject(efforts[phase]);
      if (!e) continue;
      const unit = e.kind === "time_ms" ? "time_ms" : "iterations";
      const value = typeof e.value === "number" ? e.value : Number(e.value);
      if (Number.isFinite(value)) draft.efforts[phase] = { value: String(value), unit };
      else warnings.push(`dropped a ${phase} effort with no numeric value`);
    }
  }

  const budgets = asObject(root.budgets);
  if (budgets) {
    const num = (name: string): string | null => {
      const v = budgets[name];
      const n = typeof v === "number" ? v : Number(v);
      return v === undefined ? null : Number.isFinite(n) ? String(Math.trunc(n)) : null;
    };
    draft.budgets = {
      tuningPairBudget: num("tuning_pair_budget") ?? DEFAULT_BUDGETS.tuningPairBudget,
      validationPairBudget: num("validation_pair_budget") ?? DEFAULT_BUDGETS.validationPairBudget,
      productionValidationPairs:
        num("production_validation_pairs") ?? DEFAULT_BUDGETS.productionValidationPairs,
      cohortSize: num("cohort_size") ?? "",
      finalists: num("finalists") ?? "",
    };
  }

  return { draft, warnings };
}

/** Client-side checks mirroring the launch form: a game and objective are
 * picked, every constraint row is expressible, the core budgets are positive
 * integers, and a same-unit tuning/validation effort never exceeds
 * production. The server preflight (via `validateProfile`) is authoritative. */
export function validateProfileDraft(draft: ProfileDraft, schema: ParamSchema): string[] {
  const errors: string[] = [];
  if (draft.gameKind === "") errors.push("pick a game");
  if (draft.objectiveKey === "") errors.push("pick an objective");

  for (const err of deriveConstraints(schema, draft.constraintRows).errors) {
    errors.push(`constraint ${err}`);
  }

  const budgetLabels: Array<[keyof ProfileBudgetsDraft, string]> = [
    ["tuningPairBudget", "tuning pair budget"],
    ["validationPairBudget", "validation pair budget"],
    ["productionValidationPairs", "production validation pairs"],
  ];
  for (const [key, label] of budgetLabels) {
    if (posInt(draft.budgets[key]) === undefined) errors.push(`${label} must be a positive integer`);
  }

  const prod = posInt(draft.efforts.production.value);
  if (prod !== undefined) {
    for (const phase of ["tuning", "validation"] as const) {
      const value = posInt(draft.efforts[phase].value);
      if (
        value !== undefined &&
        draft.efforts[phase].unit === draft.efforts.production.unit &&
        value > prod
      ) {
        errors.push(`${phase} effort (${value}) cannot exceed production effort (${prod})`);
      }
    }
  }
  return errors;
}
