// constraint-editor-model.ts — the pure core of the schema-driven constraint
// editor that replaces the launch form's free-text "Constrain parameters"
// textarea. Given a game's tuning schema (`tuner.parameters` +
// `tuner.conditions` from `GET /api/bench/tuner/kinds`) and the editor's
// per-parameter row state, it derives the unified `constraints` wire form and
// applies the same narrow-not-widen / non-empty-residual checks as the server
// (`tuner/src/tuner_cli/constraints.py`), so the operator sees an error before
// the debounced preflight round-trip.
//
// Zero hardcoded parameter or algorithm names: everything here is a function
// of the schema, which is what lets the editor survive future schema changes.
// No rendering, no fetch.

import type { TunerCondition, TunerParameter } from "../../types.js";
import type { Constraint, SpaceOverride } from "../tuner-types.js";

/** The schema shape this module needs — satisfied by `TunerInfo`. */
export interface ParamSchema {
  parameters: TunerParameter[];
  conditions: TunerCondition[];
}

/** How a single parameter's domain is being narrowed. `free` emits nothing. */
export type ConstraintMode = "free" | "fix" | "range" | "choices";

/** One editor row. Text inputs are held raw so a half-typed value doesn't
 * throw; `retained` is the still-allowed subset for `choices` mode; `when`
 * maps a categorical parameter to the values under which this row applies
 * (empty ⇒ unconditional). */
export interface ConstraintRow {
  mode: ConstraintMode;
  fix: string;
  low: string;
  high: string;
  retained: string[];
  when: Record<string, string[]>;
}

export type ConstraintRows = Record<string, ConstraintRow>;

export interface ConstraintAxisGroup {
  /** `null` is the top group: `algorithm`, the axis categoricals, and any
   * parameter no condition gates. Otherwise the owning axis' parent key(s). */
  axis: string | null;
  parameters: TunerParameter[];
}

export interface ConstraintDerivation {
  constraints: Constraint[];
  errors: string[];
}

const NUMERIC_TYPES = new Set(["float", "int"]);

function isNumeric(p: TunerParameter): boolean {
  return NUMERIC_TYPES.has(p.type) && Array.isArray(p.bounds);
}

function isCategorical(p: TunerParameter): boolean {
  return Array.isArray(p.choices);
}

/** The narrowing modes an operator can pick for a parameter. A
 * schema-constant parameter is not editable at all. */
export function modesFor(p: TunerParameter): ConstraintMode[] {
  if (p.type === "constant") return ["free"];
  if (isNumeric(p)) return ["free", "fix", "range"];
  if (isCategorical(p)) return ["free", "fix", "choices"];
  return ["free"];
}

/** A fresh, unconstrained row seeded from the parameter's schema domain
 * (`retained` starts as every choice so unticking is the edit). */
export function emptyRow(p: TunerParameter): ConstraintRow {
  return {
    mode: "free",
    fix: "",
    low: p.bounds ? String(p.bounds[0]) : "",
    high: p.bounds ? String(p.bounds[1]) : "",
    retained: [...(p.choices ?? [])],
    when: {},
  };
}

export function emptyRows(schema: ParamSchema): ConstraintRows {
  return Object.fromEntries(schema.parameters.map((p) => [p.name, emptyRow(p)]));
}

/** Parameters grouped by the axis that gates them, derived from the
 * conditions. The ungated top group comes first, then axes alphabetically. */
export function axisGroups(schema: ParamSchema): ConstraintAxisGroup[] {
  const owner = new Map<string, string>();
  for (const c of schema.conditions) {
    const key = Object.keys(c.if).sort().join(" & ");
    for (const child of c.then) if (!owner.has(child)) owner.set(child, key);
  }
  const groups = new Map<string | null, TunerParameter[]>();
  for (const p of schema.parameters) {
    const axis = owner.get(p.name) ?? null;
    const list = groups.get(axis) ?? [];
    list.push(p);
    groups.set(axis, list);
  }
  const out: ConstraintAxisGroup[] = [];
  if (groups.has(null)) out.push({ axis: null, parameters: groups.get(null)! });
  for (const axis of [...groups.keys()]
    .filter((a): a is string => a !== null)
    .sort()) {
    out.push({ axis, parameters: groups.get(axis)! });
  }
  return out;
}

/** Categorical parameters upstream of `name` in the condition graph — the
 * candidates for a `when` predicate on `name`'s row. */
export function predicateParents(schema: ParamSchema, name: string): TunerParameter[] {
  const parentsOf = new Map<string, string[]>();
  for (const c of schema.conditions) {
    const keys = Object.keys(c.if);
    for (const child of c.then) {
      parentsOf.set(child, [...(parentsOf.get(child) ?? []), ...keys]);
    }
  }
  const seen = new Set<string>();
  const queue = [name];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    for (const parent of parentsOf.get(cur) ?? []) {
      if (!seen.has(parent)) {
        seen.add(parent);
        queue.push(parent);
      }
    }
  }
  return schema.parameters.filter((p) => seen.has(p.name) && isCategorical(p));
}

function buildOp(p: TunerParameter, row: ConstraintRow): { op?: SpaceOverride; error?: string } {
  if (row.mode === "fix") {
    if (isNumeric(p)) {
      const raw = row.fix.trim();
      const n = Number(raw);
      if (raw === "" || !Number.isFinite(n)) return { error: "fix needs a number" };
      if (p.type === "int" && !Number.isInteger(n)) return { error: "fix needs an integer" };
      const [lo, hi] = p.bounds!;
      if (n < lo || n > hi) return { error: `fix ${n} is outside [${lo}, ${hi}]` };
      return { op: { fix: n } };
    }
    if (!p.choices!.includes(row.fix)) return { error: `fix "${row.fix}" is not a schema choice` };
    return { op: { fix: row.fix } };
  }

  if (row.mode === "range") {
    if (!isNumeric(p)) return { error: "range needs a numeric parameter" };
    const lo = Number(row.low.trim());
    const hi = Number(row.high.trim());
    if (row.low.trim() === "" || row.high.trim() === "" || !Number.isFinite(lo) || !Number.isFinite(hi))
      return { error: "range needs two numbers" };
    if (!(lo < hi)) return { error: "range low must be below high" };
    if (p.type === "int" && (!Number.isInteger(lo) || !Number.isInteger(hi)))
      return { error: "integer range needs integer bounds" };
    const [slo, shi] = p.bounds!;
    if (lo < slo || hi > shi) return { error: `range escapes schema bounds [${slo}, ${shi}]` };
    return { op: { range: [lo, hi] } };
  }

  // choices
  if (!isCategorical(p)) return { error: "choices needs a categorical parameter" };
  const all = p.choices!;
  const unknown = row.retained.filter((c) => !all.includes(c));
  if (unknown.length > 0) return { error: `"${unknown[0]}" is not a schema choice` };
  const kept = all.filter((c) => row.retained.includes(c));
  if (kept.length === 0) return { error: "must leave at least one choice" };
  if (kept.length >= all.length) return { error: "choices must drop at least one value" };
  return { op: { choices: kept } };
}

function buildWhen(
  schema: ParamSchema,
  row: ConstraintRow,
): { when?: Record<string, Array<string | number | boolean>>; error?: string } {
  const entries = Object.entries(row.when).filter(([, values]) => values.length > 0);
  if (entries.length === 0) return {};
  const when: Record<string, string[]> = {};
  for (const [parent, values] of entries) {
    const spec = schema.parameters.find((p) => p.name === parent);
    if (!spec || !isCategorical(spec))
      return { error: `when parent "${parent}" is not a categorical parameter` };
    const bad = values.filter((v) => !spec.choices!.includes(v));
    if (bad.length > 0) return { error: `when value "${bad[0]}" is not a choice of "${parent}"` };
    when[parent] = [...new Set(values)];
  }
  return { when };
}

/** Derive the `constraints` wire form from the editor rows, collecting a
 * human-readable error for every row the server preflight would reject. */
export function deriveConstraints(schema: ParamSchema, rows: ConstraintRows): ConstraintDerivation {
  const constraints: Constraint[] = [];
  const errors: string[] = [];
  for (const p of schema.parameters) {
    const row = rows[p.name];
    if (!row || row.mode === "free") continue;

    const { op, error } = buildOp(p, row);
    if (error) {
      errors.push(`${p.name}: ${error}`);
      continue;
    }
    if (!op) continue;

    const { when, error: whenError } = buildWhen(schema, row);
    if (whenError) {
      errors.push(`${p.name}: ${whenError}`);
      continue;
    }

    const entry: Constraint = { set: { [p.name]: op } };
    if (when) entry.when = when;
    constraints.push(entry);
  }
  return { constraints, errors };
}
