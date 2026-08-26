// family-activation.ts — the parent/child activation walk over a
// `TunerInfo`'s `parameters`/`conditions`, used by named-family mode to
// decide which fields are currently visible and to fill in defaults for
// newly-revealed ones. Conditions form a shallow DAG (parent -> child, never
// a cycle) but may be more than one level deep, so both functions below
// repeat passes until no new parameter activates rather than assuming one
// pass covers every depth. A child named by more than one condition is
// active if *any* of them is satisfied.

import type { TunerCondition, TunerParameter } from "@mcts/game";

/** One condition, expanded to a single parent name and the value(s) that
 * satisfy it -- `TunerCondition.if` always has exactly one entry. */
interface ParentCondition {
  parent: string;
  values: unknown[];
}

/** Every parameter name that appears as a `then` target, mapped to the
 * condition(s) that activate it. */
export function childrenOf(conditions: TunerCondition[]): Map<string, ParentCondition[]> {
  const children = new Map<string, ParentCondition[]>();
  for (const condition of conditions) {
    const entry = Object.entries(condition.if)[0]!;
    const [parent, rawValues] = entry;
    const values = Array.isArray(rawValues) ? rawValues : [rawValues];
    for (const name of condition.then) {
      const conds = children.get(name) ?? [];
      conds.push({ parent, values });
      children.set(name, conds);
    }
  }
  return children;
}

/** Which parameter names are active given the current flat value dict --
 * roots (never named as a `then` target) are always active; everything else
 * activates once some condition's parent is active with a value present in
 * `values` matching that condition. */
export function activeNames(
  parameters: TunerParameter[],
  conditions: TunerCondition[],
  values: Record<string, unknown>,
): Set<string> {
  const children = childrenOf(conditions);
  const active = new Set(parameters.filter((p) => !children.has(p.name)).map((p) => p.name));

  let changed = true;
  while (changed) {
    changed = false;
    for (const [name, conds] of children) {
      if (active.has(name)) continue;
      if (conds.some((c) => active.has(c.parent) && c.values.includes(values[c.parent]))) {
        active.add(name);
        changed = true;
      }
    }
  }

  return active;
}

/** `values` merged with schema defaults for every currently-active
 * parameter that has no value yet -- never overwrites a value already
 * present, so it's safe to call after every single field edit (only fills
 * in newly-revealed fields) or with a fresh `{family: x}` seed (full reset
 * on a family change). Fills defaults as activation is discovered rather
 * than after the fact, so a child gated on a grandparent's *default* value
 * (e.g. picking a family activates `rave_ucb`, whose own default in turn
 * activates `c`) resolves in one call instead of needing a second pass from
 * the caller. */
export function withDefaultsFilled(
  parameters: TunerParameter[],
  conditions: TunerCondition[],
  values: Record<string, unknown>,
): Record<string, unknown> {
  const byName = new Map(parameters.map((p) => [p.name, p]));
  const children = childrenOf(conditions);
  const result: Record<string, unknown> = { ...values };

  function fillDefault(name: string) {
    if (result[name] !== undefined) return;
    const param = byName.get(name);
    if (param) result[name] = param.default;
  }

  const active = new Set(parameters.filter((p) => !children.has(p.name)).map((p) => p.name));
  for (const name of active) fillDefault(name);

  let changed = true;
  while (changed) {
    changed = false;
    for (const [name, conds] of children) {
      if (active.has(name)) continue;
      if (conds.some((c) => active.has(c.parent) && c.values.includes(result[c.parent]))) {
        active.add(name);
        fillDefault(name);
        changed = true;
      }
    }
  }

  return result;
}
