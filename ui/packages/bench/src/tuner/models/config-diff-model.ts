// config-diff-model.ts — pure derivation behind `<ConfigDiff>`. Flattens a
// candidate's canonical config and the schema-default config to dotted leaf
// paths and pairs them into diff rows. The schema default comes from the
// game's tuner parameter list (`TunerInfo.parameters`, each carrying a
// `default` or `value`).

import type { JsonValue, TunerParameter } from "../../types.js";

export interface ConfigDiffRow {
  path: string;
  base: string | null;
  candidate: string | null;
  changed: boolean;
}

/** `null` leaves render as the string `"null"`; a genuinely absent side is
 * `null` in the row and rendered as "—" by the component. */
function render(value: JsonValue): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

export function flattenConfig(
  value: JsonValue,
  prefix = "",
  out: Record<string, string> = {},
): Record<string, string> {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value)) {
      flattenConfig(v, prefix ? `${prefix}.${k}` : k, out);
    }
  } else {
    out[prefix] = render(value);
  }
  return out;
}

/** The game's default config as a flat path→string map, from its tuner
 * parameter schema. */
export function schemaDefaults(parameters: TunerParameter[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const p of parameters) {
    const d = p.default !== undefined ? p.default : p.value;
    if (d !== undefined) out[p.name] = typeof d === "string" ? d : JSON.stringify(d);
  }
  return out;
}

export function configDiffRows(
  base: Record<string, string>,
  candidate: JsonValue | null,
): ConfigDiffRow[] {
  const cand = candidate === null ? {} : flattenConfig(candidate);
  const paths = [...new Set([...Object.keys(base), ...Object.keys(cand)])].sort();
  return paths.map((path) => {
    const b = base[path] ?? null;
    const c = cand[path] ?? null;
    return { path, base: b, candidate: c, changed: b !== c };
  });
}
