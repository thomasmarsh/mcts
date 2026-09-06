// json-util.ts — small readers for walking a verbatim `report.json`
// (`JsonValue`) without `as` casts scattered through the science models.
// Every reader is total: a missing / wrong-typed node yields a safe empty
// value, never a throw.

import type { JsonValue } from "../../types.js";

export function asObject(v: JsonValue | undefined): Record<string, JsonValue> | null {
  return v !== null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, JsonValue>)
    : null;
}

export function asArray(v: JsonValue | undefined): JsonValue[] {
  return Array.isArray(v) ? v : [];
}

export function asNumber(v: JsonValue | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function asString(v: JsonValue | undefined): string | null {
  return typeof v === "string" ? v : null;
}

export function asStringArray(v: JsonValue | undefined): string[] {
  return asArray(v).filter((x): x is string => typeof x === "string");
}

/** `candidate-<hex>` / `prefix-<hex>` → a short, still-distinct stem. */
export function shortId(id: string): string {
  const bare = id.replace(/^[a-z]+-/, "");
  return bare.length > 12 ? bare.slice(0, 12) : bare;
}
