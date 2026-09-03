// objective-model.ts — the bidirectional bridge between the ObjectiveEditor's
// form state and the frozen objective wire JSON, plus the client-side panel
// validator. Pure: no rendering, no fetch. The server
// (`tuner/src/tuner_cli/objective.py::resolve_objective`) stays the authority
// on panel semantics; this module just makes the legal panel the easy one to
// build and catches the common mistakes before a round-trip.

import type {
  GameConfigSchema,
  JsonValue,
  TunerCondition,
  TunerInfo,
  TunerParameter,
} from "../../types.js";

export type JsonObject = { [key: string]: JsonValue };

/** The shape `activeParamNames` / `schemaDefaultConfig` need — satisfied by
 * both `TunerInfo` (the strategy search space) and `GameConfigSchema` (the
 * game-setup axis). */
export interface ParamSchema {
  parameters: TunerParameter[];
  conditions: TunerCondition[];
}

const EMPTY_SCHEMA: GameConfigSchema = { parameters: [], conditions: [] };

export type OpponentKind = "schema_default" | "inline";

export interface OpponentDraft {
  id: string;
  label: string;
  /** `role` is derived: `schema_default` → `default`, `inline` →
   * `historical_reference`. */
  kind: OpponentKind;
  weight: number;
  /** Parsed inline config; ignored when `kind === "schema_default"`. Kept in
   * sync with `configText` whenever that buffer parses as an object. */
  config: JsonObject;
  /** Raw-JSON editor buffer — the escape hatch for values the form can't
   * express (`"q_init": "Infinity"`). */
  configText: string;
  configMode: "form" | "raw";
}

export interface ObjectiveDraft {
  objectiveId: string;
  gameKind: string;
  /** `opponents[0]` is always the pinned schema-default opponent. */
  opponents: OpponentDraft[];
  /** The optional `game_config` override, held **flat** with dotted keys
   * (`{"size.w": 7}`) so it lines up 1:1 with `config_schema.parameters`;
   * nested back to `{size: {w: 7}}` only on the wire. Empty ⇒ the key is
   * omitted and the run uses the binary's `default_config()`. */
  gameConfig: JsonObject;
  /** Raw-JSON buffer for `game_config`, in **nested** wire shape. */
  gameConfigText: string;
  gameConfigMode: "form" | "raw";
}

export interface DraftParse {
  draft: ObjectiveDraft;
  /** Non-fatal problems encountered while parsing an existing file. */
  warnings: string[];
}

const PINNED_DEFAULT_ID = "schema-default";

function asObject(v: JsonValue | undefined): JsonObject | null {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as JsonObject) : null;
}

/** Flatten a nested object to dotted leaf keys: `{size:{w:7,h:9}} →
 * {"size.w":7,"size.h":9}`. A non-object or empty object at the root yields
 * `{}`. */
export function flattenDotted(value: JsonValue | undefined): JsonObject {
  const out: JsonObject = {};
  const walk = (v: JsonValue, prefix: string): void => {
    const o = asObject(v);
    if (o && Object.keys(o).length > 0) {
      for (const key of Object.keys(o)) walk(o[key]!, prefix ? `${prefix}.${key}` : key);
    } else if (prefix) {
      out[prefix] = v;
    }
  };
  walk(value ?? {}, "");
  return out;
}

/** Inverse of `flattenDotted`: `{"size.w":7} → {size:{w:7}}`. */
export function nestDotted(flat: JsonObject): JsonObject {
  const out: JsonObject = {};
  for (const key of Object.keys(flat)) {
    const parts = key.split(".");
    let cursor = out;
    for (let i = 0; i < parts.length - 1; i++) {
      const next = asObject(cursor[parts[i]!]) ?? {};
      cursor[parts[i]!] = next;
      cursor = next;
    }
    cursor[parts[parts.length - 1]!] = flat[key]!;
  }
  return out;
}

function gcd2(a: number, b: number): number {
  a = Math.abs(Math.trunc(a));
  b = Math.abs(Math.trunc(b));
  while (b) [a, b] = [b, a % b];
  return a;
}

/** Divide a weight panel by its gcd so the panel is *reduced*
 * (`resolve_objective` rejects an unreduced panel). `[2,4,6] → [1,2,3]`,
 * `[3,5] → [3,5]`, `[7] → [1]`. */
export function reduceWeights(weights: number[]): number[] {
  const divisor = weights.reduce((acc, w) => gcd2(acc, w), 0);
  return divisor <= 1 ? weights.slice() : weights.map((w) => Math.trunc(w) / divisor);
}

function schemaDefaultOpponent(): OpponentDraft {
  return {
    id: PINNED_DEFAULT_ID,
    label: "Schema default",
    kind: "schema_default",
    weight: 1,
    config: {},
    configText: "{}",
    configMode: "form",
  };
}

export function blankInlineOpponent(index: number): OpponentDraft {
  return {
    id: `historical-${index}`,
    label: "",
    kind: "inline",
    weight: 1,
    config: {},
    configText: "{}",
    configMode: "form",
  };
}

/** A draft with just the pinned schema-default opponent and one blank inline
 * opponent — the minimum legal shape. */
export function emptyDraft(gameKind: string): ObjectiveDraft {
  return {
    objectiveId: "",
    gameKind,
    opponents: [schemaDefaultOpponent(), blankInlineOpponent(1)],
    gameConfig: {},
    gameConfigText: "{}",
    gameConfigMode: "form",
  };
}

/** Parse an existing objective file into a draft (edit / duplicate).
 * Tolerant: unparseable pieces are dropped and reported in `warnings`
 * rather than throwing. */
export function draftFromContent(content: JsonValue, fallbackGame = ""): DraftParse {
  const warnings: string[] = [];
  const root = asObject(content);
  if (!root) {
    return {
      draft: emptyDraft(fallbackGame),
      warnings: ["objective file is not a JSON object"],
    };
  }

  const objectiveId = typeof root["objective_id"] === "string" ? root["objective_id"] : "";
  const gameKind = typeof root["game_kind"] === "string" ? root["game_kind"] : fallbackGame;

  const rawOpponents = Array.isArray(root["opponents"]) ? root["opponents"] : [];
  if (!Array.isArray(root["opponents"])) warnings.push("objective has no opponents array");

  const opponents: OpponentDraft[] = [];
  for (const item of rawOpponents) {
    const o = asObject(item);
    if (!o) {
      warnings.push("skipped an opponent that is not an object");
      continue;
    }
    const cfg = asObject(o["config"]) ?? {};
    const kind: OpponentKind = cfg["source"] === "schema_default" ? "schema_default" : "inline";
    const value = asObject(cfg["value"]) ?? {};
    opponents.push({
      id: typeof o["id"] === "string" ? o["id"] : "",
      label: typeof o["label"] === "string" ? o["label"] : "",
      kind,
      weight:
        typeof o["weight"] === "number" && Number.isFinite(o["weight"])
          ? Math.trunc(o["weight"])
          : 1,
      config: value,
      configText: JSON.stringify(value, null, 2),
      configMode: "form",
    });
  }

  // The pinned schema-default opponent always sits at index 0.
  const defaultIdx = opponents.findIndex((o) => o.kind === "schema_default");
  if (defaultIdx === -1) {
    warnings.push("objective has no schema-default opponent; added one");
    opponents.unshift(schemaDefaultOpponent());
  } else if (defaultIdx !== 0) {
    const [d] = opponents.splice(defaultIdx, 1);
    opponents.unshift(d!);
  }
  if (opponents.length < 2) opponents.push(blankInlineOpponent(opponents.length));

  const rawGameConfig = asObject(root["game_config"]);
  if (root["game_config"] !== undefined && !rawGameConfig) {
    warnings.push("game_config is not a JSON object; ignored");
  }
  const gameConfig = flattenDotted(rawGameConfig ?? {});
  const gameConfigText = JSON.stringify(rawGameConfig ?? {}, null, 2);

  return {
    draft: {
      objectiveId,
      gameKind,
      opponents,
      gameConfig,
      gameConfigText,
      gameConfigMode: "form",
    },
    warnings,
  };
}

/** Recursively sort object keys so two equal configs serialise identically. */
export function canonicalize(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(canonicalize);
  const o = asObject(value);
  if (!o) return value;
  const out: JsonObject = {};
  for (const key of Object.keys(o).sort()) out[key] = canonicalize(o[key]!);
  return out;
}

/** The inline config an opponent will emit: its raw-JSON buffer if that
 * parses as an object, otherwise the last good parsed `config`. */
export function effectiveConfig(opponent: OpponentDraft): JsonObject {
  try {
    const parsed = asObject(JSON.parse(opponent.configText));
    if (parsed) return parsed;
  } catch {
    /* fall through to the last good parse */
  }
  return opponent.config;
}

/** The flat, dotted `game_config` a draft will emit: its raw-JSON buffer
 * (flattened) when that parses as an object, otherwise the last good form
 * state. */
export function effectiveGameConfig(draft: ObjectiveDraft): JsonObject {
  try {
    const parsed = asObject(JSON.parse(draft.gameConfigText));
    if (parsed) return flattenDotted(parsed);
  } catch {
    /* fall through to the last good form state */
  }
  return draft.gameConfig;
}

/** Assemble the wire JSON: reduce weights, set the fixed
 * `schema_version` / `start_distribution`, canonicalise each inline config. */
export function draftToContent(draft: ObjectiveDraft): JsonObject {
  const weights = reduceWeights(draft.opponents.map((o) => Math.max(1, Math.trunc(o.weight))));
  const opponents: JsonObject[] = draft.opponents.map((o, i): JsonObject => {
    const config: JsonObject =
      o.kind === "schema_default"
        ? { source: "schema_default" }
        : { source: "inline", value: canonicalize(effectiveConfig(o)) };
    return {
      id: o.id,
      label: o.label,
      role: o.kind === "schema_default" ? "default" : "historical_reference",
      weight: weights[i]!,
      config,
    };
  });
  const wire: JsonObject = {
    schema_version: 1,
    objective_id: draft.objectiveId,
    game_kind: draft.gameKind,
    opponents,
    start_distribution: { kind: "default_only" },
  };
  const gameConfig = nestDotted(effectiveGameConfig(draft));
  if (Object.keys(gameConfig).length > 0) {
    wire["game_config"] = canonicalize(gameConfig);
  }
  return wire;
}

/** The game's schema-default config as a plain object, best-effort from its
 * tuner parameter list (each parameter carries a `default` or `value`). */
export function schemaDefaultConfig(schema: ParamSchema): JsonObject {
  const out: JsonObject = {};
  for (const p of schema.parameters) {
    const d = p.default !== undefined ? p.default : p.value;
    if (d !== undefined) out[p.name] = d as JsonValue;
  }
  return out;
}

/** Which parameters are active for `config`, applying `tuner.conditions`
 * (a conditioned parameter is active only when some condition that lists it
 * in `then` has its `if` satisfied) — same gating the tuner uses. */
export function activeParamNames(schema: ParamSchema, config: JsonObject): Set<string> {
  const conditioned = new Set<string>();
  for (const c of schema.conditions) for (const then of c.then) conditioned.add(then);

  const active = new Set<string>();
  for (const p of schema.parameters) if (!conditioned.has(p.name)) active.add(p.name);

  for (const c of schema.conditions) {
    const satisfied = Object.entries(c.if).every(([key, want]) => {
      const have = config[key];
      const options = Array.isArray(want) ? want : [want];
      return options.some((w) => w === have || String(w) === String(have));
    });
    if (satisfied) for (const then of c.then) active.add(then);
  }
  return active;
}

/** Every §1 panel rule the UI can check locally. An empty list means the
 * client is willing to let the operator save (the server re-checks). */
export function validateDraft(draft: ObjectiveDraft, schema?: TunerInfo | null): string[] {
  const errors: string[] = [];

  if (draft.objectiveId.trim() === "") errors.push("Objective id is required.");
  if (draft.gameKind.trim() === "") errors.push("Game kind is required.");

  const defaults = draft.opponents.filter((o) => o.kind === "schema_default");
  if (defaults.length !== 1) {
    errors.push("The panel must have exactly one schema-default opponent.");
  }
  const inline = draft.opponents.filter((o) => o.kind === "inline");
  if (inline.length < 1) {
    errors.push("Add at least one historical-reference opponent (a panel needs ≥ 2).");
  }

  const ids = draft.opponents.map((o) => o.id.trim());
  ids.forEach((id, i) => {
    if (id === "") errors.push(`Opponent ${i + 1} needs an id.`);
  });
  const dupes = [...new Set(ids.filter((id, i) => id !== "" && ids.indexOf(id) !== i))];
  if (dupes.length) errors.push(`Duplicate opponent id: ${dupes.join(", ")}.`);

  draft.opponents.forEach((o, i) => {
    if (o.label.trim() === "") errors.push(`Opponent ${i + 1} needs a label.`);
    if (!Number.isInteger(o.weight) || o.weight < 1) {
      errors.push(`Opponent ${i + 1} weight must be a positive integer.`);
    }
  });

  const defaultFingerprint = schema
    ? JSON.stringify(canonicalize(schemaDefaultConfig(schema)))
    : null;
  const seen = new Map<string, number>();
  const bump = (key: string): void => {
    seen.set(key, (seen.get(key) ?? 0) + 1);
  };
  if (defaultFingerprint) bump(defaultFingerprint);

  draft.opponents.forEach((o, i) => {
    if (o.kind === "schema_default") return;
    let value: JsonValue;
    try {
      value = JSON.parse(o.configText);
    } catch {
      errors.push(`Opponent ${i + 1} config is not valid JSON.`);
      return;
    }
    const obj = asObject(value);
    if (!obj) {
      errors.push(`Opponent ${i + 1} config must be a JSON object.`);
      return;
    }
    const fingerprint = JSON.stringify(canonicalize(obj));
    if (defaultFingerprint && fingerprint === defaultFingerprint) {
      errors.push(
        `Opponent ${i + 1} re-types the schema default — use the pinned default opponent instead.`,
      );
    }
    bump(fingerprint);
    if (schema && schema.parameters.length > 0) {
      for (const key of Object.keys(obj)) {
        if (!schema.parameters.some((p) => p.name === key)) {
          errors.push(`Opponent ${i + 1}: unknown parameter "${key}" for ${draft.gameKind}.`);
        }
      }
      // The binary's `compare validate` rejects a partial config: every
      // parameter active for the chosen family must be present (it fills no
      // defaults). Mirror that here so a half-specified opponent fails in the
      // editor rather than at launch.
      const missing = [...activeParamNames(schema, obj)].filter((name) => !(name in obj));
      if (missing.length) {
        errors.push(
          `Opponent ${i + 1}: missing required parameter${missing.length > 1 ? "s" : ""} ${missing
            .map((name) => `"${name}"`)
            .join(", ")} for this family.`,
        );
      }
    }
  });
  for (const count of seen.values()) {
    if (count > 1) {
      errors.push("Two opponents have the same effective configuration.");
      break;
    }
  }

  errors.push(...validateGameConfig(draft, schema));

  return errors;
}

/** The `game_config` schema for a game, defaulting to the empty (fixed
 * board) schema when `tune describe` predates the field. */
export function gameConfigSchema(schema?: TunerInfo | null): GameConfigSchema {
  return schema?.game_config_schema ?? EMPTY_SCHEMA;
}

/** §4.1 `game_config` rules the UI checks locally: valid JSON, every key a
 * known `config_schema` parameter, numeric values in bounds, and the whole
 * override not equal to the game's default (write nothing instead). */
function validateGameConfig(draft: ObjectiveDraft, schema?: TunerInfo | null): string[] {
  const errors: string[] = [];
  const gcSchema = gameConfigSchema(schema);

  if (draft.gameConfigMode === "raw") {
    let parsed: JsonValue;
    try {
      parsed = JSON.parse(draft.gameConfigText);
    } catch {
      return ["Game setup config is not valid JSON."];
    }
    if (!asObject(parsed)) return ["Game setup config must be a JSON object."];
  }

  const config = effectiveGameConfig(draft);
  const keys = Object.keys(config);
  if (keys.length === 0) return errors;

  if (gcSchema.parameters.length === 0) {
    errors.push("This game's board is fixed — remove the game setup override.");
    return errors;
  }

  const active = activeParamNames(gcSchema, config);
  for (const key of keys) {
    const param = gcSchema.parameters.find((p) => p.name === key);
    if (!param) {
      errors.push(`Game setup: unknown field "${key}".`);
      continue;
    }
    if (!active.has(key)) {
      errors.push(`Game setup: "${key}" is not active for the current selection.`);
      continue;
    }
    const value = config[key];
    if ((param.type === "int" || param.type === "float") && param.bounds) {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        errors.push(`Game setup: "${key}" must be a number.`);
      } else if (value < param.bounds[0] || value > param.bounds[1]) {
        errors.push(
          `Game setup: "${key}" must be within [${param.bounds[0]}, ${param.bounds[1]}].`,
        );
      }
    }
  }

  const defaultConfig = flattenDotted((schema?.game_config as JsonValue | undefined) ?? {});
  const same =
    JSON.stringify(canonicalize(nestDotted(config))) ===
    JSON.stringify(canonicalize(nestDotted(defaultConfig)));
  if (same) {
    errors.push("Game setup matches the default — leave it unset to use the default.");
  }

  return errors;
}

/** A filesystem-safe key from an objective id (`^[a-zA-Z0-9._-]+$`). */
export function slugKey(objectiveId: string): string {
  return (
    objectiveId
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "objective"
  );
}
