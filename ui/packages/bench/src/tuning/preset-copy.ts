import type { JsonValue, TuningPoolAnchor, TuningPoolRevision, TuningTrialDetailView } from "../types.js";

export type JsonObject = { [key: string]: JsonValue };

export interface PresetBudgetSnapshot {
  max_time_ms?: number | null;
  max_iterations?: number | null;
}

export interface PresetSource extends PresetBudgetSnapshot {
  kind: "candidate" | "opponent";
  sourceId: string;
  sourceDescription: string;
  params: unknown;
}

/** The frozen `presets.json` entry produced by every tuning copy surface. */
export interface PresetSpec {
  id: string;
  label: string;
  description: string;
  params: JsonObject;
  max_time_ms?: number;
  max_iterations?: number;
  threads: 1;
  use_transpositions: boolean;
}

export type PresetDisabledReason =
  | { code: "legacy_missing_config"; message: string }
  | { code: "invalid_source_id"; message: string }
  | { code: "invalid_params"; message: string }
  | { code: "missing_family"; message: string }
  | { code: "invalid_mcgs"; message: string }
  | { code: "invalid_budget"; message: string }
  | { code: "multiple_budgets"; message: string };

export type PresetBuildResult =
  | { enabled: true; preset: PresetSpec; text: string }
  | { enabled: false; reason: PresetDisabledReason };

export interface ClipboardWriter {
  writeText(text: string): Promise<void>;
}

export type PresetCopyState =
  | { status: "success"; announcement: string }
  | { status: "failure"; announcement: string }
  | { status: "disabled"; announcement: string };

function isJsonValue(value: unknown): value is JsonValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(isJsonValue);
  return isJsonObject(value);
}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && Object.values(value).every(isJsonValue);
}

function validBudget(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

/** Reversible, source-stable identifier encoding for arbitrary recorded IDs. */
export function safePresetId(kind: PresetSource["kind"], sourceId: string): string | null {
  if (sourceId.length === 0) return null;
  const encoded = [...sourceId].map((character) => {
    if (/^[a-z0-9]$/.test(character)) return character;
    return `_x${character.codePointAt(0)!.toString(16)}_`;
  }).join("");
  return `${kind}-${encoded}`;
}

function stableValue(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(stableValue);
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(Object.keys(value).sort(compareKey).map((key) => [key, stableValue(value[key]!)]));
}

function compareKey(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "variant" });
}

/** Deterministic, paste-ready JSON without mutating the recorded params object. */
export function serializePresetSpec(preset: PresetSpec): string {
  const ordered: JsonObject = {
    id: preset.id,
    label: preset.label,
    description: preset.description,
    params: stableValue(preset.params),
    ...(preset.max_time_ms === undefined ? {} : { max_time_ms: preset.max_time_ms }),
    ...(preset.max_iterations === undefined ? {} : { max_iterations: preset.max_iterations }),
    threads: preset.threads,
    use_transpositions: preset.use_transpositions,
  };
  return JSON.stringify(ordered, null, 4);
}

/** Raw config copy remains available for the existing baseline CLI control. */
export function serializeRecordedParams(params: unknown): string | null {
  return isJsonObject(params) ? JSON.stringify(params) : null;
}

/** Validates one persisted candidate/opponent snapshot and resolves its one budget. */
export function buildPresetSpec(source: PresetSource): PresetBuildResult {
  const id = safePresetId(source.kind, source.sourceId);
  if (id === null) return { enabled: false, reason: { code: "invalid_source_id", message: "This snapshot has no stable source id." } };
  if (source.params === null || source.params === undefined) {
    return { enabled: false, reason: { code: "legacy_missing_config", message: "This legacy snapshot did not record a strategy configuration." } };
  }
  if (!isJsonObject(source.params)) {
    return { enabled: false, reason: { code: "invalid_params", message: "The recorded strategy configuration is not a JSON object." } };
  }
  if (typeof source.params.family !== "string" || source.params.family.length === 0) {
    return { enabled: false, reason: { code: "missing_family", message: "The recorded strategy configuration has no family." } };
  }
  if (Object.hasOwn(source.params, "mcgs") && typeof source.params.mcgs !== "boolean") {
    return { enabled: false, reason: { code: "invalid_mcgs", message: "The recorded mcgs capability is not boolean." } };
  }
  const hasTime = source.max_time_ms !== null && source.max_time_ms !== undefined;
  const hasIterations = source.max_iterations !== null && source.max_iterations !== undefined;
  if (hasTime && hasIterations) {
    return { enabled: false, reason: { code: "multiple_budgets", message: "The snapshot records both a time and an iteration budget." } };
  }
  if ((hasTime && !validBudget(source.max_time_ms)) || (hasIterations && !validBudget(source.max_iterations))) {
    return { enabled: false, reason: { code: "invalid_budget", message: "The recorded search budget must be a positive integer." } };
  }
  const preset: PresetSpec = {
    id,
    label: source.kind === "candidate" ? "Tuned candidate" : "Pool opponent",
    description: source.sourceDescription,
    params: source.params,
    ...(hasTime
      ? { max_time_ms: source.max_time_ms as number }
      : { max_iterations: hasIterations ? source.max_iterations as number : 10_000 }),
    threads: 1,
    // The recorded search-space's `mcgs` key is the durable capability
    // marker. Its boolean value chooses graph search for that trial, not
    // whether this game supports transposition tables at all.
    use_transpositions: Object.hasOwn(source.params, "mcgs"),
  };
  return { enabled: true, preset, text: serializePresetSpec(preset) };
}

export function candidatePresetSource(
  trial: Pick<TuningTrialDetailView, "trial_id" | "trial_number" | "config">,
  budget: PresetBudgetSnapshot = {},
): PresetSource {
  return {
    kind: "candidate",
    sourceId: trial.trial_id,
    sourceDescription: `Candidate snapshot from trial ${trial.trial_number} (${trial.trial_id}).`,
    params: trial.config,
    ...budget,
  };
}

export function opponentPresetSource(
  anchor: TuningPoolAnchor,
  revision: Pick<TuningPoolRevision, "display_ordinal">,
  budget: PresetBudgetSnapshot = {},
): PresetSource {
  return {
    kind: "opponent",
    sourceId: anchor.anchor_id,
    sourceDescription: `Opponent snapshot ${anchor.anchor_id} from pool revision ${revision.display_ordinal}.`,
    params: anchor.config,
    ...budget,
  };
}

/** Thin, injectable clipboard action with text suitable for an aria-live status. */
export async function copyPreset(build: PresetBuildResult, clipboard: ClipboardWriter): Promise<PresetCopyState> {
  if (!build.enabled) return { status: "disabled", announcement: build.reason.message };
  try {
    await clipboard.writeText(build.text);
    return { status: "success", announcement: "Preset copied to clipboard." };
  } catch {
    return { status: "failure", announcement: "Could not copy preset to clipboard." };
  }
}
