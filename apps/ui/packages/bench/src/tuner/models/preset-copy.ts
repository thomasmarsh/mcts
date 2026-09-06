// preset-copy.ts — pure serialisation behind `<CopyPresetButton>`. Turns a
// projection candidate's immutable `canonical_config` into the JSON blob the
// operator pastes into a game's `presets.json` to ship it. No clipboard
// access here (the button owns that) — this is the text, unit-tested against
// a hand-built config.

import type { JsonValue } from "../../types.js";
import { shortCandidateId } from "./verdict-model.js";

export interface PresetSpec {
  id: string;
  label: string;
  game: string;
  /** The candidate's canonical config, verbatim. */
  params: JsonValue;
}

export interface PresetCopyInput {
  candidateId: string;
  gameKind: string;
  config: JsonValue | null;
}

export type PresetCopyResult =
  { ok: true; preset: PresetSpec; text: string } | { ok: false; reason: string };

export function buildPreset(input: PresetCopyInput): PresetCopyResult {
  if (input.config === null || typeof input.config !== "object" || Array.isArray(input.config)) {
    return { ok: false, reason: "candidate has no recorded config" };
  }
  const short = shortCandidateId(input.candidateId);
  const preset: PresetSpec = {
    id: `tuned-${input.gameKind}-${short}`,
    label: `Tuned ${input.gameKind} (${short})`,
    game: input.gameKind,
    params: input.config,
  };
  return { ok: true, preset, text: JSON.stringify(preset, null, 2) };
}
