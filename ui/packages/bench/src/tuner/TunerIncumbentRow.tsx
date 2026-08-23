/** TunerIncumbentRow — the run's current best config (tuner's own tracked
 * champion, not just the lowest-score trial) with buttons to copy its config
 * as a `--baseline-config` argument or as a ready-to-paste `presets.json`
 * entry.
 *
 * `cost` on the wire is `-(mu - 3*sigma)`, so `-cost` recovers the score. */

import { createSignal, Show, type Component } from "solid-js";
import type { IncumbentInfo } from "../index.js";
import { fmtScore } from "./tuner-helpers.js";
import { buildPresetSpec, copyPreset, serializeRecordedParams, type PresetCopyState } from "../tuning/preset-copy.js";

export const TunerIncumbentRow: Component<{
  incumbent: IncumbentInfo | null;
}> = (props) => {
  const [copied, setCopied] = createSignal(false);
  const [presetCopy, setPresetCopy] = createSignal<PresetCopyState | null>(null);

  const displayScore = (): string => {
    if (!props.incumbent) return "—";
    return fmtScore(-props.incumbent.cost);
  };

  async function copyIncumbentConfig(): Promise<void> {
    const incumbent = props.incumbent;
    if (!incumbent) return;
    const text = serializeRecordedParams(incumbent.config);
    if (text === null) return;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function copyIncumbentAsPreset(): Promise<void> {
    const incumbent = props.incumbent;
    if (!incumbent) return;
    const status = await copyPreset(buildPresetSpec({
      kind: "candidate",
      sourceId: "incumbent",
      sourceDescription: "Candidate snapshot from the recorded incumbent.",
      params: incumbent.config,
    }), navigator.clipboard);
    setPresetCopy(status);
    if (status.status === "success") setTimeout(() => setPresetCopy(null), 1500);
  }

  return (
    <Show when={props.incumbent}>
        <div id="tuner-incumbent-row">
          <span class="tuner-stat-label">Incumbent</span>
          <span class="tuner-stat-value">{displayScore()}</span>
          <button
            id="tuner-copy-incumbent-btn"
            onClick={copyIncumbentConfig}
            title="Copy this config for a later run's --baseline-config"
          >
            {copied() ? "Copied!" : "Copy as baseline config"}
          </button>
          <button
            id="tuner-copy-incumbent-preset-btn"
            onClick={copyIncumbentAsPreset}
            title="Copy a ready-to-paste presets.json entry from this recorded configuration"
            aria-label={presetCopy()?.announcement ?? "Copy as preset"}
          >
            {presetCopy()?.status === "success" ? "Copied!" : presetCopy()?.status === "failure" ? "Copy failed" : presetCopy()?.status === "disabled" ? "Preset unavailable" : "Copy as preset"}
          </button>
        </div>
    </Show>
  );
};
