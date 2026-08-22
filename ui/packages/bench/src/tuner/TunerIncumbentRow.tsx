/** TunerIncumbentRow — the run's current best config (tuner's own tracked
 * champion, not just the lowest-score trial) with buttons to copy its config
 * as a `--baseline-config` argument or as a ready-to-paste `presets.json`
 * entry.
 *
 * `cost` on the wire is `-(mu - 3*sigma)`, so `-cost` recovers the score. */

import { createSignal, Show, type Component } from "solid-js";
import type { IncumbentInfo } from "../index.js";
import { fmtScore } from "./tuner-helpers.js";

export const TunerIncumbentRow: Component<{
  incumbent: IncumbentInfo | null;
  /** `true` when this game supports transpositions (`mcgs` appears in the
   * parameter list — it's a per-game capability flag, not a per-trial
   * boolean). Affects the preset JSON shape. */
  supportsTranspositions: boolean;
}> = (props) => {
  const [copied, setCopied] = createSignal(false);
  const [copiedPreset, setCopiedPreset] = createSignal(false);

  const displayScore = (): string => {
    if (!props.incumbent) return "—";
    return fmtScore(-props.incumbent.cost);
  };

  async function copyIncumbentConfig(): Promise<void> {
    const incumbent = props.incumbent;
    if (!incumbent) return;
    await navigator.clipboard.writeText(JSON.stringify(incumbent.config));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function copyIncumbentAsPreset(): Promise<void> {
    const incumbent = props.incumbent;
    if (!incumbent) return;
    const entry = {
      id: "tuned",
      label: "Tuned",
      description: "tuner tuned.",
      params: incumbent.config,
      use_transpositions: props.supportsTranspositions,
    };
    await navigator.clipboard.writeText(JSON.stringify(entry, null, 4));
    setCopiedPreset(true);
    setTimeout(() => setCopiedPreset(false), 1500);
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
            title="Copy a ready-to-paste presets.json entry (params plus the use_transpositions this game requires)"
          >
            {copiedPreset() ? "Copied!" : "Copy as preset"}
          </button>
        </div>
    </Show>
  );
};