/** TunerTrialsTable — scrollable, reverse-chronological listing of every
 * scored trial in a tuner run (or the full chain, when laddered), with the
 * trial number, run label (for laddered runs), the config's family, baseline
 * instance, score (`mu − 3σ`), seed, and timestamp.
 *
 * The table is inside a scrollable container with sticky headers and a
 * resize handle. Trials are listed newest-first. */

import { For, Show, type Component } from "solid-js";
import type { ChainedTrial, ChainRung, TrialRow } from "../index.js";
import { fmtScore, trialScore } from "./tuner-helpers.js";

function chainRungLabel(rung: ChainRung, index: number): string {
  return index === 0
    ? `Root (${rung.run_id})`
    : `Rung ${index + 1} (${rung.run_id})`;
}

function instanceOf(t: TrialRow): string | null {
  const extra = t.extra as { instance?: unknown } | null;
  return typeof extra?.instance === "string" ? extra.instance : null;
}

export const TunerTrialsTable: Component<{
  entries: ChainedTrial[];
  chain: ChainRung[];
  bestEntry: ChainedTrial | null;
}> = (props) => {
  const isBest = (e: ChainedTrial): boolean => {
    const b = props.bestEntry;
    return (
      b !== null &&
      b.rungIndex === e.rungIndex &&
      b.trial.trial_id === e.trial.trial_id
    );
  };

  return (
    <div id="tuner-trials-scroll">
      <table id="tuner-trials-table">
        <thead>
          <tr>
            <th>#</th>
            <Show when={props.chain.length > 1}>
              <th>Run</th>
            </Show>
            <th>Family</th>
            <th>Baseline</th>
            <th>Score</th>
            <th>Seed</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          <For each={props.entries.slice().reverse()}>
            {(e) => (
              <tr
                classList={{ "tuner-trial-best": isBest(e) }}
                title={JSON.stringify(e.trial.config)}
              >
                <td>{e.trial.trial_id}</td>
                <Show when={props.chain.length > 1}>
                  <td class="tuner-trial-rung">
                    {chainRungLabel(props.chain[e.rungIndex]!, e.rungIndex)}
                  </td>
                </Show>
                <td class="tuner-trial-family">
                  {typeof e.trial.config.family === "string"
                    ? e.trial.config.family
                    : "—"}
                </td>
                <td class="tuner-trial-baseline">
                  {instanceOf(e.trial) ?? "—"}
                </td>
                <td>{fmtScore(trialScore(e.trial))}</td>
                <td>{e.trial.seed ?? "—"}</td>
                <td>{e.trial.ts}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
};