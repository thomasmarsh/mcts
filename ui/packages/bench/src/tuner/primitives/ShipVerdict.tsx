// ShipVerdict — the run overview's headline: which candidate to ship, its
// validation estimate, how it differs from the default, a copy-preset
// button, the runner-up, unresolved ties, and every caveat that says "don't
// trust this yet". All derivation is in `verdict-model.ts`; this composes
// primitives.

import { For, Show, type Component } from "solid-js";
import type { JsonValue } from "../../types.js";
import type { ShipVerdict as ShipVerdictModel } from "../models/verdict-model.js";
import { CandidateChip } from "./CandidateChip.js";
import { ConfigDiff } from "./ConfigDiff.js";
import { CopyPresetButton } from "./CopyPresetButton.js";
import { IntervalBar } from "./IntervalBar.js";

export interface ShipVerdictProps {
  verdict: ShipVerdictModel;
  gameKind: string | null;
  /** Schema-default config as a flat path→string map. */
  baseConfig: Record<string, string>;
  onOpenCandidate: (candidateId: string) => void;
}

export const ShipVerdict: Component<ShipVerdictProps> = (props) => {
  const v = () => props.verdict;
  return (
    <section class="tuner-ship-verdict" data-testid="ship-verdict">
      <h3>Ship decision</h3>
      <Show
        when={v().finalist}
        fallback={<p class="tuner-fleet-empty">No validated finalist yet.</p>}
      >
        {(finalist) => (
          <div class="tuner-ship-finalist">
            <div class="tuner-ship-finalist-head">
              <CandidateChip
                candidateId={finalist().candidateId}
                source={finalist().source}
                rank={finalist().rank}
                onClick={props.onOpenCandidate}
              />
              <CopyPresetButton
                candidateId={finalist().candidateId}
                gameKind={props.gameKind}
                config={finalist().config}
              />
            </div>
            <IntervalBar
              mean={finalist().estimate}
              lower={finalist().lower}
              upper={finalist().upper}
              domain={v().domain}
              reference={0}
            />
            <ConfigDiff base={props.baseConfig} candidate={finalist().config as JsonValue | null} />
          </div>
        )}
      </Show>

      <Show when={v().runnerUp}>
        {(runnerUp) => (
          <p class="tuner-ship-runnerup">
            Runner-up:{" "}
            <CandidateChip
              candidateId={runnerUp().candidateId}
              source={runnerUp().source}
              rank={runnerUp().rank}
              onClick={props.onOpenCandidate}
            />
          </p>
        )}
      </Show>

      <Show when={v().ties.length > 0}>
        <div class="tuner-ship-ties" data-testid="ship-ties">
          <For each={v().ties}>
            {(tie) => (
              <p>
                Cannot distinguish {tie.leftShort} from {tie.rightShort} at this sample.
              </p>
            )}
          </For>
        </div>
      </Show>

      <Show when={v().caveats.length > 0}>
        <ul class="tuner-ship-caveats" data-testid="ship-caveats">
          <For each={v().caveats}>{(c) => <li>{c}</li>}</For>
        </ul>
      </Show>
    </section>
  );
};
