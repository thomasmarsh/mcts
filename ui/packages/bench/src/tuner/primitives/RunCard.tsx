// RunCard — one run in the fleet dashboard. A live run shows its status and
// a stop button; a completed run shows its validation claim and compute
// total. The card links into the run's overview for the full detail.

import { Show, type Component } from "solid-js";
import { RunStatusBadge } from "./RunStatusBadge.js";
import type { TerminalOutcome, TunerRunLiveness } from "../tuner-types.js";

export interface RunCardProps {
  runId: string;
  game?: string | null;
  objective?: string | null;
  status: TunerRunLiveness | null;
  terminalOutcome?: TerminalOutcome | null;
  reportStatus?: string | null;
  validationClaim?: string | null;
  totalPairs?: number | null;
  ingestError?: string | null;
  /** `launch.err` tail for a run that died before it began working. */
  errorDetail?: string | null;
  highlight?: boolean;
  onOpen: () => void;
  onStop?: () => void;
}

export const RunCard: Component<RunCardProps> = (props) => {
  return (
    <div
      class="tuner-run-card"
      classList={{ "tuner-run-card-highlight": props.highlight }}
      data-testid="run-card"
    >
      <button class="tuner-run-card-open" onClick={() => props.onOpen()}>
        <div class="tuner-run-card-top">
          <span class="tuner-run-card-id">{props.runId}</span>
          <RunStatusBadge
            status={props.status}
            terminalOutcome={props.terminalOutcome}
            reportStatus={props.reportStatus}
          />
        </div>
        <div class="tuner-run-card-meta">
          <Show when={props.game}>
            <span>{props.game}</span>
          </Show>
          <Show when={props.objective}>
            <span class="tuner-run-card-objective">{props.objective}</span>
          </Show>
        </div>
        <Show when={props.validationClaim}>
          <div class="tuner-run-card-claim">{props.validationClaim}</div>
        </Show>
        <Show when={props.ingestError}>
          <div class="tuner-run-card-error">ingest error: {props.ingestError}</div>
        </Show>
        <Show when={props.errorDetail}>
          <pre class="tuner-run-card-error tuner-run-card-error-detail">{props.errorDetail}</pre>
        </Show>
        <Show when={props.totalPairs != null && props.totalPairs > 0}>
          <div class="tuner-run-card-compute">{props.totalPairs} pair attempts</div>
        </Show>
      </button>
      <Show when={props.status === "live" && props.onStop}>
        <button class="tuner-run-card-stop" onClick={() => props.onStop?.()}>
          Stop
        </button>
      </Show>
    </div>
  );
};
