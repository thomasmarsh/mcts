// RunOverview — a run's landing view: the header, the live progress rail,
// and the launch-log tail (the sub-second feedback the old UI lacked). The
// ship verdict, validation table, cohort strip, and the science / evidence
// tabs are not built yet.

import { createMemo, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import { RunStatusBadge } from "../primitives/RunStatusBadge.js";
import { ProgressRail } from "../primitives/ProgressRail.js";

export const RunOverview: Component<{
  store: Store<TunerState, TunerAction>;
  runId: string;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const journalRow = createMemo(() =>
    (peek(state().runs) ?? []).find((r) => r.run_id === props.runId),
  );
  const projectionRow = createMemo(() =>
    (peek(state().projectionRuns) ?? []).find((r) => r.run_id === props.runId),
  );
  const status = createMemo(() => journalRow()?.status ?? (projectionRow() ? "exited" : null));
  const live = createMemo(() => status() === "live");

  return (
    <div class="tuner-run-overview" data-testid="tuner-run-overview">
      <div class="tuner-run-overview-header">
        <button class="tuner-back" onClick={() => props.navigate({ view: "fleet" })}>
          ← Fleet
        </button>
        <h2>{props.runId}</h2>
        <RunStatusBadge
          status={status()}
          terminalOutcome={journalRow()?.terminal_outcome}
          reportStatus={projectionRow()?.report_status}
        />
        <Show when={live()}>
          <button onClick={() => dispatch({ tag: "stopRun", runId: props.runId })}>Stop</button>
        </Show>
      </div>

      <Show when={journalRow()?.run_dir}>
        <p class="tuner-run-dir">{journalRow()!.run_dir}</p>
      </Show>
      <Show when={state().stopError}>
        <div class="launch-error" role="alert">
          {state().stopError}
        </div>
      </Show>

      <ProgressRail
        status={status()}
        startedAt={journalRow()?.started_at ?? null}
        nowMs={Date.now()}
      />

      <div class="tuner-run-links">
        <button onClick={() => props.navigate({ view: "run", runId: props.runId, tab: "science" })}>
          Full science →
        </button>
        <button onClick={() => props.navigate({ view: "run", runId: props.runId, tab: "evidence" })}>
          Raw evidence →
        </button>
      </div>

      <Show when={live() || state().log.lines.length > 0}>
        <section class="tuner-log-tail">
          <h3>Launch log</h3>
          <Show when={state().log.error}>
            <div class="launch-error" role="alert">
              {state().log.error}
            </div>
          </Show>
          <pre class="tuner-log-lines" data-testid="tuner-log-lines">
            <For each={state().log.lines}>{(line) => <>{line}{"\n"}</>}</For>
          </pre>
          <Show when={state().log.errLines.length > 0}>
            <pre class="tuner-log-err" data-testid="tuner-log-err">
              <For each={state().log.errLines}>{(line) => <>{line}{"\n"}</>}</For>
            </pre>
          </Show>
        </section>
      </Show>
    </div>
  );
};
