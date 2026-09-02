// FleetDashboard — the tuner landing screen. Answers "what is running and
// what finished" at a glance: a KPI row, the live runs from the operational
// journal (refreshed every few seconds by the reducer's poll loop), and the
// completed / failed runs from the projection.

import { createMemo, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import { RunCard } from "../primitives/RunCard.js";

export const FleetDashboard: Component<{
  store: Store<TunerState, TunerAction>;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const runs = createMemo(() => peek(state().runs) ?? []);
  const projection = createMemo(() => peek(state().projectionRuns) ?? []);

  const liveRuns = createMemo(() => runs().filter((r) => r.status === "live"));
  const liveIds = createMemo(() => new Set(liveRuns().map((r) => r.run_id)));
  const finishedProjection = createMemo(() =>
    projection().filter((r) => !liveIds().has(r.run_id)),
  );

  const failedCount = createMemo(
    () => finishedProjection().filter((r) => r.ingest_error || r.report_status === "failed").length,
  );

  const freshness = createMemo(() => {
    const at = state().lastProjectionRefreshAt;
    if (!at) return "not yet refreshed";
    const secs = Math.round((Date.now() - at) / 1000);
    return `refreshed ${secs}s ago`;
  });

  return (
    <div class="tuner-fleet" data-testid="tuner-fleet">
      <div class="tuner-fleet-header">
        <div class="tuner-kpi-row">
          <span data-testid="kpi-live">{liveRuns().length} live</span>
          <span data-testid="kpi-complete">{finishedProjection().length} complete</span>
          <span data-testid="kpi-failed">{failedCount()} failed</span>
        </div>
        <div class="tuner-fleet-actions">
          <span class="tuner-fleet-freshness">{freshness()}</span>
          <button
            onClick={() => dispatch({ tag: "refreshProjection" })}
            disabled={state().refreshing}
          >
            {state().refreshing ? "Refreshing…" : "Refresh science"}
          </button>
          <button onClick={() => props.navigate({ view: "objectives" })}>Manage objectives</button>
          <button class="tuner-fleet-new" onClick={() => props.navigate({ view: "launch" })}>
            New run
          </button>
        </div>
      </div>

      <Show when={state().refreshError}>
        <div class="launch-error" role="alert">
          {state().refreshError}
        </div>
      </Show>

      <section class="tuner-fleet-section">
        <h3>Live</h3>
        <Show when={liveRuns().length > 0} fallback={<p class="tuner-fleet-empty">No runs live.</p>}>
          <For each={liveRuns()}>
            {(run) => (
              <RunCard
                runId={run.run_id}
                status={run.status}
                terminalOutcome={run.terminal_outcome}
                highlight={run.run_id === state().launch.lastRunId}
                onOpen={() =>
                  props.navigate({ view: "run", runId: run.run_id, tab: "overview" })
                }
                onStop={() => dispatch({ tag: "stopRun", runId: run.run_id })}
              />
            )}
          </For>
        </Show>
        <Show when={state().stopError}>
          <div class="launch-error" role="alert">
            {state().stopError}
          </div>
        </Show>
      </section>

      <section class="tuner-fleet-section">
        <h3>Completed &amp; failed</h3>
        <Show
          when={finishedProjection().length > 0}
          fallback={<p class="tuner-fleet-empty">No completed runs in the projection.</p>}
        >
          <For each={finishedProjection()}>
            {(run) => (
              <RunCard
                runId={run.run_id}
                status="exited"
                game={run.game_kind}
                objective={run.objective_id}
                reportStatus={run.report_status}
                validationClaim={run.validation_claim}
                ingestError={run.ingest_error}
                totalPairs={run.total_pair_attempts}
                onOpen={() =>
                  props.navigate({ view: "run", runId: run.run_id, tab: "overview" })
                }
              />
            )}
          </For>
        </Show>
      </section>
    </div>
  );
};
