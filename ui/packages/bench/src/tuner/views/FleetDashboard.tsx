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

  const deleteRun = (runId: string): void => {
    if (
      typeof window !== "undefined" &&
      !window.confirm(`Permanently delete tuner run "${runId}"? This cannot be undone.`)
    ) {
      return;
    }
    dispatch({ tag: "deleteRun", runId });
  };

  const runs = createMemo(() => peek(state().runs) ?? []);
  const projection = createMemo(() => peek(state().projectionRuns) ?? []);

  const liveRuns = createMemo(() => runs().filter((r) => r.status === "live"));
  const liveIds = createMemo(() => new Set(liveRuns().map((r) => r.run_id)));

  // A run that died before it began working, from whichever source saw it
  // first: the operational journal (`status: "failed"`, carries the tuner's
  // own `launch.err` in `error_detail`) or the projection (`ingest_error`,
  // once it has refreshed). Keyed by run_id, journal reason preferred — so
  // the operator always gets a concrete why, and the run never shows up
  // twice or hides in "Completed & failed" as a deceptive green "exited".
  const failedRuns = createMemo(() => {
    const byId = new Map<
      string,
      { runId: string; reason: string; game?: string | null; objective?: string | null }
    >();
    for (const r of projection()) {
      if (r.ingest_error && !liveIds().has(r.run_id)) {
        byId.set(r.run_id, {
          runId: r.run_id,
          reason: r.ingest_error,
          game: r.game_kind,
          objective: r.objective_id,
        });
      }
    }
    for (const r of runs()) {
      if (r.status === "failed") {
        byId.set(r.run_id, {
          runId: r.run_id,
          reason:
            r.error_detail ??
            byId.get(r.run_id)?.reason ??
            "The run process exited during startup — check its launch.err.",
        });
      }
    }
    return [...byId.values()];
  });
  const failedIds = createMemo(() => new Set(failedRuns().map((r) => r.runId)));

  const finishedProjection = createMemo(() =>
    projection().filter((r) => !liveIds().has(r.run_id) && !failedIds().has(r.run_id)),
  );

  const failedCount = createMemo(() => failedRuns().length);

  const freshness = createMemo(() => {
    // This tab's own last refresh, else the server follower's last pass (so a
    // cold open on an unattended run still shows a real age).
    const at =
      state().lastProjectionRefreshAt ??
      (state().projectionLastPassAt ? Date.parse(state().projectionLastPassAt!) : NaN);
    if (!at || Number.isNaN(at)) return "not yet refreshed";
    const secs = Math.max(0, Math.round((Date.now() - at) / 1000));
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

      <Show when={failedRuns().length > 0}>
        <section class="tuner-fleet-section">
          <h3>Failed to start</h3>
          <For each={failedRuns()}>
            {(run) => (
              <RunCard
                runId={run.runId}
                status="failed"
                game={run.game}
                objective={run.objective}
                failureReason={run.reason}
                deleting={state().deletingRunId === run.runId}
                onOpen={() =>
                  props.navigate({ view: "run", runId: run.runId, tab: "overview" })
                }
                onDelete={() => deleteRun(run.runId)}
              />
            )}
          </For>
        </section>
      </Show>

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
                deleting={state().deletingRunId === run.run_id}
                onOpen={() =>
                  props.navigate({ view: "run", runId: run.run_id, tab: "overview" })
                }
                onDelete={() => deleteRun(run.run_id)}
              />
            )}
          </For>
        </Show>
        <Show when={state().deleteError}>
          <div class="launch-error" role="alert">
            {state().deleteError}
          </div>
        </Show>
      </section>
    </div>
  );
};
