// BenchApp.tsx — Top-level bench UI: run list (sidebar), plus launch form or
// run detail in the main pane.
//
// Creates its own `createStore(benchReducer, benchEnv)` independent of the
// game store. Fetches available tuner kinds and the runs list on mount.

import { createEffect, onMount, Show, type Component } from "solid-js";
import { createStore, type Store } from "@mcts/core";
import {
  benchReducer,
  createBenchApiClient,
  createBenchEnv,
  initialBenchState,
  type BenchAction,
  type BenchState,
} from "./index.js";
import { RunList } from "./RunList.js";
import { RunDetailPanel } from "./RunDetailPanel.js";
import { LaunchForm } from "./LaunchForm.js";
import { TuningSessionWorkbench } from "./tuning/TuningSessionWorkbench.js";
import type { BenchSpectatorProps } from "./types.js";

export const BenchApp: Component<{ Spectator?: Component<BenchSpectatorProps> }> = (props) => {
  const api = createBenchApiClient();
  const env = createBenchEnv(api);
  const store: Store<BenchState, BenchAction> = createStore(initialBenchState(), benchReducer, env);
  const state = store.getState();
  const dispatch = store.dispatch;

  const openRun = () => state().openRun;
  const showLaunchForm = () => state().showLaunchForm;
  const launchStatus = () => state().launch.status;
  const selectedTuningSession = () => state().tuningNavigation.selection.sessionId;

  // When a launch completes successfully, open the new run and close the form.
  // Guarded against re-dispatch: once the launched run is already open (its
  // runId matches), do nothing — otherwise the dispatch mutates state, which
  // triggers a new snapshot and re-runs this effect, creating an infinite loop
  // that starves the browser's event loop (Safari tab goes gray/unresponsive).
  createEffect(() => {
    if (launchStatus() === "done" && state().launch.result) {
      const result = state().launch.result;
      if (result && state().openRun?.runId !== result.run_id) {
        dispatch({ tag: "openRun", runId: result.run_id });
      }
    }
  });

  // Fetch tuner kinds metadata and the run list on mount.
  onMount(() => {
    store.dispatch({ tag: "tunerKinds", action: { tag: "request" } });
    store.dispatch({ tag: "runs", action: { tag: "request" } });
  });

  function onNewRun(): void {
    dispatch({ tag: "tuningNavigation", action: { tag: "clearSession" } });
    dispatch({ tag: "closeRun" });
    dispatch({ tag: "setShowLaunchForm", show: true });
  }

  return (
    <div id="bench-app">
      <div id="bench-runs-layout">
        <div id="bench-sidebar">
          <RunList store={store} onNewRun={onNewRun} />
        </div>
        <div id="bench-main-pane">
          <Show
            when={selectedTuningSession() !== null}
            fallback={
              <>
                <Show when={openRun() !== null}>
                  <RunDetailPanel store={store} Spectator={props.Spectator} />
                </Show>
                <Show when={openRun() === null && showLaunchForm()}>
                  <LaunchForm store={store} />
                </Show>
              </>
            }
          >
            <TuningSessionWorkbench store={store} Spectator={props.Spectator} />
          </Show>
        </div>
      </div>
    </div>
  );
};
