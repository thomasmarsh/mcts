// BenchApp.tsx — Top-level bench UI with tab navigation across:
//   - Runs: run list (sidebar), plus launch form or run detail in the main pane
//   - Leaderboard: win-rate table with Wilson CI, filters, commit trends chart, and two-commit comparison
//   - Projects
//
// Creates its own `createStore(benchReducer, benchEnv)` independent of the
// game store. Fetches available kinds and the runs list on mount.

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
import { LeaderboardTable } from "./LeaderboardTable.js";
import { WinRateChart } from "./WinRateChart.js";
import { CommitComparison } from "./CommitComparison.js";
import { ProjectsApp } from "./ProjectsApp.js";
import { ExperimentRunDetail } from "./ExperimentRunDetail.js";
import { TuningSessionWorkbench } from "./tuning/TuningSessionWorkbench.js";
import type { BenchSpectatorProps } from "./types.js";

export const BenchApp: Component<{ Spectator?: Component<BenchSpectatorProps> }> = (props) => {
  const api = createBenchApiClient();
  const env = createBenchEnv(api);
  const store: Store<BenchState, BenchAction> = createStore(initialBenchState(), benchReducer, env);
  const state = store.getState();
  const dispatch = store.dispatch;

  const activeTab = () => state().activeTab;
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

  // Fetch kinds metadata and the run list on mount.
  onMount(() => {
    store.dispatch({ tag: "kinds", action: { tag: "request" } });
    store.dispatch({ tag: "tunerKinds", action: { tag: "request" } });
    store.dispatch({ tag: "runs", action: { tag: "request" } });
    store.dispatch({ tag: "leaderboard", action: { tag: "request" } });
    store.dispatch({ tag: "projectsRequest" });
  });

  function onNewRun(): void {
    dispatch({ tag: "tuningNavigation", action: { tag: "clearSession" } });
    dispatch({ tag: "closeRun" });
    dispatch({ tag: "setShowLaunchForm", show: true });
  }

  return (
    <div id="bench-app">
      <div id="bench-sub-tabs">
        <button
          class="sub-tab-btn"
          classList={{ active: activeTab() === "runs" }}
          onClick={() => dispatch({ tag: "setTab", tab: "runs" })}
        >
          Runs
        </button>
        <button
          class="sub-tab-btn"
          classList={{ active: activeTab() === "leaderboard" }}
          onClick={() => dispatch({ tag: "setTab", tab: "leaderboard" })}
        >
          Leaderboard
        </button>
        <button
          class="sub-tab-btn"
          classList={{ active: activeTab() === "projects" }}
          onClick={() => dispatch({ tag: "setTab", tab: "projects" })}
        >
          Projects
        </button>
      </div>

      <Show when={activeTab() === "projects"}>
        <ProjectsApp store={store} />
      </Show>

      <Show when={activeTab() === "runs"}>
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
                    <Show
                      when={openRun()?.detail?.kind === "experiment"}
                      fallback={<RunDetailPanel store={store} Spectator={props.Spectator} />}
                    >
                      <ExperimentRunDetail store={store} Spectator={props.Spectator} />
                    </Show>
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
      </Show>

      <Show when={activeTab() === "leaderboard"}>
        <div id="bench-leaderboard-layout">
          <div id="leaderboard-left">
            <LeaderboardTable store={store} />
          </div>
          <div id="leaderboard-right">
            <WinRateChart store={store} />
            <CommitComparison store={store} />
          </div>
        </div>
      </Show>
    </div>
  );
};
