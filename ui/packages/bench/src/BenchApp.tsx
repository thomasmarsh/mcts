// BenchApp.tsx — Top-level bench UI with tab navigation across:
//   - Runs: run list, log tail, launch form (existing)
//   - Leaderboard: win-rate table with Wilson CI, filters, commit trends chart, and two-commit comparison
//
// Creates its own `createStore(benchReducer, benchEnv)` independent of the
// game store. Fetches available kinds and the runs list on mount.

import { onMount, Show, type Component } from "solid-js";
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

export const BenchApp: Component<{ Spectator?: Component<{ runId: string; game: string; kind: string; live: boolean }> }> = (props) => {
  const api = createBenchApiClient();
  const env = createBenchEnv(api);
  const store: Store<BenchState, BenchAction> = createStore(
    initialBenchState(),
    benchReducer,
    env,
  );
  const state = store.getState();

  const activeTab = () => state().activeTab;

  // Fetch kinds metadata and the run list on mount.
  onMount(() => {
    store.dispatch({ tag: "kinds", action: { tag: "request" } });
    store.dispatch({ tag: "smac3Kinds", action: { tag: "request" } });
    store.dispatch({ tag: "runs", action: { tag: "request" } });
    store.dispatch({ tag: "leaderboard", action: { tag: "request" } });
    store.dispatch({ tag: "projectsRequest" });
  });

  return (
    <div id="bench-app">
      <div id="bench-sub-tabs">
        <button
          class="sub-tab-btn"
          classList={{ active: activeTab() === "runs" }}
          onClick={() => store.dispatch({ tag: "setTab", tab: "runs" })}
        >
          Runs
        </button>
        <button
          class="sub-tab-btn"
          classList={{ active: activeTab() === "leaderboard" }}
          onClick={() => store.dispatch({ tag: "setTab", tab: "leaderboard" })}
        >
          Leaderboard
        </button>
        <button class="sub-tab-btn" classList={{ active: activeTab() === "projects" }} onClick={() => store.dispatch({ tag: "setTab", tab: "projects" })}>Projects</button>
      </div>

      <Show when={activeTab() === "projects"}><ProjectsApp store={store} /></Show>

      <Show when={activeTab() === "runs"}>
        <div id="bench-runs-layout">
          <div id="bench-sidebar">
            <LaunchForm store={store} />
            <RunList store={store} />
          </div>
          <Show when={state().openRun !== null}>
            <Show when={state().openRun?.detail?.kind === "experiment"} fallback={<RunDetailPanel store={store} Spectator={props.Spectator} />}><ExperimentRunDetail store={store} /></Show>
          </Show>
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
