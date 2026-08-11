// BenchApp.tsx — Top-level bench UI with tab navigation across:
//   - Runs: run list, log tail, launch form (existing)
//   - Leaderboard: win-rate table with Wilson CI, filters, commit trends chart, and two-commit comparison
//
// Creates its own `createStore(benchReducer, benchEnv)` independent of the
// game store. Fetches available kinds and the runs list on mount.

import { createSignal, onMount, Show, type Component } from "solid-js";
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

type BenchTab = "runs" | "leaderboard";

export const BenchApp: Component = () => {
  const api = createBenchApiClient();
  const env = createBenchEnv(api);
  const store: Store<BenchState, BenchAction> = createStore(
    initialBenchState(),
    benchReducer,
    env,
  );
  const state = store.getState();

  const [activeTab, setActiveTab] = createSignal<BenchTab>("runs");

  // Fetch kinds metadata and the run list on mount.
  onMount(() => {
    store.dispatch({ tag: "kinds", action: { tag: "request" } });
    store.dispatch({ tag: "smac3Kinds", action: { tag: "request" } });
    store.dispatch({ tag: "runs", action: { tag: "request" } });
    store.dispatch({ tag: "leaderboard", action: { tag: "request" } });
  });

  return (
    <div id="bench-app">
      <div id="bench-sub-tabs">
        <button
          class="sub-tab-btn"
          classList={{ active: activeTab() === "runs" }}
          onClick={() => setActiveTab("runs")}
        >
          Runs
        </button>
        <button
          class="sub-tab-btn"
          classList={{ active: activeTab() === "leaderboard" }}
          onClick={() => setActiveTab("leaderboard")}
        >
          Leaderboard
        </button>
      </div>

      <Show when={activeTab() === "runs"}>
        <div id="bench-runs-layout">
          <div id="bench-sidebar">
            <LaunchForm store={store} />
            <RunList store={store} />
          </div>
          <Show when={state().openRun !== null}>
            <RunDetailPanel store={store} />
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