// BenchApp.tsx — Top-level bench UI: run list, run detail/log tail, launch form.
//
// Creates its own `createStore(benchReducer, benchEnv)` — independent of
// the game store.  Fetches available kinds and the runs list on mount.
// Layout: a sidebar with the run list and launch form stacked vertically,
// and a detail panel on the right when a run is open.

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

export const BenchApp: Component = () => {
  const api = createBenchApiClient();
  const env = createBenchEnv(api);
  const store: Store<BenchState, BenchAction> = createStore(
    initialBenchState(),
    benchReducer,
    env,
  );
  const state = store.getState();

  // Fetch kinds metadata and run list on mount.
  onMount(() => {
    store.dispatch({ tag: "kinds", action: { tag: "request" } });
    store.dispatch({ tag: "runs", action: { tag: "request" } });
  });

  return (
    <div id="bench-app">
      <div id="bench-sidebar">
        <LaunchForm store={store} />
        <RunList store={store} />
      </div>
      <Show when={state().openRun !== null}>
        <RunDetailPanel store={store} />
      </Show>
    </div>
  );
};