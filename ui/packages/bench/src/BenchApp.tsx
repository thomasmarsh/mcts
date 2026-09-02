// BenchApp.tsx — Top-level bench UI. Two surfaces behind one tab:
//   - "Round-robin runs": the run list (sidebar) + launch form / run detail.
//   - "Tuner": the version-4 tuner UI (`TunerApp`), which owns its own store
//     and hash sub-routes (`#/tuner/...`).
//
// The round-robin store is `createStore(benchReducer, benchEnv)`,
// independent of both the game store and the tuner store.

import { createSignal, onCleanup, onMount, Show, type Component } from "solid-js";
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
import { TunerApp } from "./tuner/TunerApp.js";
import type { BenchSpectatorProps } from "./types.js";

const isTunerHash = (): boolean =>
  typeof window !== "undefined" && window.location.hash.startsWith("#/tuner");

export const BenchApp: Component<{ Spectator?: Component<BenchSpectatorProps> }> = (props) => {
  const api = createBenchApiClient();
  const env = createBenchEnv(api);
  const store: Store<BenchState, BenchAction> = createStore(initialBenchState(), benchReducer, env);
  const state = store.getState();
  const dispatch = store.dispatch;

  const [showTuner, setShowTuner] = createSignal(isTunerHash());
  const openRun = () => state().openRun;
  const showLaunchForm = () => state().showLaunchForm;

  onMount(() => {
    store.dispatch({ tag: "tunerKinds", action: { tag: "request" } });
    store.dispatch({ tag: "runs", action: { tag: "request" } });
    const onHash = (): void => {
      setShowTuner(isTunerHash());
    };
    window.addEventListener("hashchange", onHash);
    onCleanup(() => window.removeEventListener("hashchange", onHash));
  });

  function goTuner(): void {
    if (typeof window !== "undefined") window.location.hash = "#/tuner";
    setShowTuner(true);
  }

  function goRuns(): void {
    if (typeof window !== "undefined" && window.location.hash.startsWith("#/tuner")) {
      window.location.hash = "";
    }
    setShowTuner(false);
  }

  function onNewRun(): void {
    dispatch({ tag: "closeRun" });
    dispatch({ tag: "setShowLaunchForm", show: true });
  }

  return (
    <div id="bench-app">
      <nav id="bench-surface-nav">
        <button classList={{ active: !showTuner() }} onClick={goRuns}>
          Round-robin runs
        </button>
        <button classList={{ active: showTuner() }} onClick={goTuner}>
          Tuner
        </button>
      </nav>

      <Show
        when={showTuner()}
        fallback={
          <div id="bench-runs-layout">
            <div id="bench-sidebar">
              <RunList store={store} onNewRun={onNewRun} />
            </div>
            <div id="bench-main-pane">
              <Show when={openRun() !== null}>
                <RunDetailPanel store={store} Spectator={props.Spectator} />
              </Show>
              <Show when={openRun() === null && showLaunchForm()}>
                <RoundRobinLaunchHint />
              </Show>
            </div>
          </div>
        }
      >
        <TunerApp />
      </Show>
    </div>
  );
};

// The old bench launch form drove the retired Optuna tuner. Round-robin
// runs are launched from the CLI; the tuner has its own launch form under
// the Tuner surface.
const RoundRobinLaunchHint: Component = () => (
  <div id="launch-form">
    <h3>Launch a run</h3>
    <p>
      Round-robin bench runs are launched from the command line (<code>bin/bench</code>). To tune a
      strategy, switch to the <strong>Tuner</strong> surface above.
    </p>
  </div>
);
