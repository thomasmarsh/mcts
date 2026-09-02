// TunerApp — the tuner UI's root. Owns its own
// `createStore(tunerReducer, tunerEnv)` (independent of the round-robin
// bench store), binds `window.location.hash` to the current route, and
// dispatches `openRun` / `closeRun` as the route changes so the reducer's
// log-tail loop follows the open run.

import {
  createEffect,
  createMemo,
  createSignal,
  onCleanup,
  onMount,
  Show,
  Switch,
  Match,
  type Component,
} from "solid-js";
import { createStore, type Store } from "@mcts/core";
import { createTunerApiClient } from "./tuner-api-client.js";
import { createTunerEnv, type TunerEnv } from "./tuner-env.js";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "./tuner-reducer.js";
import { parseTunerHash, tunerHash, type TunerRoute } from "./tuner-routes.js";
import { FleetDashboard } from "./views/FleetDashboard.js";
import { LaunchForm } from "./views/LaunchForm.js";
import { RunOverview } from "./views/RunOverview.js";

function currentHash(): string {
  return typeof window === "undefined" ? "" : window.location.hash;
}

export const TunerApp: Component<{ env?: TunerEnv }> = (props) => {
  const env = props.env ?? createTunerEnv(createTunerApiClient());
  const store: Store<TunerState, TunerAction> = createStore(
    initialTunerState(),
    tunerReducer,
    env,
  );

  const [hash, setHash] = createSignal(currentHash());
  const route = createMemo(() => parseTunerHash(hash()));
  const runRoute = createMemo(() => {
    const r = route();
    return r.view === "run" ? r : null;
  });

  function navigate(next: TunerRoute): void {
    const h = tunerHash(next);
    if (typeof window !== "undefined") window.location.hash = h;
    setHash(h);
  }

  onMount(() => {
    store.dispatch({ tag: "init" });
    const onHashChange = (): void => {
      setHash(currentHash());
    };
    window.addEventListener("hashchange", onHashChange);
    onCleanup(() => window.removeEventListener("hashchange", onHashChange));
  });

  // Keep the reducer's open-run (and its log tail) aligned with the route.
  let lastOpen: string | null = null;
  createEffect(() => {
    const r = route();
    const open = r.view === "run" ? r.runId : null;
    if (open === lastOpen) return;
    lastOpen = open;
    if (open) store.dispatch({ tag: "openRun", runId: open });
    else store.dispatch({ tag: "closeRun" });
  });

  // A successful launch navigates to the new run's overview.
  let navigatedLaunch: string | null = null;
  createEffect(() => {
    const s = store.getState()();
    if (s.launch.status === "done" && s.launch.lastRunId && s.launch.lastRunId !== navigatedLaunch) {
      navigatedLaunch = s.launch.lastRunId;
      navigate({ view: "run", runId: s.launch.lastRunId, tab: "overview" });
    }
  });

  return (
    <div class="tuner-app" data-testid="tuner-app">
      <Switch>
        <Match when={route().view === "fleet"}>
          <FleetDashboard store={store} navigate={navigate} />
        </Match>
        <Match when={route().view === "launch"}>
          <div class="tuner-launch-pane">
            <button class="tuner-back" onClick={() => navigate({ view: "fleet" })}>
              ← Fleet
            </button>
            <LaunchForm store={store} />
          </div>
        </Match>
        <Match when={runRoute()}>
          {(r) => (
            <Show
              when={r().tab === "overview"}
              fallback={
                <div class="tuner-run-overview">
                  <button
                    class="tuner-back"
                    onClick={() => navigate({ view: "run", runId: r().runId, tab: "overview" })}
                  >
                    ← Overview
                  </button>
                  <p class="tuner-fleet-empty">
                    The {r().tab} view is not built yet.
                  </p>
                </div>
              }
            >
              <RunOverview store={store} runId={r().runId} navigate={navigate} />
            </Show>
          )}
        </Match>
      </Switch>
    </div>
  );
};
