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
import { ObjectiveManager } from "./views/ObjectiveManager.js";
import { ObjectiveEditor } from "./views/ObjectiveEditor.js";
import { ProfileManager } from "./views/ProfileManager.js";
import { ProfileEditor } from "./views/ProfileEditor.js";
import { RunOverview } from "./views/RunOverview.js";
import { RunScience } from "./views/RunScience.js";
import { RunEvidence } from "./views/RunEvidence.js";
import { CandidateDrawer } from "./views/CandidateDrawer.js";

function currentHash(): string {
  return typeof window === "undefined" ? "" : window.location.hash;
}

export const TunerApp: Component<{ env?: TunerEnv }> = (props) => {
  const env = props.env ?? createTunerEnv(createTunerApiClient());
  const store: Store<TunerState, TunerAction> = createStore(initialTunerState(), tunerReducer, env);

  const [hash, setHash] = createSignal(currentHash());
  const route = createMemo(() => parseTunerHash(hash()));
  const runRoute = createMemo(() => {
    const r = route();
    return r.view === "run" ? r : null;
  });
  const drawerCandidate = createMemo(() => runRoute()?.candidate ?? null);
  const objectiveRoute = createMemo(() => {
    const r = route();
    return r.view === "objective" ? r : null;
  });
  const profileRoute = createMemo(() => {
    const r = route();
    return r.view === "profile" ? r : null;
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

  // Keep the reducer's open-objective aligned with the route so the editor
  // loads (and reloads) the right file.
  let lastObjective: string | null | undefined = undefined;
  createEffect(() => {
    const r = objectiveRoute();
    if (!r) {
      if (lastObjective !== undefined) {
        lastObjective = undefined;
        store.dispatch({ tag: "closeObjective" });
      }
      return;
    }
    if (r.key === lastObjective) return;
    lastObjective = r.key;
    store.dispatch({ tag: "openObjective", key: r.key });
  });

  // Keep the reducer's open-profile aligned with the route so the editor
  // loads (and reloads) the right file.
  let lastProfile: string | null | undefined = undefined;
  createEffect(() => {
    const r = profileRoute();
    if (!r) {
      if (lastProfile !== undefined) {
        lastProfile = undefined;
        store.dispatch({ tag: "closeProfile" });
      }
      return;
    }
    if (r.key === lastProfile) return;
    lastProfile = r.key;
    store.dispatch({ tag: "openProfile", key: r.key });
  });

  // Mirror the `?candidate=` param into the reducer so the drawer's subject
  // is part of app state, not just the URL.
  let lastCandidate: string | null = null;
  createEffect(() => {
    const cid = drawerCandidate();
    if (cid === lastCandidate) return;
    lastCandidate = cid;
    if (cid) store.dispatch({ tag: "openCandidate", candidateId: cid });
    else store.dispatch({ tag: "closeCandidate" });
  });

  // A successful launch navigates to the new run's overview.
  let navigatedLaunch: string | null = null;
  createEffect(() => {
    const s = store.getState()();
    if (
      s.launch.status === "done" &&
      s.launch.lastRunId &&
      s.launch.lastRunId !== navigatedLaunch
    ) {
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
            <LaunchForm store={store} navigate={navigate} />
          </div>
        </Match>
        <Match when={route().view === "objectives"}>
          <ObjectiveManager store={store} navigate={navigate} />
        </Match>
        <Match when={objectiveRoute()}>
          {(r) => (
            <ObjectiveEditor
              store={store}
              objectiveKey={r().key}
              game={r().game}
              navigate={navigate}
            />
          )}
        </Match>
        <Match when={route().view === "profiles"}>
          <ProfileManager store={store} navigate={navigate} />
        </Match>
        <Match when={profileRoute()}>
          {(r) => (
            <ProfileEditor store={store} profileKey={r().key} navigate={navigate} />
          )}
        </Match>
        <Match when={runRoute()}>
          {(r) => (
            <div
              class="tuner-run-pane"
              classList={{ "tuner-run-pane-drawer": !!drawerCandidate() }}
            >
              <Switch
                fallback={
                  <div class="tuner-run-overview">
                    <button
                      class="tuner-back"
                      onClick={() => navigate({ view: "run", runId: r().runId, tab: "overview" })}
                    >
                      ← Overview
                    </button>
                    <p class="tuner-fleet-empty">The {r().tab} view is not built yet.</p>
                  </div>
                }
              >
                <Match when={r().tab === "overview"}>
                  <RunOverview store={store} runId={r().runId} navigate={navigate} />
                </Match>
                <Match when={r().tab === "science"}>
                  <RunScience store={store} runId={r().runId} navigate={navigate} />
                </Match>
                <Match when={r().tab === "evidence"}>
                  <RunEvidence store={store} runId={r().runId} navigate={navigate} />
                </Match>
              </Switch>
              <Show when={drawerCandidate()}>
                {(cid) => (
                  <CandidateDrawer
                    store={store}
                    candidateId={cid()}
                    onClose={() => navigate({ view: "run", runId: r().runId, tab: r().tab })}
                  />
                )}
              </Show>
            </div>
          )}
        </Match>
      </Switch>
    </div>
  );
};
