// App.tsx — Wires the real game store: creates the
// `ApiClient`/`Env` pair, the `appReducer` store seeded with a
// placeholder root state (replaced the instant `GameShell`'s bootstrap
// `newGame` request resolves — see that file's header comment), and
// mounts `GameShell`. Also fetches `GET /api/games` once on mount to
// populate the kind-picker labels (replacing the former hand-maintained
// `GAME_LABELS`).
//
// Tab navigation: "Game" tab shows the existing `GameShell`; "Bench" tab
// shows the new `BenchApp` (run list, log tail, launch form).
//
// Both `GameShell` and `BenchApp` are lazy-loaded so each tab's dependency
// tree ends up in its own chunk — the Game tab pulls in three.js (via the
// Druid renderer), while the Bench tab stays lean.

import { render } from "solid-js/web";
import { createEffect, createSignal, lazy, onMount, Show } from "solid-js";
import { createStore } from "@mcts/core";
import {
  appReducer,
  createApiClient,
  createEnv,
  initialAppState,
  type AppAction,
  type AppState,
  type Env,
} from "@mcts/game";
import { DEFAULT_GAME_KIND } from "./games.js";
import "./app.css";
import "./bench.css";

const GameShell = lazy(() => import("./GameShell.js").then((m) => ({ default: m.GameShell })));
const BenchApp = lazy(() => import("@mcts/bench").then((m) => ({ default: m.BenchApp })));

const App = () => {
  const api = createApiClient();
  const env = createEnv(api);
  const store = createStore<AppState<unknown, unknown, unknown>, AppAction<unknown, unknown, unknown>, Env>(
    initialAppState<unknown, unknown, unknown>(DEFAULT_GAME_KIND, null),
    appReducer<unknown, unknown, unknown>,
    env,
  );

  const [activeTab, setActiveTab] = createSignal<"game" | "bench">("game");

  onMount(async () => {
    try {
      const games = await api.getGames();
      store.dispatch({ tag: "setGames", games });
    } catch (e) {
      console.warn("Failed to fetch game list:", e);
    }
  });

  // Toggle body class so bench-specific CSS can hide game chrome
  // (the hud, move list, analysis panel) when the bench tab is active.
  createEffect(() => {
    document.body.classList.toggle("bench-active", activeTab() === "bench");
  });

  return (
    <>
      <div id="app-tabs">
        <button
          class="tab-btn"
          classList={{ active: activeTab() === "game" }}
          onClick={() => setActiveTab("game")}
        >
          Game
        </button>
        <button
          class="tab-btn"
          classList={{ active: activeTab() === "bench" }}
          onClick={() => setActiveTab("bench")}
        >
          Bench
        </button>
      </div>

      <Show when={activeTab() === "game"}>
        <GameShell store={store} />
      </Show>

      <Show when={activeTab() === "bench"}>
        <BenchApp />
      </Show>
    </>
  );
};

const root = document.getElementById("app");
if (root) render(() => <App />, root);