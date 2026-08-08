// App.tsx — Wires the real game store: creates the
// `ApiClient`/`Env` pair, the `appReducer` store seeded with a
// placeholder root state (replaced the instant `GameShell`'s bootstrap
// `newGame` request resolves -- see that file's header comment), and
// mounts `GameShell`. Also fetches `GET /api/games` once on mount to
// populate the kind-picker labels (replacing the former hand-maintained
// `GAME_LABELS`).

import { render } from "solid-js/web";
import { onMount, type Component } from "solid-js";
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
import { GameShell } from "./GameShell.js";
import { DEFAULT_GAME_KIND } from "./games.js";
import "./app.css";

const App: Component = () => {
  const api = createApiClient();
  const env = createEnv(api);
  const store = createStore<AppState<unknown, unknown, unknown>, AppAction<unknown, unknown, unknown>, Env>(
    initialAppState<unknown, unknown, unknown>(DEFAULT_GAME_KIND, null),
    appReducer<unknown, unknown, unknown>,
    env,
  );

  onMount(async () => {
    try {
      const games = await api.getGames();
      store.dispatch({ tag: "setGames", games });
    } catch (e) {
      console.warn("Failed to fetch game list:", e);
    }
  });

  return <GameShell store={store} />;
};

const root = document.getElementById("app");
if (root) render(() => <App />, root);