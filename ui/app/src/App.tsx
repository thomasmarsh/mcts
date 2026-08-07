// App.tsx — Session 1 placeholder. Proves the workspace links, the vendored
// @mcts/core framework compiles, and SolidJS renders through Vite.
// No game logic, no API calls, no reducer wiring beyond a trivial createStore.
// All of that arrives in Session 3+.

import { render } from "solid-js/web";
import type { Component } from "solid-js";
import { createStore } from "@mcts/core";

interface PlaceholderState {
  message: string;
}

const App: Component = () => {
  // Minimal createStore smoke test: proves the vendored framework links,
  // compiles, and drives SolidJS reactivity. No real reducers — Session 3
  // wires the game-tree reducer and API env.
  const store = createStore<PlaceholderState, never, object>(
    { message: "mcts/ui scaffold ready" },
    () => null,
    {},
  );
  const state = store.getState();
  return (
    <main id="app">
      <h1>{state().message}</h1>
      <p>Game UI workspace — Session 1 scaffolding.</p>
    </main>
  );
};

const root = document.getElementById("app");
if (root) render(() => <App />, root);