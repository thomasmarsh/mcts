// tests/helpers.tsx — Shared test utilities for GameShell/renderer component
// tests. Mirrors pb/ui/tests/helpers.tsx's `mockEnv`/`createTestStore`
// pattern: a real `Store` (createStore + appReducer), wired to a mocked
// `Env` so a test can drive the actual reducer/effect machinery -- and
// therefore the real `GameShell` component reacting to it -- without ever
// touching a live server. This is the harness `GameShell.test.tsx` uses to
// regression-test bugs that only exist at the component level (effects that
// re-dispatch based on props/store state), which `packages/game/tests/
// reducer.test.ts`'s pure-reducer `TestStore` can't reach on its own.

import { cleanup } from "@solidjs/testing-library";
import { afterEach } from "vitest";
import { Effect, createStore, type Store } from "@mcts/core";
import {
  appReducer,
  initialAppState,
  type AppAction,
  type AppState,
  type AxisSchema,
  type Env,
  type TunerInfo,
} from "@mcts/game";

// A minimal but real-shaped `AxisSchema` fixture -- one variant per axis
// (`ucb1`/`uniform`/`classic`/`robust_child`), plus an `epsilon_greedy`
// select variant wrapping `select_base` so tests exercising the New Game
// dialog's "Custom…" option can drive a wrapper's nested picker without
// pulling in `axis_schema()`'s full ~20-variant shape. Mirrors
// `packages/strategy-config/tests/schema-fixture.ts`.
export const fixtureAxisSchema: AxisSchema = {
  select: {
    variants: [
      {
        kind: "ucb1",
        fields: [{ name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 }],
      },
      {
        kind: "epsilon_greedy",
        fields: [{ name: "epsilon", type: "float", bounds: [0, 1], default: 0.1 }],
        wraps: "select_base",
      },
    ],
  },
  select_base: {
    variants: [
      {
        kind: "ucb1",
        fields: [{ name: "c", type: "float", bounds: [0, 3], default: 1.4142135623730951 }],
      },
    ],
  },
  simulate: { variants: [{ kind: "uniform", fields: [] }] },
  simulate_base: { variants: [{ kind: "uniform", fields: [] }] },
  backprop: { variants: [{ kind: "classic", fields: [] }] },
  final_action: { variants: [{ kind: "robust_child", fields: [] }] },
};

export const mockFetchStrategySchema = (): Promise<AxisSchema> =>
  Promise.resolve(fixtureAxisSchema);

// No fixture game in these tests exposes a tuner, so the default mock mirrors
// that: every `GameShell` test drives the free-composition-only path unless a
// test overrides this itself.
export const mockFetchStrategyFamilies = (): Promise<TunerInfo | null> => Promise.resolve(null);

// Every `Env` method stubbed to a no-op effect -- individual tests override
// just the methods their scenario needs (same shape as pb's `mockEnv`).
export const mockEnv: Env = {
  getGames: () => Effect.none(),
  newGame: () => Effect.none(),
  legalMoves: () => Effect.none(),
  view: () => Effect.none(),
  apply: () => Effect.none(),
  aiPresets: () => Effect.none(),
  aiMove: () => Effect.none(),
  analyze: () => Effect.none(),
};

export interface TestStoreResult {
  store: Store<AppState<unknown, unknown, unknown>, AppAction<unknown, unknown, unknown>>;
  captured: AppAction<unknown, unknown, unknown>[];
}

/** A real `appReducer`-backed store seeded at `gameKind`'s pre-bootstrap
 * placeholder root (same shape `App.tsx` itself constructs) -- `GameShell`'s
 * own `onMount` is what actually starts a game, via `env.newGame`. */
export function createTestStore(gameKind: string, env: Env = mockEnv): TestStoreResult {
  const captured: AppAction<unknown, unknown, unknown>[] = [];
  const init = initialAppState<unknown, unknown, unknown>(gameKind, null);
  const store = createStore<
    AppState<unknown, unknown, unknown>,
    AppAction<unknown, unknown, unknown>,
    Env
  >(init, appReducer<unknown, unknown, unknown>, env, (action) => {
    captured.push(action);
  });
  return { store, captured };
}

afterEach(() => {
  cleanup();
});
