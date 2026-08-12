// games.ts — The game-kind registry `GameShell` picks a renderer/module
// from. One entry per game. Game modules are loaded lazily (dynamic
// `import()`) so each game's dependencies (e.g. three.js for druid) end up
// in separate chunks rather than the main bundle.
//
// Game-kind display labels are not maintained here: they come from
// `GET /api/games` (`env.getGames()`) fetched on mount and stored in
// `AppState.gamesInfo` (see `App.tsx`). If a caller needs a display name, it
// should look up the kind in `state().gamesInfo` or fall back to the raw
// kind string.
//
// Static metadata (`GAME_META`) is kept synchronously available for the
// new-game dialog's seat pickers and kind selector — loading the full module
// (with renderer, three.js, etc.) is deferred until `GameShell` actually
// renders it.

import type { GameKindModule } from "@mcts/game";

/**
 * Lazy loaders for each game-kind module. Each `() => Promise<…>` triggers a
 * separate chunk via Vite's code-splitting. Use `createResource` in GameShell
 * to load these asynchronously — the component already has a loading fallback
 * for the (brief) gap.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const GAME_MODULES: Record<string, () => Promise<GameKindModule<any, any, any>>> = {
  atarigo: () => import("@mcts/atarigo").then((m) => m.atarigoModule),
  breakthrough: () => import("@mcts/breakthrough").then((m) => m.breakthroughModule),
  druid: () => import("@mcts/druid").then((m) => m.druidModule),
  gonnect: () => import("@mcts/gonnect").then((m) => m.gonnectModule),
  knightthrough: () => import("@mcts/knightthrough").then((m) => m.knightthroughModule),
  othello: () => import("@mcts/othello").then((m) => m.othelloModule),
  tanbo: () => import("@mcts/tanbo").then((m) => m.tanboModule),
  "traffic-lights": () => import("@mcts/traffic-lights").then((m) => m.trafficLightsModule),
  ttt: () => import("@mcts/ttt").then((m) => m.tttModule),
};

/**
 * Synchronous metadata for each game kind — available without loading the
 * full module (renderer, scene graph libs, etc.). Used by GameShell's
 * new-game dialog to build seat-picker defaults before the module has loaded.
 */
export const GAME_META: Record<string, { players: string[] }> = {
  atarigo: { players: ["Black", "White"] },
  breakthrough: { players: ["Black", "White"] },
  druid: { players: ["Black", "White"] },
  gonnect: { players: ["Black", "White"] },
  knightthrough: { players: ["Black", "White"] },
  othello: { players: ["Black", "White"] },
  tanbo: { players: ["Black", "White"] },
  "traffic-lights": { players: ["A", "B"] },
  ttt: { players: ["X", "O"] },
};

export const DEFAULT_GAME_KIND = "druid";