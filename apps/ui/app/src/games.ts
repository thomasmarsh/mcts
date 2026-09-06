// games.ts — The game-kind registry `GameShell` picks a renderer/module
// from. One entry per *variant*: most games have exactly one, registered
// under their own name (e.g. "druid"), but a game with several player
// counts/board sizes can register several colon-namespaced ids that share
// one base name (Focus: "focus", "focus:3p", "focus:4p") -- `groupIdOf`
// collapses those back to one row in the game picker, with a secondary
// variant selector for the rest (see `GameShell.tsx`'s New Game dialog).
// Game modules are loaded lazily (dynamic `import()`) so each game's
// dependencies (e.g. three.js for druid) end up in separate chunks rather
// than the main bundle.
//
// Game-kind display labels are not maintained here for a single-variant
// game: they come from `GET /api/games` (`env.getGames()`) fetched on mount
// and stored in `AppState.gamesInfo` (see `App.tsx`), looked up by
// `wireKindOf`. A multi-variant group's own picker row needs an explicit
// `groupLabel` instead (see `GameMeta` below), since `gamesInfo` only knows
// about the server's individual wire kinds ("focus-2p" etc.), not the
// group as a whole.
//
// Static metadata (`GAME_META`) is kept synchronously available for the
// new-game dialog's seat pickers and kind selector — loading the full module
// (with renderer, three.js, etc.) is deferred until `GameShell` actually
// renders it.
//
// `GameMeta.wireKind` is the one place a UI id and the server's own kind
// string can diverge: `state().gameKind` (and everything downstream of it —
// `GAME_MODULES`/`GAME_META` keys, save files) is always the UI id, but the
// actual HTTP calls need the server's kind. `App.tsx` wires `wireKindOf` into
// `createApiClient` as its `resolveKind` translator, so nothing below this
// file ever needs to know the two can differ.

import type { GameKindModule } from "@mcts/game";

/**
 * Lazy loaders for each game-kind module. Each `() => Promise<…>` triggers a
 * separate chunk via Vite's code-splitting. Use `createResource` in GameShell
 * to load these asynchronously — the component already has a loading fallback
 * for the (brief) gap.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const GAME_MODULES: Record<string, () => Promise<GameKindModule<any, any, any>>> = {
  akron: () => import("@mcts/akron").then((m) => m.akronModule),
  atarigo: () => import("@mcts/atarigo").then((m) => m.atarigoModule),
  breakthrough: () => import("@mcts/breakthrough").then((m) => m.breakthroughModule),
  congo: () => import("@mcts/congo").then((m) => m.congoModule),
  druid: () => import("@mcts/druid").then((m) => m.druidModule),
  focus: () => import("@mcts/focus").then((m) => m.focusModule2p),
  "focus:3p": () => import("@mcts/focus").then((m) => m.focusModule3p),
  "focus:4p": () => import("@mcts/focus").then((m) => m.focusModule4p),
  gonnect: () => import("@mcts/gonnect").then((m) => m.gonnectModule),
  "hex-gen": () => import("@mcts/hex-gen").then((m) => m.hexGenModule),
  ingenious: () => import("@mcts/ingenious").then((m) => m.ingeniousModule),
  knightthrough: () => import("@mcts/knightthrough").then((m) => m.knightthroughModule),
  margo: () => import("@mcts/margo").then((m) => m.margoModule),
  othello: () => import("@mcts/othello").then((m) => m.othelloModule),
  tak: () => import("@mcts/tak").then((m) => m.takModule),
  tanbo: () => import("@mcts/tanbo").then((m) => m.tanboModule),
  "traffic-lights": () => import("@mcts/traffic-lights").then((m) => m.trafficLightsModule),
  ttt: () => import("@mcts/ttt").then((m) => m.tttModule),
};

export interface GameMeta {
  players: string[];
  /** The server-side adapter kind this UI id actually talks to over HTTP.
   * Defaults to the id itself (true for every single-variant game). Differs
   * only for a colon-namespaced variant id -- Focus's three player counts
   * are three separate server adapters/binaries (see `games/focus/src/
   * adapter.rs`), but one UI game with a player-count picker. */
  wireKind?: string;
  /** This id's row label in its group's variant sub-selector (e.g.
   * "3 players"). Only meaningful when the id shares a group with others
   * (see `groupIdOf`) -- a single-variant game never renders that selector. */
  variantLabel?: string;
  /** The group's own row label in the top-level game picker. Set only on a
   * group's default/base id (the one with no `:variant` suffix -- see
   * `groupIdOf`). Omit to fall back to `state().gamesInfo`'s label (looked
   * up by `wireKind`), the same as every single-variant game already does;
   * a multi-variant group needs this explicitly since `gamesInfo` only
   * knows the server's individual wire kinds, not the group as a whole. */
  groupLabel?: string;
}

/**
 * Synchronous metadata for each game kind — available without loading the
 * full module (renderer, scene graph libs, etc.). Used by GameShell's
 * new-game dialog to build seat-picker defaults before the module has loaded.
 */
export const GAME_META: Record<string, GameMeta> = {
  akron: { players: ["Black", "White"] },
  atarigo: { players: ["Black", "White"] },
  breakthrough: { players: ["Black", "White"] },
  congo: { players: ["Black", "White"] },
  druid: { players: ["Black", "White"] },
  focus: {
    players: ["P0", "P1"],
    wireKind: "focus-2p",
    groupLabel: "Focus",
    variantLabel: "2 players",
  },
  "focus:3p": { players: ["P0", "P1", "P2"], wireKind: "focus-3p", variantLabel: "3 players" },
  "focus:4p": {
    players: ["P0", "P1", "P2", "P3"],
    wireKind: "focus-4p",
    variantLabel: "4 players",
  },
  gonnect: { players: ["Black", "White"] },
  "hex-gen": { players: ["P0", "P1"] },
  ingenious: { players: ["P0", "P1"] },
  knightthrough: { players: ["Black", "White"] },
  margo: { players: ["Black", "White"] },
  othello: { players: ["Black", "White"] },
  tak: { players: ["White", "Black"] },
  tanbo: { players: ["Black", "White"] },
  "traffic-lights": { players: ["A", "B"] },
  ttt: { players: ["X", "O"] },
};

/** The group a UI id belongs to: everything before its first `:`, or the id
 * itself if it has none. Ids sharing a group are one game's variants
 * (different player counts, board sizes, etc. -- see `GAME_META`'s doc
 * comment), collapsed to a single top-level entry in the game picker with a
 * secondary variant selector. */
export function groupIdOf(id: string): string {
  const i = id.indexOf(":");
  return i === -1 ? id : id.slice(0, i);
}

/** The actual server-side kind a UI id maps to -- see `GameMeta.wireKind`. */
export function wireKindOf(id: string): string {
  return GAME_META[id]?.wireKind ?? id;
}

export const DEFAULT_GAME_KIND = "druid";
