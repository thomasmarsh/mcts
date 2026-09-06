// packages/ingenious/src/index.ts — Ingenious's `GameKindModule`, registered
// under the "ingenious" key in `app/src/games.ts`'s
// `Record<string, GameKindModule>`. Backed server-side by `games/ingenious`
// via `games/ingenious/src/main.rs`'s subprocess adapter -- the 2-player
// board only (see that file's doc comment).

import type { GameKindModule } from "@mcts/game";
import { IngeniousRenderer } from "./IngeniousRenderer.js";
import { formatMove, summarize } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export * from "./geometry.js";
export { IngeniousRenderer } from "./IngeniousRenderer.js";
export { summarize, formatMove, COLOR_HEX, isMaxed } from "./summary.js";

export const ingeniousModule: GameKindModule<GameState, Move, GameView> = {
  kind: "ingenious",
  players: ["P0", "P1"],
  Renderer: IngeniousRenderer,
  summarize,
  formatMove,
};
