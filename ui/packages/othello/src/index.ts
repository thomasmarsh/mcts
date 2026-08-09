// packages/othello/src/index.ts — Othello's `GameKindModule`:
// everything `app/src/GameShell.tsx` needs to host it, registered
// under the "othello" key in `app/src/games.ts`'s
// `Record<string, GameKindModule>`.

import type { GameKindModule } from "@mcts/game";
import { OthelloRenderer } from "./OthelloRenderer.js";
import { formatMove, summarize } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export { OthelloRenderer } from "./OthelloRenderer.js";
export { summarize, formatMove } from "./summary.js";

export const othelloModule: GameKindModule<GameState, Move, GameView> = {
  kind: "othello",
  players: ["Black", "White"],
  Renderer: OthelloRenderer,
  summarize,
  formatMove,
};