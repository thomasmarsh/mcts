// packages/ttt/src/index.ts — tic-tac-toe's `GameKindModule` (PLAN-UI.md
// session 8): everything `app/src/GameShell.tsx` needs to host it, registered
// under the "ttt" key in `app/src/games.ts`'s `Record<string, GameKindModule>`.

import type { GameKindModule } from "@mcts/game";
import { TttRenderer } from "./TttRenderer.js";
import { formatMove, summarize } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export { TttRenderer } from "./TttRenderer.js";
export { summarize, formatMove } from "./summary.js";

export const tttModule: GameKindModule<GameState, Move, GameView> = {
  kind: "ttt",
  players: ["X", "O"],
  Renderer: TttRenderer,
  summarize,
  formatMove,
};
