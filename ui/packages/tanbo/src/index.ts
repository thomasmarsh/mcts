// packages/tanbo/src/index.ts — Tanbo's `GameKindModule`:
// everything `app/src/GameShell.tsx` needs to host it, registered
// under the "tanbo" key in `app/src/games.ts`'s `Record<string, GameKindModule>`.

import type { GameKindModule } from "@mcts/game";
import { TanboRenderer } from "./TanboRenderer.js";
import { formatMove, summarize } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export { TanboRenderer } from "./TanboRenderer.js";
export { summarize, formatMove } from "./summary.js";

export const tanboModule: GameKindModule<GameState, Move, GameView> = {
  kind: "tanbo",
  players: ["Black", "White"],
  Renderer: TanboRenderer,
  summarize,
  formatMove,
};