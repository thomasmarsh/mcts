// packages/margo/src/index.ts — Margo's `GameKindModule`: everything
// `app/src/GameShell.tsx` needs to host Margo, registered under the
// "margo" key in that file's `Record<string, GameKindModule>`.

import type { GameKindModule } from "@mcts/game";
import { MargoRenderer } from "./MargoRenderer.js";
import { NewGameFields } from "./NewGameFields.js";
import { formatMove, summarize } from "./summary.js";
import type { Action, GameState, GameView } from "./types.js";

export * from "./types.js";
export * from "./geometry.js";
export { MargoRenderer } from "./MargoRenderer.js";
export { NewGameFields } from "./NewGameFields.js";
export { summarize, formatMove } from "./summary.js";

export const margoModule: GameKindModule<GameState, Action, GameView> = {
  kind: "margo",
  players: ["Black", "White"],
  Renderer: MargoRenderer,
  summarize,
  NewGameFields,
  formatMove,
};
