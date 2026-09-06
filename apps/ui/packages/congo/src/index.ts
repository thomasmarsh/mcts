// index.ts — Assembles the `GameKindModule` GameShell loads for "congo".

import type { GameKindModule } from "@mcts/game";
import { CongoRenderer } from "./CongoRenderer.js";
import { summarize, formatMove } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export { CongoRenderer } from "./CongoRenderer.js";
export { summarize, formatMove } from "./summary.js";

export const congoModule: GameKindModule<GameState, Move, GameView> = {
  kind: "congo",
  players: ["Black", "White"],
  Renderer: CongoRenderer,
  summarize,
  formatMove,
};
