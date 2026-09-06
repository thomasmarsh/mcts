// packages/druid/src/index.ts — Druid's `GameKindModule`:
// everything `app/src/GameShell.tsx` needs to host Druid, registered
// under the "druid" key in that file's `Record<string, GameKindModule>`.

import type { GameKindModule } from "@mcts/game";
import { DruidRenderer } from "./DruidRenderer.js";
import { NewGameFields } from "./NewGameFields.js";
import { formatMove, modes, summarize } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export { buildStackModel, footprintFor } from "./layers.js";
export type { Beam, LayerEntry, StackModel } from "./layers.js";
export { DruidRenderer } from "./DruidRenderer.js";
export { NewGameFields } from "./NewGameFields.js";
export { summarize, modes, formatMove } from "./summary.js";

export const druidModule: GameKindModule<GameState, Move, GameView> = {
  kind: "druid",
  players: ["Black", "White"],
  Renderer: DruidRenderer,
  summarize,
  modes,
  NewGameFields,
  formatMove,
};
