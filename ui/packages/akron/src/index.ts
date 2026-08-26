// packages/akron/src/index.ts — Akron's `GameKindModule`: everything
// `app/src/GameShell.tsx` needs to host Akron, registered under the
// "akron" key in that file's `Record<string, GameKindModule>`.

import type { GameKindModule } from "@mcts/game";
import { AkronRenderer } from "./AkronRenderer.js";
import { NewGameFields } from "./NewGameFields.js";
import { formatMove, summarize } from "./summary.js";
import type { Action, GameState, GameView } from "./types.js";

export * from "./types.js";
export * from "@mcts/pyramid";
export { AkronRenderer } from "./AkronRenderer.js";
export { NewGameFields } from "./NewGameFields.js";
export { summarize, formatMove } from "./summary.js";

export const akronModule: GameKindModule<GameState, Action, GameView> = {
  kind: "akron",
  players: ["Black", "White"],
  Renderer: AkronRenderer,
  summarize,
  NewGameFields,
  formatMove,
};
