// packages/tak/src/index.ts — Tak's `GameKindModule`: everything
// `app/src/GameShell.tsx` needs to host Tak, registered under the "tak" key
// in that file's `Record<string, GameKindModule>` (see app/src/games.ts).

import type { GameKindModule } from "@mcts/game";
import { NewGameFields } from "./NewGameFields.js";
import { formatMove, modes, summarize } from "./summary.js";
import { TakRenderer } from "./TakRenderer.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export {
  coordFor,
  footprintFor,
  isCapPlacement,
  isFlatPlacement,
  isPlacement,
  isWallPlacement,
  notation,
  parsePtn,
  type ParsedMove,
  type Direction,
} from "./move-codec.js";
export { parseTps, type ParsedStack, type ParsedTps } from "./tps-parser.js";
export { TakRenderer } from "./TakRenderer.js";
export { NewGameFields } from "./NewGameFields.js";
export { summarize, modes, formatMove } from "./summary.js";
export { findWinningRoad } from "./TakRenderer.js";

export const takModule: GameKindModule<GameState, Move, GameView> = {
  kind: "tak",
  players: ["White", "Black"],
  Renderer: TakRenderer,
  summarize,
  modes,
  NewGameFields,
  formatMove,
};
