// packages/gonnect/src/index.ts — Gonnect's `GameKindModule`. A goban
// connection game where connecting your two opposite edges wins, playable
// on any board from 3×3 to 19×19 (see `games/gonnect/src/main.rs`'s
// `SUPPORTED_SIZES`). Shares board rendering, the size picker, and
// stone-count summary with `@mcts/goban` — only the supported size range and
// win-condition banner text are specific to Gonnect, and rules (including
// the ko/swap-rule handling in the wire state) live entirely server-side in
// `games/gonnect`.

import type { GameKindModule } from "@mcts/game";
import {
  createSimpleSummary,
  createSizeRangeField,
  formatMove,
  GobanRenderer,
  type GameState,
  type GameView,
  type Move,
} from "@mcts/goban";

export * from "@mcts/goban";

/** Mirrors `games/gonnect/src/main.rs`'s `SUPPORTED_SIZES`/`DEFAULT_SIZE`. */
const MIN_SIZE = 3;
const MAX_SIZE = 19;
const DEFAULT_SIZE = 13;

export const gonnectModule: GameKindModule<GameState, Move, GameView> = {
  kind: "gonnect",
  players: ["Black", "White"],
  Renderer: GobanRenderer,
  NewGameFields: createSizeRangeField(MIN_SIZE, MAX_SIZE, DEFAULT_SIZE),
  summarize: createSimpleSummary((winner) =>
    winner ? `${winner} wins by connecting their edges!` : "No moves left — draw.",
  ),
  formatMove,
};
