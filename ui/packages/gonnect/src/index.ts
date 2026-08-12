// packages/gonnect/src/index.ts — Gonnect's `GameKindModule`. A goban
// connection game where connecting your two opposite edges wins, playable
// on 9×9, 13×13, or 19×19 boards (see `games/gonnect/src/main.rs`'s
// `SUPPORTED_SIZES`). Shares board rendering, the size picker, and
// stone-count summary with `@mcts/goban` — only the supported sizes and
// win-condition banner text are specific to Gonnect, and rules (including
// the ko/swap-rule handling in the wire state) live entirely server-side in
// `games/gonnect`.

import type { GameKindModule } from "@mcts/game";
import {
  createSimpleSummary,
  createSizeField,
  formatMove,
  GobanRenderer,
  type GameState,
  type GameView,
  type Move,
} from "@mcts/goban";

export * from "@mcts/goban";

/** Mirrors `games/gonnect/src/main.rs`'s `SUPPORTED_SIZES`/`DEFAULT_SIZE`. */
const SIZES = [9, 13, 19];
const DEFAULT_SIZE = 13;

export const gonnectModule: GameKindModule<GameState, Move, GameView> = {
  kind: "gonnect",
  players: ["Black", "White"],
  Renderer: GobanRenderer,
  NewGameFields: createSizeField(SIZES, DEFAULT_SIZE),
  summarize: createSimpleSummary((winner) =>
    winner ? `${winner} wins by connecting their edges!` : "No moves left — draw.",
  ),
  formatMove,
};
