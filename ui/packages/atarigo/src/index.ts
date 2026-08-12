// packages/atarigo/src/index.ts — AtariGo's `GameKindModule`. A goban game
// where capturing a single stone wins, playable on 5×5, 7×7, or 9×9 boards
// (see `games/atarigo/src/main.rs`'s `SUPPORTED_SIZES`). Shares board
// rendering, the size picker, and stone-count summary with `@mcts/goban` —
// only the supported sizes and win-condition banner text are specific to
// AtariGo, and rules live entirely server-side in `games/atarigo`.

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

/** Mirrors `games/atarigo/src/main.rs`'s `SUPPORTED_SIZES`/`DEFAULT_SIZE`. */
const SIZES = [5, 7, 9];
const DEFAULT_SIZE = 9;

export const atarigoModule: GameKindModule<GameState, Move, GameView> = {
  kind: "atarigo",
  players: ["Black", "White"],
  Renderer: GobanRenderer,
  NewGameFields: createSizeField(SIZES, DEFAULT_SIZE),
  summarize: createSimpleSummary((winner) =>
    winner ? `${winner} wins by capturing a stone!` : "No moves left — draw.",
  ),
  formatMove,
};
