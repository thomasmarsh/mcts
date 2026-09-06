// packages/atarigo/src/index.ts — AtariGo's `GameKindModule`. A goban game
// where capturing a single stone wins, playable on any board from 3×3 to
// 19×19 (see `games/atarigo/src/main.rs`'s `SUPPORTED_SIZES`). Shares board
// rendering, the size picker, and stone-count summary with `@mcts/goban` —
// only the supported size range and win-condition banner text are specific
// to AtariGo, and rules live entirely server-side in `games/atarigo`.

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

/** Mirrors `games/atarigo/src/main.rs`'s `SUPPORTED_SIZES`/`DEFAULT_SIZE`. */
const MIN_SIZE = 3;
const MAX_SIZE = 19;
const DEFAULT_SIZE = 9;

export const atarigoModule: GameKindModule<GameState, Move, GameView> = {
  kind: "atarigo",
  players: ["Black", "White"],
  Renderer: GobanRenderer,
  NewGameFields: createSizeRangeField(MIN_SIZE, MAX_SIZE, DEFAULT_SIZE),
  summarize: createSimpleSummary((winner) =>
    winner ? `${winner} wins by capturing a stone!` : "No moves left — draw.",
  ),
  formatMove,
};
