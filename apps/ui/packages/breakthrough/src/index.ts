// packages/breakthrough/src/index.ts — Breakthrough's `GameKindModule`.
// A pawn-move game on an 8×8 board. Shares types, renderer, and summary
// with the `@mcts/chess-variant` package — only difference from
// Knightthrough is the move-generation rules, which live server-side.

import type { GameKindModule } from "@mcts/game";
import {
  ChessVariantRenderer,
  formatMove,
  summarize,
  type GameState,
  type GameView,
  type Move,
} from "@mcts/chess-variant";

export * from "@mcts/chess-variant";

export const breakthroughModule: GameKindModule<GameState, Move, GameView> = {
  kind: "breakthrough",
  players: ["Black", "White"],
  Renderer: ChessVariantRenderer,
  summarize,
  formatMove,
};
