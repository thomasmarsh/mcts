// packages/knightthrough/src/index.ts — Knightthrough's `GameKindModule`.
// A chess-knight-move variant of Breakthrough. Shares types, renderer, and
// summary with the `@mcts/chess-variant` package — only difference from
// Breakthrough is the move-generation rules, which live server-side.

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

export const knightthroughModule: GameKindModule<GameState, Move, GameView> = {
  kind: "knightthrough",
  players: ["Black", "White"],
  Renderer: ChessVariantRenderer,
  summarize,
  formatMove,
};
