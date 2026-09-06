// summary.ts — tic-tac-toe's `GameSummary`/`formatMove`, the per-game half
// of `GameShell`'s HUD chrome. No `modes` (unlike
// Druid's Sarsen/Lintel buttons) -- tic-tac-toe's move space has no
// meaningful subdivision, so the module simply omits that field.

import type { GameSummary } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";

export function summarize(view: GameView): GameSummary {
  if (view.terminal) {
    return view.winner
      ? {
          turnText: "Game over",
          bannerText: `${view.winner} wins!`,
          lines: [],
          currentPlayer: null,
        }
      : {
          turnText: "Game over",
          bannerText: "No moves left — draw.",
          lines: [],
          currentPlayer: null,
        };
  }
  return { turnText: `${view.turn} to move`, bannerText: "", lines: [], currentPlayer: view.turn };
}

export function formatMove(move: Move, before: GameState): string {
  const row = Math.floor(move / 3);
  const col = move % 3;
  return `${before.turn} (${row}, ${col})`;
}
