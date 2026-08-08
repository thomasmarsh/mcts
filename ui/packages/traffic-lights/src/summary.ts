// summary.ts — traffic lights' `GameSummary`/`formatMove`, the per-game half
// of `GameShell`'s HUD chrome. No `modes` (unlike Druid's Sarsen/Lintel
// buttons) — traffic lights' move space is unsubdivided, so the module
// simply omits that field.

import type { GameSummary } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";

export function summarize(view: GameView): GameSummary {
  if (view.terminal) {
    return view.winner
      ? {
          turnText: "Game over",
          bannerText: `Player ${view.winner} wins!`,
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
  return {
    turnText: `Player ${view.turn} to move`,
    bannerText: "",
    lines: [],
    currentPlayer: view.turn,
  };
}

export function formatMove(move: Move, _before: GameState): string {
  const index = move >> 2;
  const row = Math.floor(index / 3);
  const col = index % 3;
  return `(${row}, ${col})`;
}