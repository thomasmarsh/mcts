// summary.ts — Akron's `GameSummary`/`formatMove`, mirroring
// `@mcts/margo`'s `summary.ts` shape. Akron has no "modes" (a move is
// always an add, a move, or the one-off swap, never a player-chosen piece
// kind), so `GameKindModule.modes` is omitted here too.

import type { GameSummary } from "@mcts/game";
import { toCoord } from "@mcts/pyramid";
import type { Action, GameState, GameView } from "./types.js";

const BLACK_SWATCH = "#3a3d46";
const WHITE_SWATCH = "#f2e9d8";

function pieceCounts(view: GameView): { black: number; white: number } {
  let black = 0;
  let white = 0;
  for (const cell of view.cells) {
    if (!cell) continue;
    if (cell.piece === "Black") black++;
    else white++;
  }
  return { black, white };
}

function countLines(view: GameView): GameSummary["lines"] {
  const { black, white } = pieceCounts(view);
  return [
    {
      id: "black",
      text: `Black — ${black} pieces (${view.black_pile} in hand)`,
      swatch: BLACK_SWATCH,
    },
    {
      id: "white",
      text: `White — ${white} pieces (${view.white_pile} in hand)`,
      swatch: WHITE_SWATCH,
    },
  ];
}

export function summarize(view: GameView): GameSummary {
  if (view.terminal) {
    return view.winner
      ? {
          turnText: "Game over",
          bannerText: `${view.winner} wins!`,
          bannerColor: view.winner === "Black" ? "#c9cbd4" : "#f2e9d8",
          lines: countLines(view),
          currentPlayer: null,
        }
      : {
          turnText: "Game over",
          bannerText: "No legal move — draw.",
          bannerColor: "#e8e8ec",
          lines: countLines(view),
          currentPlayer: null,
        };
  }
  return {
    turnText: view.can_swap ? `${view.turn} to move (may swap colours)` : `${view.turn} to move`,
    bannerText: "",
    lines: countLines(view),
    currentPlayer: view.turn,
  };
}

function coordText(n: number, index: number): string {
  const [col, row, level] = toCoord(n, index);
  return `(${col},${row},L${level})`;
}

export function formatMove(move: Action, before: GameState): string {
  if (move === "Swap") return `${before.turn} swap`;
  if ("Add" in move) return `${before.turn} add ${coordText(before.n, move.Add[0])}`;
  const [src, dst] = move.Move;
  return `${before.turn} ${coordText(before.n, src)} → ${coordText(before.n, dst)}`;
}
