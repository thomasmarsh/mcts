// summary.ts — Margo's `GameSummary`/`formatMove`: the per-game half of
// `GameShell`'s HUD chrome (turn indicator, banner, piece-count lines),
// following Druid's `summary.ts` shape. Margo has no "modes" (a move is
// always either a placement or the one-off swap, never a player-chosen
// piece kind), so unlike Druid this module exports no `modes` array --
// `GameKindModule.modes` is optional for exactly this case.

import type { GameSummary } from "@mcts/game";
import { toCoord } from "./geometry.js";
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
    { id: "black", text: `Black — ${black} pieces`, swatch: BLACK_SWATCH },
    { id: "white", text: `White — ${white} pieces`, swatch: WHITE_SWATCH },
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
          bannerText: "Equal pieces — draw.",
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
  const [index] = move.Place;
  return `${before.turn} ${coordText(before.n, index)}`;
}
