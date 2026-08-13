// summary.ts — Congo's `GameSummary`/`formatMove`. Piece counts (not stone
// counts) are the natural per-side tally here since captures remove pieces
// of many different kinds, not a single fungible unit.

import type { GameSummary } from "@mcts/game";
import { RIVER_ROW, SIZE, type GameState, type GameView, type Move } from "./types.js";

const COL_NAMES = "abcdefg";

function squareName(square: number): string {
  const row = Math.floor(square / SIZE);
  const col = square % SIZE;
  const file = COL_NAMES[col] ?? "?";
  const rank = SIZE - row;
  return `${file}${rank}`;
}

function countPieces(state: GameState, player: "Black" | "White"): number {
  let count = 0;
  for (const cell of state.squares) if (cell?.player === player) count++;
  return count;
}

export function summarize(view: GameView): GameSummary {
  const blackCount = countPieces(view, "Black");
  const whiteCount = countPieces(view, "White");
  const lines = [
    { id: "black", text: `Black — ${blackCount} piece${blackCount !== 1 ? "s" : ""}`, swatch: "#2a2a2a" },
    { id: "white", text: `White — ${whiteCount} piece${whiteCount !== 1 ? "s" : ""}`, swatch: "#f0ead6" },
  ];

  if (view.terminal) {
    return {
      turnText: "Game over",
      bannerText: view.winner
        ? `${view.winner} wins — the ${view.winner === "Black" ? "White" : "Black"} lion has fallen!`
        : "Bare lions, neither can strike — draw.",
      bannerColor: view.winner === "Black" ? "#2a2a2a" : view.winner === "White" ? "#f0ead6" : undefined,
      lines,
      currentPlayer: null,
    };
  }
  return {
    turnText: `${view.turn} to move`,
    bannerText: "",
    lines,
    currentPlayer: view.turn,
  };
}

const PIECE_LABEL: Record<string, string> = {
  giraffe: "Giraffe",
  monkey: "Monkey",
  elephant: "Elephant",
  lion: "Lion",
  crocodile: "Crocodile",
  zebra: "Zebra",
  pawn: "Pawn",
  superpawn: "Superpawn",
};

export function formatMove(move: Move, before: GameState): string {
  const piece = before.squares[move.from]?.piece;
  const label = piece ? PIECE_LABEL[piece] : "?";
  const sep = move.captures.length > 0 ? "x" : "-";
  const river = move.to >= RIVER_ROW * SIZE && move.to < (RIVER_ROW + 1) * SIZE ? " ~" : "";
  return `${before.turn} ${label} ${squareName(move.from)}${sep}${squareName(move.to)}${river}`;
}
