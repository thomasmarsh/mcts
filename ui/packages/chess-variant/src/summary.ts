// summary.ts — Chess-variant `GameSummary`/`formatMove`, shared by
// Knightthrough and Breakthrough (both have the same piece types,
// player labels, and coordinate system).

import type { GameSummary } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";

const COL_NAMES = "abcdefgh";

export function summarize(view: GameView): GameSummary {
  const blackCount = popcount(view.black);
  const whiteCount = popcount(view.white);
  const pieceText = `● ${blackCount}  ○ ${whiteCount}`;

  if (view.terminal) {
    return view.winner
      ? {
          turnText: "Game over",
          bannerText: `${view.winner} wins by reaching the back rank!`,
          lines: [{ id: "pieces", text: pieceText }],
          currentPlayer: null,
        }
      : {
          turnText: "Game over",
          bannerText: "No moves left — draw.",
          lines: [{ id: "pieces", text: pieceText }],
          currentPlayer: null,
        };
  }
  return {
    turnText: `${view.turn} to move`,
    bannerText: "",
    lines: [{ id: "pieces", text: pieceText }],
    currentPlayer: view.turn,
  };
}

export function formatMove(move: Move, before: GameState): string {
  const [src, dst] = move;
  const srcRow = Math.floor(src / 8);
  const srcCol = src % 8;
  const dstRow = Math.floor(dst / 8);
  const dstCol = dst % 8;
  const srcFile = COL_NAMES[srcCol] ?? "?";
  const dstFile = COL_NAMES[dstCol] ?? "?";
  return `${before.turn} ${srcFile}${srcRow + 1}→${dstFile}${dstRow + 1}`;
}

/** Population count for a u64 hex string. Uses BigInt to preserve all 64 bits. */
function popcount(hex: string): number {
  let n = BigInt(`0x${hex}`);
  let count = 0;
  while (n > 0n) {
    count += 1;
    n &= n - 1n; // clear lowest set bit (Brian Kernighan's algorithm)
  }
  return count;
}
