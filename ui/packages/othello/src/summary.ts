// summary.ts — Othello's `GameSummary`/`formatMove`, the per-game half
// of `GameShell`'s HUD chrome. No `modes` (unlike Druid's Sarsen/Lintel
// buttons) — Othello's move space has no meaningful subdivision.

import type { GameSummary } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";

const COL_NAMES = "abcdefgh";
const ROW_NAMES = "12345678";

export function summarize(view: GameView): GameSummary {
  const blackCount = popcount(view.black);
  const whiteCount = popcount(view.white);
  const discText = `● ${blackCount}  ○ ${whiteCount}`;

  if (view.terminal) {
    return view.winner
      ? {
          turnText: "Game over",
          bannerText: `${view.winner} wins!`,
          lines: [{ id: "discs", text: discText }],
          currentPlayer: null,
        }
      : {
          turnText: "Game over",
          bannerText: "No moves left — draw.",
          lines: [{ id: "discs", text: discText }],
          currentPlayer: null,
        };
  }
  const turnLine = view.last_pass ? `${view.turn} to move (last move was a pass)` : `${view.turn} to move`;
  return {
    turnText: turnLine,
    bannerText: "",
    lines: [{ id: "discs", text: discText }],
    currentPlayer: view.turn,
  };
}

export function formatMove(move: Move, before: GameState): string {
  if (move === 64) return `${before.turn} passes`;
  const row = Math.floor(move / 8);
  const col = move % 8;
  const file = COL_NAMES[col] ?? "?";
  const rank = ROW_NAMES[row] ?? "?";
  return `${before.turn} ${file}${rank}`;
}

/** Population count for a hex-encoded u64 board value. Counts via `BigInt`
 * rather than JS's `>>>`/`&`/`Math.imul` -- those coerce to 32-bit ints
 * (`ToUint32`), so the Hamming-weight trick this used to use silently
 * ignored every bit above position 31 on a real (non-empty-upper-half)
 * board. */
function popcount(hex: string): number {
  let v = BigInt(`0x${hex}`);
  let count = 0;
  while (v !== 0n) {
    v &= v - 1n;
    count++;
  }
  return count;
}