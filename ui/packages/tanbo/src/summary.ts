// summary.ts — Tanbo's `GameSummary`/`formatMove`, displaying the
// group-connection score format: "Black — X stones, Y groups" style info.

import type { GameSummary } from "@mcts/game";
import type { GameState, GameView, Move } from "./types.js";

/** Count connected groups of a given colour on the board. */
function countGroups(cells: (string | null)[], player: string): number {
  const visited = new Array<boolean>(81).fill(false);
  let groups = 0;
  for (let i = 0; i < 81; i++) {
    if (cells[i] !== player || visited[i]) continue;
    groups++;
    // Flood-fill this group.
    const stack = [i];
    visited[i] = true;
    while (stack.length > 0) {
      const idx = stack.pop()!;
      const r = Math.floor(idx / 9);
      const c = idx % 9;
      for (const nb of [[r - 1, c] as const, [r + 1, c] as const, [r, c - 1] as const, [r, c + 1] as const]) {
        const nr = nb[0];
        const nc = nb[1];
        if (nr < 0 || nr >= 9 || nc < 0 || nc >= 9) continue;
        const ni = nr * 9 + nc;
        if (cells[ni] === player && !visited[ni]) {
          visited[ni] = true;
          stack.push(ni);
        }
      }
    }
  }
  return groups;
}

export function summarize(view: GameView): GameSummary {
  if (view.terminal) {
    const msg = view.winner
      ? `${view.winner} wins (all opponent stones eliminated)!`
      : "No moves left — draw.";
    return {
      turnText: "Game over",
      bannerText: msg,
      bannerColor: view.winner === "Black" ? "#1a1a1a" : "#e0e0e0",
      lines: [],
      currentPlayer: null,
    };
  }
  const blackGroups = countGroups(view.cells, "Black");
  const whiteGroups = countGroups(view.cells, "White");
  return {
    turnText: `${view.turn} to move`,
    bannerText: "",
    lines: [
      { id: "black", text: `Black — ${blackGroups} group${blackGroups !== 1 ? "s" : ""}`, swatch: "#1a1a1a" },
      { id: "white", text: `White — ${whiteGroups} group${whiteGroups !== 1 ? "s" : ""}`, swatch: "#e0e0e0" },
    ],
    currentPlayer: view.turn,
  };
}

export function formatMove(move: Move, before: GameState): string {
  const row = Math.floor(move / 9) + 1;
  const col = String.fromCharCode(65 + (move % 9)); // A-I
  return `${before.turn}${col}${row}`;
}