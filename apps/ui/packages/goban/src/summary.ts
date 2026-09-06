// summary.ts — Generic stone-count `GameSummary`/`formatMove` factories for
// goban games whose win condition is a one-line banner (AtariGo: capture a
// stone; Gonnect: connect opposite edges) rather than Tanbo's
// group-connection scoring, which stays bespoke in `@mcts/tanbo`.

import type { GameSummary } from "@mcts/game";
import { cellOf, type GameState, type GameView, type Move, type Player } from "./types.js";

function countStones(cells: (Player | null)[], player: Player): number {
  let count = 0;
  for (const cell of cells) if (cell === player) count++;
  return count;
}

/** Builds a `summarize` whose only game-specific part is `terminalMessage`,
 * which turns the winner (or `null` for a draw) into the banner's prose. */
export function createSimpleSummary(
  terminalMessage: (winner: Player | null) => string,
): (view: GameView) => GameSummary {
  return (view) => {
    const blackCount = countStones(view.cells, "Black");
    const whiteCount = countStones(view.cells, "White");
    const lines = [
      {
        id: "black",
        text: `Black — ${blackCount} stone${blackCount !== 1 ? "s" : ""}`,
        swatch: "#1a1a1a",
      },
      {
        id: "white",
        text: `White — ${whiteCount} stone${whiteCount !== 1 ? "s" : ""}`,
        swatch: "#e0e0e0",
      },
    ];

    if (view.terminal) {
      return {
        turnText: "Game over",
        bannerText: terminalMessage(view.winner),
        bannerColor:
          view.winner === "Black" ? "#1a1a1a" : view.winner === "White" ? "#e0e0e0" : undefined,
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
  };
}

const COL_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/** Formats a move as e.g. "BlackC4". Board size is derived from
 * `before.cells.length` (always `N * N`) rather than a fixed parameter,
 * since AtariGo/Gonnect's board size is chosen per new-game. */
export function formatMove(move: Move, before: GameState): string {
  const n = Math.round(Math.sqrt(before.cells.length));
  const cell = cellOf(move);
  const row = Math.floor(cell / n) + 1;
  const col = COL_NAMES[cell % n] ?? "?";
  return `${before.turn}${col}${row}`;
}
