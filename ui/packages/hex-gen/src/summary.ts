// summary.ts — Hex's `GameSummary`/`formatMove`, the per-game half of
// `GameShell`'s HUD chrome. No `modes` (like tic-tac-toe): Hex's move space
// has no meaningful subdivision.

import type { GameSummary } from "@mcts/game";
import type { GameState, GameView, Move, Player } from "./types.js";
import { SIDE } from "./types.js";

/** Prose reminding the player which pair of edges each side connects --
 * not recoverable from `view` alone, since the wire state only carries
 * `turn`/`cells`, not the edge assignment baked into
 * `games/hex-gen/src/lib.rs`'s `Position::winner`. */
function edgesFor(player: Player): string {
  return player === "P0" ? "top ↔ bottom" : "left ↔ right";
}

export function summarize(view: GameView): GameSummary {
  if (view.terminal) {
    return view.winner
      ? {
          turnText: "Game over",
          bannerText: `${view.winner} wins by connecting ${edgesFor(view.winner)}!`,
          lines: [],
          currentPlayer: null,
        }
      : { turnText: "Game over", bannerText: "No moves left — draw.", lines: [], currentPlayer: null };
  }
  return {
    turnText: `${view.turn} to move (connect ${edgesFor(view.turn)})`,
    bannerText: "",
    lines: [],
    currentPlayer: view.turn,
  };
}

export function formatMove(move: Move, before: GameState): string {
  const row = Math.floor(move / SIDE);
  const col = move % SIDE;
  return `${before.turn} (${col}, ${row})`;
}
