// summary.ts — Tak's `GameSummary`/`GameModeDef`s. `terminal`/`winner` are
// already computed server-side (`games/tak/src/main.rs`'s `view()`), and
// moves are PTN strings -- no client-side parsing needed for the summary.
// Mode filters use `move-codec.ts`'s quick classifiers (no full parse).

import type { GameModeDef, GameSummary } from "@mcts/game";
import { isCapPlacement, isFlatPlacement, isWallPlacement, isPlacement } from "./move-codec.js";
import type { GameState, GameView, Move, Player } from "./types.js";

const WHITE_SWATCH = "#f2e9d8";
const BLACK_SWATCH = "#3a3d46";

function reserveLines(view: GameView): GameSummary["lines"] {
  const label = (p: Player, idx: 0 | 1) => {
    const stones = view.stones[idx];
    const caps = view.caps[idx];
    return `${p} — ${stones} stone${stones === 1 ? "" : "s"}, ${caps} capstone${caps === 1 ? "" : "s"}`;
  };
  return [
    { id: "white", text: label("White", 0), swatch: WHITE_SWATCH },
    { id: "black", text: label("Black", 1), swatch: BLACK_SWATCH },
  ];
}

export function summarize(view: GameView): GameSummary {
  if (view.terminal) {
    return view.winner
      ? {
          turnText: "Game over",
          bannerText: `${view.winner} wins!`,
          bannerColor: view.winner === "Black" ? "#c9cbd4" : WHITE_SWATCH,
          lines: reserveLines(view),
          currentPlayer: null,
        }
      : {
          turnText: "Game over",
          bannerText: "Board full — draw.",
          bannerColor: "#e8e8ec",
          lines: reserveLines(view),
          currentPlayer: null,
        };
  }
  return {
    turnText: view.opening
      ? `${view.turn} to move (opening: place your opponent's piece)`
      : `${view.turn} to move`,
    bannerText: "",
    lines: reserveLines(view),
    currentPlayer: view.turn,
  };
}

/** Moves are PTN strings -- `formatMove` is the identity (the PTN string
 * is already human-readable notation). If the PTN string includes
 * informational marks (`*`, `'`, `!`, `?`), they're displayed as-is. */
export function formatMove(move: Move, _before: GameState): string {
  return move;
}

/** Mirrors Druid's `Sarsen`/`Lintel↔`/`Lintel↕` modes: one per placement
 * kind, plus a `Move stack` mode for spreads. `GameShell` filters
 * `legalMoves` through the active mode's `filter`.
 *
 * Filters use quick string classifiers (`isFlatPlacement` etc.) rather
 * than a full `parsePtn` on every legal move -- placement/spread
 * disambiguation only needs to check the presence of a direction glyph
 * and an optional `S`/`C` prefix. */
export const modes: GameModeDef<Move>[] = [
  { id: "flat", label: "Flat", hotkey: "1", filter: isFlatPlacement },
  { id: "wall", label: "Wall", hotkey: "2", filter: isWallPlacement },
  { id: "cap", label: "Capstone", hotkey: "3", filter: isCapPlacement },
  { id: "move", label: "Move stack", hotkey: "4", filter: (m) => !isPlacement(m) },
];
