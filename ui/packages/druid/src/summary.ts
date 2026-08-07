// summary.ts — Druid's `GameSummary`/`GameModeDef`s: the per-game half of
// `GameShell`'s HUD chrome (turn indicator, banner, hand counts, mode
// buttons), ported from app.js's `updateHud`/`movesForMode`.

import type { GameModeDef, GameSummary } from "@mcts/game";
import type { GameView, Move } from "./types.js";

const BLACK_SWATCH = "#3a3d46";
const WHITE_SWATCH = "#f2e9d8";

export function summarize(view: GameView): GameSummary {
  if (view.terminal) {
    return view.winner
      ? {
          turnText: "Game over",
          bannerText: `${view.winner} wins!`,
          bannerColor: view.winner === "Black" ? "#c9cbd4" : "#f2e9d8",
          lines: handLines(view),
          currentPlayer: null,
        }
      : {
          turnText: "Game over",
          bannerText: "No moves left — draw.",
          bannerColor: "#e8e8ec",
          lines: handLines(view),
          currentPlayer: null,
        };
  }
  return {
    turnText: `${view.player} to move`,
    bannerText: "",
    lines: handLines(view),
    currentPlayer: view.player,
  };
}

function handLines(view: GameView): GameSummary["lines"] {
  return [
    {
      id: "black",
      text: `Black — ${view.hand_black.sarsens} sarsens, ${view.hand_black.lintels} lintels`,
      swatch: BLACK_SWATCH,
    },
    {
      id: "white",
      text: `White — ${view.hand_white.sarsens} sarsens, ${view.hand_white.lintels} lintels`,
      swatch: WHITE_SWATCH,
    },
  ];
}

/** Mirrors app.js's `mode`/`movesForMode`/`HOTKEYS`: which piece a click
 * places. `GameShell` filters `legalMoves` through the active mode's
 * `filter` before handing them to `DruidRenderer`. */
export const modes: GameModeDef<Move>[] = [
  { id: "sarsen", label: "Sarsen", hotkey: "1", filter: ([piece]) => piece === "Sarsen" },
  {
    id: "lintelH",
    label: "Lintel ↔",
    hotkey: "2",
    filter: ([piece]) => typeof piece === "object" && piece.Lintel === "Horizontal",
  },
  {
    id: "lintelV",
    label: "Lintel ↕",
    hotkey: "3",
    filter: ([piece]) => typeof piece === "object" && piece.Lintel === "Vertical",
  },
];
