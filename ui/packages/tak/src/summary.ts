// summary.ts — Tak's `GameSummary`/`GameModeDef`s, mirroring
// `ui/packages/druid/src/summary.ts`. Unlike Druid, `terminal`/`winner` are
// already computed server-side (`games/tak/src/main.rs`'s `view()`, via the
// engine's own `Tak::<5>::terminal_status`) -- no client-side road/flat-count
// duplication needed here.

import type { GameModeDef, GameSummary } from "@mcts/game";
import { notation } from "./move-codec.js";
import { boardSize, type GameState, type GameView, type Move, type Player } from "./types.js";

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
    turnText: view.opening ? `${view.turn} to move (opening: place your opponent's piece)` : `${view.turn} to move`,
    bannerText: "",
    lines: reserveLines(view),
    currentPlayer: view.turn,
  };
}

/** `before` (the state a move was applied *from*, mirroring `MoveStep`'s own
 * shape) supplies the board width `notation` needs to turn a square index
 * into a PTN coordinate. */
export function formatMove(move: Move, before: GameState): string {
  return notation(move, boardSize(before));
}

/** Mirrors Druid's `Sarsen`/`Lintel↔`/`Lintel↕` modes: one per placement
 * kind, plus a `Move stack` mode for spreads. `GameShell` filters
 * `legalMoves` through the active mode's `filter` before handing them to
 * `TakRenderer`. */
export const modes: GameModeDef<Move>[] = [
  { id: "flat", label: "Flat", hotkey: "1", filter: (m) => m.tag === "Place" && m.kind === "Flat" },
  { id: "wall", label: "Wall", hotkey: "2", filter: (m) => m.tag === "Place" && m.kind === "Wall" },
  { id: "cap", label: "Capstone", hotkey: "3", filter: (m) => m.tag === "Place" && m.kind === "Cap" },
  { id: "move", label: "Move stack", hotkey: "4", filter: (m) => m.tag === "Spread" },
];
