// summary.ts — Focus's `GameSummary`/`GameModeDef`s/`formatMove`, plus the
// player-index-to-color palette every other file in this package shares.
// `terminal`/`winner`/`reserves` are already computed server-side
// (`games/focus/src/adapter.rs`'s `view()`), so this is pure formatting, no
// client-side rule logic.
//
// `players` (the module's fixed player-id list, e.g. `["P0", "P1", "P2"]`
// for the 3-player variant) isn't known by this file alone -- it varies per
// variant -- so `summarize` is built by `makeSummarize(players)` rather than
// exported directly; `index.ts`'s `makeFocusModule` calls it once per
// variant module.

import type { GameModeDef, GameSummary } from "@mcts/game";
import { coordFor } from "./geometry.js";
import { isSlideMove, moveCell, moveCount, moveDir } from "./move-codec.js";
import type { Move } from "./move-codec.js";
import type { GameState, GameView } from "./types.js";

/** One swatch per player index -- red/blue/teal/gold, chosen for mutual
 * contrast at a glance rather than any thematic meaning (Focus has no
 * canonical player colors the way Black/White games do). Indexed 0..3,
 * covering the 2/3/4-player variants. */
export const PLAYER_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"];

const DIR_LABEL = ["N", "E", "S", "W"];

export function makeSummarize(players: string[]): (view: GameView) => GameSummary {
  function reserveLines(view: GameView): GameSummary["lines"] {
    return players.map((p, i) => {
      const n = view.reserves[i] ?? 0;
      return { id: p, text: `${p} — ${n} reserve${n === 1 ? "" : "s"}`, swatch: PLAYER_COLORS[i] };
    });
  }

  return function summarize(view: GameView): GameSummary {
    if (view.terminal) {
      const winnerName = view.winner !== null ? players[view.winner] : null;
      return {
        turnText: "Game over",
        bannerText: winnerName ? `${winnerName} wins!` : "No one can move — draw.",
        bannerColor: view.winner !== null ? PLAYER_COLORS[view.winner] : undefined,
        lines: reserveLines(view),
        currentPlayer: null,
      };
    }
    return {
      turnText: `${players[view.current_player] ?? "?"} to move`,
      bannerText: "",
      lines: reserveLines(view),
      currentPlayer: players[view.current_player] ?? null,
    };
  };
}

export function formatMove(move: Move, _before: GameState): string {
  const cell = moveCell(move);
  if (!isSlideMove(move)) return `place ${coordFor(cell)}`;
  return `${coordFor(cell)} ${DIR_LABEL[moveDir(move)]}${moveCount(move)}`;
}

/** One mode per move kind: `Place` (from reserve) and `Slide` (slide/split a
 * stack). `GameShell` filters `legalMoves` through the active mode's filter
 * before handing them to the renderer, so `FocusRenderer` never needs its
 * own mode toggle. */
export const modes: GameModeDef<Move>[] = [
  { id: "place", label: "Place", hotkey: "1", filter: (m) => !isSlideMove(m) },
  { id: "slide", label: "Slide / Split", hotkey: "2", filter: isSlideMove },
];
