// packages/gonnect/src/index.ts — Gonnect's `GameKindModule`. A goban
// connection game where connecting your two opposite edges wins, playable
// on any board from 3×3 to 19×19 (see `games/gonnect/src/main.rs`'s
// `SUPPORTED_SIZES`). Shares board rendering, the size picker, and
// stone-count summary with `@mcts/goban` — only the supported size range and
// win-condition banner text are specific to Gonnect, and rules (including
// the ko/swap-rule handling in the wire state) live entirely server-side in
// `games/gonnect`.

import type { GameKindModule } from "@mcts/game";
import {
  cellOf,
  createSimpleSummary,
  createSizeRangeField,
  formatMove,
  GobanRenderer,
  type GameState,
  type GameView,
  type Move,
} from "@mcts/goban";

export * from "@mcts/goban";

/** Mirrors `games/gonnect/src/main.rs`'s `SUPPORTED_SIZES`/`DEFAULT_SIZE`. */
const MIN_SIZE = 3;
const MAX_SIZE = 19;
const DEFAULT_SIZE = 13;

/** Mirrors `games/gonnect/src/lib.rs`'s `Move::SWAP`/`Move::NO_MOVE` cell
 * sentinels -- `@mcts/goban`'s `formatMove` only knows how to turn an
 * in-range cell index into board notation, so a swap or a forced pass (no
 * legal placement) needs its own label here instead of being decoded as a
 * (garbage) board coordinate. */
const SWAP_CELL = 0xffff;
const NO_MOVE_CELL = 0xfffe;

function formatGonnectMove(move: Move, before: GameState): string {
  const cell = cellOf(move);
  if (cell === SWAP_CELL) return `${before.turn} swap`;
  if (cell === NO_MOVE_CELL) return `${before.turn} pass`;
  return formatMove(move, before);
}

export const gonnectModule: GameKindModule<GameState, Move, GameView> = {
  kind: "gonnect",
  players: ["Black", "White"],
  Renderer: GobanRenderer,
  NewGameFields: createSizeRangeField(MIN_SIZE, MAX_SIZE, DEFAULT_SIZE),
  summarize: createSimpleSummary((winner) =>
    winner ? `${winner} wins by connecting their edges!` : "No moves left — draw.",
  ),
  formatMove: formatGonnectMove,
};
