// packages/focus/src/index.ts — Focus's `GameKindModule`s, one per player
// count. Every variant shares this one package/renderer/backend
// (games/focus/src/adapter.rs's `FocusAdapter<const P: usize>`); only the
// player list and the display id differ. Registered in `app/src/games.ts`'s
// `GAME_MODULES`/`GAME_META` under the colon-namespaced ids
// `focus`/`focus:3p`/`focus:4p` -- see that file's `groupIdOf`/`wireKindOf`
// for how those collapse to one entry in the game picker.

import type { GameKindModule } from "@mcts/game";
import { FocusRenderer } from "./FocusRenderer.js";
import { formatMove, makeSummarize, modes } from "./summary.js";
import type { Move } from "./move-codec.js";
import type { GameState, GameView } from "./types.js";

export * from "./types.js";
export * from "./move-codec.js";
export * from "./geometry.js";
export { FocusRenderer } from "./FocusRenderer.js";
export { formatMove, modes, PLAYER_COLORS } from "./summary.js";

function makeFocusModule(id: string, players: string[]): GameKindModule<GameState, Move, GameView> {
  return {
    kind: id,
    players,
    Renderer: FocusRenderer,
    summarize: makeSummarize(players),
    modes,
    formatMove,
  };
}

export const focusModule2p = makeFocusModule("focus", ["P0", "P1"]);
export const focusModule3p = makeFocusModule("focus:3p", ["P0", "P1", "P2"]);
export const focusModule4p = makeFocusModule("focus:4p", ["P0", "P1", "P2", "P3"]);
