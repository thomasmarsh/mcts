// packages/hex-gen/src/index.ts — Hex's `GameKindModule`, registered under
// the "hex-gen" key in `app/src/games.ts`'s `Record<string, GameKindModule>`.
// Backed server-side by `games/hex-gen` (gdl's Core-IR-to-Rust codegen
// output, see that crate's doc comment) via `games/hex-gen/src/main.rs`'s
// subprocess adapter -- the first hexagonal-board game wired into this UI.
// Playable at 5x5, 7x7, or regulation 11x11 (mirrors
// `games/hex-gen/src/main.rs`'s `SUPPORTED_SIZES`); reuses `@mcts/goban`'s
// `createSizeField` for the new-game size picker the same way `@mcts/gonnect`
// does for its own (non-goban) board-size choice.

import type { GameKindModule } from "@mcts/game";
import { createSizeField } from "@mcts/goban";
import { HexGenRenderer } from "./HexGenRenderer.js";
import { formatMove, summarize } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export { HexGenRenderer } from "./HexGenRenderer.js";
export { summarize, formatMove } from "./summary.js";

const SIZES = [5, 7, 11];
const DEFAULT_SIZE = 11;

export const hexGenModule: GameKindModule<GameState, Move, GameView> = {
  kind: "hex-gen",
  players: ["P0", "P1"],
  Renderer: HexGenRenderer,
  NewGameFields: createSizeField(SIZES, DEFAULT_SIZE),
  summarize,
  formatMove,
};
