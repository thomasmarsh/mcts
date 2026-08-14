// packages/hex-gen/src/index.ts — Hex's `GameKindModule`, registered under
// the "hex-gen" key in `app/src/games.ts`'s `Record<string, GameKindModule>`.
// Backed server-side by `games/hex-gen` (gdl's Core-IR-to-Rust codegen
// output, see that crate's doc comment) via `games/hex-gen/src/main.rs`'s
// subprocess adapter -- the first hexagonal-board game wired into this UI.

import type { GameKindModule } from "@mcts/game";
import { HexGenRenderer } from "./HexGenRenderer.js";
import { formatMove, summarize } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export { HexGenRenderer } from "./HexGenRenderer.js";
export { summarize, formatMove } from "./summary.js";

export const hexGenModule: GameKindModule<GameState, Move, GameView> = {
  kind: "hex-gen",
  players: ["P0", "P1"],
  Renderer: HexGenRenderer,
  summarize,
  formatMove,
};
