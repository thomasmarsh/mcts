// games.ts — The game-kind registry `GameShell` picks a renderer/module
// from (PLAN-UI.md session 4). One entry per game; session 8 adds
// `ttt: tttModule` here and nowhere else. Necessarily type-erased at this
// boundary (a single `Record` can't carry each module's own concrete
// S/M/V) -- the TS-side mirror of `GameAdapter`'s `Value` erasure on the
// Rust side (see `server/adapters/mod.rs`).

import type { GameKindModule } from "@mcts/game";
import { druidModule } from "@mcts/druid";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const GAME_MODULES: Record<string, GameKindModule<any, any, any>> = {
  druid: druidModule,
};

export const DEFAULT_GAME_KIND = "druid";
