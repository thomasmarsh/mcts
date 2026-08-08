// games.ts — The game-kind registry `GameShell` picks a renderer/module
// from. One entry per game. Necessarily type-erased at this
// boundary (a single `Record` can't carry each module's own concrete
// S/M/V) -- the TS-side mirror of `GameAdapter`'s `Value` erasure on the
// Rust side (see `server/adapters/mod.rs`).

import type { GameKindModule } from "@mcts/game";
import { druidModule } from "@mcts/druid";
import { trafficLightsModule } from "@mcts/traffic-lights";
import { tttModule } from "@mcts/ttt";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const GAME_MODULES: Record<string, GameKindModule<any, any, any>> = {
  druid: druidModule,
  "traffic-lights": trafficLightsModule,
  ttt: tttModule,
};

/** Display labels for the game-kind picker (New Game dialog) --
 * `GameKindModule` itself carries no display label (Rust's
 * `GameAdapter::label` isn't mirrored on the TS side, since nothing
 * needs a kind-picker UI to show one besides this). */
export const GAME_LABELS: Record<string, string> = {
  druid: "Druid",
  "traffic-lights": "Traffic Lights",
  ttt: "Tic-Tac-Toe",
};

export const DEFAULT_GAME_KIND = "druid";
