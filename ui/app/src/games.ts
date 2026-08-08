// games.ts — The game-kind registry `GameShell` picks a renderer/module
// from. One entry per game. Necessarily type-erased at this
// boundary (a single `Record` can't carry each module's own concrete
// S/M/V) -- the TS-side mirror of `GameAdapter`'s `Value` erasure on the
// Rust side (see `server/adapters/mod.rs`).
//
// Game-kind display labels are not maintained here: they come from
// `GET /api/games` (`env.getGames()`) fetched on mount and stored in
// `AppState.gamesInfo` (see `App.tsx`). If a caller needs a display name, it
// should look up the kind in `state().gamesInfo` or fall back to the raw
// kind string.

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

export const DEFAULT_GAME_KIND = "druid";