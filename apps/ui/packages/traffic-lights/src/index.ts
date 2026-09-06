// packages/traffic-lights/src/index.ts — traffic lights' `GameKindModule`:
// everything `app/src/GameShell.tsx` needs to host it, registered
// under the "traffic-lights" key in `app/src/games.ts`'s
// `Record<string, GameKindModule>`.

import type { GameKindModule } from "@mcts/game";
import { TrafficLightsRenderer } from "./TrafficLightsRenderer.js";
import { formatMove, summarize } from "./summary.js";
import type { GameState, GameView, Move } from "./types.js";

export * from "./types.js";
export { TrafficLightsRenderer } from "./TrafficLightsRenderer.js";
export { summarize, formatMove } from "./summary.js";

export const trafficLightsModule: GameKindModule<GameState, Move, GameView> = {
  kind: "traffic-lights",
  players: ["A", "B"],
  Renderer: TrafficLightsRenderer,
  summarize,
  formatMove,
};
