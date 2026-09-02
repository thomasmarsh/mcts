// BoardViewport — the one read-only board. Given a loaded `GameKindModule`
// and a trace position (`state` + root-to-here `history`), it mounts that
// game's `Renderer` with every interaction disabled. The spectator and any
// future tuner pair playback share this component so there is a single
// place that knows how to drive a game renderer in playback mode. It takes
// an already-loaded module so it stays free of the `GAME_MODULES` registry;
// it lives here, beside that registry and its only callers, not in a
// shared package.

import { Dynamic } from "solid-js/web";
import { Show, type Component, type JSX } from "solid-js";
import type { GameKindModule, MoveStep } from "@mcts/game";

function readonlyView(state: unknown): unknown {
  return state && typeof state === "object"
    ? { ...(state as Record<string, unknown>), terminal: false, winner: null }
    : { terminal: false, winner: null };
}

export interface BoardViewportProps {
  /** The already-loaded module for this game kind, or null while it loads /
   * when the kind has no renderer. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  module: GameKindModule<any, any, any> | null;
  state: unknown;
  history: MoveStep<unknown, unknown>[];
  loading?: JSX.Element;
}

export const BoardViewport: Component<BoardViewportProps> = (props) => (
  <Show
    when={props.module}
    fallback={props.loading ?? <div class="log-empty">Loading board…</div>}
  >
    {(mod) => (
      <Dynamic
        component={mod().Renderer}
        state={props.state}
        view={readonlyView(props.state)}
        history={props.history}
        legalMoves={[]}
        busy={true}
        onMove={() => undefined}
        hoveredMove={null}
        onHover={() => undefined}
      />
    )}
  </Show>
);
