// tests/fixtures/fake-game.tsx — A minimal `GameKindModule` for GameShell
// component tests (see ../GameShell.test.tsx). Deliberately rule-free: state
// is just a move counter, the only legal move is "inc", and the game ends
// after TERMINAL_AT moves -- just enough shape to exercise GameShell's own
// tree/autoplay/renderer-mounting logic against, without dragging in a real
// renderer (Druid's three.js `WebGLRenderer` needs a real canvas context,
// which happy-dom doesn't provide).

import { type Component, onCleanup, onMount } from "solid-js";
import type { GameKindModule, GameRendererProps } from "@mcts/game";

export const TERMINAL_AT = 6;
export const PLAYERS = ["A", "B"] as const;

export interface FakeView {
  turn: string | null;
  terminal: boolean;
}

export function viewFor(state: number): FakeView {
  return { turn: state >= TERMINAL_AT ? null : PLAYERS[state % 2]!, terminal: state >= TERMINAL_AT };
}

/** Every mount/cleanup of `FakeRenderer`, in order -- what the no-remount
 * regression test in GameShell.test.tsx inspects. `GameShell` used to
 * unmount/remount its renderer (via `<Show when={position()}>`) on every
 * move, since `position` legitimately goes `null` for one reduction after
 * every move/nav while it's re-fetched (see reducer.ts) -- which, for
 * DruidRenderer, meant a fresh three.js scene/camera/OrbitControls every
 * time (a visible flash/tear and a snapped-back camera). Reset between
 * tests with `resetMountLog()`. */
export const mountLog: string[] = [];

export function resetMountLog(): void {
  mountLog.length = 0;
}

export const FakeRenderer: Component<GameRendererProps<number, string, FakeView>> = (props) => {
  onMount(() => mountLog.push("mount"));
  onCleanup(() => mountLog.push("cleanup"));
  return <div data-testid="fake-board">state:{props.state}</div>;
};

export const fakeModule: GameKindModule<number, string, FakeView> = {
  kind: "fake",
  players: [...PLAYERS],
  Renderer: FakeRenderer,
  summarize: (view) => ({
    turnText: view.terminal ? "Game over" : `${view.turn} to move`,
    bannerText: view.terminal ? "Done" : "",
    lines: [],
    currentPlayer: view.turn,
  }),
  formatMove: (move, before) => `${move}#${before}`,
};
