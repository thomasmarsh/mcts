// tests/GameShell.test.tsx — Component-level regression tests for two bugs
// only visible at the GameShell effect layer, not in appReducer's own
// contract (packages/game/tests/reducer.test.ts already covers that layer):
//
//   1. The board renderer (DruidRenderer in production) used to unmount and
//      remount on every single move, because `state().position` legitimately
//      goes `null` for one reduction after *every* move/nav while GameShell
//      re-fetches it (see reducer.ts), and the renderer was gated directly
//      on `<Show when={position()}>`. For DruidRenderer that meant a fresh
//      three.js scene/camera/OrbitControls each time -- a visible flash/tear
//      and the camera snapping back to its default framing after every AI
//      move. Fixed by `heldPosition`, a signal that only clears on a real
//      `epoch` change (see GameShell.tsx).
//
//   2 & 3. GameShell's autoplay effect (fire an aiMove whenever it's an
//      AI-controlled seat's turn) used to fire unconditionally, with no
//      check for whether `tree.currentId` was actually the live frontier of
//      play. Navigating back into history (undo/redo/jumpTo -- e.g. the
//      ArrowLeft hotkey, or clicking an earlier move in the history panel)
//      to a node whose mover happened to be AI-controlled immediately
//      re-triggered an aiMove *from that node*, which replayed the existing
//      child and snapped straight back to (or forked past) wherever the
//      user had just navigated to. Fixed by gating the effect on
//      `isFrontier(tree)` (game-tree.ts).
//
// Both are exercised here through the real `appReducer`/`createStore` (no
// TestStore stubbing) against a mocked `Env` -- a fake, rule-free game
// module stands in for Druid/tic-tac-toe (see tests/fixtures/fake-game.tsx)
// so this never needs a real renderer or a live server, only
// `@solidjs/testing-library` + happy-dom.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@solidjs/testing-library";
import { Effect } from "@mcts/core";
import type { AiMoveResult, Env, LegalMovesResult, StateAndView } from "@mcts/game";
import { createTestStore, mockEnv } from "./helpers.js";
import { mountLog, resetMountLog, TERMINAL_AT, viewFor } from "./fixtures/fake-game.js";

vi.mock("../app/src/games.js", () => import("./fixtures/fake-games-registry.js"));

const { GameShell } = await import("../app/src/GameShell.js");

function makeFakeEnv(): Env {
  return {
    ...mockEnv,
    newGame: <S2, V2 = unknown>() => Effect.send({ state: 0, view: viewFor(0) }) as unknown as Effect<StateAndView<S2, V2>>,
    legalMoves: <S2, M2>(_kind: string, state: S2) =>
      Effect.send({ moves: (state as unknown as number) >= TERMINAL_AT ? [] : ["inc"] }) as unknown as Effect<
        LegalMovesResult<M2>
      >,
    view: <S2, V2 = unknown>(_kind: string, state: S2) => Effect.send(viewFor(state as unknown as number)) as unknown as Effect<V2>,
    apply: <S2, M2, V2 = unknown>(_kind: string, state: S2) => {
      const next = (state as unknown as number) + 1;
      return Effect.send({ state: next, view: viewFor(next) }) as unknown as Effect<StateAndView<S2, V2>>;
    },
    aiPresets: () => Effect.send([]),
    aiMove: <S2, M2, V2 = unknown>(_kind: string, state: S2) => {
      const cur = state as unknown as number;
      // If the frontier guard regresses, autoplay could fire from a
      // terminal (or otherwise past-the-end) position -- fail loudly rather
      // than silently returning nonsense.
      if (cur >= TERMINAL_AT) throw new Error("aiMove called at/after TERMINAL_AT -- the frontier guard regressed");
      const next = cur + 1;
      return Effect.send({ move: "inc", state: next, view: viewFor(next) }) as unknown as Effect<AiMoveResult<S2, M2, V2>>;
    },
  };
}

/** Both seats AI-controlled -- lets autoplay alone drive the game to
 * TERMINAL_AT, deterministically, one linear branch (n0..n6, no forks). */
function makeBothSeatsAi(store: ReturnType<typeof createTestStore>["store"]): void {
  store.dispatch({ tag: "setSeat", player: "A", control: "ai" });
  store.dispatch({ tag: "setSeat", player: "B", control: "ai" });
}

/** Polls `getValue()` for `ms` of real wall-clock time, asserting it equals
 * `expected` on every poll -- fails at the first tick it drifts, rather than
 * just checking the final value (which the old autoplay bug could, after a
 * few ticks, settle back to something indistinguishable from "never
 * navigated at all" -- see GameShell.test.tsx's header comment). */
async function holdsSteadyAt(getValue: () => unknown, expected: unknown, ms = 300): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < ms) {
    expect(getValue()).toBe(expected);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

beforeEach(() => {
  resetMountLog();
});

describe("GameShell autoplay/history bugs (fake game, no real server)", () => {
  it("never remounts the board renderer across autoplay moves or history navigation", async () => {
    const { store } = createTestStore("fake", makeFakeEnv());
    render(() => <GameShell store={store} />);

    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());
    expect(mountLog).toEqual(["mount"]);

    makeBothSeatsAi(store);
    await vi.waitFor(() => expect(store.state.tree.nextId).toBe(TERMINAL_AT + 1), { timeout: 5000 });
    await vi.waitFor(() => expect(store.state.position?.nodeId).toBe(store.state.tree.currentId));
    expect(mountLog).toEqual(["mount"]);

    store.dispatch({ tag: "tree", action: { tag: "undo" } });
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n5"));

    store.dispatch({ tag: "tree", action: { tag: "jumpTo", id: "n0" } });
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n0"));

    // Every move above (6 of them) plus two history jumps each put
    // `position` through the transient-null gap the old code unmounted the
    // renderer for -- it should still be exactly the one mount from initial
    // load.
    expect(mountLog).toEqual(["mount"]);
  });

  it("undo (ArrowLeft) moves back one ply and holds -- it used to immediately snap forward again", async () => {
    const { store } = createTestStore("fake", makeFakeEnv());
    render(() => <GameShell store={store} />);
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    makeBothSeatsAi(store);
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n6"), { timeout: 5000 });

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));

    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n5"));
    // n5's mover ("B", AI-controlled) is a leaf-adjacent AI turn -- exactly
    // the case that used to re-fire an aiMove from n5 and snap straight
    // back to n6, one reactive tick later.
    await holdsSteadyAt(() => store.state.tree.currentId, "n5");
  });

  it("clicking a move in the history panel jumps there and holds -- it used to go nowhere the user could see", async () => {
    const { store } = createTestStore("fake", makeFakeEnv());
    render(() => <GameShell store={store} />);
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    makeBothSeatsAi(store);
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n6"), { timeout: 5000 });

    const startRow = screen.getByText("Start").closest("button");
    expect(startRow).not.toBeNull();
    startRow!.click();

    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n0"));
    // Root's mover ("A", AI-controlled) used to cascade autoplay all the
    // way back to n6 within a couple of reactive ticks, making the click
    // look like it did nothing.
    await holdsSteadyAt(() => store.state.tree.currentId, "n0");
  });
});
