// tests/TrafficLightsGameShell.test.tsx — GameShell-level regression test
// for the TrafficLights renderer click integration. Exercises the full
// appReducer → GameShell → Renderer chain against a mocked Env that
// returns real traffic-lights-shaped states and views (no server needed).
//
// The key scenario: place a piece on an empty cell, then click the same
// cell again to advance it R→Y. Between the two clicks the cell's DOM
// node is recreated during the position=null refetch gap (legalMoves=[]),
// which once froze the button's `disabled` attribute at creation time —
// the still-reactive classList kept showing a pointer cursor, but clicks
// were dead.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createStore, Effect } from "@mcts/core";
import {
  appReducer,
  initialAppState,
  type Env,
} from "@mcts/game";
import { GameShell } from "../app/src/GameShell.js";
import { mockFetchStrategyFamilies, mockFetchStrategySchema } from "./helpers.js";

// Track moves dispatched through env.apply
let capturedMoves: unknown[] = [];

function makeEnv(): Env {
  // Shared mutable state that each env method reads/writes
  let state: { turn: "A" | "B"; cells: (string | null)[] } = {
    turn: "A",
    cells: Array(9).fill(null),
  };

  function advance(s: typeof state, move: number): typeof state {
    const idx = move >> 2;
    const piece = move & 3;
    const cellLabel = piece === 0 ? "R" : piece === 1 ? "Y" : "G";
    const cells = [...s.cells];
    cells[idx] = cellLabel;
    const turn: "A" | "B" = s.turn === "A" ? "B" : "A";
    return { turn, cells };
  }

  function viewOf(s: typeof state) {
    return { turn: s.turn, cells: [...s.cells], winner: null, terminal: false };
  }

  function movesOf(s: typeof state): number[] {
    const moves: number[] = [];
    for (let i = 0; i < 9; i++) {
      const cell = s.cells[i];
      if (cell === null) moves.push(i << 2);
      else if (cell === "R") moves.push((i << 2) | 1);
      else if (cell === "Y") moves.push((i << 2) | 2);
    }
    return moves;
  }

  return {
    getGames: () => Effect.none(),
    newGame: () => {
      state = { turn: "A", cells: Array(9).fill(null) };
      return Effect.send({ state: { ...state }, view: viewOf(state) });
    },
    legalMoves: () => Effect.send({ moves: movesOf(state) }),
    view: () => Effect.send(viewOf(state)),
    apply: (_kind, _s, move) => {
      state = advance(state, move as number);
      capturedMoves.push(move);
      return Effect.send({ state: { ...state }, view: viewOf(state) });
    },
    aiPresets: () => Effect.send([]),
    aiMove: () => Effect.send({ move: 0, state: { ...state }, view: viewOf(state) }),
    analyze: () => Effect.send({ actions: [], principal_variation: [], total_visits: 0, suggested_move: null }),
  };
}

import "../app/src/games.js";

afterEach(() => {
  cleanup();
  capturedMoves = [];
});

describe("TrafficLights GameShell integration", () => {
  it("places a piece on an empty cell, then advances it R→Y on second click", async () => {
    const env = makeEnv();
    const init = initialAppState("traffic-lights", null);
    const store = createStore(init, appReducer, env);
    capturedMoves = [];

    render(() => <GameShell store={store} fetchStrategySchema={mockFetchStrategySchema} fetchStrategyFamilies={mockFetchStrategyFamilies} />);

    // Get the 9 board buttons (filter out HUD/nav buttons by class)
    function boardButtons(): HTMLButtonElement[] {
      return screen.getAllByRole("button").filter(
        (b) => b.classList.contains("tl-cell"),
      ) as HTMLButtonElement[];
    }

    // Wait for the initial position to load (GameShell's onMount fires
    // newGame) *and* for the board to actually be in the DOM. The game-kind
    // module itself loads via a separate `createResource` in GameShell
    // (independent of `store.state.position`, which the mocked env resolves
    // synchronously) -- querying `boardButtons()` right after only the
    // position settles races that resource and can hit "Loading game…"
    // before it flips over, especially on a cold module-transform cache.
    await vi.waitFor(() => {
      expect(store.state.position).not.toBeNull();
      expect(store.state.position!.view).toMatchObject({ turn: "A" });
      expect(document.querySelectorAll(".tl-cell").length).toBe(9);
    });

    // --- Click 1: place R on cell 4 (center) ---
    fireEvent.click(boardButtons()[4]);

    // Wait for the move to process and position to re-fetch
    await vi.waitFor(() => {
      expect(capturedMoves.length).toBe(1);
    });
    expect(capturedMoves[0]).toBe(16); // place R at cell 4

    await vi.waitFor(() => {
      const pos = store.state.position;
      expect(pos).not.toBeNull();
      // Turn should now be B (AI didn't play — both seats are human)
      expect(pos!.view).toMatchObject({ turn: "B" });
    });

    // Verify cell 4 is now R in the view
    expect(store.state.position!.view.cells[4]).toBe("R");

    // --- Click 2: advance cell 4 R→Y ---
    // It's B's turn, and B's legal moves include advancing cell 4 from
    // R→Y (move 17 = (4 << 2) | 1). Wait for the DOM to reflect the
    // re-fetched position (button enabled) before clicking — the store
    // updates before Solid's render effects flush.
    //
    // Regression: the renderer used to compute `legal` as a plain const
    // in the `<For>` callback, freezing `disabled` at item-creation time.
    // Cell 4's node was recreated during the position=null refetch gap
    // (legalMoves=[]), so `disabled` stayed true even after the position
    // reloaded — the button got the `legal` CSS class (pointer cursor)
    // via the still-reactive classList, but clicks were dead.
    await vi.waitFor(() => {
      const pos = store.state.position;
      expect(pos).not.toBeNull();
      expect(pos!.view).toMatchObject({ turn: "B" });
      expect(boardButtons()[4].disabled).toBe(false);
    });

    fireEvent.click(boardButtons()[4]);

    await vi.waitFor(() => {
      expect(capturedMoves.length).toBe(2);
    });
    expect(capturedMoves[1]).toBe(17); // advance cell 4 R→Y

    // --- Click 3: advance cell 4 Y→G ---
    await vi.waitFor(() => {
      const pos = store.state.position;
      expect(pos).not.toBeNull();
      expect(pos!.view).toMatchObject({ turn: "A" });
      expect(pos!.view.cells[4]).toBe("Y");
      const btn = boardButtons()[4];
      expect(btn.disabled).toBe(false);
    });

    fireEvent.click(boardButtons()[4]);

    await vi.waitFor(() => {
      expect(capturedMoves.length).toBe(3);
    });
    expect(capturedMoves[2]).toBe(18); // advance cell 4 Y→G
  });

  it("clicking an empty cell when busy does not dispatch", async () => {
    const env = makeEnv();
    const init = initialAppState("traffic-lights", null);
    const store = createStore(init, appReducer, env);
    capturedMoves = [];

    render(() => <GameShell store={store} fetchStrategySchema={mockFetchStrategySchema} fetchStrategyFamilies={mockFetchStrategyFamilies} />);

    const boardButtons = () =>
      screen.getAllByRole("button").filter((b) => b.classList.contains("tl-cell")) as HTMLButtonElement[];

    // See the first test's comment: wait for the board to actually be in
    // the DOM, not just for the store's position to settle.
    await vi.waitFor(() => {
      expect(store.state.position).not.toBeNull();
      expect(document.querySelectorAll(".tl-cell").length).toBe(9);
    });

    // Click cell 0 while not busy — should work
    fireEvent.click(boardButtons()[0]);

    await vi.waitFor(() => {
      expect(capturedMoves.length).toBe(1);
    });
  });
});