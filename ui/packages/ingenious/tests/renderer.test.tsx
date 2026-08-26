// tests/renderer.test.tsx — IngeniousRenderer interaction tests: does
// selecting a rack tile, optionally flipping it, and clicking the resulting
// board overlay actually dispatch the matching `Action::Place` move? The
// renderer resolves a tile's ambiguous orientation via a type-selection +
// flip step (see IngeniousRenderer.tsx's doc comment) rather than per-click
// disambiguation, so this is the path most likely to silently drop a click.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { IngeniousRenderer } from "../src/index.js";
import { neighborOf } from "../src/geometry.js";
import { SIDE, type Color, type GameState, type GameView, type Move } from "../src/types.js";

afterEach(() => cleanup());

const CENTER_CELL = 84; // row 6, col 6 -- the grid center, always on the 2-player board.
const DIR_N = 0;
const NEIGHBOR_CELL = neighborOf(CENTER_CELL, DIR_N)!;

function emptyBoard(): (Color | null)[] {
  return new Array(SIDE * SIDE).fill(null);
}

function makeState(overrides: Partial<GameState> = {}): GameState {
  return {
    board: emptyBoard(),
    board_tile_counts: [],
    racks: [
      [["Red", "Blue"], null, null, null, null, null],
      [null, null, null, null, null, null],
    ],
    score: [
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
    ],
    bonus_used: [
      [false, false, false, false, false, false],
      [false, false, false, false, false, false],
    ],
    has_moved: [false, false],
    claimed_symbols: [false, false, false, false, false, false],
    current_player: 0,
    phase: "place",
    pending_bonus: 0,
    winner_immediate: null,
    rng: 0,
    ...overrides,
  };
}

function makeView(state: GameState, overrides: Partial<GameView> = {}): GameView {
  return { ...state, winner: null, terminal: false, ...overrides };
}

function placeMove(colorA: Color, colorB: Color): Move {
  return { Place: { cell: CENTER_CELL, dir: DIR_N, color_a: colorA, color_b: colorB } };
}

function renderBoard(props: {
  state: GameState;
  legalMoves: Move[];
  busy?: boolean;
  onMove?: (move: Move) => void;
}) {
  const onMove = props.onMove ?? vi.fn();
  render(() => (
    <IngeniousRenderer
      state={props.state}
      view={makeView(props.state)}
      history={[]}
      legalMoves={props.legalMoves}
      busy={props.busy ?? false}
      onMove={onMove}
      hoveredMove={null}
      onHover={vi.fn()}
    />
  ));
  return onMove;
}

function rackTileButton(): HTMLButtonElement {
  return document.querySelector(".ingenious-tile") as HTMLButtonElement;
}

function placementOverlays(): NodeListOf<Element> {
  return document.querySelectorAll(".ingenious-placement");
}

describe("IngeniousRenderer tile placement", () => {
  it("clicking the board overlay after selecting a rack tile dispatches the matching Place move", () => {
    const state = makeState();
    const legalMoves = [placeMove("Red", "Blue"), placeMove("Blue", "Red")];
    const onMove = renderBoard({ state, legalMoves });

    // No tile selected yet -- no placement overlay should be interactive.
    expect(placementOverlays()).toHaveLength(0);

    fireEvent.click(rackTileButton());

    // Selecting the rack's only tile type pins color_a/color_b to its
    // unflipped orientation, so exactly one of the two legal moves matches.
    const overlays = placementOverlays();
    expect(overlays).toHaveLength(1);

    fireEvent.click(overlays[0]!);
    expect(onMove).toHaveBeenCalledTimes(1);
    expect(onMove).toHaveBeenCalledWith(placeMove("Red", "Blue"));
  });

  it("flipping the selected tile dispatches the swapped-color move instead", () => {
    const state = makeState();
    const legalMoves = [placeMove("Red", "Blue"), placeMove("Blue", "Red")];
    const onMove = renderBoard({ state, legalMoves });

    fireEvent.click(rackTileButton());
    fireEvent.click(screen.getByRole("button", { name: "Flip" }));

    const overlays = placementOverlays();
    expect(overlays).toHaveLength(1);
    fireEvent.click(overlays[0]!);
    expect(onMove).toHaveBeenCalledWith(placeMove("Blue", "Red"));
  });

  it("a same-color tile offers no flip button", () => {
    const state = makeState({
      racks: [
        [["Red", "Red"], null, null, null, null, null],
        [null, null, null, null, null, null],
      ],
    });
    renderBoard({ state, legalMoves: [placeMove("Red", "Red")] });

    fireEvent.click(rackTileButton());
    expect(screen.queryByRole("button", { name: "Flip" })).toBeNull();
  });

  it("does not dispatch while busy", () => {
    const state = makeState();
    const legalMoves = [placeMove("Red", "Blue"), placeMove("Blue", "Red")];
    const onMove = renderBoard({ state, legalMoves, busy: true });

    const tileButton = rackTileButton();
    expect(tileButton.disabled).toBe(true);
    fireEvent.click(tileButton);
    expect(placementOverlays()).toHaveLength(0);
    expect(onMove).not.toHaveBeenCalled();
  });
});

describe("IngeniousRenderer swap decision", () => {
  it("dispatches KeepRack, and disables Swap when it isn't legal", () => {
    const state = makeState({ phase: "swap_decision" });
    const onMove = renderBoard({ state, legalMoves: ["KeepRack"] });

    const keep = screen.getByRole("button", { name: "Keep & refill" });
    const swap = screen.getByRole("button", { name: "Swap rack" }) as HTMLButtonElement;
    expect(swap.disabled).toBe(true);

    fireEvent.click(keep);
    expect(onMove).toHaveBeenCalledWith("KeepRack");
  });

  it("dispatches Swap when it is legal", () => {
    const state = makeState({ phase: "swap_decision" });
    const onMove = renderBoard({ state, legalMoves: ["KeepRack", "Swap"] });

    fireEvent.click(screen.getByRole("button", { name: "Swap rack" }));
    expect(onMove).toHaveBeenCalledWith("Swap");
  });
});

// Sanity check on this test file's own geometry assumption -- if this ever
// fails, `CENTER_CELL`/`DIR_N` above no longer name real adjacent cells.
describe("test fixture geometry", () => {
  it("CENTER_CELL and NEIGHBOR_CELL are real adjacent cells on the board", () => {
    expect(NEIGHBOR_CELL).not.toBeNull();
    expect(neighborOf(CENTER_CELL, DIR_N)).toBe(NEIGHBOR_CELL);
  });
});
