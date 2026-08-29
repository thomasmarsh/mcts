// tests/renderer.test.tsx — IngeniousRenderer interaction tests: does
// selecting a rack tile and performing the two-click board placement (pick a
// start cell, then a completing neighbor) dispatch the matching `Action::Place`
// move? Placement is resolved entirely on the board (no popup), so this is the
// path most likely to silently drop or misroute a click.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { IngeniousRenderer } from "../src/index.js";
import { neighborOf } from "../src/geometry.js";
import { SIDE, type Color, type GameState, type GameView, type Move } from "../src/types.js";

afterEach(() => cleanup());

const CENTER_CELL = 84; // row 6, col 6 -- the grid center, always on the 2-player board.
const DIR_N = 0;
const DIR_E = 2;
const DIR_NE = 4;
const NEIGHBOR_CELL = neighborOf(CENTER_CELL, DIR_N)!; // 97
const EAST_NEIGHBOR_CELL = neighborOf(CENTER_CELL, DIR_E)!; // 85

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

function placeMove(colorA: Color, colorB: Color, dir: number = DIR_N): Move {
  return { Place: { cell: CENTER_CELL, dir, color_a: colorA, color_b: colorB } };
}

function renderBoard(props: {
  state: GameState;
  legalMoves: Move[];
  busy?: boolean;
  onMove?: (move: Move) => void;
  hoveredMove?: Move | null;
  onHover?: (move: Move | null) => void;
}) {
  const onMove = props.onMove ?? vi.fn();
  const onHover = props.onHover ?? vi.fn();
  render(() => (
    <IngeniousRenderer
      state={props.state}
      view={makeView(props.state)}
      history={[]}
      legalMoves={props.legalMoves}
      busy={props.busy ?? false}
      onMove={onMove}
      hoveredMove={props.hoveredMove ?? null}
      onHover={onHover}
    />
  ));
  return onMove;
}

function rackTileButton(): HTMLButtonElement {
  return document.querySelector(".ingenious-tile-half") as HTMLButtonElement;
}

function rackTileOtherHalf(): HTMLButtonElement {
  return document.querySelectorAll(".ingenious-tile-half")[1] as HTMLButtonElement;
}

/** A board cell associated with a placement interaction. */
function cellGhost(role: string, c: number): HTMLElement {
  return document.querySelector(`[data-role="${role}"][data-cell="${c}"]`) as HTMLElement;
}

function countCells(role: string): number {
  return document.querySelectorAll(`[data-role="${role}"]`).length;
}

describe("IngeniousRenderer tile placement", () => {
  it("keeps the board clear until hover, then dispatches a two-click placement", () => {
    const state = makeState();
    const move = placeMove("Red", "Blue");
    const onMove = renderBoard({ state, legalMoves: [move] });

    // No tile selected yet -- the board has no placement ghosts.
    expect(countCells("preview")).toBe(0);

    fireEvent.click(rackTileButton());
    // Choosing a tile must not wash the board out with every possible move.
    expect(countCells("preview")).toBe(0);
    fireEvent.mouseEnter(cellGhost("cell-hit", CENTER_CELL));
    expect(cellGhost("preview", NEIGHBOR_CELL)).not.toBeNull();
    expect(
      document.querySelectorAll(".ingenious-preview-anchor .ingenious-ghost-hex"),
    ).toHaveLength(1);
    expect(document.querySelectorAll(".ingenious-preview .ingenious-ghost-hex")).toHaveLength(1);

    // First click anchors the hovered endpoint.
    fireEvent.click(cellGhost("cell-hit", CENTER_CELL));
    expect(cellGhost("anchor", CENTER_CELL)).not.toBeNull();
    // Its completing neighbor lights up as the required second click.
    expect(cellGhost("target", NEIGHBOR_CELL)).not.toBeNull();
    expect(cellGhost("target", NEIGHBOR_CELL).querySelector(".ingenious-ghost-hex")).toBeNull();
    expect(onMove).not.toHaveBeenCalled();

    // Second click finalizes the move.
    fireEvent.click(cellGhost("target", NEIGHBOR_CELL));
    expect(onMove).toHaveBeenCalledTimes(1);
    expect(onMove).toHaveBeenCalledWith(move);
  });

  it("choosing the other tile half selects the swapped-color move", () => {
    const state = makeState();
    // Only the Blue/Red placement is legal; it must use the Blue tile half.
    const flippedMove = placeMove("Blue", "Red");
    const onMove = renderBoard({ state, legalMoves: [flippedMove] });

    fireEvent.click(rackTileButton()); // Red half
    // Red cannot anchor this move at the engine's encoded first endpoint.
    fireEvent.mouseEnter(cellGhost("cell-hit", CENTER_CELL));
    expect(countCells("preview")).toBe(0);

    fireEvent.click(rackTileOtherHalf()); // Blue half
    fireEvent.mouseEnter(cellGhost("cell-hit", CENTER_CELL));
    expect(countCells("preview")).toBe(1);

    fireEvent.click(cellGhost("cell-hit", CENTER_CELL));
    fireEvent.click(cellGhost("target", NEIGHBOR_CELL));
    expect(onMove).toHaveBeenCalledWith(flippedMove);
  });

  it("offers every legal direction from a tile as a separate second click", () => {
    const state = makeState();
    const northMove = placeMove("Red", "Blue", DIR_N);
    const eastMove = placeMove("Red", "Blue", DIR_E);
    const onMove = renderBoard({ state, legalMoves: [northMove, eastMove] });

    fireEvent.click(rackTileButton());
    fireEvent.click(cellGhost("cell-hit", CENTER_CELL));
    // Two distinct completing neighbors, one per direction.
    expect(cellGhost("target", NEIGHBOR_CELL)).not.toBeNull();
    expect(cellGhost("target", EAST_NEIGHBOR_CELL)).not.toBeNull();

    // Clicking the east neighbor finishes the east placement.
    fireEvent.click(cellGhost("target", EAST_NEIGHBOR_CELL));
    expect(onMove).toHaveBeenCalledWith(eastMove);
  });

  it("allows either endpoint to anchor a legal edge", () => {
    const state = makeState();
    const move = placeMove("Red", "Blue");
    const flipped = placeMove("Blue", "Red");
    const onMove = renderBoard({ state, legalMoves: [move, flipped] });

    fireEvent.click(rackTileButton()); // Red half
    // The engine records this edge as 84 -> 97, but 97 is equally usable as
    // the first click, exposing the opposite three directions on the board.
    fireEvent.click(cellGhost("cell-hit", NEIGHBOR_CELL));
    expect(cellGhost("anchor", NEIGHBOR_CELL)).not.toBeNull();
    expect(cellGhost("target", CENTER_CELL)).not.toBeNull();

    fireEvent.click(cellGhost("target", CENTER_CELL));
    expect(onMove).toHaveBeenCalledWith(flipped);
  });

  it("gives each orientation through one start cell its own completing neighbor", () => {
    // Three placements {84,97}, {97,98}, {84,98} share cells pairwise. The
    // {84,98} placement shares 84 with one sibling and 98 with the other, so it
    // has no cell to itself -- it must still be reachable, now via its own
    // distinct completing neighbor rather than being unreachable.
    const state = makeState();
    const aN: Move = placeMove("Red", "Blue", DIR_N); // 84 -> 97
    const bE: Move = {
      Place: { cell: NEIGHBOR_CELL, dir: DIR_E, color_a: "Red", color_b: "Blue" },
    }; // 97 -> 98
    const aNE: Move = placeMove("Red", "Blue", DIR_NE); // 84 -> 98
    const onMove = renderBoard({ state, legalMoves: [aN, bE, aNE] });

    fireEvent.click(rackTileButton());
    // The board remains clear until one of its cells is hovered or clicked.
    expect(countCells("preview")).toBe(0);

    fireEvent.click(cellGhost("cell-hit", CENTER_CELL));
    // Both completing neighbors are offered.
    expect(cellGhost("target", NEIGHBOR_CELL)).not.toBeNull();
    expect(cellGhost("target", 98)).not.toBeNull();

    // Clicking 98 finishes the {84,98} placement (the no-unique-cell one).
    fireEvent.click(cellGhost("target", 98));
    expect(onMove).toHaveBeenCalledWith(aNE);
  });

  it("clicking the start cell again cancels the in-progress placement", () => {
    const state = makeState();
    const move = placeMove("Red", "Blue");
    const onMove = renderBoard({ state, legalMoves: [move] });

    fireEvent.click(rackTileButton());
    fireEvent.click(cellGhost("cell-hit", CENTER_CELL));
    expect(cellGhost("target", NEIGHBOR_CELL)).not.toBeNull();

    // Re-click the start cell to cancel.
    fireEvent.click(cellGhost("anchor", CENTER_CELL));
    expect(cellGhost("cell-hit", CENTER_CELL)).not.toBeNull();
    expect(cellGhost("target", NEIGHBOR_CELL)).toBeNull();
    expect(onMove).not.toHaveBeenCalled();
  });

  it("does not dispatch while busy", () => {
    const state = makeState();
    const legalMoves = [placeMove("Red", "Blue"), placeMove("Blue", "Red")];
    const onMove = renderBoard({ state, legalMoves, busy: true });

    const tileButton = rackTileButton();
    expect(tileButton.disabled).toBe(true);
    fireEvent.click(tileButton);
    expect(countCells("cell-hit")).toBe(0);
    expect(onMove).not.toHaveBeenCalled();
  });

  it("uses the same anchor color for both halves of a same-color tile", () => {
    const state = makeState({
      racks: [
        [["Red", "Red"], null, null, null, null, null],
        [null, null, null, null, null, null],
      ],
    });
    renderBoard({ state, legalMoves: [placeMove("Red", "Red")] });

    fireEvent.click(rackTileButton());
    expect(screen.getAllByRole("button", { name: "Place Red first" })).toHaveLength(2);
  });
});

describe("IngeniousRenderer swap decision", () => {
  it("automatically keeps the rack when swapping is not an option", () => {
    const state = makeState({ phase: "swap_decision" });
    const onMove = renderBoard({ state, legalMoves: ["KeepRack"] });

    expect(onMove).toHaveBeenCalledWith("KeepRack");
    expect(screen.queryByRole("button", { name: "Keep & refill" })).toBeNull();
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
