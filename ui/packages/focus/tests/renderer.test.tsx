// tests/renderer.test.tsx — FocusRenderer interaction tests: does clicking a
// board cell dispatch the right move for both the placement path (single
// click) and the two-stage slide path (select a source, then a destination)?
// This is the path most likely to silently drop a click, same rationale as
// ingenious's own renderer.test.tsx.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@solidjs/testing-library";
import { FocusRenderer } from "../src/FocusRenderer.js";
import { VALID_CELLS } from "../src/geometry.js";
import { placeMove, slideMove } from "../src/move-codec.js";
import type { GameState, GameView, Move } from "../src/index.js";

afterEach(() => cleanup());

function emptyBoard(): number[][] {
  return Array.from({ length: 64 }, () => []);
}

function makeState(overrides: Partial<GameState> = {}): GameState {
  return {
    cells: new Array(64).fill(0),
    reserves: [4, 4],
    captured: [
      [0, 0],
      [0, 0],
    ],
    turn: 0,
    hash: 0,
    ...overrides,
  };
}

function makeView(overrides: Partial<GameView> = {}): GameView {
  return {
    board: emptyBoard(),
    reserves: [4, 4],
    captured: [
      [0, 0],
      [0, 0],
    ],
    current_player: 0,
    winner: null,
    terminal: false,
    ...overrides,
  };
}

function renderBoard(props: {
  state?: GameState;
  view?: GameView;
  legalMoves: Move[];
  busy?: boolean;
  hoveredMove?: Move | null;
}) {
  const onMove = vi.fn();
  const onHover = vi.fn();
  render(() => (
    <FocusRenderer
      state={props.state ?? makeState()}
      view={props.view ?? makeView()}
      history={[]}
      legalMoves={props.legalMoves}
      busy={props.busy ?? false}
      onMove={onMove}
      hoveredMove={props.hoveredMove ?? null}
      onHover={onHover}
    />
  ));
  return { onMove, onHover };
}

function cellButtons(): HTMLButtonElement[] {
  return Array.from(document.querySelectorAll(".focus-cell"));
}

// Board cell index -> its position among rendered `.focus-cell` buttons
// (invalid corner cells render as a plain, unbuttoned `.focus-gap` div, so
// the Nth valid cell is not cell index N).
function buttonForCell(cell: number): HTMLButtonElement {
  const idx = VALID_CELLS.indexOf(cell);
  const btn = cellButtons()[idx];
  if (!btn) throw new Error(`cell ${cell} has no rendered button`);
  return btn;
}

describe("FocusRenderer placement", () => {
  it("clicking a placeable cell dispatches the matching Place move", () => {
    const legalMoves = [placeMove(27), placeMove(28)];
    const { onMove } = renderBoard({ legalMoves });

    fireEvent.click(buttonForCell(27));

    expect(onMove).toHaveBeenCalledTimes(1);
    expect(onMove).toHaveBeenCalledWith(placeMove(27));
  });

  it("does not dispatch while busy", () => {
    const legalMoves = [placeMove(27)];
    const { onMove } = renderBoard({ legalMoves, busy: true });

    const btn = buttonForCell(27);
    expect(btn.disabled).toBe(true);
    fireEvent.click(btn);
    expect(onMove).not.toHaveBeenCalled();
  });
});

describe("FocusRenderer slide two-stage selection", () => {
  it("selecting a source then its destination dispatches the matching Slide move", () => {
    // Source cell 27 (row 3, col 3) has two legal slides: East 1 -> cell 28,
    // South 2 -> cell 43.
    const legalMoves = [slideMove(27, 1, 1), slideMove(27, 2, 2)];
    const { onMove } = renderBoard({ legalMoves });

    // No source selected yet -- clicking the destination cells directly
    // does nothing (they're not sources or placements).
    fireEvent.click(buttonForCell(28));
    expect(onMove).not.toHaveBeenCalled();

    fireEvent.click(buttonForCell(27));
    fireEvent.click(buttonForCell(28));

    expect(onMove).toHaveBeenCalledTimes(1);
    expect(onMove).toHaveBeenCalledWith(slideMove(27, 1, 1));
  });

  it("clicking the selected source again deselects it instead of moving", () => {
    const legalMoves = [slideMove(27, 1, 1)];
    const { onMove } = renderBoard({ legalMoves });

    fireEvent.click(buttonForCell(27));
    fireEvent.click(buttonForCell(27));
    fireEvent.click(buttonForCell(28)); // no longer a live destination

    expect(onMove).not.toHaveBeenCalled();
  });
});
