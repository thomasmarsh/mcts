// tests/chess-variant-renderer.test.tsx — Component-level test of the
// chess-variant board's two-click source→destination interaction. Verifies
// that clicking a movable piece selects it (source highlight), that legal
// destinations then light up, and that clicking a destination dispatches the
// correct move. This matters because Breakthrough destinations can be
// reached by multiple pieces, so a one-click-destination model is ambiguous.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import type { GameRendererProps } from "@mcts/game";
import { ChessVariantRenderer } from "../packages/chess-variant/src/ChessVariantRenderer.js";
import type { GameState, GameView, Move } from "../packages/chess-variant/src/types.js";

// Starting Breakthrough position: black at top (engine rows 6-7), white at
// bottom (engine rows 0-1). Black to move.
const START_BLACK = "ffff000000000000";
const START_WHITE = "000000000000ffff";

function startState(): GameState {
  return { black: START_BLACK, white: START_WHITE, turn: "Black", winner: false };
}

// Legal moves for the black front rank (engine row 6). Each black piece on
// row 6 can move forward to row 5, or diagonally.
const BLACK_LEGAL_MOVES: Move[] = [
  [48, 40], [48, 41], // a7 → a6, b6
  [49, 40], [49, 41], [49, 42], // b7 → a6, b6, c6
  [50, 41], [50, 42], [50, 43], // c7 → b6, c6, d6
  [51, 42], [51, 43], [51, 44], // d7 → c6, d6, e6
  [52, 43], [52, 44], [52, 45], // e7 → d6, e6, f6
  [53, 44], [53, 45], [53, 46], // f7 → e6, f6, g6
  [54, 45], [54, 46], [54, 47], // g7 → f6, g6, h6
  [55, 46], [55, 47], // h7 → g6, h6
];

function startView(): GameView {
  return { black: START_BLACK, white: START_WHITE, turn: "Black", winner: null, terminal: false };
}

function makeProps(overrides: Partial<GameRendererProps<GameState, Move, GameView>> = {}): GameRendererProps<GameState, Move, GameView> {
  return {
    state: startState(),
    view: startView(),
    history: [],
    legalMoves: BLACK_LEGAL_MOVES,
    busy: false,
    onMove: () => {},
    hoveredMove: null,
    onHover: () => {},
    ...overrides,
  };
}

/** All cells are `<button class="cv-cell">`; find the one whose display index
 * matches. Display index 0 = top-left (a8). `engineIndex` flips rows, so
 * display index and engine index differ. */
function cellAtDisplay(displayIndex: number): HTMLButtonElement {
  return screen.getAllByRole("button")[displayIndex]! as HTMLButtonElement;
}

describe("ChessVariantRenderer two-click interaction", () => {
  beforeEach(() => {
    cleanup(); // ensure a fresh DOM between tests
  });

  it("shows the hand (movable) class on pieces that have legal moves", () => {
    render(() => <ChessVariantRenderer {...makeProps()} />);
    // Black front rank is engine row 6 = display row 1 (since we flip rows).
    // Engine index 48 = display index? engine index 48 = row 6, col 0.
    // displayRow = 8 - 1 - 6 = 1, col 0 → display index 8.
    const cell = cellAtDisplay(8);
    expect(cell.className).toContain("cv-movable");
  });

  it("has no explicit disabled attribute on movable pieces", () => {
    render(() => <ChessVariantRenderer {...makeProps()} />);
    const cell = cellAtDisplay(8);
    expect(cell.disabled).toBe(false);
  });

  it("selects a source piece on click", () => {
    render(() => <ChessVariantRenderer {...makeProps()} />);
    const cell = cellAtDisplay(8);
    fireEvent.click(cell);
    // After clicking, the piece should have the selected-source style.
    expect(cell.className).toContain("cv-selected-source");
  });

  it("shows legal destinations after selecting a source", () => {
    render(() => <ChessVariantRenderer {...makeProps()} />);
    fireEvent.click(cellAtDisplay(8)); // select a7 (engine 48)
    // Engine 48's legal destinations are engine 40 and 41.
    // Engine 40 = row 5, col 0 → displayRow = 8-1-5 = 2, display index 16.
    // Engine 41 = row 5, col 1 → display index 17.
    const dst1 = cellAtDisplay(16);
    const dst2 = cellAtDisplay(17);
    expect(dst1.className).toContain("legal");
    expect(dst2.className).toContain("legal");
    // Other cells should not be legal.
    expect(cellAtDisplay(18).className).not.toContain("legal");
  });

  it("dispatches the move when clicking a legal destination", () => {
    const onMove = () => {};
    const spy = vi.fn();
    render(() => <ChessVariantRenderer {...makeProps({ onMove: spy })} />);
    fireEvent.click(cellAtDisplay(8)); // select a7 (engine 48)
    fireEvent.click(cellAtDisplay(16)); // click destination a6 (engine 40)
    expect(spy).toHaveBeenCalledWith([48, 40]);
  });

  it("deselects when clicking the same piece again", () => {
    render(() => <ChessVariantRenderer {...makeProps()} />);
    const cell = cellAtDisplay(8);
    fireEvent.click(cell);
    expect(cell.className).toContain("cv-selected-source");
    fireEvent.click(cell);
    expect(cell.className).not.toContain("cv-selected-source");
  });

  it("switches selection when clicking another movable piece", () => {
    render(() => <ChessVariantRenderer {...makeProps()} />);
    fireEvent.click(cellAtDisplay(8)); // a7 (engine 48)
    fireEvent.click(cellAtDisplay(9)); // b7 (engine 49)
    expect(cellAtDisplay(8).className).not.toContain("cv-selected-source");
    expect(cellAtDisplay(9).className).toContain("cv-selected-source");
  });
});