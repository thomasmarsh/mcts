// tests/renderer.test.tsx — TrafficLightsRenderer click-handling tests.
// Unlike GameShell.test.tsx (which tests GameShell's effect integration
// against a fake game module), this tests the real TrafficLightsRenderer
// with controlled props: does clicking a cell with a piece actually fire
// the correct move, or does something silently prevent the dispatch?
//
// The user reported "I can't play on a red position to change it to Y" —
// clicking a cell with R dispatches move 17 (R→Y on cell 4) via
// `moveByCell().get(4)`. This test verifies that path works.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createSignal } from "solid-js";
import { TrafficLightsRenderer } from "../src/index.js";
import type { GameView } from "../src/types.js";

afterEach(() => cleanup());

function makeView(cells: (string | null)[], turn: "A" | "B", terminal = false): GameView {
  return { turn, cells: cells as GameView["cells"], winner: terminal ? turn : null, terminal };
}

// Test state: A's turn, R at cell 2 (AI's piece) and cell 4 (user's piece).
const cellsWithTwoReds: (string | null)[] = [null, null, "R", null, "R", null, null, null, null];
// Legal moves: advance cell 2 (move 9), advance cell 4 (move 17), place R on any empty cell
const legalMoves = [0, 4, 9, 12, 17, 20, 24, 28, 32];

describe("TrafficLightsRenderer cell clicks", () => {
  it("clicking a red cell dispatches the advance move (R→Y)", () => {
    const onMove = vi.fn();
    render(() => (
      <TrafficLightsRenderer
        state={{ turn: "A", cells: cellsWithTwoReds as GameView["cells"] }}
        view={makeView(cellsWithTwoReds, "A")}
        history={[]}
        legalMoves={legalMoves}
        busy={false}
        onMove={onMove}
        hoveredMove={null}
        onHover={vi.fn()}
      />
    ));

    const buttons = screen.getAllByRole("button") as HTMLButtonElement[];
    expect(buttons).toHaveLength(9);

    // Cell 4 has "R" → legal advance is move 17 (R→Y)
    fireEvent.click(buttons[4]!);
    expect(onMove).toHaveBeenCalledTimes(1);
    expect(onMove).toHaveBeenCalledWith(17);
  });

  it("clicking a different red cell dispatches the correct advance move", () => {
    const onMove = vi.fn();
    render(() => (
      <TrafficLightsRenderer
        state={{ turn: "A", cells: cellsWithTwoReds as GameView["cells"] }}
        view={makeView(cellsWithTwoReds, "A")}
        history={[]}
        legalMoves={legalMoves}
        busy={false}
        onMove={onMove}
        hoveredMove={null}
        onHover={vi.fn()}
      />
    ));

    const buttons = screen.getAllByRole("button") as HTMLButtonElement[];

    // Cell 2 has "R" → legal advance is move 9 (R→Y on cell 2)
    fireEvent.click(buttons[2]!);
    expect(onMove).toHaveBeenCalledTimes(1);
    expect(onMove).toHaveBeenCalledWith(9);
  });

  it("clicking a yellow cell dispatches the advance move (Y→G)", () => {
    const onMove = vi.fn();
    const cells: (string | null)[] = [null, null, null, null, "Y", null, null, null, null];
    // Cell 4 is Y → legal advance is move 18 (Y→G)
    const moves = [0, 4, 8, 12, 18, 20, 24, 28, 32];

    render(() => (
      <TrafficLightsRenderer
        state={{ turn: "A", cells: cells as GameView["cells"] }}
        view={makeView(cells, "A")}
        history={[]}
        legalMoves={moves}
        busy={false}
        onMove={onMove}
        hoveredMove={null}
        onHover={vi.fn()}
      />
    ));

    const buttons = screen.getAllByRole("button") as HTMLButtonElement[];

    fireEvent.click(buttons[4]!);
    expect(onMove).toHaveBeenCalledTimes(1);
    expect(onMove).toHaveBeenCalledWith(18);
  });

  it("clicking a green cell does not dispatch (G has no legal move)", () => {
    const onMove = vi.fn();
    const cells: (string | null)[] = [null, null, null, null, null, null, null, null, "G"];
    // Cell 8 is G → no legal move for it (moves are for empty cells only)
    const moves = [0, 4, 8, 12, 16, 20, 24, 28];

    render(() => (
      <TrafficLightsRenderer
        state={{ turn: "A", cells: cells as GameView["cells"] }}
        view={makeView(cells, "A")}
        history={[]}
        legalMoves={moves}
        busy={false}
        onMove={onMove}
        hoveredMove={null}
        onHover={vi.fn()}
      />
    ));

    const buttons = screen.getAllByRole("button") as HTMLButtonElement[];

    fireEvent.click(buttons[8]!);
    expect(onMove).not.toHaveBeenCalled();
  });

  it("clicking is suppressed when busy", () => {
    const onMove = vi.fn();
    render(() => (
      <TrafficLightsRenderer
        state={{ turn: "A", cells: cellsWithTwoReds as GameView["cells"] }}
        view={makeView(cellsWithTwoReds, "A")}
        history={[]}
        legalMoves={legalMoves}
        busy={true}
        onMove={onMove}
        hoveredMove={null}
        onHover={vi.fn()}
      />
    ));

    fireEvent.click(screen.getAllByRole("button")[4] as HTMLButtonElement);
    expect(onMove).not.toHaveBeenCalled();
  });

  it("clicking an empty cell with a legal move dispatches the correct move", () => {
    const onMove = vi.fn();
    const cells: (string | null)[] = [null, null, null, null, "R", null, null, null, null];
    // Cell 0 is empty → legal move is 0 (place R on cell 0)
    const moves = [0, 4, 8, 12, 17, 20, 24, 28, 32];

    render(() => (
      <TrafficLightsRenderer
        state={{ turn: "A", cells: cells as GameView["cells"] }}
        view={makeView(cells, "A")}
        history={[]}
        legalMoves={moves}
        busy={false}
        onMove={onMove}
        hoveredMove={null}
        onHover={vi.fn()}
      />
    ));

    const buttons = screen.getAllByRole("button") as HTMLButtonElement[];

    fireEvent.click(buttons[0]!);
    expect(onMove).toHaveBeenCalledWith(0);
  });

  it("enables a cell whose node was created while legalMoves was empty", async () => {
    // Regression: GameShell nulls the position (and with it legalMoves)
    // for one reduction after every move while it re-fetches. A cell that
    // changed during that gap (null→"R") gets its `<For>` node created
    // with legalMoves=[]; when the real legal moves arrive a moment later,
    // the button must become enabled. Derived values computed as plain
    // consts in the `<For>` callback froze `disabled` at creation time —
    // the classList stayed reactive (pointer cursor) but clicks were dead.
    const cells: (string | null)[] = [null, null, null, null, "R", null, null, null, null];
    const onMove = vi.fn();
    const [moves, setMoves] = createSignal<number[]>([]);

    render(() => (
      <TrafficLightsRenderer
        state={{ turn: "A", cells: cells as GameView["cells"] }}
        view={makeView(cells, "A")}
        history={[]}
        legalMoves={moves()}
        busy={false}
        onMove={onMove}
        hoveredMove={null}
        onHover={vi.fn()}
      />
    ));

    const button = screen.getAllByRole("button")[4] as HTMLButtonElement;
    expect(button.disabled).toBe(true);

    // The re-fetch completes: advance cell 4 (move 17) is now legal.
    setMoves([0, 4, 8, 12, 17, 20, 24, 28, 32]);

    await vi.waitFor(() => {
      expect(button.disabled).toBe(false);
    });

    fireEvent.click(button);
    expect(onMove).toHaveBeenCalledTimes(1);
    expect(onMove).toHaveBeenCalledWith(17);
  });
});