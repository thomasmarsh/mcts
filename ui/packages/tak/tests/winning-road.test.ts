// tests/winning-road.test.ts — findWinningRoad is a real reimplementation of
// games/tak/src/lib.rs's road-connectivity check (for the post-game glow
// highlight only -- the server's view() is the authoritative terminal/winner
// source, see TakRenderer.tsx's file header), so it gets its own test
// against known board layouts rather than only exercising it through a full
// component render.

import { describe, it, expect } from "vitest";
import { findWinningRoad } from "../src/TakRenderer.js";
import type { GameState } from "../src/types.js";

function emptyState(n: number): GameState {
  return {
    cells: Array.from({ length: n * n }, () => null),
    stones: [21, 21],
    caps: [1, 1],
    turn: "White",
    opening: false,
  };
}

function place(state: GameState, square: number, color: "White" | "Black", topKind: "Flat" | "Wall" | "Cap" = "Flat"): void {
  state.cells[square] = { colors: [color], top_kind: topKind };
}

describe("findWinningRoad", () => {
  it("finds a vertical (south-north) road down one column", () => {
    const s = emptyState(5);
    for (let row = 0; row < 5; row++) place(s, row * 5, "White");
    const path = findWinningRoad(s, "White");
    expect(path).not.toBeNull();
    expect(path).toEqual([0, 5, 10, 15, 20]);
  });

  it("finds a horizontal (west-east) road along one row", () => {
    const s = emptyState(5);
    for (let col = 0; col < 5; col++) place(s, col, "Black");
    const path = findWinningRoad(s, "Black");
    expect(path).not.toBeNull();
    expect(path).toEqual([0, 1, 2, 3, 4]);
  });

  it("returns null when there's no connecting road", () => {
    const s = emptyState(5);
    place(s, 0, "White");
    place(s, 24, "White");
    expect(findWinningRoad(s, "White")).toBeNull();
  });

  it("a wall on the path breaks the road (walls don't count)", () => {
    const s = emptyState(5);
    for (let row = 0; row < 5; row++) place(s, row * 5, "White");
    s.cells[10] = { colors: ["White"], top_kind: "Wall" };
    expect(findWinningRoad(s, "White")).toBeNull();
  });
});
