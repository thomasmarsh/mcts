// tests/winning-road.test.ts — findWinningRoad is a real reimplementation of
// games/tak/src/lib.rs's road-connectivity check (for the post-game glow
// highlight only -- the server's view() is the authoritative terminal/winner
// source, see TakRenderer.tsx's file header), so it gets its own test
// against known board layouts rather than only exercising it through a full
// component render.

import { describe, it, expect } from "vitest";
import { findWinningRoad } from "../src/TakRenderer.js";
import type { ParsedStack } from "../src/tps-parser.js";
import type { Player } from "../src/types.js";

function emptyCells(n: number): (ParsedStack | null)[] {
  return Array.from({ length: n * n }, () => null);
}

function place(
  cells: (ParsedStack | null)[],
  square: number,
  color: Player,
  topKind: "Flat" | "Wall" | "Cap" = "Flat",
): void {
  cells[square] = { colors: [color], topKind };
}

describe("findWinningRoad", () => {
  it("finds a vertical (south-north) road down one column", () => {
    const cells = emptyCells(5);
    for (let row = 0; row < 5; row++) place(cells, row * 5, "White");
    const path = findWinningRoad(cells, 5, "White");
    expect(path).not.toBeNull();
    expect(path).toEqual([0, 5, 10, 15, 20]);
  });

  it("finds a horizontal (west-east) road along one row", () => {
    const cells = emptyCells(5);
    for (let col = 0; col < 5; col++) place(cells, col, "Black");
    const path = findWinningRoad(cells, 5, "Black");
    expect(path).not.toBeNull();
    expect(path).toEqual([0, 1, 2, 3, 4]);
  });

  it("returns null when there's no connecting road", () => {
    const cells = emptyCells(5);
    place(cells, 0, "White");
    place(cells, 24, "White");
    expect(findWinningRoad(cells, 5, "White")).toBeNull();
  });

  it("a wall on the path breaks the road (walls don't count)", () => {
    const cells = emptyCells(5);
    for (let row = 0; row < 5; row++) place(cells, row * 5, "White");
    cells[10] = { colors: ["White"], topKind: "Wall" };
    expect(findWinningRoad(cells, 5, "White")).toBeNull();
  });
});
