// tests/move-codec.test.ts — Bit-packing round trips and destination-cell
// math against known inputs, cross-checked against games/focus/src/lib.rs's
// own `Move::place`/`Move::slide` layout and `DIRS` ordering.

import { describe, expect, it } from "vitest";
import {
  destinationCell,
  isSlideMove,
  moveCell,
  moveCount,
  moveDir,
  placeMove,
  slideMove,
} from "../src/move-codec.js";

describe("placeMove / slideMove round trips", () => {
  it("placeMove encodes a bare cell with the slide bit clear", () => {
    const m = placeMove(17);
    expect(isSlideMove(m)).toBe(false);
    expect(moveCell(m)).toBe(17);
  });

  it("slideMove round-trips cell/dir/count and sets the slide bit", () => {
    const m = slideMove(26, 1, 3);
    expect(isSlideMove(m)).toBe(true);
    expect(moveCell(m)).toBe(26);
    expect(moveDir(m)).toBe(1);
    expect(moveCount(m)).toBe(3);
  });
});

describe("destinationCell", () => {
  it("a place move's destination is its own cell", () => {
    expect(destinationCell(placeMove(30))).toBe(30);
  });

  it("dir 0 (N) decreases the row", () => {
    // Cell 34 = row 4, col 2. North 2 -> row 2, col 2 = cell 18.
    expect(destinationCell(slideMove(34, 0, 2))).toBe(18);
  });

  it("dir 1 (E) increases the column", () => {
    // Cell 26 = row 3, col 2. East 1 -> row 3, col 3 = cell 27.
    expect(destinationCell(slideMove(26, 1, 1))).toBe(27);
  });

  it("dir 2 (S) increases the row", () => {
    // Cell 18 = row 2, col 2. South 3 -> row 5, col 2 = cell 42.
    expect(destinationCell(slideMove(18, 2, 3))).toBe(42);
  });

  it("dir 3 (W) decreases the column", () => {
    // Cell 27 = row 3, col 3. West 2 -> row 3, col 1 = cell 25.
    expect(destinationCell(slideMove(27, 3, 2))).toBe(25);
  });
});
