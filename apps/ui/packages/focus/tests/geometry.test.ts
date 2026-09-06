// tests/geometry.test.ts — Board-shape sanity checks against
// games/focus/src/lib.rs's own `valid_cell_count_is_52` test, so a
// transcription slip in `rowRange` is caught here too, not just server-side.

import { describe, expect, it } from "vitest";
import { coordFor, isValidCell, VALID_CELLS } from "../src/geometry.js";

describe("isValidCell / VALID_CELLS", () => {
  it("52 of the 64 cells are playable", () => {
    expect(VALID_CELLS).toHaveLength(52);
  });

  it("a corner-notch cell (row 0, col 0) is invalid", () => {
    expect(isValidCell(0)).toBe(false);
  });

  it("a center cell (row 3, col 3 = cell 27) is valid", () => {
    expect(isValidCell(27)).toBe(true);
  });
});

describe("coordFor", () => {
  it("row 0 (top) is the highest rank number", () => {
    expect(coordFor(2)).toBe("c8"); // row 0, col 2
  });

  it("row 7 (bottom) is rank 1", () => {
    expect(coordFor(58)).toBe("c1"); // row 7, col 2
  });
});
