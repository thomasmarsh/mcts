// tests/move-codec.test.ts — PTN parse/format helpers against known inputs,
// cross-checked against games/tak/src/lib.rs's own `Tak::notation`
// doc-comment examples (`a1`, `Sa1`, `a1>`, `3c3>12`).

import { describe, it, expect } from "vitest";
import { coordFor, footprintFor, notation, parsePtn } from "../src/move-codec.js";
import type { ParsedMove } from "../src/move-codec.js";

describe("coordFor", () => {
  it("turns a row-major index into a PTN coordinate (row 0 = south/'1')", () => {
    expect(coordFor(0, 5)).toBe("a1");
    expect(coordFor(4, 5)).toBe("e1");
    expect(coordFor(5, 5)).toBe("a2");
    expect(coordFor(24, 5)).toBe("e5");
  });
});

describe("footprintFor", () => {
  it("a placement's footprint is just its own square", () => {
    const move: ParsedMove = { tag: "Place", square: 12, kind: "Flat" };
    expect(footprintFor(move, 5)).toEqual([12]);
  });

  it("a spread walks the direction, one cell per drop", () => {
    // From a1 (square 0) moving East, dropping (2, 1) -> touches a1, b1, c1.
    const move: ParsedMove = { tag: "Spread", square: 0, direction: "East", drop_sizes: [2, 1] };
    expect(footprintFor(move, 5)).toEqual([0, 1, 2]);
  });

  it("North increases the row (row 0 is the south edge)", () => {
    const move: ParsedMove = { tag: "Spread", square: 0, direction: "North", drop_sizes: [1, 1] };
    expect(footprintFor(move, 5)).toEqual([0, 5, 10]);
  });
});

describe("parsePtn", () => {
  it("parses placements: bare/S/C prefix + coordinate", () => {
    expect(parsePtn("a1", 5)).toEqual({ tag: "Place", square: 0, kind: "Flat" });
    expect(parsePtn("Sa1", 5)).toEqual({ tag: "Place", square: 0, kind: "Wall" });
    expect(parsePtn("Ca1", 5)).toEqual({ tag: "Place", square: 0, kind: "Cap" });
  });

  it("parses a single-piece spread with no count/drop suffix", () => {
    expect(parsePtn("a1>", 5)).toEqual({
      tag: "Spread",
      square: 0,
      direction: "East",
      drop_sizes: [1],
    });
  });

  it("parses a multi-piece spread: take count + drop sizes", () => {
    // take 3 from c3 (square 12 on a 5x5 board) moving east, dropping 1 then 2.
    expect(parsePtn("3c3>12", 5)).toEqual({
      tag: "Spread",
      square: 12,
      direction: "East",
      drop_sizes: [1, 2],
    });
  });

  it("parses spreads in all four directions", () => {
    expect(parsePtn("a1+", 5)).toEqual({
      tag: "Spread",
      square: 0,
      direction: "North",
      drop_sizes: [1],
    });
    expect(parsePtn("a1>", 5)).toEqual({
      tag: "Spread",
      square: 0,
      direction: "East",
      drop_sizes: [1],
    });
    expect(parsePtn("a1-", 5)).toEqual({
      tag: "Spread",
      square: 0,
      direction: "South",
      drop_sizes: [1],
    });
    expect(parsePtn("a1<", 5)).toEqual({
      tag: "Spread",
      square: 0,
      direction: "West",
      drop_sizes: [1],
    });
  });
});

describe("notation (ParsedMove -> PTN round-trip)", () => {
  it("round-trips placements", () => {
    for (const kind of ["Flat", "Wall", "Cap"] as const) {
      const pm: ParsedMove = { tag: "Place", square: 0, kind };
      const ptn = notation(pm, 5);
      expect(parsePtn(ptn, 5)).toEqual(pm);
    }
  });

  it("round-trips spreads", () => {
    const cases: ParsedMove[] = [
      { tag: "Spread", square: 0, direction: "East", drop_sizes: [1] },
      { tag: "Spread", square: 12, direction: "East", drop_sizes: [1, 2] },
      { tag: "Spread", square: 12, direction: "North", drop_sizes: [3] },
    ];
    for (const pm of cases) {
      const ptn = notation(pm, 5);
      expect(parsePtn(ptn, 5)).toEqual(pm);
    }
  });

  it("formats placements: bare/S/C prefix + coordinate", () => {
    expect(notation({ tag: "Place", square: 0, kind: "Flat" }, 5)).toBe("a1");
    expect(notation({ tag: "Place", square: 0, kind: "Wall" }, 5)).toBe("Sa1");
    expect(notation({ tag: "Place", square: 0, kind: "Cap" }, 5)).toBe("Ca1");
  });

  it("formats a single-piece spread with no count/drop suffix", () => {
    expect(notation({ tag: "Spread", square: 0, direction: "East", drop_sizes: [1] }, 5)).toBe(
      "a1>",
    );
  });

  it("formats a multi-piece spread: take-count prefix + per-square drop suffix", () => {
    const move: ParsedMove = { tag: "Spread", square: 12, direction: "East", drop_sizes: [1, 2] };
    expect(notation(move, 5)).toBe("3c3>12");
  });
});
