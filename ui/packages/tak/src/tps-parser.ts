// tps-parser.ts — Parse a TPS (Tak Positional System) string into a
// structure the TakRenderer needs to build its 3D board: per-cell stacks
// (colors bottom-to-top, top piece kind) and the board size. TPS is the
// standard board-state format in the Tak ecosystem (see plan/tak/tps-spec.md);
// the server sends it as the `tps` field of `GameState`/`GameView`.

import type { Player } from "./types.js";

export interface ParsedStack {
  colors: Player[];
  topKind: "Flat" | "Wall" | "Cap";
}

/** Parsed TPS: board size and per-cell stacks (row-major, row 0 = south
 * edge, index = row * size + col). An empty cell is `null`. */
export interface ParsedTps {
  size: number;
  cells: (ParsedStack | null)[];
}

/**
 * Parse a TPS string into `ParsedTps`. The format (see the spec):
 *   <rows>/<rows>/... <turn> <move-counter>
 * Rows are listed from top (highest row number) to bottom, `/`-separated.
 * Within a row, cells are `,`-separated: `x` or `xN` for empty runs,
 * digits `1`/`2` bottom-to-top with optional `S`/`C` suffix.
 *
 * This returns cells in our row-major order (row 0 = south edge) so the
 * renderer can index directly by `row * size + col`.
 */
export function parseTps(tps: string): ParsedTps {
  const trimmed = tps.trim();
  const spaceIdx = trimmed.indexOf(" ");
  const boardPart = spaceIdx >= 0 ? trimmed.slice(0, spaceIdx) : trimmed;

  const tpsRows = boardPart.split("/");
  const size = tpsRows.length;
  const cells: (ParsedStack | null)[] = new Array(size * size).fill(null);

  for (let tpsRowIdx = 0; tpsRowIdx < size; tpsRowIdx++) {
    // TPS row 0 = top = our row (size - 1)
    const ourRow = size - 1 - tpsRowIdx;
    const rowStr = tpsRows[tpsRowIdx]!;
    const cellParts = rowStr.split(",");
    let col = 0;
    for (const part of cellParts) {
      if (part.startsWith("x")) {
        const run: number = part.length === 1 ? 1 : parseInt(part.slice(1), 10);
        if (isNaN(run) || run <= 0 || col + run > size) {
          throw new Error(`Invalid empty-run '${part}' in TPS row ${tpsRowIdx}`);
        }
        col += run;
        continue;
      }
      if (col >= size) {
        throw new Error(`Too many cells in TPS row '${rowStr}'`);
      }
      const idx = ourRow * size + col;
      cells[idx] = parseStack(part);
      col++;
    }
    if (col !== size) {
      throw new Error(`TPS row '${rowStr}' has ${col} columns, expected ${size}`);
    }
  }

  return { size, cells };
}

/** Parse one TPS cell description like `"12"`, `"2S"`, `"1212S"`, `"1"`. */
function parseStack(raw: string): ParsedStack {
  let topKind: "Flat" | "Wall" | "Cap" = "Flat";
  let digits = raw;
  if (raw.endsWith("S")) {
    topKind = "Wall";
    digits = raw.slice(0, -1);
  } else if (raw.endsWith("C")) {
    topKind = "Cap";
    digits = raw.slice(0, -1);
  }
  if (digits.length === 0 || !/^[12]+$/.test(digits)) {
    throw new Error(`Invalid TPS stack '${raw}'`);
  }
  const colors: Player[] = [];
  for (const ch of digits) {
    colors.push(ch === "1" ? "White" : "Black");
  }
  return { colors, topKind };
}