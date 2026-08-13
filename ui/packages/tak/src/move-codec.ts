// move-codec.ts — Pure helpers over the `Move` wire shape (types.ts):
// which cells a move touches (`footprintFor`, for highlighting/stacking) and
// a PTN-style human-readable label (`notation`, mirroring
// `games/tak/src/lib.rs`'s `Tak::notation` exactly). No bit-level decoding
// needed here -- `games/tak/src/main.rs` already hands over a self-describing
// tagged JSON shape instead of the engine's packed `u32`.

import type { Move } from "./types.js";

type Direction = "North" | "East" | "South" | "West";

/** Matches `games/tak/src/lib.rs`'s `DIRS` (indexed the same way `Move::dir()`
 * is: 0 = North, 1 = East, 2 = South, 3 = West), in board coordinates where
 * row 0 is the south edge -- so North is `+row`. */
const DIR_DELTA: Record<Direction, [number, number]> = {
  North: [0, 1],
  East: [1, 0],
  South: [0, -1],
  West: [-1, 0],
};

/** The ordered path of board indices a move touches: `[square]` for a
 * placement, `[src, ...one per drop]` for a spread (in walk order, one entry
 * per `drop_sizes` element). */
export function footprintFor(move: Move, n: number): number[] {
  if (move.tag === "Place") return [move.square];
  const [dc, dr] = DIR_DELTA[move.direction];
  let col = move.square % n;
  let row = Math.floor(move.square / n);
  const path = [move.square];
  for (let i = 0; i < move.drop_sizes.length; i++) {
    col += dc;
    row += dr;
    path.push(row * n + col);
  }
  return path;
}

export function coordFor(square: number, n: number): string {
  const col = square % n;
  const row = Math.floor(square / n);
  return `${String.fromCharCode(97 + col)}${row + 1}`;
}

const DIR_GLYPH: Record<Direction, string> = {
  North: "+",
  East: ">",
  South: "-",
  West: "<",
};

const KIND_PREFIX: Record<"Flat" | "Wall" | "Cap", string> = { Flat: "", Wall: "S", Cap: "C" };

/** PTN-style notation: placements are `a1`/`Sa1`/`Ca1`; spreads are `a1>` or,
 * for a multi-piece take, `3c3>12` (take 3 from c3 moving east, dropping 1
 * then 2). Mirrors `games/tak/src/lib.rs`'s `Tak::notation` field for field. */
export function notation(move: Move, n: number): string {
  const at = coordFor(move.square, n);
  if (move.tag === "Place") return `${KIND_PREFIX[move.kind]}${at}`;
  const take = move.drop_sizes.reduce((a, b) => a + b, 0);
  const prefix = take > 1 ? String(take) : "";
  const suffix = take > 1 ? move.drop_sizes.join("") : "";
  return `${prefix}${at}${DIR_GLYPH[move.direction]}${suffix}`;
}
