// geometry.ts — Pure re-implementation, in TypeScript, of `pyramid::{
// level_side, level_size, level_offset, total_cells, to_coord, index }`'s
// index math (see that crate's module doc comment for the coordinate
// system: an `n`x`n` base, level `k` an `(n-k)`x`(n-k)` square centered over
// a 2x2 block of level-`(k-1)` cells, apex at level `n-1`). Kept in sync by
// hand rather than shared/generated -- the same "small, stable index math,
// reimplemented client-side" precedent `layers.ts`'s `footprintFor` and
// `summary.ts`'s `coordFor` set for Druid, since there's no code-generation
// pipeline in this repo bridging Rust and TS types.
//
// Game-agnostic: every pyramid-family board (Margo, Akron, ...) built on
// `pyramid::Pyramid` shares this exact index math and physical layout, so
// this module lives in `@mcts/pyramid` rather than being duplicated per
// game package -- see `render.ts` for the shared three.js board/marble
// building that sits on top of it.
//
// `positionFor` additionally fixes the *physical* layout a renderer places
// spheres at: level `k`'s grid is offset by `0.5` in both `x`/`z` from level
// `k-1` (each ball nests in the pocket formed by the 4 balls below it) and
// raised by `Math.SQRT1_2` (the vertical rise between touching same-radius
// spheres offset horizontally by `sqrt(2)/2` -- see this file's own
// `LEVEL_RISE` doc comment).

function sumSquares(x: number): number {
  return (x * (x + 1) * (2 * x + 1)) / 6;
}

export function levelSide(n: number, level: number): number {
  return n - level;
}

export function levelSize(n: number, level: number): number {
  const side = levelSide(n, level);
  return side * side;
}

export function levelOffset(n: number, level: number): number {
  return sumSquares(n) - sumSquares(n - level);
}

export function totalCells(n: number): number {
  return sumSquares(n);
}

export function cellIndex(n: number, col: number, row: number, level: number): number {
  return levelOffset(n, level) + row * levelSide(n, level) + col;
}

export function toCoord(n: number, index: number): [col: number, row: number, level: number] {
  let level = 0;
  while (index >= levelOffset(n, level) + levelSize(n, level)) level++;
  const local = index - levelOffset(n, level);
  const side = levelSide(n, level);
  return [local % side, Math.floor(local / side), level];
}

/** Vertical rise per pyramid level: two same-radius touching spheres (unit
 * diameter, so unit center-to-center distance) offset horizontally by
 * `sqrt(2)/2` (a ball nested diagonally over the gap between 4 base balls)
 * have `sqrt(1^2 - (sqrt(2)/2)^2) == sqrt(1/2)` of vertical separation. */
export const LEVEL_RISE = Math.SQRT1_2;

/** World-space center of the sphere at flat `index` on a base-`n` board,
 * with level 0's grid occupying integer `(x, z)` coordinates `0..n-1` at
 * `y = 0`. */
export function positionFor(n: number, index: number): [x: number, y: number, z: number] {
  const [col, row, level] = toCoord(n, index);
  const offset = level * 0.5;
  return [col + offset, level * LEVEL_RISE, row + offset];
}
