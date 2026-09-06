// geometry.ts — Hex-grid layout for the board, independently re-derived from
// games/ingenious/src/lib.rs's own doc comment (offset coordinates, six
// adjacency deltas, Chebyshev-style hex distance) rather than sent over the
// wire -- it's a fixed, documented formula, the same kind of duplication
// @mcts/hex-gen's own renderer already accepts for its (different) board.

import { PLAYABLE_RADIUS, SIDE } from "./types.js";

const CENTER = (SIDE - 1) / 2;

export function hexDistance(row: number, col: number): number {
  const dr = row - CENTER;
  const dc = col - CENTER;
  return Math.max(Math.abs(dc), Math.abs(dr), Math.abs(dr - dc));
}

export function isValidCell(row: number, col: number): boolean {
  return hexDistance(row, col) <= PLAYABLE_RADIUS;
}

/** Every playable cell index (row-major, `row * SIDE + col`) on the
 * 2-player board -- 91 cells, matching
 * `geometry::<2>().valid.len()` in `games/ingenious/src/lib.rs`'s own tests. */
export const VALID_CELLS: number[] = (() => {
  const cells: number[] = [];
  for (let row = 0; row < SIDE; row++) {
    for (let col = 0; col < SIDE; col++) {
      if (isValidCell(row, col)) cells.push(row * SIDE + col);
    }
  }
  return cells;
})();

/** Six hex-adjacency directions as `(row, col)` deltas, indexed identically
 * to `games/ingenious/src/lib.rs`'s `DELTAS`: N=0, S=1, E=2, W=3, NE=4,
 * SW=5. */
const DELTAS: [number, number][] = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [-1, -1],
];

/** `cell`'s neighbor in direction `dir`, or `null` if that neighbor falls
 * outside the grid or off the playable disc. */
export function neighborOf(cell: number, dir: number): number | null {
  const row = Math.floor(cell / SIDE);
  const col = cell % SIDE;
  const delta = DELTAS[dir];
  if (!delta) return null;
  const nr = row + delta[0];
  const nc = col + delta[1];
  if (nr < 0 || nc < 0 || nr >= SIDE || nc >= SIDE) return null;
  if (!isValidCell(nr, nc)) return null;
  return nr * SIDE + nc;
}

/** Pixel center of `cell`'s hex, before any board-wide translation. Uses
 * axial coordinates `(q, r) = (col, row - col)` -- the substitution that
 * turns this grid's `(dr, dc)` adjacency deltas (`N/S` along the row axis,
 * `E/W` along the column axis, `NE/SW` along `row == col`) into the
 * standard six unit axial directions, so the usual pointy-top axial pixel
 * formula applies directly. */
export function centerOf(cell: number, hexSize: number): { x: number; y: number } {
  const row = Math.floor(cell / SIDE);
  const col = cell % SIDE;
  const q = col;
  const r = row - col;
  return {
    x: hexSize * Math.sqrt(3) * (q + r / 2),
    y: hexSize * 1.5 * r,
  };
}
