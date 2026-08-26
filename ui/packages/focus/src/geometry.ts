// geometry.ts — Board shape, mirroring `games/focus/src/lib.rs`'s
// `row_range`/`is_valid_cell`/`VALID` exactly: an 8x8 grid (row 0 = top,
// index = row*8+col) with 3 squares notched off each corner, 52 playable
// cells. `GameView.board` (see types.ts) is already a flat 64-entry array
// indexed the same way, so nothing here needs to touch the wire format --
// this is purely which of those 64 slots the renderer should draw as a
// clickable square versus an empty gap.

export const BOARD_SIZE = 8;

/** Valid column range `[lo, hi]` for a board row, 0-indexed -- see
 * `games/focus/src/lib.rs`'s `row_range`. */
function rowRange(row: number): [number, number] {
  if (row === 0 || row === 7) return [2, 5];
  if (row === 1 || row === 6) return [1, 6];
  return [0, 7];
}

export function isValidCell(idx: number): boolean {
  const row = Math.floor(idx / BOARD_SIZE);
  const col = idx % BOARD_SIZE;
  const [lo, hi] = rowRange(row);
  return col >= lo && col <= hi;
}

/** All 64 row-major indices, in order -- the renderer iterates this once to
 * lay out the grid, rendering an inert gap for anything `isValidCell` rejects. */
export const ALL_CELLS: number[] = Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, i) => i);

/** The 52 playable cells. */
export const VALID_CELLS: number[] = ALL_CELLS.filter(isValidCell);

export function coordFor(cell: number): string {
  const row = Math.floor(cell / BOARD_SIZE);
  const col = cell % BOARD_SIZE;
  const letter = String.fromCharCode(97 + col);
  const rank = BOARD_SIZE - row; // row 0 = top = the highest rank number
  return `${letter}${rank}`;
}
