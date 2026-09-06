// move-codec.ts — Bit layout for a packed `Move`, mirroring
// `games/focus/src/lib.rs`'s `Move(u16)` exactly: `serde_json` serializes a
// single-field tuple struct transparently, so the wire value really is a
// bare number, not `{0: ...}`.
//
//   bit 0        : 1 = slide/split, 0 = place from reserve
//   bits [1, 7)  : placement cell, or a slide's source cell (0..64)
//   bits [7, 9)  : slide direction (0=N, 1=E, 2=S, 3=W) -- unused for a place
//   bits [9, 12) : slide/split count, 1..=5              -- unused for a place

import { BOARD_SIZE } from "./geometry.js";

export type Move = number;

export function isSlideMove(m: Move): boolean {
  return (m & 1) !== 0;
}

/** Placement cell, or a slide's source cell. */
export function moveCell(m: Move): number {
  return (m >> 1) & 63;
}

export function moveDir(m: Move): number {
  return (m >> 7) & 3;
}

export function moveCount(m: Move): number {
  return (m >> 9) & 7;
}

export function placeMove(cell: number): Move {
  return cell << 1;
}

export function slideMove(cell: number, dir: number, count: number): Move {
  return 1 | (cell << 1) | (dir << 7) | (count << 9);
}

/** Direction deltas `[dRow, dCol]`, indexed 0=N/1=E/2=S/3=W (row 0 = top of
 * the board) -- matches `games/focus/src/lib.rs`'s `DIRS` constant exactly. */
const DIR_DELTA: [number, number][] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
];

/** Where a move lands: the placement cell itself, or a slide's source cell
 * offset `count` squares in `dir`. A slide's (direction, count) pair maps to
 * a unique landing cell for a given source (no two legal slides from the
 * same source can share a destination), so this alone is enough to
 * disambiguate a board click -- no drop-schedule-style candidate list is
 * needed the way Tak's spreads need one. */
export function destinationCell(m: Move): number {
  const cell = moveCell(m);
  if (!isSlideMove(m)) return cell;
  const [dr, dc] = DIR_DELTA[moveDir(m)]!;
  const row = Math.floor(cell / BOARD_SIZE) + dr * moveCount(m);
  const col = (cell % BOARD_SIZE) + dc * moveCount(m);
  return row * BOARD_SIZE + col;
}
