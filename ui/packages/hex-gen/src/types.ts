// types.ts — Concrete Hex wire types, mirroring games/hex-gen/src/main.rs's
// `WireState`/`GameView` shapes. `P0` connects the top edge (row 0) to the
// bottom edge (row `side - 1`); `P1` connects the left edge (col 0) to the
// right edge (col `side - 1`) -- see that file's `winner()` edge sets.
//
// Board side isn't a fixed constant: `games/hex-gen` is const-generic over
// board size (5/7/11, see `games/hex-gen/src/main.rs`'s `SUPPORTED_SIZES`),
// so every consumer derives `side` from `cells.length` via `sideOf` below
// rather than assuming one fixed board.

export type Player = "P0" | "P1";

/** A cell index 0..side*side (row-major: `row * side + col`), matching
 * `games/hex-gen/src/lib.rs`'s `Move(pub u8)`. */
export type Move = number;

/** Recovers the board side from a cell count -- `cells.length` is always a
 * perfect square (one of `SUPPORTED_SIZES`' `N * N`), so this is exact. */
export function sideOf(cellCount: number): number {
  return Math.round(Math.sqrt(cellCount));
}

export interface GameState {
  turn: Player;
  cells: (Player | null)[];
}

export interface GameView {
  turn: Player;
  cells: (Player | null)[];
  winner: Player | null;
  terminal: boolean;
}
