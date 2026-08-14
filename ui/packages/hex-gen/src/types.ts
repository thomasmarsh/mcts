// types.ts — Concrete Hex wire types, mirroring games/hex-gen/src/main.rs's
// `WireState`/`GameView` shapes. `P0` connects the top edge (row 0) to the
// bottom edge (row `SIDE - 1`); `P1` connects the left edge (col 0) to the
// right edge (col `SIDE - 1`) -- see that file's `winner()` edge sets.

export type Player = "P0" | "P1";

/** A cell index 0..SIDE*SIDE (row-major: `row * SIDE + col`), matching
 * `games/hex-gen/src/lib.rs`'s `Move(pub u8)`. */
export type Move = number;

export const SIDE = 11;

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
