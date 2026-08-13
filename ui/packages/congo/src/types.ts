// types.ts — Wire types for Congo, mirroring `games/congo/src/main.rs`'s
// `WireState`/`WireMove`/`WireCell` JSON shape exactly (field names and
// string codes match, so no translation layer is needed at the API boundary).

export const SIZE = 7;
export const NUM_SQUARES = SIZE * SIZE;
export const RIVER_ROW = 3;

export type Player = "Black" | "White";

export type PieceCode =
  | "giraffe"
  | "monkey"
  | "elephant"
  | "lion"
  | "crocodile"
  | "zebra"
  | "pawn"
  | "superpawn";

export interface Cell {
  player: Player;
  piece: PieceCode;
}

/** `squares`/`river_since` are always exactly `NUM_SQUARES` (49) long,
 * row-major with row 0 = Black's home rank (see `games/congo/src/lib.rs`'s
 * module doc comment). */
export interface GameState {
  squares: (Cell | null)[];
  river_since: number[];
  turn: Player;
}

export interface GameView extends GameState {
  winner: Player | null;
  terminal: boolean;
}

export interface Move {
  from: number;
  to: number;
  captures: number[];
}
