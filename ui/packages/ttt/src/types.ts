// types.ts — Concrete tic-tac-toe wire types, mirroring server/adapters/ttt.rs's
// `WireState`/`GameView` shapes (PLAN-UI.md session 8). Deliberately not the
// engine's internal packed-`u32` board encoding -- the adapter already
// converts to/from a plain 9-cell array, so this side just mirrors that
// plain shape.

export type Piece = "X" | "O";

/** A cell index 0..9 (row-major: `row * 3 + col`), matching
 * `src/games/ttt.rs`'s `Move(pub u8)`. */
export type Move = number;

export interface GameState {
  turn: Piece;
  cells: (Piece | null)[];
}

export interface GameView {
  turn: Piece;
  cells: (Piece | null)[];
  winner: Piece | null;
  terminal: boolean;
}
