// types.ts — Concrete Othello wire types, mirroring
// server/adapters/othello.rs's `WireState`/`GameView` shapes. Bitboards are
// transmitted as raw u64 values for compactness; the renderer decodes them
// into an 8×8 display grid.

export type Player = "Black" | "White";

/** A square index 0..63 (row-major: `row * 8 + col`), or 64 for a pass.
 * 0 = a1 (top-left), 63 = h8 (bottom-right). */
export type Move = number;

export interface GameState {
  black: number;
  white: number;
  turn: Player;
  last_pass: boolean;
}

export interface GameView {
  black: number;
  white: number;
  turn: Player;
  last_pass: boolean;
  winner: Player | null;
  terminal: boolean;
}