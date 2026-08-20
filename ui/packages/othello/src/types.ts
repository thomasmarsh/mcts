// types.ts — Concrete Othello wire types, mirroring
// games/othello/src/main.rs's `WireState`/`GameView` shapes. Bitboards are
// transmitted as 16-digit lowercase hex strings, not raw JSON numbers: a
// full-board u64 routinely has bits scattered across its whole width (not
// just near the end of the game), which commonly exceeds JS's 2^53
// safe-integer range -- a plain numeric encoding would let `JSON.parse`
// silently round such a value, corrupting the board, and since the client
// echoes its current state back on every subsequent move request that
// corruption compounds instead of self-correcting. The renderer/summary
// code parses the hex string into a `BigInt` before doing any bit
// arithmetic on it.
export type Player = "Black" | "White";

/** A square index 0..63 (row-major: `row * 8 + col`), or 64 for a pass.
 * 0 = a1 (top-left), 63 = h8 (bottom-right). */
export type Move = number;

export interface GameState {
  black: string;
  white: string;
  turn: Player;
  last_pass: boolean;
}

export interface GameView {
  black: string;
  white: string;
  turn: Player;
  last_pass: boolean;
  winner: Player | null;
  terminal: boolean;
}