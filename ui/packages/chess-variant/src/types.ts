// types.ts — Concrete wire types for chess-variant games (Knightthrough,
// Breakthrough, and any future piece-movement variants sharing the same
// bitboard+turn+winner wire shape).
//
// Both Knightthrough and Breakthrough share this exact wire format on the
// Rust side — the same `State<8, 8>` struct with `black`/`white` bitboards
// and identical `Move(src, dst)` encoding. Only the move-generation rules
// differ (knight jumps vs. pawn-like forward/diagonal), which lives entirely
// server-side. The UI treats both identically.

export type Player = "Black" | "White";

/** A move is `[sourceIndex, destinationIndex]` — row-major 0..63 on an 8×8 board. */
export type Move = [number, number];

/** Bitboard fields are 16-character hex strings, not raw numbers: 64-bit
 * values are too large for JavaScript's `Number` (max safe integer = 2⁵³),
 * which would silently round away lower bits. Using hex strings through
 * `BigInt(hex)` preserves all 64 bits. */
export interface GameState {
  black: string;
  white: string;
  turn: Player;
  winner: boolean;
}

export interface GameView {
  black: string;
  white: string;
  turn: Player;
  winner: Player | null;
  terminal: boolean;
}