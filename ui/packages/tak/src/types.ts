// types.ts — Wire types mirroring games/tak/src/main.rs's `WireState`/
// `GameView`. Board state is a TPS (Tak Positional System) string; moves
// are PTN (Portable Tak Notation) strings. Both are standard formats
// understood by the Tak ecosystem (PlayTak.com, ptn.ninja, etc.) -- nothing
// on the TS side needs to know about the engine's internal packed
// representation.

export type Player = "White" | "Black";

/** Wire state: a TPS string for the board layout plus pre-computed metadata
 * fields. The `tps` field is canonical; `stones`/`caps`/`turn`/`opening`
 * are redundant convenience values the server pre-computes so the client
 * doesn't need to derive them from the TPS. */
export interface GameState {
  tps: string;
  stones: [number, number]; // [White, Black]
  caps: [number, number];
  turn: Player;
  opening: boolean;
}

/** `GameView` adds the display-only `terminal`/`winner` fields a renderer
 * needs but `GameState` doesn't round-trip. */
export interface GameView extends GameState {
  terminal: boolean;
  winner: Player | null;
}

/** A move is a PTN string: `a1`, `Sa1`, `Ca1`, `a1>`, `3c3>12`, etc. */
export type Move = string;

export interface NewGameConfig {
  size: number;
}

/** The full range the engine's `State<const N: usize>` supports (see
 * `games/tak/src/lib.rs`'s `MAX_SIZE`/`piece_counts`). `games/tak/src/main.rs`
 * is hardcoded to `State<5>` today (`new_state` ignores `config` and always
 * returns a 5x5 board) -- see `NewGameFields.tsx`'s comment for what that
 * means for this picker. */
export const BOARD_SIZES: number[] = [3, 4, 5, 6];
export const DEFAULT_SIZE = 5;

/** Board width/height, derived from the TPS string's row count rather than
 * trusted from a request config -- correct regardless of whether the server
 * honored a requested size. */
export function boardSizeFromTps(tps: string): number {
  return tps.split(" ")[0]!.split("/").length;
}
