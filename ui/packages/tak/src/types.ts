// types.ts — Wire types mirroring games/tak/src/main.rs's `WireState`/
// `GameView`/`WireMove`. `main.rs` decodes the engine's packed `u64` cell
// words and `Move`'s packed `u32` (see games/tak/src/lib.rs's file header for
// that internal encoding) into these self-describing shapes at the adapter
// boundary -- nothing on the TS side needs to know about that packing.

export type Player = "White" | "Black";

/** One board cell's stack, bottom-to-top; `null` = empty. Every piece below
 * the top is always flat (walls/capstones can never be covered), so only the
 * top piece needs a `topKind`. */
export interface Stack {
  colors: Player[];
  top_kind: "Flat" | "Wall" | "Cap";
}

/** `cells[i]` is row-major (`i = row * size + col`), row 0 = south edge --
 * `size` itself isn't sent on the wire, so callers derive it from
 * `Math.sqrt(cells.length)` (see `boardSize` below). */
export interface GameState {
  cells: (Stack | null)[];
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

export type Move =
  | { tag: "Place"; square: number; kind: "Flat" | "Wall" | "Cap" }
  | { tag: "Spread"; square: number; direction: "North" | "East" | "South" | "West"; drop_sizes: number[] };

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

/** Board width/height, derived from the actual cell count rather than
 * trusted from a request config -- correct regardless of whether the server
 * honored a requested size. */
export function boardSize(state: GameState): number {
  return Math.round(Math.sqrt(state.cells.length));
}
