// types.ts — Concrete Ingenious wire types, mirroring
// games/ingenious/src/main.rs's `WireState`/`GameView` shapes. Board storage
// is fixed-size regardless of player count (`games/ingenious/src/lib.rs`'s
// `BOARD_RADIUS` doc comment) -- this package only serves the 2-player game
// (`games/ingenious/src/main.rs`'s own doc comment), so every constant below
// is the 2-player board's, not a general-`P` formula.

export type Color = "Red" | "Green" | "Blue" | "Orange" | "Yellow" | "Purple";

export const COLORS: Color[] = ["Red", "Green", "Blue", "Orange", "Yellow", "Purple"];

export const NUM_PLAYERS = 2;
export const RACK_SIZE = 6;
export const TARGET_SCORE = 18;

/** Row/col grid side -- fixed across every player count this crate builds
 * (`games/ingenious/src/lib.rs`'s `SIDE`), not just the 2-player board. */
export const SIDE = 13;

/** How far from the grid center the 2-player board's playable disc extends
 * (`games/ingenious/src/lib.rs`'s `playable_radius(2)`). */
export const PLAYABLE_RADIUS = 5;

export type Phase = "place" | "swap_decision";

export interface PlaceMove {
  cell: number;
  dir: number;
  color_a: Color;
  color_b: Color;
}

/** Mirrors `Action`'s serde default (externally-tagged) representation:
 * a tuple variant serializes as `{ Variant: payload }`, a unit variant as a
 * bare string. */
export type Move = { Place: PlaceMove } | "KeepRack" | "Swap";

export function isPlaceMove(move: Move): move is { Place: PlaceMove } {
  return typeof move === "object" && "Place" in move;
}

/** One rack tile: an unordered color pair, or an empty slot. */
export type RackSlot = [Color, Color] | null;

export interface GameState {
  board: (Color | null)[];
  board_tile_counts: number[][];
  racks: RackSlot[][];
  score: number[][];
  bonus_used: boolean[][];
  has_moved: boolean[];
  claimed_symbols: boolean[];
  current_player: number;
  phase: Phase;
  pending_bonus: number;
  winner_immediate: number | null;
  rng: number;
}

export interface GameView extends GameState {
  winner: number | null;
  terminal: boolean;
}
