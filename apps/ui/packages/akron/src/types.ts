// types.ts — Concrete Akron wire types, mirroring `games/akron/src/
// main.rs`'s plain (no `#[serde(...)]` attributes) JSON shapes:
//   - `Player` (unit-variant enum) -> a bare string.
//   - `Action` (mixed enum: `Add(u16, u8)`/`Move(u16, u16, u8)` tuples,
//     `Swap` unit) -> externally tagged: `{ "Add": [index, n] }`,
//     `{ "Move": [src, dst, n] }`, or `"Swap"`.
//   - `WireState`/`GameView`/`CellView` (plain structs) -> plain JSON
//     objects, unrenamed field names.

export type Player = "Black" | "White";

/** Tagged union mirroring `game_akron::Action`. `Add`'s `[index, n]` pair
 * and `Move`'s `[src, dst, n]` triple carry the board's base width alongside
 * the flat index/indices the same way Margo's `Place` does -- see that
 * package's `types.ts` doc comment for why (`apply_to_action`/
 * `invert_action` need it and have no `State` in scope). */
export type Action = { Add: [number, number] } | { Move: [number, number, number] } | "Swap";

export type Move = Action;

export interface GameState {
  n: number;
  occupied: number[];
  black: number[];
  white_pile: number;
  black_pile: number;
  turn: Player;
  can_swap: boolean;
}

export interface CellView {
  piece: Player;
}

export interface GameView {
  n: number;
  /** One entry per flat pyramid index (`0..totalCells(n)`, see
   * `@mcts/pyramid`'s `toCoord`/`totalCells`), `null` for an empty cell. */
  cells: (CellView | null)[];
  white_pile: number;
  black_pile: number;
  turn: Player;
  can_swap: boolean;
  winner: Player | null;
  terminal: boolean;
}

export interface NewGameConfig {
  n: number;
}

/** Mirrors `game_akron::{MIN_N, MAX_N, DEFAULT_N}`. */
export const MIN_N = 4;
export const MAX_N = 10;
export const DEFAULT_N = 7;

export function isSwap(action: Action): action is "Swap" {
  return action === "Swap";
}

export function isAdd(action: Action): action is { Add: [number, number] } {
  return typeof action === "object" && "Add" in action;
}

export function isMove(action: Action): action is { Move: [number, number, number] } {
  return typeof action === "object" && "Move" in action;
}
