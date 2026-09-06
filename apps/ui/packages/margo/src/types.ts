// types.ts — Concrete Margo wire types, mirroring `games/margo/src/main.rs`'s
// plain (no `#[serde(...)]` attributes anywhere but the derives Rust's
// `Action`/`Player` already carry) JSON shapes:
//   - `Player` (unit-variant enum) -> a bare string.
//   - `Action` (mixed enum: `Place(u16, u8)` tuple, `Swap` unit) ->
//     externally tagged: `{ "Place": [index, n] }` or `"Swap"`.
//   - `WireState`/`GameView`/`CellView` (plain structs) -> plain JSON
//     objects, unrenamed field names.
// `GameView` additionally mirrors `main.rs`'s `GameView` struct -- the same
// fields as `GameState` (minus `previous`, which is round-trip-only wire
// state a renderer never needs) plus `winner`/`terminal`.

export type Player = "Black" | "White";

/** Tagged union mirroring `game_margo::Action`. `Place`'s `[index, n]` pair
 * carries the board's base width alongside the flat index the same way
 * Druid's `Move` carries its own board size -- see that crate's doc comment
 * for why (`apply_to_action`/`invert_action` need it and have no `State` in
 * scope). */
export type Action = { Place: [number, number] } | "Swap";

export type Move = Action;

export interface GameState {
  n: number;
  occupied: number[];
  black: number[];
  zombie: number[];
  previous: [number[], number[]] | null;
  turn: Player;
  can_swap: boolean;
}

export interface CellView {
  piece: Player;
  zombie: boolean;
}

export interface GameView {
  n: number;
  /** One entry per flat pyramid index (`0..totalCells(n)`, see
   * `geometry.ts`'s `toCoord`/`totalCells`), `null` for an empty cell. */
  cells: (CellView | null)[];
  turn: Player;
  can_swap: boolean;
  winner: Player | null;
  terminal: boolean;
}

export interface NewGameConfig {
  n: number;
}

/** Mirrors `game_margo::MIN_N`/`MAX_N`/`DEFAULT_N`. */
export const MIN_N = 4;
export const MAX_N = 10;
export const DEFAULT_N = 7;

/** One-click board-size presets, spanning the full supported range --
 * unlike Druid's curated subset, every Margo size the server accepts is
 * worth offering directly since there are only seven of them. */
export const BOARD_SIZES: number[] = Array.from({ length: MAX_N - MIN_N + 1 }, (_, i) => MIN_N + i);

export function isSwap(action: Action): action is "Swap" {
  return action === "Swap";
}
