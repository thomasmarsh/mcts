// types.ts — Concrete Druid wire types, mirroring src/games/druid.rs's plain
// (no #[serde(...)] attributes anywhere) derive-default JSON shapes:
//   - `Player`/`Orientation` (unit-variant enums) -> bare strings.
//   - `Piece` (mixed enum: `Sarsen` unit, `Lintel(Orientation)` newtype) ->
//     externally tagged: `"Sarsen"` or `{ "Lintel": "Horizontal" }`.
//   - `Move(Piece, u8)` (tuple struct) -> a plain 2-element JSON array.
//   - `Square`/`Hand`/`State`/`Size` (plain structs) -> plain JSON objects,
//     unrenamed field names.
// `GameView` additionally mirrors `server/adapters/druid.rs`'s `GameView`
// struct -- the same fields as `GameState` plus `winner`/`terminal`, which
// only exist on the view (a renderer needs both: `GameState` for the raw
// `apply`/`legal_moves` round trip and mover derivation, `GameView` for
// anything display-only).

export type Player = "Black" | "White";

export type Orientation = "Horizontal" | "Vertical";

export type Piece = "Sarsen" | { Lintel: Orientation };

/** Tuple struct `Move(Piece, u8)` -> `[piece, index]`. */
export type Move = [Piece, number];

export interface Square {
  height: number;
  piece: Player | null;
}

export interface Hand {
  sarsens: number;
  lintels: number;
}

export interface GameState {
  player: Player;
  board: Square[];
  hand_black: Hand;
  hand_white: Hand;
  size: Size;
}

export interface GameView {
  size: Size;
  player: Player;
  board: Square[];
  hand_black: Hand;
  hand_white: Hand;
  winner: Player | null;
  terminal: boolean;
}

export interface Size {
  w: number;
  h: number;
}

export interface NewGameConfig {
  size: Size;
}

/** Mirrors `mcts::games::druid::DEFAULT_SIZE`. */
export const DEFAULT_SIZE: Size = { w: 5, h: 5 };

/** The board sizes app.js's new-game dialog offered -- `Size::is_supported()`
 * (server/side, `src/games/druid.rs`) accepts a wider range, but this is the
 * curated set worth surfacing as one-click presets. */
export const BOARD_SIZES: Size[] = [
  { w: 5, h: 5 },
  { w: 7, h: 7 },
  { w: 9, h: 9 },
  { w: 10, h: 10 },
];
