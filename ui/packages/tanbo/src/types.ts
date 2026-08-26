// types.ts — Concrete Tanbo wire types, mirroring server/adapters/tanbo.rs's
// `WireState`/`GameView` shapes. The board is transmitted as a flat
// 81-element array (row-major) of null/"Black"/"White".

export type Player = "Black" | "White";

/** A cell index 0..80 (row-major: `row * 9 + col`), matching
 * `src/games/tanbo.rs`'s `Move(pub u16)`. */
export type Move = number;

export interface GameState {
  turn: Player;
  cells: (Player | null)[];
}

export interface GameView {
  turn: Player;
  cells: (Player | null)[];
  winner: Player | null;
  terminal: boolean;
}
