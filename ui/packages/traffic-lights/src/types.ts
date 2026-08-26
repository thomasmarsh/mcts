// types.ts — Concrete traffic lights wire types, mirroring
// server/adapters/traffic_lights.rs's `WireState`/`GameView` shapes.
// Cell colours are "R" / "Y" / "G" (not the engine's internal 2-bit
// encoding), players are "A" / "B".

/** A cell colour — `null` for empty, or one of `"R"`, `"Y"`, `"G"`. */
export type CellState = "R" | "Y" | "G" | null;

/** A player label: `"A"` (First) or `"B"` (Second). */
export type Player = "A" | "B";

/** A move is the raw u8 encoding `(index << 2) | piece` that
 * `traffic_lights::Move` uses on the Rust side. The renderer decodes
 * the cell index (bits 2..) to know which cell was played. */
export type Move = number;

export interface GameState {
  turn: Player;
  cells: CellState[];
}

export interface GameView {
  turn: Player;
  cells: CellState[];
  winner: Player | null;
  terminal: boolean;
}
