// types.ts — Wire types mirroring games/focus/src/adapter.rs's `WireState`/
// `GameView`. A player is a plain index (0..P), not a color/name union like
// tic-tac-toe's `Piece` -- Focus is symmetric under player relabeling (see
// that file's own doc comment), and P varies by variant (2/3/4 players), so
// there's no fixed enum to name here. `move-codec.ts` has `Move`'s bit
// layout; kept separate since it's pure codec logic, not a wire shape.

export interface GameState {
  /** 64 packed `u16` cell words, row-major (`row*8+col`). Opaque here --
   * only `GameAdapter::apply` needs to round-trip this; the renderer reads
   * the already-decoded `GameView.board` instead. */
  cells: number[];
  /** Reserve (off-board, replayable) piece count per player, length P. */
  reserves: number[];
  /** Player to move, 0..P. */
  turn: number;
  hash: number;
}

export interface GameView {
  /** One entry per board cell (64, row-major), bottom-to-top player indices
   * -- empty for both an empty cell and a notched-off invalid corner (see
   * geometry.ts's `isValidCell` for which indices those are; not
   * transmitted here, same convention as this repo's other fixed-shape
   * boards). */
  board: number[][];
  reserves: number[];
  current_player: number;
  winner: number | null;
  terminal: boolean;
}
