// types.ts — Generic wire types for stone-on-intersection Go-variant games
// (AtariGo, Gonnect, and any future game sharing this wire shape). The board
// is transmitted as a flat, row-major `N * N`-element array of
// null/"Black"/"White", the same shape Tanbo's server adapter also uses —
// only the board size `N` and the win condition differ per game, both of
// which live entirely server-side.

export type Player = "Black" | "White";

/** A placement move: `[cellIndex, capturedWordsHex]`. `cellIndex` is a
 * row-major `0..N*N` index; `capturedWordsHex` is an opaque snapshot the
 * server precomputes (the stones this placement captures, as hex-encoded
 * bitboard words — see `games/atarigo`/`games/gonnect`'s `Move` doc comment
 * for why hex, not raw numbers) and needs back byte-for-byte unchanged on
 * `apply`. The UI never constructs or inspects a `Move` itself — it only
 * ever reads one from `legalMoves`/`analysisOverlay` and passes it straight
 * back; `cellOf` is the one thing it needs to read off one for display. */
export type Move = [cellIndex: number, capturedWordsHex: string[]];

export function cellOf(move: Move): number {
  return move[0];
}

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
