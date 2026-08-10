// packages/chess-variant/src/index.ts — Shared chess-variant module exports.
// Both knightthrough and breakthrough re-export from here into their own
// `GameKindModule`-shaped packages.

export * from "./types.js";
export { ChessVariantRenderer } from "./ChessVariantRenderer.js";
export { summarize, formatMove } from "./summary.js";