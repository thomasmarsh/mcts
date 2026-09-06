// packages/goban/src/index.ts — Shared goban (Go-style intersection board)
// building blocks for `@mcts/atarigo`/`@mcts/gonnect`-style thin wrapper
// packages, mirroring the `@mcts/chess-variant` + `@mcts/breakthrough`/
// `@mcts/knightthrough` split. Each consuming package picks a board size and
// win-condition message, then assembles its own `GameKindModule`.

export * from "./types.js";
export { GobanRenderer } from "./GobanRenderer.js";
export { createSimpleSummary, formatMove } from "./summary.js";
export { standardStarPoints } from "./star-points.js";
export { createSizeField, createSizeRangeField, type SizeConfig } from "./NewGameFields.js";
