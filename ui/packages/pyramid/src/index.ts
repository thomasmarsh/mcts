// packages/pyramid/src/index.ts — Shared pyramid-family building blocks
// (index math, three.js board/marble helpers, pie-rule swap panel, board-
// size picker) consumed by `@mcts/margo` and `@mcts/akron`. Not a
// `GameKindModule` itself -- there is no bare "pyramid" game to host.

export * from "./geometry.js";
export * from "./render.js";
export * from "./BoardSizeFields.js";
export { PieRuleSwap } from "./PieRuleSwap.js";
