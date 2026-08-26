// Placeholder test — proves the vitest toolchain works.
// Workspace linking and framework compilation are proven by
// `pnpm build` and `pnpm typecheck`, both of which exercise
// the @mcts/core import inside the app/ package where it resolves.

import { describe, it, expect } from "vitest";

describe("placeholder smoke test", () => {
  it("vitest toolchain works", () => {
    expect(1 + 1).toBe(2);
  });
});
