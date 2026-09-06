// tests/fixtures/fake-games-registry.ts — Drop-in replacement for
// `app/src/games.js`'s shape, `vi.mock`'d in by GameShell.test.tsx so
// GameShell resolves to `fake-game.tsx`'s `fakeModule` instead of the real
// Druid/tic-tac-toe registry.

import { fakeModule } from "./fake-game.js";

export const GAME_MODULES: Record<string, () => Promise<typeof fakeModule>> = {
  fake: () => Promise.resolve(fakeModule),
};

export const GAME_META: Record<string, { players: string[]; wireKind?: string }> = {
  fake: { players: ["A", "B"] },
};

// Mirrors app/src/games.js's real implementation -- see that file's doc
// comment for what these do. Duplicated here (not imported) since this
// fixture stands in for the whole module `vi.mock`'d in above.
export function groupIdOf(id: string): string {
  const i = id.indexOf(":");
  return i === -1 ? id : id.slice(0, i);
}

export function wireKindOf(id: string): string {
  return GAME_META[id]?.wireKind ?? id;
}

export const DEFAULT_GAME_KIND = "fake";
