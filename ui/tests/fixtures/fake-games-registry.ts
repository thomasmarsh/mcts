// tests/fixtures/fake-games-registry.ts — Drop-in replacement for
// `app/src/games.js`'s shape, `vi.mock`'d in by GameShell.test.tsx so
// GameShell resolves to `fake-game.tsx`'s `fakeModule` instead of the real
// Druid/tic-tac-toe registry.

import { fakeModule } from "./fake-game.js";

export const GAME_MODULES = { fake: fakeModule };
export const DEFAULT_GAME_KIND = "fake";
