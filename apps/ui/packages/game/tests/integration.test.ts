// @vitest-environment node
//
// tests/integration.test.ts — Exercises the real ApiClient against a live
// `cargo run` server: proves the TS
// types actually match the live Rust contract, not just each other. Gated
// behind the server actually being up (`describe.runIf`) so `pnpm test`
// still passes when nothing is running on 127.0.0.1:7878 -- start the
// server with `cargo run` from the repo root to exercise this file for
// real. Forced to the "node" environment (overriding the suite's default
// happy-dom) because happy-dom's `fetch` enforces browser same-origin/CORS
// policy, which rejects a cross-origin plain HTTP call to 127.0.0.1:7878
// even though this is exactly the kind of request `createApiClient`'s
// `baseUrl` override exists for -- Node's native `fetch` has no such
// browser-only restriction.

import { describe, it, expect } from "vitest";
import { createApiClient } from "../src/api-client.js";

const BASE = "http://127.0.0.1:7878";

async function serverIsUp(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/api/games`);
    return r.ok;
  } catch {
    return false;
  }
}

const up = await serverIsUp();

describe.runIf(up)("createApiClient against a live cargo run server", () => {
  it("newGame('druid', ...) returns the same {state, view} shape curling /api/games/druid/new does", async () => {
    const config = { size: { w: 5, h: 5 } };
    const api = createApiClient(BASE);

    const viaClient = await api.newGame("druid", config);

    const raw = await fetch(`${BASE}/api/games/druid/new`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    });
    const viaCurl: unknown = await raw.json();

    expect(viaClient).toEqual(viaCurl);
  });

  it("getGames() lists druid", async () => {
    const api = createApiClient(BASE);
    const games = await api.getGames();
    expect(games.map((g) => g.kind)).toContain("druid");
  });
});
