// tests/api-client.test.ts — URL/body construction for the `AiStrategyRef`
// flattening this phase adds to `aiMove`/`analyze` (a named preset becomes
// `{preset: id}`, a custom spec becomes `{preset: "custom", custom: spec}`,
// matching `server::main::AiMoveRequest`/`AnalyzeRequest`'s actual shape --
// see api-client.ts's `strategyBody` doc comment), plus the new
// `fetchStrategySchema` route. Against a stubbed `fetch`, same convention as
// `packages/bench/tests/api-client.test.ts` -- no live server involved.

import { afterEach, describe, expect, it, vi } from "vitest";
import { createApiClient } from "../src/api-client.js";
import type { AiStrategyRef } from "../src/types.js";

interface CapturedCall {
  url: string;
  init?: RequestInit;
}

function stubFetch(body: unknown): CapturedCall[] {
  const calls: CapturedCall[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) } as unknown as Response;
    }),
  );
  return calls;
}

function bodyOf(call: CapturedCall): unknown {
  return JSON.parse(call.init!.body as string);
}

afterEach(() => vi.unstubAllGlobals());

describe("createApiClient / AiStrategyRef wire shape", () => {
  it("aiMove sends {preset: id} for a named preset, no custom key", async () => {
    const calls = stubFetch({ move: "x", state: {}, view: {} });
    const api = createApiClient();
    const strategy: AiStrategyRef = { kind: "preset", id: "master" };

    await api.aiMove("druid", { some: "state" }, strategy);

    expect(calls[0]!.url).toBe("/api/games/druid/ai_move");
    expect(bodyOf(calls[0]!)).toEqual({ state: { some: "state" }, preset: "master" });
  });

  it("aiMove sends {preset: 'custom', custom: spec} for a custom strategy", async () => {
    const calls = stubFetch({ move: "x", state: {}, view: {} });
    const api = createApiClient();
    const strategy: AiStrategyRef = {
      kind: "custom",
      spec: {
        search: {
          select: { kind: "ucb1", c: 1.4 },
          simulate: { kind: "uniform" },
          backprop: { kind: "classic" },
          final_action: { kind: "robust_child" },
        },
        max_iterations: 500,
      },
    };

    await api.aiMove("nim", { some: "state" }, strategy);

    expect(bodyOf(calls[0]!)).toEqual({ state: { some: "state" }, preset: "custom", custom: strategy.spec });
  });

  it("analyze forwards the strategy the same way plus budget_ms", async () => {
    const calls = stubFetch({ actions: [], principal_variation: [], total_visits: 0, suggested_move: null });
    const api = createApiClient();

    await api.analyze("druid", { some: "state" }, { kind: "preset", id: "strong" }, 1500);

    expect(bodyOf(calls[0]!)).toEqual({ state: { some: "state" }, preset: "strong", budget_ms: 1500 });
  });

  it("fetchStrategySchema GETs /api/strategy-schema", async () => {
    const schema = { select: { variants: [] } };
    const calls = stubFetch(schema);
    const api = createApiClient();

    const result = await api.fetchStrategySchema();

    expect(calls[0]!.url).toBe("/api/strategy-schema");
    expect(result).toEqual(schema);
  });
});
