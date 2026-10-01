// tests/api-client.test.ts — URL/body construction for the `AiStrategyRef`
// flattening this phase adds to `aiMove`/`analyze` (a named preset becomes
// `{preset: id}`, a custom spec becomes `{preset: "custom", custom: spec}`,
// matching `server::main::AiMoveRequest`/`AnalyzeRequest`'s actual shape --
// see api-client.ts's `strategyBody` doc comment), the new
// `fetchStrategySchema` route, and `aiMove`/`analyze`'s submit-then-poll
// envelope (`{status: "done", result}` / `{status: "pending", jobId}`,
// polled via `GET /api/jobs/{id}`). Against a stubbed `fetch`, same
// convention as `packages/bench/tests/api-client.test.ts` -- no live server
// involved.

import { afterEach, describe, expect, it, vi } from "vitest";
import { createApiClient } from "../src/api-client.js";
import type { AiStrategyRef, SearchReport } from "../src/types.js";

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
      return {
        ok: true,
        status: 200,
        json: async () => body,
        text: async () => JSON.stringify(body),
      } as unknown as Response;
    }),
  );
  return calls;
}

function bodyOf(call: CapturedCall): unknown {
  return JSON.parse(call.init!.body as string);
}

/** Wraps a fixture in `aiMove`/`analyze`'s `done` envelope -- what the
 * server actually sends when a search finishes within its grace period
 * (see `apps/server/src/jobs.rs`). */
function done(result: unknown): unknown {
  return { status: "done", result };
}

const nullablePartialReport: SearchReport<string> = {
  schema_version: 1,
  status: "partial",
  reason: null,
  elapsed_seconds: null,
  iteration_limit: null,
  time_limit_seconds: null,
  completed_iterations: 0,
  termination: null,
  selected_action: null,
  actions: [],
  principal_variation: [],
  root_visits: 0,
  tree_nodes: 0,
  mean_depth: null,
  max_depth: null,
  graph_mode: null,
  tt_reads: 0,
  tt_writes: 0,
  tt_hits: 0,
  tt_hit_ratio: null,
  iterations_per_second: null,
  warnings: [],
};

const unavailableReport: SearchReport<string> = {
  ...nullablePartialReport,
  status: "unavailable",
  reason: "strategy_unsupported",
};

afterEach(() => vi.unstubAllGlobals());

describe("createApiClient / AiStrategyRef wire shape", () => {
  it("aiMove sends {preset: id} for a named preset, no custom key", async () => {
    const calls = stubFetch(done({ move: "x", state: {}, view: {} }));
    const api = createApiClient();
    const strategy: AiStrategyRef = { kind: "preset", id: "master" };

    await api.aiMove("druid", { some: "state" }, strategy);

    expect(calls[0]!.url).toBe("/api/games/druid/ai_move");
    expect(bodyOf(calls[0]!)).toEqual({ state: { some: "state" }, preset: "master" });
  });

  it("aiMove sends {preset: 'custom', custom: spec} for a custom strategy", async () => {
    const calls = stubFetch(done({ move: "x", state: {}, view: {} }));
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

    expect(bodyOf(calls[0]!)).toEqual({
      state: { some: "state" },
      preset: "custom",
      custom: strategy.spec,
    });
  });

  it("analyze forwards the strategy the same way plus budget_ms", async () => {
    const calls = stubFetch(
      done({
        actions: [],
        principal_variation: [],
        total_visits: 0,
        suggested_move: null,
      }),
    );
    const api = createApiClient();

    await api.analyze("druid", { some: "state" }, { kind: "preset", id: "strong" }, 1500);

    expect(bodyOf(calls[0]!)).toEqual({
      state: { some: "state" },
      preset: "strong",
      budget_ms: 1500,
    });
  });

  it("preserves complete snake_case search reports and legacy search forms", async () => {
    const aiCalls = stubFetch(
      done({ move: "x", state: {}, view: {}, search: nullablePartialReport }),
    );
    const api = createApiClient();

    const aiMove = await api.aiMove("druid", { some: "state" }, { kind: "preset", id: "strong" });

    expect(aiCalls[0]!.url).toBe("/api/games/druid/ai_move");
    expect(bodyOf(aiCalls[0]!)).toEqual({ state: { some: "state" }, preset: "strong" });
    expect(aiMove.status).toBe("done");
    if (aiMove.status !== "done") throw new Error("unreachable");
    expect(aiMove.result.search).toEqual(nullablePartialReport);
    expect(aiMove.result.search?.elapsed_seconds).toBeNull();
    expect(aiMove.result.search?.tt_hit_ratio).toBeNull();

    const analysisCalls = stubFetch(
      done({
        actions: [],
        principal_variation: [],
        total_visits: 0,
        suggested_move: null,
        search: unavailableReport,
      }),
    );
    const analysis = await api.analyze(
      "druid",
      { some: "state" },
      { kind: "preset", id: "random" },
    );

    expect(analysisCalls[0]!.url).toBe("/api/games/druid/analyze");
    expect(bodyOf(analysisCalls[0]!)).toEqual({ state: { some: "state" }, preset: "random" });
    expect(analysis.status).toBe("done");
    if (analysis.status !== "done") throw new Error("unreachable");
    expect(analysis.result.search).toEqual(unavailableReport);

    const legacyCalls = stubFetch(
      done({
        actions: [],
        principal_variation: [],
        total_visits: 0,
        suggested_move: null,
        search: null,
      }),
    );
    const legacy = await api.analyze("druid", { some: "state" }, { kind: "preset", id: "easy" });

    expect(legacyCalls[0]!.url).toBe("/api/games/druid/analyze");
    if (legacy.status !== "done") throw new Error("unreachable");
    expect(legacy.result.search).toBeNull();

    const absentCalls = stubFetch(done({ move: "x", state: {}, view: {} }));
    const absent = await api.aiMove("druid", { some: "state" }, { kind: "preset", id: "easy" });

    expect(absentCalls[0]!.url).toBe("/api/games/druid/ai_move");
    if (absent.status !== "done") throw new Error("unreachable");
    expect(absent.result.search).toBeUndefined();
  });

  it("aiMove/analyze pass through a pending envelope unchanged", async () => {
    stubFetch({ status: "pending", jobId: "job-1" });
    const api = createApiClient();

    const aiMove = await api.aiMove("druid", { some: "state" }, { kind: "preset", id: "master" });
    expect(aiMove).toEqual({ status: "pending", jobId: "job-1" });

    stubFetch({ status: "pending", jobId: "job-2" });
    const analysis = await api.analyze("druid", { some: "state" }, { kind: "preset", id: "strong" });
    expect(analysis).toEqual({ status: "pending", jobId: "job-2" });
  });

  it("pollAiMove/pollAnalyze both GET /api/jobs/{id}", async () => {
    const calls = stubFetch({ status: "pending" });
    const api = createApiClient();

    await api.pollAiMove("job-1");
    expect(calls[0]!.url).toBe("/api/jobs/job-1");

    await api.pollAnalyze("job-2");
    expect(calls[1]!.url).toBe("/api/jobs/job-2");
  });

  it("pollAiMove resolves a done/error poll result the same way submit's inline done does", async () => {
    stubFetch(done({ move: "x", state: {}, view: {} }));
    const api = createApiClient();
    const polled = await api.pollAiMove("job-1");
    expect(polled).toEqual({ status: "done", result: { move: "x", state: {}, view: {} } });

    stubFetch({ status: "error", error: "subprocess crashed" });
    const errored = await api.pollAnalyze("job-2");
    expect(errored).toEqual({ status: "error", error: "subprocess crashed" });
  });

  it("fetchStrategySchema GETs /api/strategy-schema", async () => {
    const schema = { select: { variants: [] } };
    const calls = stubFetch(schema);
    const api = createApiClient();

    const result = await api.fetchStrategySchema();

    expect(calls[0]!.url).toBe("/api/strategy-schema");
    expect(result).toEqual(schema);
  });

  it("fetchStrategyAlgorithms GETs /api/games/{kind}/strategy-algorithms", async () => {
    const info = {
      id: "druid",
      baselines: [],
      eval_rounds: 1,
      parameters: [],
      conditions: [],
      game_config: null,
    };
    const calls = stubFetch(info);
    const api = createApiClient();

    const result = await api.fetchStrategyAlgorithms("druid");

    expect(calls[0]!.url).toBe("/api/games/druid/strategy-algorithms");
    expect(result).toEqual(info);
  });

  it("fetchStrategyAlgorithms passes through a null response", async () => {
    stubFetch(null);
    const api = createApiClient();

    const result = await api.fetchStrategyAlgorithms("traffic-lights");

    expect(result).toBeNull();
  });
});

describe("createApiClient / resolveKind", () => {
  it("sends the resolved kind in the URL, not the id it was called with", async () => {
    const calls = stubFetch({ state: {}, view: {} });
    const api = createApiClient(undefined, (kind) => (kind === "focus:3p" ? "focus-3p" : kind));

    await api.newGame("focus:3p");

    expect(calls[0]!.url).toBe("/api/games/focus-3p/new");
  });

  it("defaults to the identity function when omitted", async () => {
    const calls = stubFetch([]);
    const api = createApiClient();

    await api.aiPresets("druid");

    expect(calls[0]!.url).toBe("/api/games/druid/ai_presets");
  });
});
