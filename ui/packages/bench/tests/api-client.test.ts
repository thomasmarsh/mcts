// tests/api-client.test.ts — URL/body construction and error mapping for
// createBenchApiClient, against a stubbed `fetch`. The wire contract these
// pin down (route paths, snake_case query params, `{error}` body mapping)
// is what server/bench/mod.rs serves; the stub stands in for the network,
// so no live server is involved.

import { afterEach, describe, expect, it, vi } from "vitest";
import { createBenchApiClient } from "../src/api-client.js";

interface CapturedCall {
  url: string;
  init?: RequestInit;
}

/** Stub global fetch to capture calls and respond with `body` as JSON. */
function stubFetch(body: unknown, opts: { ok?: boolean; status?: number } = {}): CapturedCall[] {
  const calls: CapturedCall[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return {
        ok: opts.ok ?? true,
        status: opts.status ?? 200,
        json: async () => body,
        text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
      } as unknown as Response;
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe("createBenchApiClient", () => {
  it("listRuns maps filters to query params, omitting nulls", async () => {
    const calls = stubFetch([]);
    const client = createBenchApiClient();

    await client.listRuns({ status: "running", game: "druid", limit: 5 });
    expect(calls[0]!.url).toBe("/api/bench/runs?status=running&game=druid&limit=5");

    calls.length = 0;
    await client.listRuns();
    expect(calls[0]!.url).toBe("/api/bench/runs");
  });

  it("getRunLog passes the offset cursor as `since`", async () => {
    const calls = stubFetch({ lines: [], next_offset: 42 });
    const client = createBenchApiClient();

    await client.getRunLog("rr-1", 42);
    expect(calls[0]!.url).toBe("/api/bench/runs/rr-1/log?since=42");

    calls.length = 0;
    await client.getRunLog("rr-1");
    expect(calls[0]!.url).toBe("/api/bench/runs/rr-1/log");
  });

  it("getLeaderboard maps gitSha to the wire's git_sha", async () => {
    const calls = stubFetch([]);
    const client = createBenchApiClient();

    await client.getLeaderboard({ game: "druid", gitSha: "abc1234", since: null });
    expect(calls[0]!.url).toBe("/api/bench/leaderboard?game=druid&git_sha=abc1234");
  });

  it("launchRun POSTs {kind, game, config}", async () => {
    const calls = stubFetch({ run_id: "r", pid: 1, log_path: "/x" });
    const client = createBenchApiClient();

    await client.launchRun("round_robin", "druid", { rounds: 2 });
    expect(calls[0]!.url).toBe("/api/bench/launch");
    expect(calls[0]!.init?.method).toBe("POST");
    expect(JSON.parse(String(calls[0]!.init?.body))).toEqual({
      kind: "round_robin",
      game: "druid",
      config: { rounds: 2 },
    });
  });

  it("stopRun POSTs to the run's stop route", async () => {
    const calls = stubFetch({ run_id: "rr-1", message: "stopped" });
    const client = createBenchApiClient();

    await client.stopRun("rr-1");
    expect(calls[0]!.url).toBe("/api/bench/runs/rr-1/stop");
    expect(calls[0]!.init?.method).toBe("POST");
  });

  it("surfaces the server's structured {error} body as the rejection message", async () => {
    stubFetch({ error: "run 'nope' not found", code: 404 }, { ok: false, status: 404 });
    const client = createBenchApiClient();

    await expect(client.getRun("nope")).rejects.toThrow("run 'nope' not found");
  });
});
