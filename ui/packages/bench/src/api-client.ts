// api-client.ts — Typed fetch wrapper for the bench server's `/api/bench/*`
// routes (server/bench/mod.rs). Hard rule: this is the *only* file in this
// package allowed to reference `fetch` — enforced by the fetch-ban eslint
// rule in ui/eslint.config.js. Three layers, mirroring @mcts/game's
// api-client.ts:
//   1. `BenchApiClient` — a plain interface of `Promise`-returning methods.
//   2. `createBenchApiClient(): BenchApiClient` — the one concrete
//      implementation.
//   3. `createBenchEnv(api): BenchEnv` — lifts every method into an
//      `Effect`. `BenchEnv` (the type the reducer actually receives) is
//      defined in reducer.ts, not here — see that file's header comment.

import { Effect } from "@mcts/core";
import type { BenchEnv } from "./reducer.js";
import type {
  BenchKindInfo,
  CommitTrendData,
  LaunchResponse,
  LeaderboardEntry,
  LeaderboardFilters,
  RunDetail,
  RunFilters,
  RunLogResponse,
  RunSummary,
  StopResponse,
} from "./types.js";

export interface BenchApiClient {
  listRuns(filters?: { status?: string | null; game?: string | null; limit?: number }): Promise<RunSummary[]>;
  getRun(runId: string): Promise<RunDetail>;
  getRunLog(runId: string, since?: number): Promise<RunLogResponse>;
  /** Fetch the full raw content of the run's stdout.log file (stderr
   * output redirected by the launcher). */
  getRunStdout(runId: string): Promise<string>;
  getLeaderboard(filters?: Partial<LeaderboardFilters>): Promise<LeaderboardEntry[]>;
  /** Fetch the leaderboard for each distinct git SHA that has run data
   * for the given game. Returns a map of SHA -> entries. */
  fetchCommitTrends(game: string | null): Promise<CommitTrendData>;
  launchRun(kind: string, game: string, config?: unknown): Promise<LaunchResponse>;
  stopRun(runId: string): Promise<StopResponse>;
  getBenchKinds(): Promise<BenchKindInfo[]>;
}

/** The server (`BenchError`'s `IntoResponse` impl, server/bench/mod.rs)
 * returns a structured `{error, code}` JSON body. Read as text first (a
 * body-limit/timeout rejection, or anything below the `BenchError` layer,
 * may not be JSON at all) and only then try to parse it as `{error}`,
 * falling back to the raw text. Same helper as @mcts/game's api-client. */
async function errorMessage(r: Response): Promise<string> {
  const text = await r.text().catch(() => "");
  if (text) {
    try {
      const body: unknown = JSON.parse(text);
      if (body && typeof body === "object" && typeof (body as { error?: unknown }).error === "string") {
        return (body as { error: string }).error;
      }
    } catch {
      // Not JSON -- fall through to the raw text below.
    }
  }
  return text || `API ${r.status}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await errorMessage(r));
  return r.json() as Promise<T>;
}

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new Error(await errorMessage(r));
  return r.json() as Promise<T>;
}

/** Build a `?k=v&...` suffix, skipping null/undefined values. */
function queryString(params: Record<string, string | number | null | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined) q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

/** `baseUrl` defaults to `""` (relative URLs, resolved against whatever
 * origin the page loads from — the Vite dev proxy or the server's own
 * `ServeDir` in production), same convention as @mcts/game's client. */
export function createBenchApiClient(baseUrl = ""): BenchApiClient {
  const url = (path: string): string => baseUrl + path;
  return {
    async listRuns(filters = {}): Promise<RunSummary[]> {
      return fetchJson(url(`/api/bench/runs${queryString({ status: filters.status, game: filters.game, limit: filters.limit })}`));
    },
    async getRun(runId: string): Promise<RunDetail> {
      return fetchJson(url(`/api/bench/runs/${encodeURIComponent(runId)}`));
    },
    async getRunLog(runId: string, since?: number): Promise<RunLogResponse> {
      return fetchJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/log${queryString({ since })}`));
    },
    async getRunStdout(runId: string): Promise<string> {
      const r = await fetch(url(`/api/bench/runs/${encodeURIComponent(runId)}/stdout`));
      if (!r.ok) throw new Error(await errorMessage(r));
      return r.text();
    },
    async getLeaderboard(filters: Partial<LeaderboardFilters> = {}): Promise<LeaderboardEntry[]> {
      return fetchJson(
        url(`/api/bench/leaderboard${queryString({ game: filters.game, git_sha: filters.gitSha, since: filters.since })}`),
      );
    },
    async fetchCommitTrends(game: string | null): Promise<CommitTrendData> {
      // First fetch runs to discover distinct git SHAs.
      const runs = await this.listRuns({ game, limit: 1000 });
      const shaSet = new Set<string>();
      for (const r of runs) {
        if (!r.git_dirty) shaSet.add(r.git_sha);
      }
      const shas = Array.from(shaSet).sort();
      // Fetch leaderboard for each SHA in parallel.
      const results = await Promise.all(
        shas.map((sha) => this.getLeaderboard({ game, gitSha: sha, since: null })),
      );
      const data: CommitTrendData = {};
      for (let i = 0; i < shas.length; i++) {
        data[shas[i]!] = results[i]!;
      }
      return data;
    },
    async launchRun(kind: string, game: string, config?: unknown): Promise<LaunchResponse> {
      return postJson(url("/api/bench/launch"), { kind, game, config });
    },
    async stopRun(runId: string): Promise<StopResponse> {
      return postJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/stop`));
    },
    async getBenchKinds(): Promise<BenchKindInfo[]> {
      return fetchJson(url("/api/bench/kinds"));
    },
  };
}

export function createBenchEnv(api: BenchApiClient): BenchEnv {
  const lift = <T>(thunk: () => Promise<T>): Effect<T> => Effect.fromPromise(thunk);
  return {
    listRuns: (filters: RunFilters) => lift(() => api.listRuns(filters)),
    getRun: (runId: string) => lift(() => api.getRun(runId)),
    getRunLog: (runId: string, since: number) => lift(() => api.getRunLog(runId, since)),
    getRunStdout: (runId: string) => lift(() => api.getRunStdout(runId)),
    getLeaderboard: (filters: LeaderboardFilters) => lift(() => api.getLeaderboard(filters)),
    fetchCommitTrends: (game: string | null) => lift(() => api.fetchCommitTrends(game)),
    launchRun: (kind: string, game: string, config?: unknown) => lift(() => api.launchRun(kind, game, config)),
    stopRun: (runId: string) => lift(() => api.stopRun(runId)),
    getBenchKinds: () => lift(() => api.getBenchKinds()),
  };
}
