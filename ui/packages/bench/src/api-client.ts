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
  LaunchResponse,
  RunDetail,
  RunFilters,
  RunLogResponse,
  RunSummary,
  TunerGameInfo,
  StopResponse,
  TrialRow,
  GameTraceSummary,
  GameMove,
} from "./types.js";

export interface BenchApiClient {
  listRuns(filters?: {
    status?: string | null;
    game?: string | null;
    limit?: number;
    project_id?: string | null;
    experiment_id?: string | null;
  }): Promise<RunSummary[]>;
  getRun(runId: string): Promise<RunDetail>;
  getRunLog(runId: string, since?: number): Promise<RunLogResponse>;
  /** Fetch the full raw content of the run's stdout.log file (stderr
   * output redirected by the launcher). */
  getRunStdout(runId: string): Promise<string>;
  launchRun(kind: string, game: string, config?: unknown): Promise<LaunchResponse>;
  stopRun(runId: string): Promise<StopResponse>;
  /** Per-game tuner metadata for every game that supports tuner tuning. */
  getTunerKinds(): Promise<TunerGameInfo[]>;
  /** Trial rows for one run, oldest first. */
  getRunTrials(runId: string, limit?: number): Promise<TrialRow[]>;
  getRunGames(runId: string, limit?: number, cellId?: string | null): Promise<GameTraceSummary[]>;
  getRunGameMoves(runId: string, gameSeq: number): Promise<GameMove[]>;
  deleteRun(runId: string): Promise<void>;
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
      if (
        body &&
        typeof body === "object" &&
        typeof (body as { error?: unknown }).error === "string"
      ) {
        const fields = (body as { fields?: unknown }).fields;
        if (Array.isArray(fields)) {
          const fieldMessages = fields.flatMap((field): string[] => {
            if (!field || typeof field !== "object") return [];
            const path = (field as { path?: unknown }).path;
            const message = (field as { message?: unknown }).message;
            return typeof path === "string" && typeof message === "string"
              ? [`${path}: ${message}`]
              : [];
          });
          if (fieldMessages.length > 0) return fieldMessages.join("; ");
        }
        return (body as { error: string }).error;
      }
    } catch {
      // Not JSON -- fall through to the raw text below.
    }
  }
  return text || `API ${r.status}`;
}

class BenchApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "BenchApiError";
  }
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new BenchApiError(await errorMessage(r), r.status);
  return r.json() as Promise<T>;
}

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new BenchApiError(await errorMessage(r), r.status);
  return r.json() as Promise<T>;
}

async function deleteRequest(url: string): Promise<void> {
  const r = await fetch(url, { method: "DELETE" });
  if (!r.ok) throw new BenchApiError(await errorMessage(r), r.status);
}

/** Build a `?k=v&...` suffix, skipping null/undefined values. */
function queryString(params: object): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (typeof v === "string" || typeof v === "number") q.set(k, String(v));
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
      return fetchJson(
        url(
          `/api/bench/runs${queryString({ status: filters.status, game: filters.game, limit: filters.limit, project_id: filters.project_id, experiment_id: filters.experiment_id })}`,
        ),
      );
    },
    async getRun(runId: string): Promise<RunDetail> {
      return fetchJson(url(`/api/bench/runs/${encodeURIComponent(runId)}`));
    },
    async getRunLog(runId: string, since?: number): Promise<RunLogResponse> {
      return fetchJson(
        url(`/api/bench/runs/${encodeURIComponent(runId)}/log${queryString({ since })}`),
      );
    },
    async getRunStdout(runId: string): Promise<string> {
      const r = await fetch(url(`/api/bench/runs/${encodeURIComponent(runId)}/stdout`));
      if (!r.ok) throw new Error(await errorMessage(r));
      return r.text();
    },
    async launchRun(kind: string, game: string, config?: unknown): Promise<LaunchResponse> {
      return postJson(url("/api/bench/launch"), { kind, game, config });
    },
    async stopRun(runId: string): Promise<StopResponse> {
      return postJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/stop`));
    },
    async getTunerKinds(): Promise<TunerGameInfo[]> {
      return fetchJson(url("/api/bench/tuner/kinds"));
    },
    async getRunTrials(runId: string, limit?: number): Promise<TrialRow[]> {
      return fetchJson(
        url(`/api/bench/runs/${encodeURIComponent(runId)}/trials${queryString({ limit })}`),
      );
    },
    async getRunGames(
      runId: string,
      limit?: number,
      cellId?: string | null,
    ): Promise<GameTraceSummary[]> {
      return fetchJson(
        url(
          `/api/bench/runs/${encodeURIComponent(runId)}/games${queryString({ limit, cell_id: cellId })}`,
        ),
      );
    },
    async getRunGameMoves(runId: string, gameSeq: number): Promise<GameMove[]> {
      return fetchJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/games/${gameSeq}/moves`));
    },
    async deleteRun(runId: string): Promise<void> {
      return deleteRequest(url(`/api/bench/runs/${encodeURIComponent(runId)}`));
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
    launchRun: (kind: string, game: string, config?: unknown) =>
      lift(() => api.launchRun(kind, game, config)),
    stopRun: (runId: string) => lift(() => api.stopRun(runId)),
    getTunerKinds: () => lift(() => api.getTunerKinds()),
    getRunTrials: (runId: string, limit?: number) => lift(() => api.getRunTrials(runId, limit)),
    getRunGames: (runId: string, limit?: number, cellId?: string | null) =>
      lift(() => api.getRunGames(runId, limit, cellId)),
    getRunGameMoves: (runId: string, gameSeq: number) =>
      lift(() => api.getRunGameMoves(runId, gameSeq)),
    deleteRun: (runId: string) => lift(() => api.deleteRun(runId)),
  };
}
