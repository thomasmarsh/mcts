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
  ChainRung,
  CommitTrendData,
  LaunchResponse,
  LeaderboardEntry,
  LeaderboardFilters,
  RunDetail,
  RunFilters,
  RunLogResponse,
  RunSummary,
  Smac3GameInfo,
  StopResponse,
  TrialRow,
  GameTraceSummary,
  GameMove,
  Project,
  Experiment,
  ExperimentSpecV1,
  ExperimentCell,
} from "./types.js";

export interface BenchApiClient {
  listRuns(filters?: { status?: string | null; game?: string | null; limit?: number; project_id?: string | null; experiment_id?: string | null }): Promise<RunSummary[]>;
  listProjects(): Promise<Project[]>;
  createProject(name: string, description: string): Promise<Project>;
  getProject(projectId: string): Promise<Project>;
  updateProject(projectId: string, body: { name?: string; description?: string; archived?: boolean }): Promise<Project>;
  listExperiments(projectId: string): Promise<Experiment[]>;
  createExperiment(projectId: string, body: { name: string; description: string; spec: ExperimentSpecV1 }): Promise<Experiment>;
  getExperiment(experimentId: string): Promise<Experiment>;
  updateExperiment(experimentId: string, body: { name: string; description: string; spec: ExperimentSpecV1 }): Promise<Experiment>;
  launchExperiment(experimentId: string): Promise<LaunchResponse>;
  getRunCells(runId: string): Promise<ExperimentCell[]>;
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
  /** Relaunch a finished/stopped SMAC3 run with a bigger trial budget,
   * seeded from its saved state (`POST /api/bench/runs/{run_id}/resume`). */
  resumeRun(runId: string, nTrials: number, nWorkers?: number): Promise<LaunchResponse>;
  /** Promote this run's current incumbent to a new baseline instance and
   * relaunch as the next rung in its ladder chain (`POST
   * /api/bench/runs/{run_id}/advance-baseline`). Stops the run first if
   * it's still running. `nTrials` defaults server-side when omitted. */
  advanceBaseline(runId: string, nTrials?: number, nWorkers?: number): Promise<LaunchResponse>;
  getBenchKinds(): Promise<BenchKindInfo[]>;
  /** Per-game tuner metadata for every game that supports SMAC3 tuning. */
  getSmac3Kinds(): Promise<Smac3GameInfo[]>;
  /** Trial rows for one run, oldest first. */
  getRunTrials(runId: string, limit?: number): Promise<TrialRow[]>;
  /** Every rung of the ladder chain `runId` belongs to, oldest first (`GET
   * /api/bench/runs/{run_id}/chain`) -- a one-element list containing just
   * `runId` for a plain (non-laddered) run. */
  getRunChain(runId: string): Promise<ChainRung[]>;
  getRunGames(runId: string, limit?: number): Promise<GameTraceSummary[]>;
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
      if (body && typeof body === "object" && typeof (body as { error?: unknown }).error === "string") {
        const fields = (body as { fields?: unknown }).fields;
        if (Array.isArray(fields)) {
          const fieldMessages = fields.flatMap((field): string[] => {
            if (!field || typeof field !== "object") return [];
            const path = (field as { path?: unknown }).path;
            const message = (field as { message?: unknown }).message;
            return typeof path === "string" && typeof message === "string" ? [`${path}: ${message}`] : [];
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

async function deleteRequest(url: string): Promise<void> {
  const r = await fetch(url, { method: "DELETE" });
  if (!r.ok) throw new Error(await errorMessage(r));
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
      return fetchJson(url(`/api/bench/runs${queryString({ status: filters.status, game: filters.game, limit: filters.limit, project_id: filters.project_id, experiment_id: filters.experiment_id })}`));
    },
    async listProjects(): Promise<Project[]> { return fetchJson(url("/api/bench/projects")); },
    async createProject(name: string, description: string): Promise<Project> { return postJson(url("/api/bench/projects"), { name, description }); },
    async getProject(projectId: string): Promise<Project> { return fetchJson(url(`/api/bench/projects/${encodeURIComponent(projectId)}`)); },
    async updateProject(projectId: string, body): Promise<Project> { const r = await fetch(url(`/api/bench/projects/${encodeURIComponent(projectId)}`), { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); if (!r.ok) throw new Error(await errorMessage(r)); return r.json() as Promise<Project>; },
    async listExperiments(projectId: string): Promise<Experiment[]> { return fetchJson(url(`/api/bench/projects/${encodeURIComponent(projectId)}/experiments`)); },
    async createExperiment(projectId: string, body): Promise<Experiment> { return postJson(url(`/api/bench/projects/${encodeURIComponent(projectId)}/experiments`), body); },
    async getExperiment(experimentId: string): Promise<Experiment> { return fetchJson(url(`/api/bench/experiments/${encodeURIComponent(experimentId)}`)); },
    async updateExperiment(experimentId: string, body): Promise<Experiment> { const r = await fetch(url(`/api/bench/experiments/${encodeURIComponent(experimentId)}`), { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); if (!r.ok) throw new Error(await errorMessage(r)); return r.json() as Promise<Experiment>; },
    async launchExperiment(experimentId: string): Promise<LaunchResponse> { return postJson(url(`/api/bench/experiments/${encodeURIComponent(experimentId)}/runs`), {}); },
    async getRunCells(runId: string): Promise<ExperimentCell[]> { return fetchJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/cells`)); },
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
    async resumeRun(runId: string, nTrials: number, nWorkers?: number): Promise<LaunchResponse> {
      return postJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/resume`), {
        n_trials: nTrials,
        n_workers: nWorkers,
      });
    },
    async advanceBaseline(runId: string, nTrials?: number, nWorkers?: number): Promise<LaunchResponse> {
      return postJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/advance-baseline`), {
        n_trials: nTrials,
        n_workers: nWorkers,
      });
    },
    async getBenchKinds(): Promise<BenchKindInfo[]> {
      return fetchJson(url("/api/bench/kinds"));
    },
    async getSmac3Kinds(): Promise<Smac3GameInfo[]> {
      return fetchJson(url("/api/bench/smac3/kinds"));
    },
    async getRunTrials(runId: string, limit?: number): Promise<TrialRow[]> {
      return fetchJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/trials${queryString({ limit })}`));
    },
    async getRunChain(runId: string): Promise<ChainRung[]> {
      return fetchJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/chain`));
    },
    async getRunGames(runId: string, limit?: number): Promise<GameTraceSummary[]> {
      return fetchJson(url(`/api/bench/runs/${encodeURIComponent(runId)}/games${queryString({ limit })}`));
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
    listProjects: () => lift(() => api.listProjects()),
    createProject: (name: string, description: string) => lift(() => api.createProject(name, description)),
    getProject: (projectId: string) => lift(() => api.getProject(projectId)),
    updateProject: (projectId: string, body) => lift(() => api.updateProject(projectId, body)),
    listExperiments: (projectId: string) => lift(() => api.listExperiments(projectId)),
    createExperiment: (projectId: string, body) => lift(() => api.createExperiment(projectId, body)),
    getExperiment: (experimentId: string) => lift(() => api.getExperiment(experimentId)),
    updateExperiment: (experimentId: string, body) => lift(() => api.updateExperiment(experimentId, body)),
    launchExperiment: (experimentId: string) => lift(() => api.launchExperiment(experimentId)),
    getRunCells: (runId: string) => lift(() => api.getRunCells(runId)),
    getRun: (runId: string) => lift(() => api.getRun(runId)),
    getRunLog: (runId: string, since: number) => lift(() => api.getRunLog(runId, since)),
    getRunStdout: (runId: string) => lift(() => api.getRunStdout(runId)),
    getLeaderboard: (filters: LeaderboardFilters) => lift(() => api.getLeaderboard(filters)),
    fetchCommitTrends: (game: string | null) => lift(() => api.fetchCommitTrends(game)),
    launchRun: (kind: string, game: string, config?: unknown) => lift(() => api.launchRun(kind, game, config)),
    stopRun: (runId: string) => lift(() => api.stopRun(runId)),
    resumeRun: (runId: string, nTrials: number, nWorkers?: number) =>
      lift(() => api.resumeRun(runId, nTrials, nWorkers)),
    advanceBaseline: (runId: string, nTrials?: number, nWorkers?: number) =>
      lift(() => api.advanceBaseline(runId, nTrials, nWorkers)),
    getBenchKinds: () => lift(() => api.getBenchKinds()),
    getSmac3Kinds: () => lift(() => api.getSmac3Kinds()),
    getRunTrials: (runId: string, limit?: number) => lift(() => api.getRunTrials(runId, limit)),
    getRunChain: (runId: string) => lift(() => api.getRunChain(runId)),
    getRunGames: (runId: string, limit?: number) => lift(() => api.getRunGames(runId, limit)),
    getRunGameMoves: (runId: string, gameSeq: number) => lift(() => api.getRunGameMoves(runId, gameSeq)),
    deleteRun: (runId: string) => lift(() => api.deleteRun(runId)),
  };
}
