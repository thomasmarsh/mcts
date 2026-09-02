// tuner-api-client.ts — the only file in the tuner UI allowed to call
// `fetch` (eslint.config.js's fetch ban whitelists `**/tuner-api-client.ts`
// alongside `api-client.ts`). One method per version-4 tuner route; typed
// request/response from `tuner-types.ts`. `tuner-env.ts` lifts each method
// to an `Effect`; the reducer and components only ever touch the env.

import type {
  ProjectionCandidate,
  ProjectionCohort,
  ProjectionPairQuery,
  ProjectionPairRow,
  ProjectionRefreshResult,
  ProjectionRunDetail,
  ProjectionRunListItem,
  ProjectionValidation,
  TunerBudgetExtension,
  TunerLaunchRequest,
  TunerObjectiveFile,
  TunerRunLog,
  TunerRunView,
} from "./tuner-types.js";
import type { JsonValue, TunerGameInfo } from "../types.js";

export interface TunerApiClient {
  // Launch metadata.
  listKinds(): Promise<TunerGameInfo[]>;
  listObjectives(): Promise<TunerObjectiveFile[]>;
  // Operational journal.
  listRuns(): Promise<TunerRunView[]>;
  getRun(runId: string): Promise<TunerRunView>;
  launchRun(body: TunerLaunchRequest): Promise<TunerRunView>;
  stopRun(runId: string): Promise<TunerRunView>;
  extendRun(runId: string, body: TunerBudgetExtension): Promise<TunerRunView>;
  getRunLog(runId: string, since?: number): Promise<TunerRunLog>;
  // Projection.
  refreshProjection(): Promise<ProjectionRefreshResult>;
  listProjectionRuns(): Promise<ProjectionRunListItem[]>;
  getProjectionRun(runId: string): Promise<ProjectionRunDetail>;
  getProjectionCohorts(runId: string): Promise<ProjectionCohort[]>;
  getProjectionCandidates(runId: string): Promise<ProjectionCandidate[]>;
  getProjectionCandidate(runId: string, candidateId: string): Promise<ProjectionCandidate>;
  getProjectionPairs(runId: string, query?: ProjectionPairQuery): Promise<ProjectionPairRow[]>;
  getProjectionValidation(runId: string): Promise<ProjectionValidation>;
  getProjectionReport(runId: string): Promise<JsonValue>;
}

class TunerApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "TunerApiError";
  }
}

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
        return (body as { error: string }).error;
      }
    } catch {
      // Not JSON -- fall through to raw text.
    }
  }
  return text || `API ${r.status}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new TunerApiError(await errorMessage(r), r.status);
  return r.json() as Promise<T>;
}

async function sendJson<T>(url: string, method: "POST", body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new TunerApiError(await errorMessage(r), r.status);
  return r.json() as Promise<T>;
}

function queryString(params: object): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (typeof v === "string" || typeof v === "number") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

/** `baseUrl` defaults to `""` (relative URLs), same convention as
 * `createBenchApiClient`. */
export function createTunerApiClient(baseUrl = ""): TunerApiClient {
  const url = (path: string): string => baseUrl + path;
  const runPath = (runId: string): string =>
    `/api/bench/tuner/runs/${encodeURIComponent(runId)}`;
  const projPath = (runId: string): string =>
    `/api/bench/tuner/projection/runs/${encodeURIComponent(runId)}`;
  return {
    listKinds: () => fetchJson(url("/api/bench/tuner/kinds")),
    listObjectives: () => fetchJson(url("/api/bench/tuner/objectives")),
    listRuns: () => fetchJson(url("/api/bench/tuner/runs")),
    getRun: (runId) => fetchJson(url(runPath(runId))),
    launchRun: (body) => sendJson(url("/api/bench/tuner/runs"), "POST", body),
    stopRun: (runId) => sendJson(url(`${runPath(runId)}/stop`), "POST"),
    extendRun: (runId, body) => sendJson(url(`${runPath(runId)}/extend`), "POST", body),
    getRunLog: (runId, since) =>
      fetchJson(url(`${runPath(runId)}/log${queryString({ since })}`)),
    refreshProjection: () =>
      sendJson(url("/api/bench/tuner/projection/refresh"), "POST"),
    listProjectionRuns: () => fetchJson(url("/api/bench/tuner/projection/runs")),
    getProjectionRun: (runId) => fetchJson(url(projPath(runId))),
    getProjectionCohorts: (runId) => fetchJson(url(`${projPath(runId)}/cohorts`)),
    getProjectionCandidates: (runId) => fetchJson(url(`${projPath(runId)}/candidates`)),
    getProjectionCandidate: (runId, candidateId) =>
      fetchJson(url(`${projPath(runId)}/candidates/${encodeURIComponent(candidateId)}`)),
    getProjectionPairs: (runId, query = {}) =>
      fetchJson(url(`${projPath(runId)}/pairs${queryString(query)}`)),
    getProjectionValidation: (runId) => fetchJson(url(`${projPath(runId)}/validation`)),
    getProjectionReport: (runId) => fetchJson(url(`${projPath(runId)}/report`)),
  };
}
